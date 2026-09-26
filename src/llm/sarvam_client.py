"""Resilient HTTP client for Sarvam AI inference API (GLM-5.3)."""

import json
import time
from typing import Any, Generator
import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.core.exceptions import LLMClientError
from src.core.logging import setup_logger
from src.llm.base import BaseLLMClient

logger = setup_logger(__name__)


def _is_retryable_error(exception: BaseException) -> bool:
    """Checks if an error is transient and should trigger a retry."""
    if isinstance(exception, (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on Rate Limit (429) and Server Errors (5xx)
        return exception.response.status_code in {429, 500, 502, 503, 504}
    return False


class SarvamGLMClient(BaseLLMClient):
    """Client for querying GLM-5.3 via Sarvam AI API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.sarvam.ai",
        model: str = "glm5.3",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        self._endpoint = f"{self.base_url}/v2/chat/completions"
        self._client = httpx.Client(timeout=self.timeout)
        self._async_client = async_client

    @property
    def model_name(self) -> str:
        return self._model

    def _build_payload_and_headers(
        self, prompt: str, system_prompt: str | None, stream: bool = False
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if not self.api_key or self.api_key.startswith("your_sarvam"):
            raise LLMClientError(
                "SARVAM_API_KEY is not configured or contains placeholder text. "
                "Please add your active Sarvam AI key in the .env file."
            )

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if stream:
            payload["stream"] = True

        headers = {
            "api-subscription-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return payload, headers

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Executes synchronous inference call to Sarvam AI."""
        payload, headers = self._build_payload_and_headers(prompt, system_prompt, stream=False)

        try:
            return self._call_api_with_retry(payload, headers)
        except Exception as e:
            if isinstance(e, LLMClientError):
                raise
            raise LLMClientError(f"Failed to generate response from Sarvam AI: {e}") from e

    async def agenerate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Executes asynchronous inference call to Sarvam AI using connection pool."""
        payload, headers = self._build_payload_and_headers(prompt, system_prompt, stream=False)
        client = self._async_client or httpx.AsyncClient(timeout=self.timeout)
        should_close = self._async_client is None

        try:
            response = await client.post(self._endpoint, json=payload, headers=headers)
            if response.status_code != 200:
                raise LLMClientError(f"Sarvam API error: HTTP {response.status_code} - {response.text}")
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if isinstance(e, LLMClientError):
                raise
            raise LLMClientError(f"Failed async generation from Sarvam AI: {e}") from e
        finally:
            if should_close:
                await client.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=10),
        retry=retry_if_exception(_is_retryable_error),
    )
    def _call_api_with_retry(self, payload: dict[str, Any], headers: dict[str, str]) -> str:
        logger.debug(f"Sending request to Sarvam endpoint: {self._endpoint} (model: {self._model})")
        response = self._client.post(self._endpoint, json=payload, headers=headers)

        if response.status_code != 200:
            logger.error(
                f"Sarvam API error: HTTP {response.status_code} - {response.text}"
            )
            response.raise_for_status()

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
            return content.strip()
        except (KeyError, IndexError) as err:
            raise LLMClientError(
                f"Unexpected response format from Sarvam API: {data}"
            ) from err

    def generate_stream(
        self, prompt: str, system_prompt: str | None = None
    ) -> Generator[str, None, None]:
        """Streams text tokens synchronously."""
        payload, headers = self._build_payload_and_headers(prompt, system_prompt, stream=True)

        try:
            with self._client.stream("POST", self._endpoint, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    err_text = response.read().decode("utf-8")
                    raise LLMClientError(
                        f"Sarvam API streaming error: HTTP {response.status_code} - {err_text}"
                    )
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_str)
                            choices = parsed.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    yield token
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            if isinstance(e, LLMClientError):
                raise
            raise LLMClientError(f"Failed during streaming generation from Sarvam: {e}") from e

    async def agenerate_stream(
        self, prompt: str, system_prompt: str | None = None
    ):
        """Asynchronously streams text tokens from Sarvam AI without blocking the event loop."""
        payload, headers = self._build_payload_and_headers(prompt, system_prompt, stream=True)
        client = self._async_client or httpx.AsyncClient(timeout=self.timeout)
        should_close = self._async_client is None

        try:
            async with client.stream("POST", self._endpoint, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    err_text = (await response.aread()).decode("utf-8")
                    raise LLMClientError(
                        f"Sarvam API async streaming error: HTTP {response.status_code} - {err_text}"
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            parsed = json.loads(data_str)
                            choices = parsed.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content", "")
                                if token:
                                    yield token
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            if isinstance(e, LLMClientError):
                raise
            raise LLMClientError(f"Failed during async streaming from Sarvam: {e}") from e
        finally:
            if should_close:
                await client.aclose()

    def close(self) -> None:
        """Closes underlying HTTP client."""
        self._client.close()


class MockLLMClient(BaseLLMClient):
    """Mock client used for automated unit tests and offline testing."""

    def __init__(self, model_name: str = "mock-glm5.3", canned_response: str | None = None) -> None:
        self._model_name = model_name
        self.canned_response = canned_response

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        if self.canned_response:
            return self.canned_response
        return (
            "A contribution of **₹35,000 / US$365** can support approximately **500 children** in one school.\n\n"
            "Your contribution provides children with access to structured digital-skills learning through the Spoken Tutorial approach."
        )

    async def agenerate(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.generate(prompt, system_prompt)

    def generate_stream(
        self, prompt: str, system_prompt: str | None = None
    ) -> Generator[str, None, None]:
        full_text = self.generate(prompt, system_prompt)
        words = full_text.split(" ")
        for word in words:
            time.sleep(0.02)
            yield word + " "

    async def agenerate_stream(
        self, prompt: str, system_prompt: str | None = None
    ):
        import asyncio
        full_text = self.generate(prompt, system_prompt)
        words = full_text.split(" ")
        for word in words:
            await asyncio.sleep(0.015)
            yield word + " "

