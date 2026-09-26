"""Unit tests for SarvamGLMClient."""

from unittest.mock import MagicMock, patch
import httpx
import pytest
from src.core.exceptions import LLMClientError
from src.llm.sarvam_client import SarvamGLMClient


def test_sarvam_client_missing_key_raises_error():
    client = SarvamGLMClient(api_key="")
    with pytest.raises(LLMClientError, match="SARVAM_API_KEY is not configured"):
        client.generate("Hello")


@patch("httpx.Client.post")
def test_sarvam_client_successful_generation(mock_post):
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "**Direct Answer**:\nSponsoring a school costs ₹35,000 / US$365.\n\n**Champion Talking Point**:\nWith ₹35,000, we empower 500 children."
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    client = SarvamGLMClient(api_key="valid-test-key-123", model="glm5.3")
    result = client.generate("How much does a school cost?", system_prompt="System prompt")

    assert "₹35,000" in result
    assert mock_post.called

    # Verify headers and payload
    call_kwargs = mock_post.call_args.kwargs
    headers = call_kwargs["headers"]
    assert headers["api-subscription-key"] == "valid-test-key-123"
    assert call_kwargs["json"]["model"] == "glm5.3"
