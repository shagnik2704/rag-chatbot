import type { QueryParams, QueryResponse, SystemStatus } from "../types/chat";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) {
    throw new Error(`Failed to fetch system status: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchSections(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/sections`);
  if (!res.ok) {
    throw new Error(`Failed to fetch sections: ${res.statusText}`);
  }
  const data = await res.json();
  return data.sections;
}

export async function sendQuery(params: QueryParams): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Server error: ${res.status}`);
  }

  return res.json();
}

export async function triggerReindex(): Promise<{ message: string; total_chunks: number }> {
  const res = await fetch(`${API_BASE}/reindex`, {
    method: "POST",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Re-indexing failed: ${res.status}`);
  }

  return res.json();
}

export async function streamQuery(
  params: QueryParams,
  onToken: (token: string) => void,
  onComplete: () => void,
  onError: (err: string) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Server error: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) {
    throw new Error("Streaming not supported in browser environment.");
  }

  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.startsWith("data: ")) continue;

      const dataStr = trimmed.slice(6).trim();
      if (dataStr === "[DONE]") {
        onComplete();
        return;
      }

      try {
        const parsed = JSON.parse(dataStr);
        if (parsed.error) {
          onError(parsed.error);
          return;
        }
        if (parsed.token) {
          onToken(parsed.token);
        }
      } catch {
        // Continue parsing subsequent lines
      }
    }
  }

  onComplete();
}
