import { useState, useEffect, useCallback, useRef } from "react";
import type { ChatMessage, SystemStatus } from "../types/chat";
import { fetchSystemStatus, fetchSections, streamQuery, triggerReindex } from "../api/client";

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isReindexing, setIsReindexing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [sections, setSections] = useState<string[]>([]);
  const [selectedSection, setSelectedSection] = useState<string>("");
  const [topK, setTopK] = useState<number>(4);
  const [apiKey, setApiKey] = useState<string>("");

  const abortControllerRef = useRef<AbortController | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const [sysStatus, secList] = await Promise.all([
        fetchSystemStatus(),
        fetchSections(),
      ]);
      setStatus(sysStatus);
      setSections(secList);
    } catch (err) {
      console.error(err);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [refreshStatus]);

  const stopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsLoading(false);
  };

  const sendMessage = async (query: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: query,
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);
    setError(null);

    const assistantId = `assistant-${Date.now()}`;
    let isFirstToken = true;
    let targetText = "";
    let displayedText = "";

    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    // Fluid 60fps typewriter smoothing queue
    timerRef.current = setInterval(() => {
      if (displayedText.length < targetText.length) {
        const lag = targetText.length - displayedText.length;
        const step = lag > 100 ? 6 : lag > 40 ? 4 : lag > 15 ? 2 : 1;
        displayedText = targetText.slice(0, displayedText.length + step);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId ? { ...msg, content: displayedText } : msg
          )
        );
      }
    }, 16);

    try {
      await streamQuery(
        {
          query,
          top_k: topK,
          filter_section: selectedSection || null,
          api_key: apiKey || undefined,
        },
        (token: string) => {
          targetText += token;
          if (isFirstToken) {
            isFirstToken = false;
            setIsLoading(false);
            displayedText = targetText.slice(0, 1);
            setMessages((prev) => [
              ...prev,
              {
                id: assistantId,
                role: "assistant",
                content: displayedText,
                timestamp: new Date().toLocaleTimeString(),
              },
            ]);
          }
        },
        () => {
          if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
          }
          displayedText = targetText;
          setIsLoading(false);
          abortControllerRef.current = null;
          const lower = targetText.toLowerCase();
          const isFallback =
            lower.includes("not covered in the") ||
            lower.includes("please reach out directly");

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    content: targetText,
                    ...(isFallback
                      ? {
                          is_fallback: true,
                          fallback_contacts: [
                            "sujathan@wheelsglobal.org",
                            "saisudha@edupyramids.org",
                          ],
                        }
                      : {}),
                  }
                : msg
            )
          );
        },
        (errText: string) => {
          if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
          }
          setIsLoading(false);
          abortControllerRef.current = null;
          setError(errText);
          setMessages((prev) => [
            ...prev,
            {
              id: assistantId,
              role: "assistant",
              content: `Error: ${errText}`,
              timestamp: new Date().toLocaleTimeString(),
            },
          ]);
        },
        controller.signal
      );
    } catch (err: any) {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      if (err.name === "AbortError") {
        setIsLoading(false);
        abortControllerRef.current = null;
        return;
      }
      setIsLoading(false);
      abortControllerRef.current = null;
      setError(err.message);
      setMessages((prev) => [
        ...prev,
        {
          id: assistantId,
          role: "assistant",
          content: `Error: ${err.message || "Failed to communicate with service."}`,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    }
  };

  const reindex = async () => {
    setIsReindexing(true);
    try {
      await triggerReindex();
      await refreshStatus();
    } catch (err: any) {
      setError(`Re-indexing failed: ${err.message}`);
    } finally {
      setIsReindexing(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  return {
    messages,
    isLoading,
    isReindexing,
    error,
    status,
    sections,
    selectedSection,
    setSelectedSection,
    topK,
    setTopK,
    apiKey,
    setApiKey,
    sendMessage,
    stopStreaming,
    reindex,
    clearChat,
  };
}
