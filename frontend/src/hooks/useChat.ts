import { useState, useEffect, useCallback } from "react";
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
  }, [refreshStatus]);

  const sendMessage = async (query: string) => {
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
    let accumulated = "";

    try {
      await streamQuery(
        {
          query,
          top_k: topK,
          filter_section: selectedSection || null,
          api_key: apiKey || undefined,
        },
        (token: string) => {
          accumulated += token;
          if (isFirstToken) {
            isFirstToken = false;
            setIsLoading(false);
            setMessages((prev) => [
              ...prev,
              {
                id: assistantId,
                role: "assistant",
                content: accumulated,
                timestamp: new Date().toLocaleTimeString(),
              },
            ]);
          } else {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId ? { ...msg, content: accumulated } : msg
              )
            );
          }
        },
        () => {
          setIsLoading(false);
          const lower = accumulated.toLowerCase();
          const isFallback =
            lower.includes("not covered in the") ||
            lower.includes("please reach out directly");

          if (isFallback) {
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantId
                  ? {
                      ...msg,
                      is_fallback: true,
                      fallback_contacts: [
                        "sujathan@wheelsglobal.org",
                        "saisudha@edupyramids.org",
                      ],
                    }
                  : msg
              )
            );
          }
        },
        (errText: string) => {
          setIsLoading(false);
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
        }
      );
    } catch (err: any) {
      setIsLoading(false);
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
    reindex,
    clearChat,
  };
}
