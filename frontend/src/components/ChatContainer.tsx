import React, { useEffect, useRef } from "react";
import type { ChatMessage } from "../types/chat";
import { MessageItem } from "./MessageItem";
import { EmptyState } from "./EmptyState";

interface ChatContainerProps {
  messages: ChatMessage[];
  isLoading: boolean;
  onSelectQuery: (query: string) => void;
}

export const ChatContainer: React.FC<ChatContainerProps> = ({
  messages,
  isLoading,
  onSelectQuery,
}) => {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <main className="chat-container">
      <div className="chat-content-limit">
        {messages.length === 0 ? (
          <EmptyState onSelectQuery={onSelectQuery} />
        ) : (
          messages.map((msg) => <MessageItem key={msg.id} message={msg} />)
        )}

        {isLoading && (
          <div className="message-row assistant">
            <div className="assistant-card">
              <div className="loading-row">
                <div className="spinner" />
                <span>Looking up verified campaign information...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </main>
  );
};
