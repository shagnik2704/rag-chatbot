import React, { useState } from "react";
import { Send, Square } from "lucide-react";

interface InputBarProps {
  onSendMessage: (query: string) => void;
  isLoading: boolean;
  onStop?: () => void;
}

export const InputBar: React.FC<InputBarProps> = ({
  onSendMessage,
  isLoading,
  onStop,
}) => {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (isLoading) {
      if (onStop) onStop();
      return;
    }
    const trimmed = input.trim();
    if (!trimmed) return;
    onSendMessage(trimmed);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <footer className="input-section">
      <form className="input-container" onSubmit={handleSubmit}>
        <input
          type="text"
          className="query-input"
          placeholder={
            isLoading
              ? "Generating answer... click stop to interrupt."
              : "Ask a question about campaign pedagogy, costs, or implementation..."
          }
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
        />
        {isLoading ? (
          <button
            type="button"
            className="send-btn stop-btn"
            onClick={onStop}
            aria-label="Stop generating"
            title="Stop generating"
            style={{ backgroundColor: "#b91c1c", color: "#ffffff" }}
          >
            <Square size={13} fill="currentColor" />
          </button>
        ) : (
          <button
            type="submit"
            className="send-btn"
            disabled={!input.trim()}
            aria-label="Send query"
          >
            <Send size={16} />
          </button>
        )}
      </form>
    </footer>
  );
};
