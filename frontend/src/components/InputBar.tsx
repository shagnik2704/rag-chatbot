import React, { useState } from "react";
import { Send } from "lucide-react";

interface InputBarProps {
  onSendMessage: (query: string) => void;
  isLoading: boolean;
}

export const InputBar: React.FC<InputBarProps> = ({ onSendMessage, isLoading }) => {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
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
          placeholder="Ask a question about campaign pedagogy, costs, or implementation..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
        />
        <button
          type="submit"
          className="send-btn"
          disabled={!input.trim() || isLoading}
          aria-label="Send query"
        >
          <Send size={16} />
        </button>
      </form>
    </footer>
  );
};
