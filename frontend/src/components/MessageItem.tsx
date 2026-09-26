import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../types/chat";
import { FallbackNotice } from "./FallbackNotice";

interface MessageItemProps {
  message: ChatMessage;
}

export const MessageItem: React.FC<MessageItemProps> = ({ message }) => {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="message-row user">
        <div className="user-bubble">{message.content}</div>
      </div>
    );
  }

  let cleanAnswer = message.content;

  // Clean any residual internal labels if present in legacy responses
  cleanAnswer = cleanAnswer
    .replace(/(?:^|\n)(?:#{1,3}\s*|\*\*)Champion Talking Point.*$/is, "")
    .replace(/(?:^|\n)(?:#{1,3}\s*|\*\*)Citations.*$/is, "")
    .replace(/^(?:#{1,3}\s*|\*\*)Direct Answer(?:\*\*)?[:\s]*/i, "")
    .trim();

  // Do not render an empty card if tokens have not arrived yet
  if (!cleanAnswer && !message.is_fallback) {
    return null;
  }

  return (
    <div className="message-row assistant">
      <div className="assistant-card">
        <div className="markdown-prose">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              a: ({ href, children }) => (
                <a href={href} target="_blank" rel="noopener noreferrer">
                  {children}
                </a>
              ),
            }}
          >
            {cleanAnswer}
          </ReactMarkdown>
        </div>

        {message.is_fallback && message.fallback_contacts && (
          <FallbackNotice contacts={message.fallback_contacts} />
        )}
      </div>
    </div>
  );
};
