import React, { useState } from "react";
import { Mic, Copy, Check } from "lucide-react";

interface TalkingPointCardProps {
  talkingPoint: string;
}

export const TalkingPointCard: React.FC<TalkingPointCardProps> = ({ talkingPoint }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(talkingPoint);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback ignore
    }
  };

  return (
    <div className="talking-point-container">
      <div className="talking-point-header">
        <span className="talking-point-label">
          <Mic size={14} />
          Champion Talking Point
        </span>
        <button
          className="copy-button"
          onClick={handleCopy}
          aria-label="Copy talking point"
        >
          {copied ? (
            <>
              <Check size={12} color="#16A34A" />
              <span>Copied</span>
            </>
          ) : (
            <>
              <Copy size={12} />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>
      <p className="talking-point-text">{talkingPoint}</p>
    </div>
  );
};
