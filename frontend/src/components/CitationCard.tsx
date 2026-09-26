import React, { useState } from "react";
import { BookOpen, ChevronDown, ChevronUp } from "lucide-react";
import type { Citation } from "../types/chat";

interface CitationCardProps {
  citations: Citation[];
}

export const CitationCard: React.FC<CitationCardProps> = ({ citations }) => {
  const [isOpen, setIsOpen] = useState(false);

  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div className="citations-wrapper">
      <button
        className="citations-toggle"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
      >
        <BookOpen size={14} />
        <span>Verified Citations ({citations.length})</span>
        {isOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {isOpen && (
        <div className="citations-list">
          {citations.map((c, index) => {
            const qTitle = c.question_number
              ? `Q${c.question_number}: ${c.question_text || ""}`
              : "Overview & Guidelines";

            return (
              <div key={`${c.chunk_id}-${index}`} className="citation-item">
                <div className="citation-header-row">
                  <span className="citation-source-tag">{c.section}</span>
                  <span className="citation-q-number">{c.chunk_id}</span>
                </div>
                <div style={{ fontWeight: 600, fontSize: "0.85rem", marginBottom: "0.25rem" }}>
                  {qTitle}
                </div>
                <p className="citation-excerpt">{c.excerpt}</p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
