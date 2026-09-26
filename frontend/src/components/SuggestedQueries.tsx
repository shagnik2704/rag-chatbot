import React from "react";

interface SuggestedQueriesProps {
  onSelectQuery: (query: string) => void;
}

const SAMPLE_QUERIES = [
  "How much does it cost to support a school?",
  "What is the Spoken Tutorial pedagogy and IEEE standard?",
  "What is cYAAG and how does team participation work?",
  "What are the implementation numbers in Maharashtra and MP?",
];

export const SuggestedQueries: React.FC<SuggestedQueriesProps> = ({ onSelectQuery }) => {
  return (
    <div className="suggestions-grid">
      {SAMPLE_QUERIES.map((query) => (
        <button
          key={query}
          className="suggestion-card"
          onClick={() => onSelectQuery(query)}
        >
          {query}
        </button>
      ))}
    </div>
  );
};
