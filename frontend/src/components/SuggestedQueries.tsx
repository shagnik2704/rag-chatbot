import React from "react";

interface SuggestedQueriesProps {
  onSelectQuery: (query: string) => void;
}

const SAMPLE_QUERIES = [
  "Can I specify a particular school?",
  "Will my contribution cover costs incurred by the schools in supporting this project?",
  "Why can’t I pay in INR even though I have a bank account in India?",
  "What courses will you offer? How many?",
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
