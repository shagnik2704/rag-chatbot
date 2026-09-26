import React from "react";
import { FileText } from "lucide-react";
import { SuggestedQueries } from "./SuggestedQueries";

interface EmptyStateProps {
  onSelectQuery: (query: string) => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ onSelectQuery }) => {
  return (
    <div className="empty-state">
      <FileText className="empty-state-icon" strokeWidth={1.5} />
      <h2 className="empty-state-title">Future-Ready Children Knowledge Base</h2>
      <p className="empty-state-description">
        Query verified FAQs, implementation statistics, pedagogy standards, and talking points
        for the WHEELS Global Foundation and EduPyramids campaign.
      </p>
      <SuggestedQueries onSelectQuery={onSelectQuery} />
    </div>
  );
};
