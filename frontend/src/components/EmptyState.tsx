import React from "react";
import { SuggestedQueries } from "./SuggestedQueries";

interface EmptyStateProps {
  onSelectQuery: (query: string) => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ onSelectQuery }) => {
  return (
    <div className="empty-state">
      <div className="empty-state-logos">
        <img
          src="/wheels-logo.png"
          alt="WHEELS Global Foundation"
          className="empty-logo wheels-empty-logo"
        />
        <span className="empty-logo-divider" />
        <img
          src="/edupyramids-logo.png"
          alt="EduPyramids"
          className="empty-logo edupyramids-empty-logo"
        />
      </div>
      <h2 className="empty-state-title">Future-Ready Children</h2>
      <p className="empty-state-description">
        Query verified FAQs, implementation statistics, pedagogy standards, and talking points
        for the WHEELS Global Foundation and EduPyramids campaign.
      </p>
      <SuggestedQueries onSelectQuery={onSelectQuery} />
    </div>
  );
};
