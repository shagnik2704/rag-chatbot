import React from "react";
import { PlusCircle } from "lucide-react";

interface HeaderProps {
  onNewChat: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onNewChat }) => {
  return (
    <header className="top-header">
      <div className="header-left">
        <div className="header-title-container">
          <span className="header-main-title">Future-Ready Children</span>
          <span className="header-sub-title">
            WHEELS Global Foundation & EduPyramids
          </span>
        </div>
      </div>

      <div className="header-right">
        <button
          className="btn-secondary"
          onClick={onNewChat}
          title="Start a new conversation"
        >
          <PlusCircle size={15} />
          <span>New Conversation</span>
        </button>
      </div>
    </header>
  );
};
