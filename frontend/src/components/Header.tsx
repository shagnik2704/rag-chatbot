import React from "react";
import { PlusCircle } from "lucide-react";

interface HeaderProps {
  onNewChat: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onNewChat }) => {
  return (
    <header className="top-header">
      <div className="header-left">
        <div className="header-logos">
          <img
            src="/wheels-logo.png"
            alt="WHEELS Global Foundation"
            className="header-logo wheels-header-logo"
          />
          <span className="header-logo-divider" />
          <img
            src="/edupyramids-logo.png"
            alt="EduPyramids"
            className="header-logo edupyramids-header-logo"
          />
        </div>
        <div className="header-title-container">
          <span className="header-main-title">Future-Ready Children</span>
          <span className="header-sub-title">
            WHEELS Global Foundation & EduPyramids
          </span>
        </div>
      </div>

      <div className="header-right">
        <button
          className="btn-secondary new-chat-btn"
          onClick={onNewChat}
          title="Start a new conversation"
        >
          <PlusCircle size={15} />
          <span className="btn-label-desktop">New Conversation</span>
          <span className="btn-label-mobile">New</span>
        </button>
      </div>
    </header>
  );
};
