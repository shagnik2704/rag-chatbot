import React from "react";
import { X, RefreshCw, Sliders, Key, Filter, CheckCircle2, AlertTriangle } from "lucide-react";
import type { SystemStatus } from "../types/chat";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  status: SystemStatus | null;
  sections: string[];
  selectedSection: string;
  onSelectSection: (section: string) => void;
  topK: number;
  onTopKChange: (k: number) => void;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
  onReindex: () => Promise<void>;
  isReindexing: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  status,
  sections,
  selectedSection,
  onSelectSection,
  topK,
  onTopKChange,
  apiKey,
  onApiKeyChange,
  onReindex,
  isReindexing,
}) => {
  return (
    <>
      <div
        className={`sidebar-overlay ${isOpen ? "active" : ""}`}
        onClick={onClose}
      />
      <aside className={`sidebar ${isOpen ? "open" : ""}`}>
        <div className="sidebar-header">
          <span className="sidebar-title">System Settings</span>
          <button
            className="menu-toggle-btn"
            onClick={onClose}
            aria-label="Close sidebar"
          >
            <X size={18} />
          </button>
        </div>

        <div className="sidebar-content">
          <div className="config-group">
            <span className="config-label">System Status</span>
            <div className="status-badge-container">
              <span
                className={`status-dot ${
                  status?.status === "ready" ? "active" : "inactive"
                }`}
              />
              <span>
                {status?.status === "ready" ? "Index Ready" : "Unindexed / Empty"}
              </span>
            </div>
            <div className="status-badge-container">
              {status?.has_api_key || apiKey ? (
                <>
                  <CheckCircle2 size={14} color="#16A34A" />
                  <span>Sarvam API Key Configured</span>
                </>
              ) : (
                <>
                  <AlertTriangle size={14} color="#D97706" />
                  <span>Running in Preview Mode</span>
                </>
              )}
            </div>
          </div>

          <div className="config-group">
            <label className="config-label" htmlFor="apiKeyInput">
              <Key size={12} style={{ display: "inline", marginRight: 4 }} />
              Sarvam AI API Key
            </label>
            <input
              id="apiKeyInput"
              type="password"
              className="input-field"
              placeholder="Paste custom API key..."
              value={apiKey}
              onChange={(e) => onApiKeyChange(e.target.value)}
            />
          </div>

          <div className="config-group">
            <label className="config-label" htmlFor="sectionSelect">
              <Filter size={12} style={{ display: "inline", marginRight: 4 }} />
              Filter by Section
            </label>
            <select
              id="sectionSelect"
              className="select-field"
              value={selectedSection}
              onChange={(e) => onSelectSection(e.target.value)}
            >
              <option value="">All Document Sections</option>
              {sections.map((sec) => (
                <option key={sec} value={sec}>
                  {sec}
                </option>
              ))}
            </select>
          </div>

          <div className="config-group">
            <div className="slider-labels">
              <span className="config-label">
                <Sliders size={12} style={{ display: "inline", marginRight: 4 }} />
                Retrieval Depth (k)
              </span>
              <span>{topK} chunks</span>
            </div>
            <input
              type="range"
              className="range-slider"
              min={1}
              max={8}
              value={topK}
              onChange={(e) => onTopKChange(Number(e.target.value))}
            />
          </div>

          <div className="config-group" style={{ marginTop: "auto" }}>
            <button
              className="btn-secondary"
              onClick={onReindex}
              disabled={isReindexing}
            >
              <RefreshCw
                size={14}
                className={isReindexing ? "spinner" : ""}
              />
              {isReindexing ? "Re-indexing..." : "Rebuild Index from Document"}
            </button>
          </div>
        </div>
      </aside>
    </>
  );
};
