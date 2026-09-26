import React from "react";
import { AlertCircle, Mail } from "lucide-react";

interface FallbackNoticeProps {
  contacts: string[];
}

export const FallbackNotice: React.FC<FallbackNoticeProps> = ({ contacts }) => {
  return (
    <div className="fallback-alert">
      <AlertCircle size={18} style={{ flexShrink: 0, marginTop: 2 }} />
      <div className="fallback-alert-content">
        <div>
          This inquiry requires information beyond the current Champion FAQ document.
        </div>
        <div style={{ fontSize: "0.8rem", color: "var(--color-text-secondary)" }}>
          Official campaign escalation contacts:
        </div>
        <div className="fallback-contacts">
          {contacts.map((contact) => (
            <div key={contact} style={{ display: "flex", alignItems: "center", gap: "0.3rem" }}>
              <Mail size={12} />
              <a
                href={`mailto:${contact}`}
                style={{ color: "var(--color-primary)", textDecoration: "underline" }}
              >
                {contact}
              </a>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
