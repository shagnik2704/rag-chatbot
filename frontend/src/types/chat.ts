export interface Citation {
  chunk_id: string;
  section: string;
  question_number: number | null;
  question_text: string | null;
  excerpt: string;
}

export interface QueryResponse {
  query: string;
  answer: string;
  champion_talking_point: string | null;
  citations: Citation[];
  model_used: string;
  is_fallback: boolean;
  fallback_contacts: string[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  champion_talking_point?: string | null;
  citations?: Citation[];
  is_fallback?: boolean;
  fallback_contacts?: string[];
  timestamp: string;
}

export interface SystemStatus {
  status: "ready" | "empty" | "error";
  indexed_chunks: number;
  model_name: string;
  has_api_key: boolean;
}

export interface QueryParams {
  query: string;
  top_k?: number;
  filter_section?: string | null;
  generate_talking_points?: boolean;
  api_key?: string;
}
