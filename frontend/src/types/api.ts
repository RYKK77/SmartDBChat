export interface ProviderPreset {
  name: string;
  base_url: string;
  default_model: string;
  models: string[];
  requires_api_key: boolean;
}

export interface ConnectionMeta {
  db_type?: string;
  database?: string;
  host?: string;
  provider?: string;
  model?: string;
  base_url?: string;
  dialect?: string;
  embedding_available?: boolean;
  has_api_key?: boolean;
}

export interface EngineConfig {
  query_mode: 'agent' | 'linked' | 'pipeline';
  agent_max_steps: number;
  enable_explain_check: boolean;
  max_retries: number;
  high_accuracy_mode: boolean;
  num_candidates: number;
  enable_preflight: boolean;
  enable_ast_schema_check: boolean;
  max_limit: number;
  similarity_threshold: number;
  selected_module: string | null;
  model: string;
}

export interface ConnectPayload {
  db_type: 'sandbox' | 'mysql';
  host: string;
  port: number;
  user: string;
  password: string;
  database: string;
  provider: string;
  api_key: string;
  base_url: string;
  model: string;
  embedding_mode: 'openai_compatible' | 'ollama_legacy';
  embedding_url: string;
  embedding_model: string;
  similarity_threshold: number;
  query_mode: 'agent' | 'linked' | 'pipeline';
  agent_max_steps: number;
  max_retries: number;
  high_accuracy_mode: boolean;
  num_candidates: number;
  enable_preflight: boolean;
  enable_ast_schema_check: boolean;
  enable_explain_check: boolean;
  max_limit: number;
  semantic_yaml_path: string;
}

export interface ChartItem {
  type: 'bar' | 'line' | 'pie' | 'area' | 'scatter' | 'radar' | 'funnel' | 'treemap' | 'heatmap' | string;
  title?: string;
  description?: string;
  data: Record<string, any>;
}

export interface ChartConfig {
  charts: ChartItem[];
}

export interface CacheHit {
  sql: string;
  matched_query: string;
  similarity: number;
  match_type: string;
}

export interface ExecutionAttempt {
  attempt: number;
  sql: string;
  row_count: number;
  error?: string | null;
}

export interface AgentStep {
  step: number;
  thought: string;
  tool_name?: string | null;
  tool_args?: Record<string, any> | null;
  observation: string;
  error?: string | null;
  duration_ms: number;
}

export interface ScalingCandidate {
  strategy: string;
  sql: string;
  is_valid: boolean;
  row_count: number;
  error?: string | null;
}

export interface ScalingInfo {
  consensus_votes: number;
  total_valid: number;
  candidates: ScalingCandidate[];
}

export interface QueryResultPayload {
  status: string;
  is_success: boolean;
  sql: string;
  rows: Record<string, any>[];
  columns: string[];
  row_count: number;
  markdown_answer: string;
  clean_markdown: string;
  chart_config: ChartConfig | null;
  recommended_chart_config: ChartConfig | null;
  cache_hit: CacheHit | null;
  attempts: ExecutionAttempt[];
  error?: string | null;
  linked_tables: string[];
  agent_steps: AgentStep[];
  mode: string;
  preflight_status: string;
  follow_up_suggestions: string[];
  workflow_trace: Record<string, any>[];
  scaling_info: ScalingInfo | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  result?: QueryResultPayload;
  elapsedSec?: number;
  statusLogs?: Array<{ level: string; message: string }>;
  streamingText?: string;
  isStreaming?: boolean;
}

export interface SchemaColumn {
  name: string;
  data_type: string;
  comment: string;
  is_primary_key: boolean;
  is_nullable: boolean;
}

export interface SchemaNode {
  name: string;
  comment: string;
  ddl: string;
  columns: SchemaColumn[];
}

export interface SchemaEdge {
  from_table: string;
  from_column: string;
  to_table: string;
  to_column: string;
  inferred: boolean;
}

export interface SchemaGraphResponse {
  nodes: SchemaNode[];
  edges: SchemaEdge[];
  table_count: number;
  edge_count: number;
}

export interface SemanticLayerData {
  database_description: string;
  user_rules: string[];
  metrics: Array<{
    name: string;
    expression: string;
    description?: string;
    aliases?: string[];
    tables?: string[];
  }>;
  glossary: Record<string, string>;
  table_aliases: Record<string, string[]>;
  table_descriptions: Record<string, string>;
  column_descriptions: Record<string, Record<string, string>>;
  column_enums: Record<string, Record<string, string>>;
  golden_examples: Array<{
    question: string;
    sql: string;
    tables?: string[];
    tags?: string[];
  }>;
}

export interface CacheStats {
  total_entries: number;
  total_hits: number;
  top_queries: Array<{ query: string; hits: number }>;
}
