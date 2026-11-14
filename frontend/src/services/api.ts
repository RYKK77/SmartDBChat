import type {
  CacheStats,
  ConnectionMeta,
  ConnectPayload,
  EngineConfig,
  ProviderPreset,
  QueryResultPayload,
  SchemaGraphResponse,
  SemanticLayerData,
} from '../types/api'

const API_BASE = '/api'

async function requestJson<T>(url: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...(options?.headers || {}) },
    ...options,
  })
  if (!resp.ok) {
    let detail = `请求失败 (${resp.status})`
    try {
      const errBody = await resp.json()
      if (errBody?.detail) detail = String(errBody.detail)
    } catch {
      // ignore
    }
    throw new Error(detail)
  }
  return resp.json() as Promise<T>
}

export const api = {
  getHealth: () =>
    requestJson<{ ok: boolean; connected: boolean; connection_meta: ConnectionMeta }>('/health'),

  getProviders: () => requestJson<{ providers: ProviderPreset[] }>('/providers'),

  connect: (payload: ConnectPayload) =>
    requestJson<{ connected: boolean; connection_meta: ConnectionMeta; config: EngineConfig }>(
      '/connect',
      { method: 'POST', body: JSON.stringify(payload) },
    ),

  disconnect: () => requestJson<{ connected: boolean }>('/disconnect', { method: 'POST' }),

  getConfig: () => requestJson<EngineConfig>('/config'),

  updateConfig: (patch: Partial<EngineConfig>) =>
    requestJson<EngineConfig>('/config', { method: 'PUT', body: JSON.stringify(patch) }),

  getModules: () => requestJson<{ modules: string[] }>('/modules'),

  getSchemaGraph: () => requestJson<SchemaGraphResponse>('/schema/graph'),

  sampleTable: (tableName: string, limit = 5) =>
    requestJson<{ table: string; rows: Record<string, any>[] }>(
      `/schema/sample/${encodeURIComponent(tableName)}?limit=${limit}`,
    ),

  exploreColumnValues: (tableName: string, columnName: string, keyword = '', limit = 20) =>
    requestJson<{ table: string; column: string; keyword: string; values: any[] }>(
      `/schema/values/${encodeURIComponent(tableName)}/${encodeURIComponent(
        columnName,
      )}?keyword=${encodeURIComponent(keyword)}&limit=${limit}`,
    ),

  findJoinPath: (fromTable: string, toTable: string) =>
    requestJson<{ from_table: string; to_table: string; path: string[]; join_conditions: string[] }>(
      `/schema/join-path?from_table=${encodeURIComponent(fromTable)}&to_table=${encodeURIComponent(
        toTable,
      )}`,
    ),

  getSemanticLayer: () =>
    requestJson<{ data: SemanticLayerData; yaml_text: string; yaml_path: string }>('/semantic'),

  updateSemanticLayer: (payload: { yaml_text?: string; data?: SemanticLayerData }) =>
    requestJson<{ data: SemanticLayerData; yaml_text: string; yaml_path: string }>('/semantic', {
      method: 'PUT',
      body: JSON.stringify(payload),
    }),

  addGoldenExample: (payload: {
    question: string
    sql: string
    tables?: string[]
    tags?: string[]
  }) =>
    requestJson<{ count: number; data: SemanticLayerData }>('/semantic/golden', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  executeSql: (sql: string) =>
    requestJson<{
      sql: string
      rows: Record<string, any>[]
      columns: string[]
      row_count: number
      tables: string[]
      recommended_chart_config: QueryResultPayload['recommended_chart_config']
    }>('/sql/execute', {
      method: 'POST',
      body: JSON.stringify({ sql }),
    }),

  getCacheStats: () => requestJson<CacheStats>('/cache/stats'),

  clearCache: () => requestJson<{ cleared: boolean }>('/cache', { method: 'DELETE' }),

  streamQuery: async (
    payload: {
      question: string
      mode?: string
      history?: Array<{ role: string; content: string }>
    },
    handlers: {
      onStatus?: (level: string, message: string) => void
      onToken?: (accumulated: string) => void
      onResult?: (result: QueryResultPayload) => void
      onError?: (message: string) => void
    },
  ) => {
    const resp = await fetch(`${API_BASE}/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!resp.ok || !resp.body) {
      let msg = `请求失败 (${resp.status})`
      try {
        const err = await resp.json()
        if (err?.detail) msg = String(err.detail)
      } catch {
        // ignore
      }
      throw new Error(msg)
    }

    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const chunks = buffer.split('\n\n')
      buffer = chunks.pop() || ''

      for (const chunk of chunks) {
        const line = chunk
          .split('\n')
          .find((l) => l.startsWith('data: '))
        if (!line) continue
        try {
          const evt = JSON.parse(line.slice(6))
          if (evt.type === 'status') {
            handlers.onStatus?.(evt.level || 'info', evt.message || '')
          } else if (evt.type === 'token') {
            handlers.onToken?.(evt.accumulated || '')
          } else if (evt.type === 'result') {
            handlers.onResult?.(evt.result)
          } else if (evt.type === 'error') {
            handlers.onError?.(evt.message || '未知错误')
          }
        } catch {
          // ignore malformed SSE packet
        }
      }
    }
  },
}
