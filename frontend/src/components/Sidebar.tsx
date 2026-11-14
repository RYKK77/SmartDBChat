import {
  Database,
  Layers,
  MessageSquare,
  PlugZap,
  Plus,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Table2,
  Trash2,
  Unplug,
} from 'lucide-react'
import type {
  CacheStats,
  ConnectionMeta,
  EngineConfig,
  SchemaGraphResponse,
} from '../types/api'
import type { ChatSession } from '../utils/storage'

interface SidebarProps {
  collapsed: boolean
  connected: boolean
  connectionMeta: ConnectionMeta
  config: EngineConfig | null
  modules: string[]
  cacheStats: CacheStats | null
  schemaGraph: SchemaGraphResponse | null
  selectedTable: string | null
  sessions: ChatSession[]
  activeSessionId: string | null
  onNewSession: () => void
  onSelectSession: (id: string) => void
  onDeleteSession: (id: string) => void
  onOpenConnect: () => void
  onDisconnect: () => void
  onUpdateConfig: (patch: Partial<EngineConfig>) => void
  onRefreshCache: () => void
  onClearCache: () => void
  onSelectTable: (tableName: string) => void
}

export function Sidebar({
  collapsed,
  connected,
  connectionMeta,
  config,
  modules,
  cacheStats,
  schemaGraph,
  selectedTable,
  sessions,
  activeSessionId,
  onNewSession,
  onSelectSession,
  onDeleteSession,
  onOpenConnect,
  onDisconnect,
  onUpdateConfig,
  onRefreshCache,
  onClearCache,
  onSelectTable,
}: SidebarProps) {
  if (collapsed) return null

  return (
    <aside className="sidebar">
      {/* 对话记录 */}
      <div className="side-section">
        <div className="side-section-title">
          <span>对话记录</span>
          <button
            type="button"
            className="btn btn-sm"
            style={{ padding: '2px 7px', fontSize: 11.5 }}
            onClick={onNewSession}
            title="新建对话"
          >
            <Plus size={12} />
            新对话
          </button>
        </div>
        {sessions.length === 0 ? (
          <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>暂无历史对话</div>
        ) : (
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
              maxHeight: 168,
              overflowY: 'auto',
            }}
          >
            {sessions.map((sess) => {
              const isActive = sess.id === activeSessionId
              return (
                <div
                  key={sess.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    borderRadius: 'var(--radius-sm)',
                    border: `1px solid ${isActive ? 'var(--accent)' : 'transparent'}`,
                    background: isActive ? 'var(--accent-soft)' : 'transparent',
                    padding: '4px 6px',
                  }}
                >
                  <button
                    type="button"
                    onClick={() => onSelectSession(sess.id)}
                    style={{
                      flex: 1,
                      minWidth: 0,
                      border: 'none',
                      background: 'transparent',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      textAlign: 'left',
                      cursor: 'pointer',
                      padding: 0,
                      color: isActive ? 'var(--accent-strong)' : 'var(--text-primary)',
                      fontSize: 12,
                      fontWeight: isActive ? 600 : 400,
                    }}
                    title={sess.title}
                  >
                    <MessageSquare size={12} style={{ flexShrink: 0, opacity: 0.7 }} />
                    <span
                      style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {sess.title}
                    </span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onDeleteSession(sess.id)}
                    title="删除对话"
                    style={{
                      border: 'none',
                      background: 'transparent',
                      color: 'var(--text-muted)',
                      cursor: 'pointer',
                      padding: 2,
                      display: 'inline-flex',
                      flexShrink: 0,
                    }}
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* 连接状态 */}
      <div className="side-section">
        <div className="side-section-title">
          <span>数据源连接</span>
          {connected ? (
            <span className="pill pill-success" style={{ textTransform: 'none' }}>
              已连接
            </span>
          ) : (
            <span className="pill">未连接</span>
          )}
        </div>

        {connected ? (
          <>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12.5 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600 }}>
                <Database size={14} color="var(--accent)" />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {connectionMeta.database || 'smartdb'}
                </span>
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11.5 }}>
                方言: {(connectionMeta.dialect || 'sqlite').toUpperCase()} · {connectionMeta.host || 'local'}
              </div>
              <div style={{ color: 'var(--text-muted)', fontSize: 11.5 }}>
                模型: {config?.model || connectionMeta.model || '-'}
              </div>
              <div style={{ display: 'flex', gap: 6, marginTop: 2, flexWrap: 'wrap' }}>
                <span className={`pill ${connectionMeta.embedding_available ? 'pill-info' : ''}`}>
                  {connectionMeta.embedding_available ? '向量语义缓存就绪' : '精确文本缓存'}
                </span>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                type="button"
                className="btn btn-sm"
                style={{ flex: 1 }}
                onClick={onOpenConnect}
              >
                <PlugZap size={13} />
                重配连接
              </button>
              <button
                type="button"
                className="btn btn-sm"
                onClick={onDisconnect}
                title="断开当前连接"
              >
                <Unplug size={13} />
              </button>
            </div>
          </>
        ) : (
          <button type="button" className="btn btn-primary" onClick={onOpenConnect}>
            <PlugZap size={14} />
            连接数据库 / 启用沙箱
          </button>
        )}
      </div>

      {/* 引擎运行参数（热更新） */}
      {connected && config && (
        <div className="side-section">
          <div className="side-section-title">
            <span>推理与安全参数</span>
            <ShieldCheck size={14} color="var(--accent)" />
          </div>

          <div className="field-group">
            <label className="field-label">推理模式 (Query Mode)</label>
            <div className="segmented">
              {(['agent', 'linked', 'pipeline'] as const).map((m) => (
                <button
                  key={m}
                  type="button"
                  className={config.query_mode === m ? 'active' : ''}
                  onClick={() => onUpdateConfig({ query_mode: m })}
                >
                  {m === 'agent' ? 'Agent' : m === 'linked' ? 'Linked' : 'Pipeline'}
                </button>
              ))}
            </div>
          </div>

          {modules.length > 0 && (
            <div className="field-group">
              <label className="field-label">
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <Layers size={12} /> 业务模块范围
                </span>
              </label>
              <select
                className="select"
                value={config.selected_module || ''}
                onChange={(e) => onUpdateConfig({ selected_module: e.target.value })}
              >
                <option value="">全部模块</option>
                {modules.map((mod) => (
                  <option key={mod} value={mod}>
                    {mod}
                  </option>
                ))}
              </select>
            </div>
          )}

          <div
            style={{
              padding: 8,
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-subtle)',
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
            }}
          >
            <label
              className="toggle-row"
              title="Linked / Pipeline 模式下，生成多路候选 SQL 并执行一致性投票"
              style={{ opacity: config.query_mode === 'agent' ? 0.6 : 1 }}
            >
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                <Sparkles size={13} color="var(--accent)" />
                高精度多候选投票
              </span>
              <input
                type="checkbox"
                checked={config.high_accuracy_mode}
                disabled={config.query_mode === 'agent'}
                onChange={(e) => onUpdateConfig({ high_accuracy_mode: e.target.checked })}
              />
            </label>
            {config.query_mode === 'agent' ? (
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                多候选共识投票适用于 Linked / Pipeline 模式
              </div>
            ) : (
              config.high_accuracy_mode && (
                <div className="field-group">
                  <label className="field-label">
                    <span>候选 SQL 路数</span>
                    <span>{config.num_candidates} 路</span>
                  </label>
                  <div className="segmented">
                    {[1, 2, 3].map((n) => (
                      <button
                        key={n}
                        type="button"
                        className={config.num_candidates === n ? 'active' : ''}
                        onClick={() => onUpdateConfig({ num_candidates: n })}
                      >
                        {n} 路
                      </button>
                    ))}
                  </div>
                </div>
              )
            )}
          </div>

          <label className="toggle-row">
            <span>问前 Preflight 澄清门控</span>
            <input
              type="checkbox"
              checked={config.enable_preflight}
              onChange={(e) => onUpdateConfig({ enable_preflight: e.target.checked })}
            />
          </label>

          <label className="toggle-row">
            <span>AST 表/列幻觉校验</span>
            <input
              type="checkbox"
              checked={config.enable_ast_schema_check}
              onChange={(e) => onUpdateConfig({ enable_ast_schema_check: e.target.checked })}
            />
          </label>

          <label className="toggle-row">
            <span>Agent EXPLAIN 预检</span>
            <input
              type="checkbox"
              checked={config.enable_explain_check}
              onChange={(e) => onUpdateConfig({ enable_explain_check: e.target.checked })}
            />
          </label>

          <div className="grid-2">
            <div className="field-group">
              <label className="field-label">Agent 步数</label>
              <input
                className="input"
                type="number"
                min={2}
                max={15}
                value={config.agent_max_steps}
                onChange={(e) =>
                  onUpdateConfig({ agent_max_steps: Math.max(2, Math.min(15, Number(e.target.value) || 8)) })
                }
              />
            </div>
            <div className="field-group">
              <label className="field-label">自愈重试</label>
              <input
                className="input"
                type="number"
                min={0}
                max={5}
                value={config.max_retries}
                onChange={(e) =>
                  onUpdateConfig({ max_retries: Math.max(0, Math.min(5, Number(e.target.value) || 0)) })
                }
              />
            </div>
          </div>

          <div className="field-group">
            <label className="field-label">
              <span>最大返回行数 (0 = 不限)</span>
              <span>{config.max_limit || '不限'}</span>
            </label>
            <input
              className="input"
              type="number"
              min={0}
              step={100}
              value={config.max_limit}
              onChange={(e) =>
                onUpdateConfig({ max_limit: Math.max(0, Number(e.target.value) || 0) })
              }
            />
          </div>

          <div className="field-group">
            <label className="field-label">
              <span>缓存相似度阈值</span>
              <span>{config.similarity_threshold.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0.5}
              max={0.99}
              step={0.01}
              value={config.similarity_threshold}
              onChange={(e) =>
                onUpdateConfig({ similarity_threshold: parseFloat(e.target.value) })
              }
            />
          </div>
        </div>
      )}

      {/* 语义缓存统计 */}
      <div className="side-section">
        <div className="side-section-title">
          <span>语义缓存</span>
          <div style={{ display: 'flex', gap: 4 }}>
            <button
              type="button"
              className="icon-btn"
              style={{ width: 24, height: 24 }}
              onClick={onRefreshCache}
              title="刷新缓存统计"
            >
              <RefreshCw size={12} />
            </button>
            <button
              type="button"
              className="icon-btn"
              style={{ width: 24, height: 24 }}
              onClick={onClearCache}
              title="清空缓存"
            >
              <Trash2 size={12} />
            </button>
          </div>
        </div>
        <div className="grid-2">
          <div className="kpi-card" style={{ padding: '7px 10px' }}>
            <div className="kpi-label" style={{ fontSize: 11 }}>缓存条数</div>
            <div className="kpi-value" style={{ fontSize: 16 }}>
              {cacheStats?.total_entries ?? 0}
            </div>
          </div>
          <div className="kpi-card" style={{ padding: '7px 10px' }}>
            <div className="kpi-label" style={{ fontSize: 11 }}>累计命中</div>
            <div className="kpi-value" style={{ fontSize: 16 }}>
              {cacheStats?.total_hits ?? 0}
            </div>
          </div>
        </div>
      </div>

      {/* 快速数据表目录 */}
      {schemaGraph && schemaGraph.nodes.length > 0 && (
        <div className="side-section">
          <div className="side-section-title">
            <span>数据表目录 ({schemaGraph.table_count})</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {schemaGraph.nodes.map((n) => (
              <button
                key={n.name}
                type="button"
                className={`btn btn-sm ${selectedTable === n.name ? 'btn-primary' : 'btn-ghost'}`}
                style={{ justifyContent: 'space-between', width: '100%' }}
                onClick={() => onSelectTable(n.name)}
              >
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, overflow: 'hidden' }}>
                  <Table2 size={13} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{n.name}</span>
                </span>
                <span style={{ fontSize: 11, opacity: 0.75 }}>{n.columns.length} 列</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </aside>
  )
}
