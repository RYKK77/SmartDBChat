import { useEffect, useMemo, useState } from 'react'
import { Bookmark, ChevronDown, ChevronRight, Database, KeyRound, PlugZap, Server, Trash2, X } from 'lucide-react'
import { api } from '../services/api'
import type {
  ConnectionMeta,
  ConnectPayload,
  EngineConfig,
  ProviderPreset,
} from '../types/api'
import {
  deleteConnectionProfile,
  loadLastConnection,
  loadSavedProfiles,
  saveConnectionProfile,
  saveLastConnection,
  setAutoConnectDisabled,
  type SavedConnectionProfile,
} from '../utils/storage'

interface ConnectionModalProps {
  open: boolean
  onClose: () => void
  providers: ProviderPreset[]
  initialConfig?: EngineConfig | null
  initialMeta?: ConnectionMeta | null
  onConnected: (res: {
    connected: boolean
    connection_meta: ConnectionMeta
    config: EngineConfig
  }) => void
  onNotify: (msg: string, isError?: boolean) => void
}

const DEFAULT_PAYLOAD: ConnectPayload = {
  db_type: 'sandbox',
  host: 'localhost',
  port: 3306,
  user: 'root',
  password: '',
  database: '',
  provider: 'DeepSeek',
  api_key: '',
  base_url: 'https://api.deepseek.com/v1',
  model: 'deepseek-chat',
  embedding_mode: 'openai_compatible',
  embedding_url: 'http://localhost:11434/v1',
  embedding_model: 'nomic-embed-text',
  similarity_threshold: 0.85,
  query_mode: 'agent',
  agent_max_steps: 8,
  max_retries: 2,
  high_accuracy_mode: false,
  num_candidates: 3,
  enable_preflight: true,
  enable_ast_schema_check: true,
  enable_explain_check: true,
  max_limit: 1000,
  semantic_yaml_path: '',
}

export function ConnectionModal({
  open,
  onClose,
  providers,
  initialConfig,
  initialMeta,
  onConnected,
  onNotify,
}: ConnectionModalProps) {
  const [form, setForm] = useState<ConnectPayload>(() => loadLastConnection() || DEFAULT_PAYLOAD)
  const [savedProfiles, setSavedProfiles] = useState<SavedConnectionProfile[]>(() => loadSavedProfiles())
  const [activeProfileId, setActiveProfileId] = useState<string | null>(null)
  const [profileName, setProfileName] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (!open) return
    setSavedProfiles(loadSavedProfiles())
    const lastSaved = loadLastConnection()
    setForm((prev) => {
      const next = { ...DEFAULT_PAYLOAD, ...(lastSaved || prev) }
      if (providers.length > 0 && !providers.some((p) => p.name === next.provider)) {
        const first = providers[0]
        next.provider = first.name
        next.base_url = first.base_url
        next.model = first.default_model
      }
      if (initialConfig) {
        next.query_mode = initialConfig.query_mode
        next.agent_max_steps = initialConfig.agent_max_steps
        next.max_retries = initialConfig.max_retries
        next.high_accuracy_mode = initialConfig.high_accuracy_mode
        next.num_candidates = initialConfig.num_candidates
        next.enable_preflight = initialConfig.enable_preflight
        next.enable_ast_schema_check = initialConfig.enable_ast_schema_check
        next.enable_explain_check = initialConfig.enable_explain_check
        next.max_limit = initialConfig.max_limit
        next.similarity_threshold = initialConfig.similarity_threshold
        if (initialConfig.model) next.model = initialConfig.model
      }
      if (initialMeta?.db_type === 'mysql' || initialMeta?.db_type === 'sandbox') {
        next.db_type = initialMeta.db_type
      }
      if (initialMeta?.provider) {
        next.provider = initialMeta.provider
      }
      if (initialMeta?.base_url) {
        next.base_url = initialMeta.base_url
      }
      return next
    })
  }, [open, providers, initialConfig, initialMeta])

  const activeProvider = useMemo(
    () => providers.find((p) => p.name === form.provider) || null,
    [providers, form.provider],
  )

  if (!open) return null

  const handleProviderChange = (providerName: string) => {
    const preset = providers.find((p) => p.name === providerName)
    if (!preset) {
      setForm((prev) => ({ ...prev, provider: providerName }))
      return
    }
    setForm((prev) => ({
      ...prev,
      provider: preset.name,
      base_url: preset.base_url,
      model: preset.default_model || preset.models[0] || prev.model,
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (form.db_type === 'mysql' && !form.database.trim()) {
      onNotify('MySQL 模式下请填写数据库名', true)
      return
    }
    if (activeProvider?.requires_api_key && !form.api_key.trim()) {
      onNotify(`${form.provider} 需要填写 API Key（若使用本地模型可切换至 Ollama）`, true)
      return
    }
    setSubmitting(true)
    try {
      const res = await api.connect(form)
      saveLastConnection(form)
      setAutoConnectDisabled(false)
      if (activeProfileId) {
        const existing = savedProfiles.find((p) => p.id === activeProfileId)
        setSavedProfiles(saveConnectionProfile(existing?.name || '', form, activeProfileId))
      } else if (savedProfiles.length === 0) {
        const updated = saveConnectionProfile('', form)
        setSavedProfiles(updated)
        if (updated[0]) setActiveProfileId(updated[0].id)
      }
      onConnected(res)
      onNotify(
        form.db_type === 'sandbox'
          ? '已接入内置高校教务 SQLite 沙箱'
          : `已成功连接 MySQL 数据库: ${form.database}`,
      )
      onClose()
    } catch (err: any) {
      onNotify(err.message || '连接失败', true)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 9 }}>
            <div className="brand-icon" style={{ width: 28, height: 28 }}>
              <PlugZap size={15} />
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15 }}>数据源与模型连接配置</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                支持 1-Click 内置沙箱演示或自有 MySQL 数据库
              </div>
            </div>
          </div>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="关闭">
            <X size={16} />
          </button>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', minHeight: 0, flex: 1 }}>
          <div className="modal-body">
            {/* 已保存的连接配置管理 */}
            <div
              style={{
                background: 'var(--bg-subtle)',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)',
                padding: '10px 12px',
                display: 'flex',
                flexDirection: 'column',
                gap: 8,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                <span className="field-label" style={{ margin: 0, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                  <Bookmark size={12} color="var(--accent)" />
                  已保存的连接配置（本地持久化）
                </span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                  {savedProfiles.length} 个配置
                </span>
              </div>

              {savedProfiles.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {savedProfiles.map((prof) => {
                    const isSelected = activeProfileId === prof.id
                    return (
                      <div
                        key={prof.id}
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: 4,
                          borderRadius: 'var(--radius-sm)',
                          border: `1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
                          background: isSelected ? 'var(--accent-soft)' : 'var(--bg-surface)',
                          padding: '3px 6px',
                          fontSize: 12,
                        }}
                      >
                        <button
                          type="button"
                          onClick={() => {
                            setForm({ ...DEFAULT_PAYLOAD, ...prof.payload })
                            setActiveProfileId(prof.id)
                            setProfileName(prof.name)
                          }}
                          style={{
                            border: 'none',
                            background: 'transparent',
                            color: isSelected ? 'var(--accent-strong)' : 'var(--text-primary)',
                            fontWeight: isSelected ? 600 : 500,
                            cursor: 'pointer',
                            padding: 0,
                            fontSize: 12,
                          }}
                        >
                          {prof.name}
                        </button>
                        <button
                          type="button"
                          title="删除该配置"
                          onClick={() => {
                            setSavedProfiles(deleteConnectionProfile(prof.id))
                            if (activeProfileId === prof.id) {
                              setActiveProfileId(null)
                              setProfileName('')
                            }
                          }}
                          style={{
                            border: 'none',
                            background: 'transparent',
                            color: 'var(--text-muted)',
                            cursor: 'pointer',
                            display: 'inline-flex',
                            padding: 2,
                          }}
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>
                    )
                  })}
                </div>
              )}

              <div style={{ display: 'flex', gap: 6 }}>
                <input
                  className="input"
                  style={{ flex: 1, fontSize: 12, padding: '5px 8px' }}
                  value={profileName}
                  onChange={(e) => setProfileName(e.target.value)}
                  placeholder={`配置名称（默认: ${form.db_type === 'sandbox' ? '沙箱' : form.database || 'MySQL'} · ${form.provider}）`}
                />
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => {
                    const updated = saveConnectionProfile(
                      profileName,
                      form,
                      activeProfileId || undefined,
                    )
                    setSavedProfiles(updated)
                    saveLastConnection(form)
                    if (!activeProfileId && updated[0]) {
                      setActiveProfileId(updated[0].id)
                    }
                    onNotify('连接配置已保存到本地')
                  }}
                >
                  {activeProfileId ? '更新配置' : '保存配置'}
                </button>
                {activeProfileId && (
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() => {
                      const updated = saveConnectionProfile(profileName, form)
                      setSavedProfiles(updated)
                      if (updated[0]) setActiveProfileId(updated[0].id)
                      onNotify('已另存为新配置')
                    }}
                  >
                    另存为
                  </button>
                )}
              </div>
            </div>

            {/* 数据源切换 */}
            <div className="field-group">
              <label className="field-label">
                <span>数据源模式</span>
              </label>
              <div className="segmented" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <button
                  type="button"
                  className={form.db_type === 'sandbox' ? 'active' : ''}
                  onClick={() => setForm((p) => ({ ...p, db_type: 'sandbox' }))}
                >
                  内置高校教务 SQLite 沙箱 (Demo)
                </button>
                <button
                  type="button"
                  className={form.db_type === 'mysql' ? 'active' : ''}
                  onClick={() => setForm((p) => ({ ...p, db_type: 'mysql' }))}
                >
                  MySQL 数据库
                </button>
              </div>
            </div>

            {form.db_type === 'sandbox' ? (
              <div
                style={{
                  padding: '10px 12px',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--accent-soft)',
                  border: '1px solid var(--accent-border)',
                  fontSize: 12.5,
                  color: 'var(--text-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                }}
              >
                <Database size={16} color="var(--accent)" style={{ flexShrink: 0 }} />
                <span>
                  内置 7 张关联表（学院、学生、教师、课程、选课成绩、奖学金、校园消费流水）及预置语义层规则，无需本地数据库即可体验完整图谱与 Agent 分析。
                </span>
              </div>
            ) : (
              <div className="grid-2">
                <div className="field-group">
                  <label className="field-label">主机地址 (Host)</label>
                  <input
                    className="input"
                    value={form.host}
                    onChange={(e) => setForm((p) => ({ ...p, host: e.target.value }))}
                    placeholder="localhost"
                  />
                </div>
                <div className="field-group">
                  <label className="field-label">端口 (Port)</label>
                  <input
                    className="input"
                    type="number"
                    value={form.port}
                    onChange={(e) => setForm((p) => ({ ...p, port: Number(e.target.value) || 3306 }))}
                  />
                </div>
                <div className="field-group">
                  <label className="field-label">用户名 (User)</label>
                  <input
                    className="input"
                    value={form.user}
                    onChange={(e) => setForm((p) => ({ ...p, user: e.target.value }))}
                  />
                </div>
                <div className="field-group">
                  <label className="field-label">密码 (Password)</label>
                  <input
                    className="input"
                    type="password"
                    value={form.password}
                    onChange={(e) => setForm((p) => ({ ...p, password: e.target.value }))}
                  />
                </div>
                <div className="field-group" style={{ gridColumn: '1 / -1' }}>
                  <label className="field-label">数据库名称 (Database)</label>
                  <input
                    className="input"
                    value={form.database}
                    onChange={(e) => setForm((p) => ({ ...p, database: e.target.value }))}
                    placeholder="例如：ecommerce_db"
                  />
                </div>
              </div>
            )}

            {/* 模型供应商 */}
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: 12 }} className="grid-2">
              <div className="field-group">
                <label className="field-label">
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                    <Server size={13} /> 模型供应商
                  </span>
                </label>
                <select
                  className="select"
                  value={form.provider}
                  onChange={(e) => handleProviderChange(e.target.value)}
                >
                  {providers.map((p) => (
                    <option key={p.name} value={p.name}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field-group">
                <label className="field-label">模型名称 (Model)</label>
                <input
                  className="input"
                  list="smartdb-model-presets"
                  value={form.model}
                  onChange={(e) => setForm((p) => ({ ...p, model: e.target.value }))}
                  placeholder="选择或输入模型名"
                />
                <datalist id="smartdb-model-presets">
                  {(activeProvider?.models || []).map((m) => (
                    <option key={m} value={m} />
                  ))}
                </datalist>
              </div>

              <div className="field-group">
                <label className="field-label">Base URL</label>
                <input
                  className="input"
                  value={form.base_url}
                  onChange={(e) => setForm((p) => ({ ...p, base_url: e.target.value }))}
                  placeholder="https://api.openai.com/v1"
                />
              </div>

              <div className="field-group">
                <label className="field-label">
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                    <KeyRound size={13} /> API Key
                  </span>
                  {!activeProvider?.requires_api_key && (
                    <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>本地免密可选</span>
                  )}
                </label>
                <input
                  className="input"
                  type="password"
                  value={form.api_key}
                  onChange={(e) => setForm((p) => ({ ...p, api_key: e.target.value }))}
                  placeholder={activeProvider?.requires_api_key ? 'sk-...' : '本地模型可留空'}
                />
              </div>
            </div>

            {/* 高级配置折叠 */}
            <div style={{ borderTop: '1px solid var(--border)', paddingTop: 10 }}>
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                style={{ paddingLeft: 0, color: 'var(--text-secondary)' }}
                onClick={() => setShowAdvanced((v) => !v)}
              >
                {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                高级引擎、安全护栏与向量缓存配置
              </button>

              {showAdvanced && (
                <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div className="grid-2">
                    <div className="field-group">
                      <label className="field-label">默认推理模式</label>
                      <select
                        className="select"
                        value={form.query_mode}
                        onChange={(e) =>
                          setForm((p) => ({
                            ...p,
                            query_mode: e.target.value as ConnectPayload['query_mode'],
                          }))
                        }
                      >
                        <option value="agent">Agent 自主工具探索 (推荐)</option>
                        <option value="linked">Schema Graph 链接模式</option>
                        <option value="pipeline">全量 Pipeline 模式</option>
                      </select>
                    </div>
                    <div className="field-group">
                      <label className="field-label">结果集默认上限 (max_limit)</label>
                      <input
                        className="input"
                        type="number"
                        min={0}
                        step={100}
                        value={form.max_limit}
                        onChange={(e) =>
                          setForm((p) => ({ ...p, max_limit: Math.max(0, Number(e.target.value) || 0) }))
                        }
                      />
                    </div>
                    <div className="field-group">
                      <label className="field-label">Agent 最大步数 (2-15)</label>
                      <input
                        className="input"
                        type="number"
                        min={2}
                        max={15}
                        value={form.agent_max_steps}
                        onChange={(e) =>
                          setForm((p) => ({
                            ...p,
                            agent_max_steps: Math.max(2, Math.min(15, Number(e.target.value) || 8)),
                          }))
                        }
                      />
                    </div>
                    <div className="field-group">
                      <label className="field-label">SQL 自动自愈重试次数</label>
                      <input
                        className="input"
                        type="number"
                        min={0}
                        max={5}
                        value={form.max_retries}
                        onChange={(e) =>
                          setForm((p) => ({
                            ...p,
                            max_retries: Math.max(0, Math.min(5, Number(e.target.value) || 0)),
                          }))
                        }
                      />
                    </div>
                    <div className="field-group">
                      <label className="field-label">Embedding 接口协议</label>
                      <select
                        className="select"
                        value={form.embedding_mode}
                        onChange={(e) =>
                          setForm((p) => ({
                            ...p,
                            embedding_mode: e.target.value as ConnectPayload['embedding_mode'],
                          }))
                        }
                      >
                        <option value="openai_compatible">OpenAI Compatible (/v1/embeddings)</option>
                        <option value="ollama_legacy">Ollama Legacy (/api/embeddings)</option>
                      </select>
                    </div>
                    <div className="field-group">
                      <label className="field-label">Embedding 服务地址</label>
                      <input
                        className="input"
                        value={form.embedding_url}
                        onChange={(e) => setForm((p) => ({ ...p, embedding_url: e.target.value }))}
                      />
                    </div>
                    <div className="field-group">
                      <label className="field-label">Embedding 模型</label>
                      <input
                        className="input"
                        value={form.embedding_model}
                        onChange={(e) => setForm((p) => ({ ...p, embedding_model: e.target.value }))}
                      />
                    </div>
                    <div className="field-group">
                      <label className="field-label">
                        语义缓存相似度阈值 ({form.similarity_threshold.toFixed(2)})
                      </label>
                      <input
                        type="range"
                        min={0.5}
                        max={0.99}
                        step={0.01}
                        value={form.similarity_threshold}
                        onChange={(e) =>
                          setForm((p) => ({
                            ...p,
                            similarity_threshold: parseFloat(e.target.value),
                          }))
                        }
                      />
                    </div>
                  </div>

                  <div className="field-group">
                    <label className="field-label">自定义语义层 YAML 路径（可选）</label>
                    <input
                      className="input"
                      value={form.semantic_yaml_path}
                      onChange={(e) =>
                        setForm((p) => ({ ...p, semantic_yaml_path: e.target.value }))
                      }
                      placeholder="例如：./semantic_layer.yaml（留空则使用默认或内置配置）"
                    />
                  </div>

                  <div className="grid-2">
                    <label className="toggle-row">
                      <span>启用问前 Preflight 澄清门控</span>
                      <input
                        type="checkbox"
                        checked={form.enable_preflight}
                        onChange={(e) =>
                          setForm((p) => ({ ...p, enable_preflight: e.target.checked }))
                        }
                      />
                    </label>
                    <label className="toggle-row">
                      <span>启用 AST 表/列幻觉校验</span>
                      <input
                        type="checkbox"
                        checked={form.enable_ast_schema_check}
                        onChange={(e) =>
                          setForm((p) => ({ ...p, enable_ast_schema_check: e.target.checked }))
                        }
                      />
                    </label>
                    <label className="toggle-row">
                      <span>启用 Agent EXPLAIN 执行计划检查</span>
                      <input
                        type="checkbox"
                        checked={form.enable_explain_check}
                        onChange={(e) =>
                          setForm((p) => ({ ...p, enable_explain_check: e.target.checked }))
                        }
                      />
                    </label>
                    <label className="toggle-row">
                      <span>高精度多候选共识投票</span>
                      <input
                        type="checkbox"
                        checked={form.high_accuracy_mode}
                        onChange={(e) =>
                          setForm((p) => ({ ...p, high_accuracy_mode: e.target.checked }))
                        }
                      />
                    </label>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="btn" onClick={onClose}>
              取消
            </button>
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              <PlugZap size={14} />
              {submitting ? '正在初始化引擎...' : '连接并初始化引擎'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
