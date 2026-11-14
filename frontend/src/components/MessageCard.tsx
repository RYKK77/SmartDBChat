import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Check,
  Code2,
  Copy,
  Database,
  Play,
  Sparkles,
  Star,
  Table2,
  Terminal,
  Zap,
} from 'lucide-react'
import { api } from '../services/api'
import type { ChartConfig, ChatMessage } from '../types/api'
import { ChartStudio, KpiStrip } from './ChartStudio'
import { DataTableView } from './DataTableView'

interface MessageCardProps {
  message: ChatMessage
  userQuestion?: string
  onSendPrompt: (text: string) => void
  onFocusTables: (tables: string[]) => void
  onNotify: (msg: string, isError?: boolean) => void
}

type CardTab = 'analysis' | 'table' | 'sql' | 'trace'

export function MessageCard({
  message,
  userQuestion,
  onSendPrompt,
  onFocusTables,
  onNotify,
}: MessageCardProps) {
  const [activeTab, setActiveTab] = useState<CardTab>('analysis')
  const [editableSql, setEditableSql] = useState('')
  const [runningCustomSql, setRunningCustomSql] = useState(false)
  const [customResult, setCustomResult] = useState<{
    sql: string
    rows: Record<string, any>[]
    recommended_chart_config: ChartConfig | null
  } | null>(null)
  const [copied, setCopied] = useState(false)
  const [savingGolden, setSavingGolden] = useState(false)

  const res = message.result

  useEffect(() => {
    if (res?.sql) {
      setEditableSql(res.sql)
    }
    setCustomResult(null)
  }, [res?.sql])

  if (message.role === 'user') {
    return (
      <div className="msg-row user">
        <div className="user-bubble">{message.content}</div>
      </div>
    )
  }

  // 流式加载态
  if (message.isStreaming && !res) {
    return (
      <div className="msg-row">
        <div className="msg-avatar">AI</div>
        <div className="assistant-card">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 600, fontSize: 13 }}>
            <Activity size={15} color="var(--accent)" />
            <span>正在推理解析与执行查询...</span>
          </div>
          {message.statusLogs && message.statusLogs.length > 0 && (
            <div
              style={{
                background: 'var(--bg-subtle)',
                borderRadius: 'var(--radius-sm)',
                padding: '8px 12px',
                display: 'flex',
                flexDirection: 'column',
                gap: 4,
                fontSize: 12,
                color: 'var(--text-secondary)',
              }}
            >
              {message.statusLogs.map((log, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: '50%',
                      background:
                        log.level === 'error'
                          ? 'var(--danger)'
                          : log.level === 'warning'
                          ? 'var(--warning)'
                          : 'var(--accent)',
                    }}
                  />
                  <span>{log.message}</span>
                </div>
              ))}
            </div>
          )}
          {message.streamingText && (
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.streamingText}</ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    )
  }

  // 纯文本兜底（如报错气泡）
  if (!res) {
    return (
      <div className="msg-row">
        <div className="msg-avatar">AI</div>
        <div className="assistant-card">
          <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          </div>
        </div>
      </div>
    )
  }

  const displayRows = customResult ? customResult.rows : res.rows
  const displayRecommendedChart = customResult
    ? customResult.recommended_chart_config
    : res.recommended_chart_config
  const displayLlmChart = customResult ? null : res.chart_config

  const handleCopySql = async () => {
    const target = editableSql || res.sql
    if (!target) return
    try {
      await navigator.clipboard.writeText(target)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      onNotify('复制失败，请手动选中复制', true)
    }
  }

  const handleRunCustomSql = async () => {
    if (!editableSql.trim()) return
    setRunningCustomSql(true)
    try {
      const out = await api.executeSql(editableSql)
      setCustomResult({
        sql: out.sql,
        rows: out.rows,
        recommended_chart_config: out.recommended_chart_config,
      })
      setEditableSql(out.sql)
      if (out.tables?.length) {
        onFocusTables(out.tables)
      }
      onNotify(`自定义 SQL 执行成功，返回 ${out.row_count} 行`)
    } catch (err: any) {
      onNotify(err.message || 'SQL 校验或执行失败', true)
    } finally {
      setRunningCustomSql(false)
    }
  }

  const handleSaveGolden = async () => {
    const targetSql = editableSql || res.sql
    if (!targetSql.trim()) return
    setSavingGolden(true)
    try {
      await api.addGoldenExample({
        question: userQuestion || '历史查询示例',
        sql: targetSql,
        tables: res.linked_tables || [],
        tags: [res.mode || 'verified'],
      })
      onNotify('已收藏为 Golden SQL，将自动增强后续语义检索')
    } catch (err: any) {
      onNotify(err.message || '收藏失败', true)
    } finally {
      setSavingGolden(false)
    }
  }

  const isWarningStatus =
    res.status === 'clarification_needed' ||
    res.status === 'off_topic' ||
    res.status === 'rejected' ||
    res.status === 'empty_result'

  return (
    <div className="msg-row">
      <div className="msg-avatar">AI</div>
      <div className="assistant-card">
        {/* 状态标签栏 */}
        <div className="card-meta-bar">
          <div className="card-badges">
            {res.is_success ? (
              <span className="pill pill-success">
                {res.row_count} 行结果
              </span>
            ) : isWarningStatus ? (
              <span className="pill pill-warning">{res.status}</span>
            ) : (
              <span className="pill pill-danger">执行异常</span>
            )}

            <span className="pill">{res.mode.toUpperCase()}</span>

            {typeof message.elapsedSec === 'number' && (
              <span className="pill">{message.elapsedSec.toFixed(2)}s</span>
            )}

            {res.cache_hit && (
              <span className="pill pill-info" title={`匹配历史问题: ${res.cache_hit.matched_query}`}>
                <Zap size={11} />
                缓存命中 ({(res.cache_hit.similarity * 100).toFixed(0)}%)
              </span>
            )}

            {res.scaling_info && (
              <span className="pill pill-accent">
                <Sparkles size={11} />
                共识投票 {res.scaling_info.consensus_votes}/{res.scaling_info.total_valid}
              </span>
            )}

            {res.linked_tables?.length > 0 && (
              <button
                type="button"
                className="pill pill-accent"
                style={{ cursor: 'pointer' }}
                onClick={() => onFocusTables(res.linked_tables)}
                title="点击在 Schema 图谱中高亮关联表"
              >
                <Database size={11} />
                关联表: {res.linked_tables.join(', ')}
              </button>
            )}
          </div>
        </div>

        {/* 异常 / Preflight 提示条 */}
        {!res.is_success && res.error && !res.clean_markdown && (
          <div
            style={{
              padding: '10px 12px',
              borderRadius: 'var(--radius-sm)',
              background: isWarningStatus ? 'var(--warning-soft)' : 'var(--danger-soft)',
              color: isWarningStatus ? 'var(--warning)' : 'var(--danger)',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              fontSize: 13,
            }}
          >
            <AlertTriangle size={15} />
            <span>{res.error}</span>
          </div>
        )}

        {/* 标签页导航 */}
        <div className="card-tabs">
          <button
            type="button"
            className={`card-tab ${activeTab === 'analysis' ? 'active' : ''}`}
            onClick={() => setActiveTab('analysis')}
          >
            <BarChart3 size={14} />
            分析与图表
          </button>
          <button
            type="button"
            className={`card-tab ${activeTab === 'table' ? 'active' : ''}`}
            onClick={() => setActiveTab('table')}
          >
            <Table2 size={14} />
            数据表格 ({displayRows.length})
          </button>
          <button
            type="button"
            className={`card-tab ${activeTab === 'sql' ? 'active' : ''}`}
            onClick={() => setActiveTab('sql')}
          >
            <Code2 size={14} />
            SQL 工作台
          </button>
          <button
            type="button"
            className={`card-tab ${activeTab === 'trace' ? 'active' : ''}`}
            onClick={() => setActiveTab('trace')}
          >
            <Terminal size={14} />
            推理与轨迹
          </button>
        </div>

        {/* Tab 1: 分析与图表 */}
        {activeTab === 'analysis' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {(res.clean_markdown || res.markdown_answer) && (
              <div className="markdown-body">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {res.clean_markdown || res.markdown_answer}
                </ReactMarkdown>
              </div>
            )}
            {displayRows.length > 0 && <KpiStrip rows={displayRows} />}
            {displayRows.length >= 2 && (
              <ChartStudio
                key={`${customResult?.sql || res.sql}-${displayRows.length}`}
                rows={displayRows}
                llmChartConfig={displayLlmChart}
                recommendedChartConfig={displayRecommendedChart}
              />
            )}
          </div>
        )}

        {/* Tab 2: 数据表格 */}
        {activeTab === 'table' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {customResult && (
              <div className="pill pill-info" style={{ alignSelf: 'flex-start' }}>
                当前展示为自定义修改 SQL 的重跑结果
              </div>
            )}
            <DataTableView rows={displayRows} />
          </div>
        )}

        {/* Tab 3: SQL 工作台 */}
        {activeTab === 'sql' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                可直接编辑 SQL 并通过 AST 只读安全护栏重跑验证
              </span>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <button type="button" className="btn btn-sm" onClick={handleCopySql}>
                  {copied ? <Check size={13} color="var(--success)" /> : <Copy size={13} />}
                  {copied ? '已复制' : '复制 SQL'}
                </button>
                <button
                  type="button"
                  className="btn btn-sm"
                  disabled={savingGolden || !editableSql.trim()}
                  onClick={handleSaveGolden}
                >
                  <Star size={13} color="var(--warning)" />
                  {savingGolden ? '收藏中...' : '收藏为 Golden SQL'}
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  disabled={runningCustomSql || !editableSql.trim()}
                  onClick={handleRunCustomSql}
                >
                  <Play size={13} />
                  {runningCustomSql ? '校验执行中...' : 'AST 校验并重跑'}
                </button>
              </div>
            </div>

            <textarea
              className="textarea textarea-mono"
              rows={6}
              value={editableSql}
              onChange={(e) => setEditableSql(e.target.value)}
              placeholder="SELECT ..."
            />

            {customResult && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--accent)' }}>
                  自定义 SQL 执行结果 ({customResult.rows.length} 行)
                </div>
                <DataTableView rows={customResult.rows} fileName="custom_sql_result.csv" />
              </div>
            )}
          </div>
        )}

        {/* Tab 4: 推理与轨迹 */}
        {activeTab === 'trace' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {res.workflow_trace?.length > 0 && (
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
                  Workflow 状态阶段轨迹
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {res.workflow_trace.map((item, idx) => (
                    <span key={idx} className="pill">
                      {String(item.stage || item.step || `Stage ${idx + 1}`)}
                      {item.status ? `: ${item.status}` : ''}
                      {typeof item.duration_ms === 'number' ? ` (${item.duration_ms}ms)` : ''}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {res.scaling_info?.candidates && res.scaling_info.candidates.length > 0 && (
              <div>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
                  多候选 Test-Time Scaling 对比 ({res.scaling_info.consensus_votes}/{res.scaling_info.total_valid})
                </div>
                <div className="data-table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>策略</th>
                        <th>校验</th>
                        <th>行数</th>
                        <th>SQL 预览</th>
                      </tr>
                    </thead>
                    <tbody>
                      {res.scaling_info.candidates.map((c, idx) => (
                        <tr key={idx}>
                          <td>{c.strategy}</td>
                          <td>
                            <span className={`pill ${c.is_valid ? 'pill-success' : 'pill-danger'}`}>
                              {c.is_valid ? '通过' : c.error || '失败'}
                            </span>
                          </td>
                          <td>{c.row_count}</td>
                          <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11.5 }}>{c.sql}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {res.agent_steps?.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 600 }}>
                  Agent ReAct 工具探索轨迹 ({res.agent_steps.length} 步)
                </div>
                {res.agent_steps.map((st, idx) => (
                  <div
                    key={idx}
                    style={{
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      padding: 10,
                      background: 'var(--bg-subtle)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 5,
                      fontSize: 12,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <strong>
                        Step {st.step}
                        {st.tool_name ? ` · ${st.tool_name}` : ''}
                      </strong>
                      <span style={{ color: 'var(--text-muted)' }}>{st.duration_ms} ms</span>
                    </div>
                    {st.thought && <div style={{ color: 'var(--text-secondary)' }}>{st.thought}</div>}
                    {st.tool_args && (
                      <pre className="sql-box" style={{ fontSize: 11, padding: '6px 8px' }}>
                        {JSON.stringify(st.tool_args, null, 2)}
                      </pre>
                    )}
                    {st.error ? (
                      <div style={{ color: 'var(--danger)' }}>错误: {st.error}</div>
                    ) : (
                      st.observation && (
                        <div style={{ color: 'var(--text-muted)', whiteSpace: 'pre-wrap' }}>
                          {st.observation}
                        </div>
                      )
                    )}
                  </div>
                ))}
              </div>
            )}

            {res.attempts?.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ fontSize: 12, fontWeight: 600 }}>
                  SQL 执行与自愈记录 ({res.attempts.length} 次)
                </div>
                {res.attempts.map((at, idx) => (
                  <div
                    key={idx}
                    style={{
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      padding: 8,
                      fontSize: 12,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span>第 {at.attempt} 次尝试</span>
                      {at.error ? (
                        <span style={{ color: 'var(--danger)' }}>{at.error}</span>
                      ) : (
                        <span style={{ color: 'var(--success)' }}>成功 ({at.row_count} 行)</span>
                      )}
                    </div>
                    <pre className="sql-box" style={{ fontSize: 11.5, padding: 8 }}>
                      {at.sql}
                    </pre>
                  </div>
                ))}
              </div>
            )}

            {!res.workflow_trace?.length &&
              !res.scaling_info?.candidates?.length &&
              !res.agent_steps?.length &&
              !res.attempts?.length && (
                <div style={{ color: 'var(--text-muted)', fontSize: 12.5 }}>
                  当前查询未记录额外的中间推理轨迹（可能直接命中缓存或由 Preflight 拦截）。
                </div>
              )}
          </div>
        )}

        {/* 追问建议芯片 (QueryWeaver 风格) */}
        {res.follow_up_suggestions && res.follow_up_suggestions.length > 0 && (
          <div className="followup-strip">
            {res.follow_up_suggestions.map((q, idx) => (
              <button
                key={idx}
                type="button"
                className="followup-chip"
                onClick={() => onSendPrompt(q)}
              >
                <Sparkles size={11} />
                {q}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
