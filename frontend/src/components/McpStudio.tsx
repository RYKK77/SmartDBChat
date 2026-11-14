import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Check,
  Code2,
  Copy,
  Eye,
  EyeOff,
  KeyRound,
  Play,
  Plus,
  RefreshCw,
  Server,
  ShieldCheck,
  Terminal,
  Trash2,
  Wrench,
} from 'lucide-react'
import { api } from '../services/api'
import type { ConnectPayload, McpApiKeyItem, McpServerInfo, McpToolMeta } from '../types/api'

interface McpStudioProps {
  lastConnection: ConnectPayload | null
  onOpenConnect: () => void
  onNotify: (msg: string, isError?: boolean) => void
}

function maskKey(key: string): string {
  if (!key || key.length <= 14) return key
  return `${key.slice(0, 10)}••••••••••••${key.slice(-4)}`
}

function formatTime(ts?: number | null): string {
  if (!ts) return '从未使用'
  try {
    return new Date(ts).toLocaleString('zh-CN', { hour12: false })
  } catch {
    return '-'
  }
}

export function McpStudio({ lastConnection, onOpenConnect, onNotify }: McpStudioProps) {
  const [info, setInfo] = useState<McpServerInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState<'keys' | 'playground'>('keys')
  const [configSnippetTab, setConfigSnippetTab] = useState<'cursor' | 'http' | 'cli'>('cursor')

  // Key 管理状态
  const [newKeyName, setNewKeyName] = useState('')
  const [creatingKey, setCreatingKey] = useState(false)
  const [revealedKeys, setRevealedKeys] = useState<Record<string, boolean>>({})
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [selectedKeyValue, setSelectedKeyValue] = useState<string>('')
  const [llmApiKeyOverride, setLlmApiKeyOverride] = useState<string>(
    () => lastConnection?.api_key || '',
  )

  // Playground 状态
  const [selectedToolName, setSelectedToolName] = useState<string>('list_tables')
  const [argsJson, setArgsJson] = useState<string>('{}')
  const [callingTool, setCallingTool] = useState(false)
  const [callResult, setCallResult] = useState<{
    tool_name: string
    duration_ms: number
    result: any
  } | null>(null)

  const loadInfo = useCallback(async () => {
    setLoading(true)
    try {
      const data = await api.getMcpInfo()
      setInfo(data)
      const activeKeys = (data.keys || []).filter((k) => k.is_active)
      if (activeKeys.length > 0 && !selectedKeyValue) {
        setSelectedKeyValue(activeKeys[0].api_key)
      }
      if (data.tools?.length && !data.tools.some((t) => t.name === selectedToolName)) {
        const first = data.tools[0]
        setSelectedToolName(first.name)
        setArgsJson(JSON.stringify(first.sample_args || {}, null, 2))
      }
    } catch (err: any) {
      onNotify(err.message || '加载 MCP 服务信息失败', true)
    } finally {
      setLoading(false)
    }
  }, [onNotify, selectedKeyValue, selectedToolName])

  useEffect(() => {
    loadInfo()
  }, [loadInfo])

  useEffect(() => {
    if (lastConnection?.api_key && !llmApiKeyOverride) {
      setLlmApiKeyOverride(lastConnection.api_key)
    }
  }, [lastConnection?.api_key, llmApiKeyOverride])

  const selectedTool: McpToolMeta | undefined = useMemo(
    () => info?.tools.find((t) => t.name === selectedToolName),
    [info?.tools, selectedToolName],
  )

  const handleSelectTool = (tool: McpToolMeta) => {
    setSelectedToolName(tool.name)
    setArgsJson(JSON.stringify(tool.sample_args || {}, null, 2))
    setCallResult(null)
  }

  const handleCopyText = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopiedId(id)
      setTimeout(() => setCopiedId(null), 1800)
      onNotify('已复制到剪贴板')
    } catch {
      onNotify('复制失败，请手动复制', true)
    }
  }

  const handleToggleRequireKey = async (nextVal: boolean) => {
    try {
      const res = await api.updateMcpSettings(nextVal)
      setInfo((prev) => (prev ? { ...prev, settings: res.settings } : prev))
      onNotify(nextVal ? '已开启 MCP 访问密钥强制校验' : '已关闭 MCP 访问密钥强制校验')
    } catch (err: any) {
      onNotify(err.message || '更新设置失败', true)
    }
  }

  const handleCreateKey = async () => {
    const label = newKeyName.trim() || `MCP Key ${(info?.keys.length || 0) + 1}`
    setCreatingKey(true)
    try {
      const res = await api.createMcpKey(label)
      setNewKeyName('')
      setInfo((prev) => (prev ? { ...prev, keys: res.keys } : prev))
      setSelectedKeyValue(res.created.api_key)
      setRevealedKeys((prev) => ({ ...prev, [res.created.id]: true }))
      onNotify(`已生成新密钥: ${res.created.name}`)
    } catch (err: any) {
      onNotify(err.message || '生成密钥失败', true)
    } finally {
      setCreatingKey(false)
    }
  }

  const handleToggleKeyStatus = async (item: McpApiKeyItem) => {
    try {
      const res = await api.toggleMcpKey(item.id, !item.is_active)
      setInfo((prev) => (prev ? { ...prev, keys: res.keys } : prev))
      onNotify(`密钥 "${item.name}" 已${!item.is_active ? '启用' : '禁用'}`)
    } catch (err: any) {
      onNotify(err.message || '操作失败', true)
    }
  }

  const handleDeleteKey = async (item: McpApiKeyItem) => {
    try {
      const res = await api.deleteMcpKey(item.id)
      setInfo((prev) => (prev ? { ...prev, keys: res.keys } : prev))
      if (selectedKeyValue === item.api_key) {
        const nextActive = res.keys.find((k) => k.is_active)
        setSelectedKeyValue(nextActive ? nextActive.api_key : '')
      }
      onNotify(`已删除密钥 "${item.name}"`)
    } catch (err: any) {
      onNotify(err.message || '删除密钥失败', true)
    }
  }

  const handleRunTool = async () => {
    if (!info?.connected) {
      onOpenConnect()
      onNotify('请先连接数据库或启用内置沙箱后再调试 MCP 工具', true)
      return
    }
    let parsedArgs: Record<string, any> = {}
    try {
      parsedArgs = argsJson.trim() ? JSON.parse(argsJson) : {}
    } catch (err: any) {
      onNotify(`参数 JSON 格式有误: ${err.message}`, true)
      return
    }

    setCallingTool(true)
    try {
      const out = await api.callMcpTool({
        tool_name: selectedToolName,
        arguments: parsedArgs,
        api_key: selectedKeyValue || undefined,
      })
      setCallResult({
        tool_name: out.tool_name,
        duration_ms: out.duration_ms,
        result: out.result,
      })
      if (out.keys) {
        setInfo((prev) => (prev ? { ...prev, keys: out.keys! } : prev))
      }
      onNotify(`MCP 工具 ${selectedToolName} 调用成功 (${out.duration_ms} ms)`)
    } catch (err: any) {
      onNotify(err.message || 'MCP 工具调用失败', true)
    } finally {
      setCallingTool(false)
    }
  }

  const effectiveMcpKey = selectedKeyValue || '<请先在上方生成并选择 MCP Key>'
  const effectiveLlmKey = llmApiKeyOverride.trim() || '<你的大模型_API_KEY>'
  const dbType = lastConnection?.db_type || (info?.connection_meta.db_type === 'mysql' ? 'mysql' : 'sandbox')

  const cursorConfigJson = useMemo(() => {
    const envBlock: Record<string, string> = {
      SMARTDB_DB_TYPE: dbType,
      SMARTDB_MCP_KEY: effectiveMcpKey,
      SMARTDB_API_KEY: effectiveLlmKey,
      SMARTDB_BASE_URL:
        lastConnection?.base_url || info?.connection_meta.base_url || 'https://api.deepseek.com/v1',
      SMARTDB_MODEL:
        lastConnection?.model || info?.connection_meta.model || 'deepseek-chat',
    }
    if (dbType === 'mysql') {
      envBlock.SMARTDB_HOST = lastConnection?.host || 'localhost'
      envBlock.SMARTDB_PORT = String(lastConnection?.port || 3306)
      envBlock.SMARTDB_USER = lastConnection?.user || 'root'
      envBlock.SMARTDB_PASSWORD = lastConnection?.password || '<数据库密码>'
      envBlock.SMARTDB_DATABASE =
        lastConnection?.database || info?.connection_meta.database || '<数据库名>'
    }
    return JSON.stringify(
      {
        mcpServers: {
          smartdbchat: {
            command: info?.python_executable || 'python',
            args: ['-m', 'smartdbchat.mcp_server', '--transport', 'stdio'],
            cwd: info?.workspace_dir || '/path/to/SmartDBChat',
            env: envBlock,
          },
        },
      },
      null,
      2,
    )
  }, [dbType, effectiveLlmKey, effectiveMcpKey, info, lastConnection])

  const httpCurlSnippet = useMemo(() => {
    const origin = typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8000'
    let sampleObj: Record<string, any> = {}
    try {
      sampleObj = argsJson.trim() ? JSON.parse(argsJson) : {}
    } catch {
      sampleObj = selectedTool?.sample_args || {}
    }
    const payload = JSON.stringify(
      {
        tool_name: selectedToolName,
        arguments: sampleObj,
      },
      null,
      2,
    )
    return `curl -X POST "${origin}/api/mcp/call" \\
  -H "Content-Type: application/json" \\
  -H "X-MCP-API-Key: ${effectiveMcpKey}" \\
  -d '${payload}'`
  }, [argsJson, effectiveMcpKey, selectedTool, selectedToolName])

  const cliStartSnippet = useMemo(() => {
    const py = info?.python_executable || 'python'
    return `# 方式 1：Stdio 协议启动（适用于 Claude Desktop / Cursor 本地进程通信）
SMARTDB_DB_TYPE=${dbType} SMARTDB_API_KEY="${effectiveLlmKey}" ${py} -m smartdbchat.mcp_server --transport stdio

# 方式 2：Streamable-HTTP 协议启动（适用于远程 Agent 集群接入）
SMARTDB_DB_TYPE=${dbType} SMARTDB_API_KEY="${effectiveLlmKey}" ${py} -m smartdbchat.mcp_server --transport streamable-http

# 方式 3：SSE 协议启动
SMARTDB_DB_TYPE=${dbType} SMARTDB_API_KEY="${effectiveLlmKey}" ${py} -m smartdbchat.mcp_server --transport sse`
  }, [dbType, effectiveLlmKey, info?.python_executable])

  if (loading && !info) {
    return <div style={{ padding: 24, color: 'var(--text-muted)' }}>正在加载 MCP 服务中心...</div>
  }

  return (
    <div
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px 28px',
        background: 'var(--bg-app)',
      }}
    >
      <div style={{ maxWidth: 1040, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
        {/* 头部标题与模式切换 */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 12,
          }}
        >
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>
                MCP 开放服务中心 (Model Context Protocol)
              </h2>
              <span className="pill pill-success">
                <Server size={11} />
                {info?.tools.length || 6} 个标准工具已就绪
              </span>
              <span className={`pill ${info?.connected ? 'pill-info' : 'pill-warning'}`}>
                {info?.connected
                  ? `已挂载数据源: ${info.connection_meta.database || '沙箱'}`
                  : '未连接数据源'}
              </span>
            </div>
            <p style={{ margin: '4px 0 0', fontSize: 13, color: 'var(--text-secondary)' }}>
              管理 MCP 访问密钥（API Key）、一键生成 Cursor / Claude Desktop 接入配置，并在线调试全部 MCP 工具
            </p>
          </div>

          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <div className="nav-tabs">
              <button
                type="button"
                className={`nav-tab ${activeTab === 'keys' ? 'active' : ''}`}
                onClick={() => setActiveTab('keys')}
              >
                <KeyRound size={14} />
                密钥与接入配置 ({info?.keys.length || 0})
              </button>
              <button
                type="button"
                className={`nav-tab ${activeTab === 'playground' ? 'active' : ''}`}
                onClick={() => setActiveTab('playground')}
              >
                <Wrench size={14} />
                工具目录与在线调试
              </button>
            </div>

            <button
              type="button"
              className="btn btn-sm"
              onClick={loadInfo}
              title="刷新 MCP 状态与密钥统计"
            >
              <RefreshCw size={13} />
              刷新
            </button>
          </div>
        </div>

        {activeTab === 'keys' ? (
          <>
            {/* 1. MCP 访问密钥管理 */}
            <div className="side-section" style={{ padding: 18 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: 10,
                  marginBottom: 14,
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <KeyRound size={15} color="var(--accent)" />
                    MCP 服务端访问密钥 (MCP API Keys)
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                    用于外部 Agent、HTTP 客户端或调试台调用 <code>/api/mcp/call</code> 时的身份鉴权与调用审计
                  </div>
                </div>

                <label className="switch-row" style={{ gap: 8 }}>
                  <span style={{ fontSize: 12.5, fontWeight: 500 }}>强制校验 MCP Key</span>
                  <input
                    type="checkbox"
                    checked={Boolean(info?.settings.require_api_key)}
                    onChange={(e) => handleToggleRequireKey(e.target.checked)}
                  />
                </label>
              </div>

              {/* 创建新 Key 表单 */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 14 }}>
                <input
                  className="input"
                  placeholder="输入密钥备注名称，例如：Cursor 本地开发 / Claude Desktop / 生产 Agent..."
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleCreateKey()
                  }}
                />
                <button
                  type="button"
                  className="btn btn-primary"
                  style={{ flexShrink: 0 }}
                  disabled={creatingKey}
                  onClick={handleCreateKey}
                >
                  <Plus size={14} />
                  {creatingKey ? '生成中...' : '生成 MCP Key'}
                </button>
              </div>

              {/* 密钥表格 */}
              {info?.keys && info.keys.length > 0 ? (
                <div className="data-table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>备注名称</th>
                        <th>访问密钥 (API Key)</th>
                        <th>状态</th>
                        <th>调用次数</th>
                        <th>最近使用</th>
                        <th>操作</th>
                      </tr>
                    </thead>
                    <tbody>
                      {info.keys.map((item) => {
                        const revealed = Boolean(revealedKeys[item.id])
                        const isSelected = selectedKeyValue === item.api_key
                        return (
                          <tr key={item.id}>
                            <td style={{ fontWeight: 600 }}>
                              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <span>{item.name}</span>
                                {isSelected && (
                                  <span className="pill pill-accent" style={{ fontSize: 10 }}>
                                    当前选中
                                  </span>
                                )}
                              </div>
                            </td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                              <span>{revealed ? item.api_key : maskKey(item.api_key)}</span>
                            </td>
                            <td>
                              <span className={`pill ${item.is_active ? 'pill-success' : 'pill-danger'}`}>
                                {item.is_active ? '已启用' : '已禁用'}
                              </span>
                            </td>
                            <td>{item.call_count} 次</td>
                            <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                              {formatTime(item.last_used_at)}
                            </td>
                            <td>
                              <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                                <button
                                  type="button"
                                  className="btn btn-sm"
                                  onClick={() =>
                                    setRevealedKeys((prev) => ({ ...prev, [item.id]: !revealed }))
                                  }
                                  title={revealed ? '隐藏密钥' : '显示完整密钥'}
                                >
                                  {revealed ? <EyeOff size={12} /> : <Eye size={12} />}
                                </button>
                                <button
                                  type="button"
                                  className="btn btn-sm"
                                  onClick={() => handleCopyText(item.api_key, item.id)}
                                  title="复制密钥"
                                >
                                  {copiedId === item.id ? (
                                    <Check size={12} color="var(--success)" />
                                  ) : (
                                    <Copy size={12} />
                                  )}
                                  复制
                                </button>
                                <button
                                  type="button"
                                  className="btn btn-sm"
                                  onClick={() => setSelectedKeyValue(item.api_key)}
                                  disabled={isSelected}
                                >
                                  {isSelected ? '已用于配置' : '填入下方配置'}
                                </button>
                                <button
                                  type="button"
                                  className="btn btn-sm"
                                  onClick={() => handleToggleKeyStatus(item)}
                                >
                                  {item.is_active ? '禁用' : '启用'}
                                </button>
                                <button
                                  type="button"
                                  className="icon-btn"
                                  onClick={() => handleDeleteKey(item)}
                                  title="删除该密钥"
                                >
                                  <Trash2 size={13} />
                                </button>
                              </div>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div
                  style={{
                    padding: '16px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--bg-subtle)',
                    color: 'var(--text-secondary)',
                    fontSize: 13,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                  }}
                >
                  <span>当前尚未创建 MCP 访问密钥。点击上方「生成 MCP Key」即可创建首个客户端密钥。</span>
                </div>
              )}
            </div>

            {/* 2. 独立进程 LLM Key 与环境变量注入配置 */}
            <div className="side-section" style={{ padding: 18 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: 10,
                  marginBottom: 12,
                }}
              >
                <div>
                  <div style={{ fontWeight: 600, fontSize: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Code2 size={15} color="var(--accent)" />
                    一键生成客户端接入配置（自动注入 Key）
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                    选择上方生成的 MCP Key 并填写大模型 API Key，自动生成可直接粘贴到 Cursor / Claude Desktop 的配置文件
                  </div>
                </div>

                <div className="nav-tabs">
                  <button
                    type="button"
                    className={`nav-tab ${configSnippetTab === 'cursor' ? 'active' : ''}`}
                    onClick={() => setConfigSnippetTab('cursor')}
                  >
                    Cursor / Claude Desktop (JSON)
                  </button>
                  <button
                    type="button"
                    className={`nav-tab ${configSnippetTab === 'http' ? 'active' : ''}`}
                    onClick={() => setConfigSnippetTab('http')}
                  >
                    HTTP 鉴权调用 (cURL)
                  </button>
                  <button
                    type="button"
                    className={`nav-tab ${configSnippetTab === 'cli' ? 'active' : ''}`}
                    onClick={() => setConfigSnippetTab('cli')}
                  >
                    命令行启动 (Stdio / HTTP / SSE)
                  </button>
                </div>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                  gap: 12,
                  marginBottom: 12,
                }}
              >
                <div className="field">
                  <label className="field-label">当前注入的 MCP 访问密钥 (SMARTDB_MCP_KEY)</label>
                  <select
                    className="select"
                    value={selectedKeyValue}
                    onChange={(e) => setSelectedKeyValue(e.target.value)}
                  >
                    <option value="">-- 请选择已生成的 MCP Key --</option>
                    {info?.keys.map((k) => (
                      <option key={k.id} value={k.api_key}>
                        {k.name} ({maskKey(k.api_key)}) {!k.is_active ? '[已禁用]' : ''}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="field">
                  <label className="field-label">
                    大模型 API Key (SMARTDB_API_KEY，供独立 stdio 进程问数使用)
                  </label>
                  <input
                    type="password"
                    className="input"
                    placeholder="填入 DeepSeek / OpenAI / Qwen 等 API Key..."
                    value={llmApiKeyOverride}
                    onChange={(e) => setLlmApiKeyOverride(e.target.value)}
                  />
                </div>
              </div>

              <div style={{ position: 'relative' }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: 6,
                    fontSize: 12,
                    color: 'var(--text-muted)',
                  }}
                >
                  <span>
                    {configSnippetTab === 'cursor' &&
                      '粘贴至 ~/.cursor/mcp.json 或 Claude Desktop 配置文件 claude_desktop_config.json：'}
                    {configSnippetTab === 'http' &&
                      '通过 HTTP Header X-MCP-API-Key 直接远程调用 SmartDBChat MCP 工具：'}
                    {configSnippetTab === 'cli' && '在终端独立启动 MCP Server 进程：'}
                  </span>
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={() =>
                      handleCopyText(
                        configSnippetTab === 'cursor'
                          ? cursorConfigJson
                          : configSnippetTab === 'http'
                          ? httpCurlSnippet
                          : cliStartSnippet,
                        `snippet-${configSnippetTab}`,
                      )
                    }
                  >
                    {copiedId === `snippet-${configSnippetTab}` ? (
                      <Check size={13} color="var(--success)" />
                    ) : (
                      <Copy size={13} />
                    )}
                    复制配置代码
                  </button>
                </div>

                <pre className="sql-box" style={{ padding: 14, fontSize: 12, overflowX: 'auto' }}>
                  {configSnippetTab === 'cursor'
                    ? cursorConfigJson
                    : configSnippetTab === 'http'
                    ? httpCurlSnippet
                    : cliStartSnippet}
                </pre>
              </div>
            </div>
          </>
        ) : (
          /* 工具目录与在线调试台 (Playground) */
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '340px 1fr',
              gap: 16,
              alignItems: 'start',
            }}
          >
            {/* 左侧工具列表 */}
            <div className="side-section" style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ fontWeight: 600, fontSize: 13.5, marginBottom: 4 }}>
                已暴露的 MCP 工具 ({info?.tools.length || 0})
              </div>
              {info?.tools.map((tool) => {
                const active = tool.name === selectedToolName
                return (
                  <button
                    key={tool.name}
                    type="button"
                    onClick={() => handleSelectTool(tool)}
                    style={{
                      textAlign: 'left',
                      padding: '10px 12px',
                      borderRadius: 'var(--radius-sm)',
                      border: active ? '1px solid var(--accent)' : '1px solid var(--border)',
                      background: active ? 'var(--accent-soft)' : 'var(--bg-subtle)',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 4,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span
                        style={{
                          fontFamily: 'var(--font-mono)',
                          fontWeight: 600,
                          fontSize: 13,
                          color: active ? 'var(--accent)' : 'var(--text-primary)',
                        }}
                      >
                        {tool.name}
                      </span>
                      <span className="pill" style={{ fontSize: 10 }}>
                        {tool.parameters.length} 参数
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                      {tool.description}
                    </div>
                  </button>
                )
              })}
            </div>

            {/* 右侧在线调试器 */}
            <div className="side-section" style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Terminal size={16} color="var(--accent)" />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 15, fontWeight: 700 }}>
                      {selectedTool?.name}
                    </span>
                  </div>
                  <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', marginTop: 4 }}>
                    {selectedTool?.description}
                  </div>
                </div>

                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={callingTool}
                  onClick={handleRunTool}
                >
                  <Play size={14} />
                  {callingTool ? '正在调用...' : '执行 MCP 工具'}
                </button>
              </div>

              {/* 参数说明表 */}
              {selectedTool && selectedTool.parameters.length > 0 && (
                <div>
                  <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>工具参数定义</div>
                  <div className="data-table-scroll">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>参数名</th>
                          <th>类型</th>
                          <th>必填</th>
                          <th>说明</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedTool.parameters.map((p) => (
                          <tr key={p.name}>
                            <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{p.name}</td>
                            <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{p.type}</td>
                            <td>
                              {p.required ? (
                                <span className="pill pill-warning">必填</span>
                              ) : (
                                <span className="pill">可选</span>
                              )}
                            </td>
                            <td style={{ fontSize: 12 }}>{p.description}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* 鉴权 Key 选择 + JSON 参数输入 */}
              <div className="field">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label className="field-label">
                    调试使用的 MCP Key
                    {info?.settings.require_api_key ? '（当前已开启强制校验）' : '（当前可选）'}
                  </label>
                  <span style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                    <ShieldCheck size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
                    通过请求头 <code>X-MCP-API-Key</code> 发送
                  </span>
                </div>
                <select
                  className="select"
                  value={selectedKeyValue}
                  onChange={(e) => setSelectedKeyValue(e.target.value)}
                >
                  <option value="">-- 不携带 Key / 请选择 MCP Key --</option>
                  {info?.keys.map((k) => (
                    <option key={k.id} value={k.api_key}>
                      {k.name} ({maskKey(k.api_key)}) {!k.is_active ? '[已禁用]' : ''}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label className="field-label">调用参数 (JSON)</label>
                  {selectedTool && (
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() =>
                        setArgsJson(JSON.stringify(selectedTool.sample_args || {}, null, 2))
                      }
                    >
                      重置为示例参数
                    </button>
                  )}
                </div>
                <textarea
                  className="textarea textarea-mono"
                  rows={6}
                  value={argsJson}
                  onChange={(e) => setArgsJson(e.target.value)}
                />
              </div>

              {/* 返回结果展示 */}
              {callResult && (
                <div>
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: 6,
                    }}
                  >
                    <span style={{ fontSize: 12, fontWeight: 600 }}>
                      MCP 响应结果 ({callResult.duration_ms} ms)
                    </span>
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() =>
                        handleCopyText(
                          JSON.stringify(callResult.result, null, 2),
                          'call-result',
                        )
                      }
                    >
                      {copiedId === 'call-result' ? (
                        <Check size={12} color="var(--success)" />
                      ) : (
                        <Copy size={12} />
                      )}
                      复制响应 JSON
                    </button>
                  </div>
                  <pre
                    className="sql-box"
                    style={{
                      padding: 14,
                      fontSize: 12,
                      maxHeight: 380,
                      overflowY: 'auto',
                      whiteSpace: 'pre-wrap',
                    }}
                  >
                    {typeof callResult.result === 'string'
                      ? callResult.result
                      : JSON.stringify(callResult.result, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
