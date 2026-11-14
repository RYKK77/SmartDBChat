import { useCallback, useEffect, useRef, useState } from 'react'
import {
  BookOpen,
  Database,
  GitFork,
  LogOut,
  MessageSquare,
  Moon,
  PanelLeft,
  PanelRight,
  PlugZap,
  Send,
  Server,
  Shield,
  Sparkles,
  Sun,
  Trash2,
} from 'lucide-react'
import { AuthModal } from './components/AuthModal'
import { ConnectionModal } from './components/ConnectionModal'
import { McpStudio } from './components/McpStudio'
import { MessageCard } from './components/MessageCard'
import { SchemaGraphCanvas } from './components/SchemaGraphCanvas'
import { SemanticLayerStudio } from './components/SemanticLayerStudio'
import { Sidebar } from './components/Sidebar'
import { api } from './services/api'
import type {
  CacheStats,
  ChatMessage,
  ConnectionMeta,
  ConnectPayload,
  EngineConfig,
  ProviderPreset,
  SchemaGraphResponse,
} from './types/api'

type ActiveView = 'chat' | 'schema' | 'semantic' | 'mcp'

const STARTER_PROMPTS = [
  {
    title: '各学院在读学生人数与平均 GPA',
    sub: '多表关联统计 + 学院维度对比柱状图',
  },
  {
    title: '校园净消费总额最高的 5 个消费类别',
    sub: '验证语义层排除已退款订单规则与饼/柱可视化',
  },
  {
    title: '统计各课程平均分、及格率及选课人数',
    sub: '课程目录与选课成绩关联计算分析',
  },
  {
    title: '哪些高绩点学生同时获得过奖学金？列出院系与总金额',
    sub: '触发语义术语“高绩点学生”与多表 JOIN 探索',
  },
]

import {
  consumeLegacyBrowserData,
  deriveSessionTitle,
  getAuthToken,
  setAuthToken,
  type AuthUser,
  type ChatSession,
  type SavedConnectionProfile,
} from './utils/storage'

export function App() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    const saved = localStorage.getItem('smartdb_theme')
    return saved === 'dark' ? 'dark' : 'light'
  })
  const [activeView, setActiveView] = useState<ActiveView>('chat')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [splitGraphOpen, setSplitGraphOpen] = useState(true)
  const [connectModalOpen, setConnectModalOpen] = useState(false)

  const [connected, setConnected] = useState(false)
  const [connectionMeta, setConnectionMeta] = useState<ConnectionMeta>({})
  const [providers, setProviders] = useState<ProviderPreset[]>([])
  const [config, setConfig] = useState<EngineConfig | null>(null)
  const [modules, setModules] = useState<string[]>([])
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null)
  const [schemaGraph, setSchemaGraph] = useState<SchemaGraphResponse | null>(null)

  const [highlightedTables, setHighlightedTables] = useState<string[]>([])
  const [selectedTable, setSelectedTable] = useState<string | null>(null)

  const [authUser, setAuthUser] = useState<AuthUser | null>(null)
  const [authChecking, setAuthChecking] = useState<boolean>(() => Boolean(getAuthToken()))
  const [lastConnection, setLastConnection] = useState<ConnectPayload | null>(null)
  const [savedProfiles, setSavedProfiles] = useState<SavedConnectionProfile[]>([])
  const [sessions, setSessions] = useState<ChatSession[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isSending, setIsSending] = useState(false)

  const [toast, setToast] = useState<{ msg: string; isError?: boolean } | null>(null)
  const toastTimerRef = useRef<number | null>(null)
  const chatScrollRef = useRef<HTMLDivElement | null>(null)
  const mainContentRef = useRef<HTMLElement | null>(null)

  const [splitGraphWidth, setSplitGraphWidth] = useState<number>(() => {
    const saved = Number(localStorage.getItem('smartdb_split_graph_width'))
    if (Number.isFinite(saved) && saved >= 300) return saved
    return typeof window !== 'undefined'
      ? Math.min(640, Math.max(380, Math.round(window.innerWidth * 0.42)))
      : 500
  })
  const [isResizingGraph, setIsResizingGraph] = useState(false)

  const handleStartResizeGraph = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault()
    setIsResizingGraph(true)
    const prevUserSelect = document.body.style.userSelect
    const prevCursor = document.body.style.cursor
    document.body.style.userSelect = 'none'
    document.body.style.cursor = 'col-resize'

    let latestWidth = splitGraphWidth

    const onPointerMove = (moveEvent: PointerEvent) => {
      const container = mainContentRef.current
      if (!container) return
      const rect = container.getBoundingClientRect()
      const maxWidth = Math.max(340, rect.width - 340)
      const nextWidth = Math.round(Math.min(maxWidth, Math.max(300, rect.right - moveEvent.clientX)))
      latestWidth = nextWidth
      setSplitGraphWidth(nextWidth)
    }

    const onPointerUp = () => {
      window.removeEventListener('pointermove', onPointerMove)
      window.removeEventListener('pointerup', onPointerUp)
      document.body.style.userSelect = prevUserSelect
      document.body.style.cursor = prevCursor
      setIsResizingGraph(false)
      localStorage.setItem('smartdb_split_graph_width', String(latestWidth))
    }

    window.addEventListener('pointermove', onPointerMove)
    window.addEventListener('pointerup', onPointerUp)
  }, [splitGraphWidth])

  const notify = useCallback((msg: string, isError = false) => {
    setToast({ msg, isError })
    if (toastTimerRef.current) {
      window.clearTimeout(toastTimerRef.current)
    }
    toastTimerRef.current = window.setTimeout(() => {
      setToast(null)
    }, 3600)
  }, [])

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('smartdb_theme', theme)
  }, [theme])

  const loadWorkspaceData = useCallback(async () => {
    try {
      const [cfg, mods, stats, graph] = await Promise.all([
        api.getConfig(),
        api.getModules().catch(() => ({ modules: [] })),
        api.getCacheStats().catch(() => null),
        api.getSchemaGraph().catch(() => null),
      ])
      setConfig(cfg)
      setModules(mods.modules || [])
      if (stats) setCacheStats(stats)
      if (graph) setSchemaGraph(graph)
    } catch {
      // ignore if disconnected
    }
  }, [])

  const syncUserRemoteState = useCallback(async () => {
    // 若浏览器曾存有旧版连接或会话，首次登录后自动迁移至后端并彻底清除本地残留
    const legacy = consumeLegacyBrowserData()
    if (legacy.lastConnection) {
      await api
        .updateUserPreferences({
          last_connection: legacy.lastConnection,
          update_last_connection: true,
        })
        .catch(() => null)
    }
    for (const prof of legacy.profiles) {
      await api
        .saveConnectionProfile({ id: prof.id, name: prof.name, payload: prof.payload })
        .catch(() => null)
    }
    for (const sess of legacy.sessions) {
      await api
        .saveChatSession(sess.id, { title: sess.title, messages: sess.messages })
        .catch(() => null)
    }

    const userState = await api.getUserState()
    setLastConnection(userState.last_connection)
    setSavedProfiles(userState.profiles || [])
    const remoteSessions = userState.sessions || []
    setSessions(remoteSessions)

    const targetSess =
      (userState.active_session_id &&
        remoteSessions.find((s) => s.id === userState.active_session_id)) ||
      remoteSessions[0] ||
      null
    setActiveSessionId(targetSess ? targetSess.id : null)
    setMessages(targetSess ? targetSess.messages : [])

    const [health, stats] = await Promise.all([
      api.getHealth().catch(() => null),
      api.getCacheStats().catch(() => null),
    ])
    if (stats) setCacheStats(stats)
    if (health?.connected) {
      setConnected(true)
      setConnectionMeta(health.connection_meta || {})
      loadWorkspaceData()
    } else if (!userState.auto_connect_disabled && userState.last_connection) {
      try {
        const res = await api.connect(userState.last_connection)
        setConnected(res.connected)
        setConnectionMeta(res.connection_meta)
        setConfig(res.config)
        loadWorkspaceData()
      } catch {
        // 自动重连失败时保留已保存配置供弹窗使用
      }
    }
  }, [loadWorkspaceData])

  useEffect(() => {
    api
      .getProviders()
      .then((prov) => setProviders(prov.providers || []))
      .catch(() => null)

    const token = getAuthToken()
    if (!token) {
      setAuthChecking(false)
      return
    }
    api
      .getMe()
      .then(async (res) => {
        setAuthUser(res.user)
        await syncUserRemoteState()
      })
      .catch(() => {
        setAuthToken(null)
        setAuthUser(null)
      })
      .finally(() => {
        setAuthChecking(false)
      })
  }, [syncUserRemoteState])

  useEffect(() => {
    if (!authUser) return
    const completed = messages.filter((m) => !m.isStreaming)
    if (completed.length > 0 && !messages.some((m) => m.isStreaming)) {
      const targetId = activeSessionId || `sess-${Date.now()}`
      const title = deriveSessionTitle(completed)
      if (!activeSessionId) {
        setActiveSessionId(targetId)
      }
      setSessions((prev) => {
        const existingIdx = prev.findIndex((s) => s.id === targetId)
        const updatedSession: ChatSession = {
          id: targetId,
          title,
          updatedAt: Date.now(),
          messages: completed,
        }
        return existingIdx >= 0
          ? prev.map((s) => (s.id === targetId ? updatedSession : s))
          : [updatedSession, ...prev]
      })
      api.saveChatSession(targetId, { title, messages: completed }).catch(() => null)
    }
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight
    }
  }, [messages, activeSessionId, authUser])

  const handleNewSession = () => {
    setActiveSessionId(null)
    setMessages([])
    setHighlightedTables([])
    setActiveView('chat')
    api
      .updateUserPreferences({ active_session_id: null, update_active_session: true })
      .catch(() => null)
  }

  const handleSelectSession = (id: string) => {
    const target = sessions.find((s) => s.id === id)
    if (!target) return
    setActiveSessionId(id)
    setMessages(target.messages)
    setHighlightedTables([])
    setActiveView('chat')
    api
      .updateUserPreferences({ active_session_id: id, update_active_session: true })
      .catch(() => null)
  }

  const handleDeleteSession = async (id: string) => {
    const next = sessions.filter((s) => s.id !== id)
    setSessions(next)
    if (id === activeSessionId) {
      const fallback = next[0] || null
      setActiveSessionId(fallback?.id || null)
      setMessages(fallback ? fallback.messages : [])
      setHighlightedTables([])
      api
        .updateUserPreferences({
          active_session_id: fallback?.id || null,
          update_active_session: true,
        })
        .catch(() => null)
    }
    await api.deleteChatSession(id).catch(() => null)
  }

  const handleAuthenticated = async (token: string, user: AuthUser) => {
    setAuthToken(token)
    setAuthUser(user)
    await syncUserRemoteState().catch(() => null)
  }

  const handleLogout = async () => {
    await api.logout().catch(() => null)
    setAuthToken(null)
    setAuthUser(null)
    setSessions([])
    setActiveSessionId(null)
    setMessages([])
    setSavedProfiles([])
    setLastConnection(null)
    notify('已退出登录')
  }

  const handleConnected = (res: {
    connected: boolean
    connection_meta: ConnectionMeta
    config: EngineConfig
  }) => {
    setConnected(res.connected)
    setConnectionMeta(res.connection_meta)
    setConfig(res.config)
    setHighlightedTables([])
    setSelectedTable(null)
    loadWorkspaceData()
  }

  const handleDisconnect = async () => {
    try {
      await api.disconnect()
      setConnected(false)
      setConnectionMeta({})
      setConfig(null)
      setSchemaGraph(null)
      setModules([])
      setHighlightedTables([])
      setSelectedTable(null)
      notify('已断开数据库连接')
    } catch (err: any) {
      notify(err.message || '断开连接失败', true)
    }
  }

  const handleUpdateConfig = async (patch: Partial<EngineConfig>) => {
    try {
      const next = await api.updateConfig(patch)
      setConfig(next)
    } catch (err: any) {
      notify(err.message || '更新配置失败', true)
    }
  }

  const handleRefreshCache = async () => {
    try {
      const stats = await api.getCacheStats()
      setCacheStats(stats)
    } catch {
      // ignore
    }
  }

  const handleClearCache = async () => {
    try {
      await api.clearCache()
      const stats = await api.getCacheStats()
      setCacheStats(stats)
      notify('语义缓存已清空')
    } catch (err: any) {
      notify(err.message || '清空缓存失败', true)
    }
  }

  const handleSelectTableFromSidebar = (tableName: string) => {
    setSelectedTable(tableName)
    setHighlightedTables([tableName])
    if (activeView === 'chat') {
      setSplitGraphOpen(true)
    } else if (activeView !== 'schema') {
      setActiveView('schema')
    }
  }

  const handleFocusTables = (tables: string[]) => {
    setHighlightedTables(tables)
    if (tables.length > 0) {
      setSelectedTable(tables[0])
    }
    if (activeView === 'chat') {
      setSplitGraphOpen(true)
    } else {
      setActiveView('schema')
    }
  }

  const handleSend = async (promptOverride?: string) => {
    const text = (promptOverride ?? input).trim()
    if (!text || isSending) return
    if (!connected) {
      setConnectModalOpen(true)
      notify('请先连接数据库或启用内置沙箱', true)
      return
    }

    if (!promptOverride) setInput('')
    setIsSending(true)
    const startedAt = performance.now()

    const userMsgId = `u-${Date.now()}`
    const aiMsgId = `a-${Date.now() + 1}`

    const history = messages
      .filter((m) => !m.isStreaming)
      .slice(-8)
      .map((m) => ({
        role: m.role,
        content: m.result?.sql
          ? `${m.content}\n[SQL]: ${m.result.sql}`
          : m.content,
      }))

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: 'user', content: text },
      {
        id: aiMsgId,
        role: 'assistant',
        content: '',
        statusLogs: [],
        streamingText: '',
        isStreaming: true,
      },
    ])

    try {
      await api.streamQuery(
        {
          question: text,
          mode: config?.query_mode,
          history,
        },
        {
          onStatus: (level, message) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aiMsgId
                  ? { ...m, statusLogs: [...(m.statusLogs || []), { level, message }] }
                  : m,
              ),
            )
          },
          onToken: (accumulated) => {
            setMessages((prev) =>
              prev.map((m) => (m.id === aiMsgId ? { ...m, streamingText: accumulated } : m)),
            )
          },
          onResult: (result) => {
            const elapsedSec = (performance.now() - startedAt) / 1000
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aiMsgId
                  ? {
                      ...m,
                      content: result.clean_markdown || result.markdown_answer || result.error || '',
                      result,
                      elapsedSec,
                      isStreaming: false,
                    }
                  : m,
              ),
            )
            if (result.linked_tables?.length) {
              setHighlightedTables(result.linked_tables)
            }
            handleRefreshCache()
          },
          onError: (errMsg) => {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aiMsgId
                  ? {
                      ...m,
                      content: `查询执行失败：${errMsg}`,
                      isStreaming: false,
                    }
                  : m,
              ),
            )
            notify(errMsg, true)
          },
        },
      )
    } catch (err: any) {
      const msg = err.message || '请求异常'
      setMessages((prev) =>
        prev.map((m) =>
          m.id === aiMsgId
            ? { ...m, content: `请求失败：${msg}`, isStreaming: false }
            : m,
        ),
      )
      notify(msg, true)
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className="app-shell">
      {/* 顶部导航栏 */}
      <header className="topbar">
        <div className="topbar-left">
          <button
            type="button"
            className="icon-btn"
            onClick={() => setSidebarCollapsed((v) => !v)}
            title={sidebarCollapsed ? '展开配置栏' : '收起配置栏'}
          >
            <PanelLeft size={16} />
          </button>

          <div className="brand-mark">
            <div className="brand-icon">
              <Database size={16} />
            </div>
            <span>SmartDBChat</span>
          </div>

          <nav className="nav-tabs" aria-label="主工作区导航">
            <button
              type="button"
              className={`nav-tab ${activeView === 'chat' ? 'active' : ''}`}
              onClick={() => setActiveView('chat')}
            >
              <MessageSquare size={14} />
              对话工作台
            </button>
            <button
              type="button"
              className={`nav-tab ${activeView === 'schema' ? 'active' : ''}`}
              onClick={() => setActiveView('schema')}
            >
              <GitFork size={14} />
              Schema 图谱
            </button>
            <button
              type="button"
              className={`nav-tab ${activeView === 'semantic' ? 'active' : ''}`}
              onClick={() => setActiveView('semantic')}
            >
              <BookOpen size={14} />
              语义层中心
            </button>
            <button
              type="button"
              className={`nav-tab ${activeView === 'mcp' ? 'active' : ''}`}
              onClick={() => setActiveView('mcp')}
            >
              <Server size={14} />
              MCP 服务中心
            </button>
          </nav>
        </div>

        <div className="topbar-right">
          {activeView === 'chat' && connected && (
            <button
              type="button"
              className={`btn btn-sm ${splitGraphOpen ? 'btn-primary' : ''}`}
              onClick={() => setSplitGraphOpen((v) => !v)}
              title="切换右侧 Schema 图谱分屏联动 (QueryWeaver 模式)"
            >
              <PanelRight size={14} />
              分屏图谱
            </button>
          )}

          <button
            type="button"
            className="btn btn-sm"
            onClick={() => setConnectModalOpen(true)}
          >
            <PlugZap size={14} color={connected ? 'var(--success)' : 'var(--text-muted)'} />
            {connected ? connectionMeta.database || '已连接' : '连接数据源'}
          </button>

          {messages.length > 0 && activeView === 'chat' && (
            <button
              type="button"
              className="icon-btn"
              onClick={() => {
                if (activeSessionId) {
                  handleDeleteSession(activeSessionId)
                } else {
                  handleNewSession()
                }
              }}
              title="删除当前对话"
            >
              <Trash2 size={15} />
            </button>
          )}

          <button
            type="button"
            className="icon-btn"
            onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
            title={theme === 'light' ? '切换深色模式' : '切换浅色模式'}
          >
            {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
          </button>

          {authUser && (
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span className="pill pill-info" style={{ textTransform: 'none' }}>
                <Shield size={11} />
                {authUser.username} · {authUser.role.toUpperCase()}
              </span>
              <button
                type="button"
                className="icon-btn"
                onClick={handleLogout}
                title="退出登录"
              >
                <LogOut size={15} />
              </button>
            </div>
          )}
        </div>
      </header>

      {/* 主体区域 */}
      <div className="workspace-body">
        <Sidebar
          collapsed={sidebarCollapsed}
          connected={connected}
          connectionMeta={connectionMeta}
          config={config}
          modules={modules}
          cacheStats={cacheStats}
          schemaGraph={schemaGraph}
          selectedTable={selectedTable}
          sessions={sessions}
          activeSessionId={activeSessionId}
          onNewSession={handleNewSession}
          onSelectSession={handleSelectSession}
          onDeleteSession={handleDeleteSession}
          onOpenConnect={() => setConnectModalOpen(true)}
          onDisconnect={handleDisconnect}
          onUpdateConfig={handleUpdateConfig}
          onRefreshCache={handleRefreshCache}
          onClearCache={handleClearCache}
          onSelectTable={handleSelectTableFromSidebar}
        />

        <main className="main-content" ref={mainContentRef}>
          {activeView === 'schema' && (
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
              <SchemaGraphCanvas
                graph={schemaGraph}
                highlightedTables={highlightedTables}
                selectedTable={selectedTable}
                onSelectTable={setSelectedTable}
              />
            </div>
          )}

          {activeView === 'semantic' && <SemanticLayerStudio onNotify={notify} />}

          {activeView === 'mcp' && (
            <McpStudio
              lastConnection={lastConnection}
              onOpenConnect={() => setConnectModalOpen(true)}
              onNotify={notify}
            />
          )}

          {activeView === 'chat' && (
            <>
              <section className="chat-pane">
                <div className="chat-scroll" ref={chatScrollRef}>
                  {messages.length === 0 ? (
                    <div className="welcome-panel">
                      <div className="welcome-hero">
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                          <span className="pill pill-accent">
                            <Sparkles size={12} />
                            NL2SQL 智能数据分析工作台
                          </span>
                          {connected && (
                            <span className="pill pill-success">
                              当前库：{connectionMeta.database}
                            </span>
                          )}
                        </div>
                        <h1>用自然语言对话探索关系型数据库</h1>
                        <p>
                          融合 Schema 关系图谱、ReAct Agent 自主探查、AST 只读安全护栏、业务语义层与交互式 Chart Studio。
                        </p>

                        {!connected && (
                          <div style={{ marginTop: 18, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                            <button
                              type="button"
                              className="btn btn-primary"
                              onClick={() => setConnectModalOpen(true)}
                            >
                              <PlugZap size={15} />
                              连接数据库 / 一键启用演示沙箱
                            </button>
                          </div>
                        )}
                      </div>

                      {connected && (
                        <div>
                          <div
                            style={{
                              fontSize: 12,
                              fontWeight: 600,
                              color: 'var(--text-muted)',
                              marginBottom: 10,
                            }}
                          >
                            推荐分析问题（点击直接运行）
                          </div>
                          <div className="starter-grid">
                            {STARTER_PROMPTS.map((item) => (
                              <button
                                key={item.title}
                                type="button"
                                className="starter-card"
                                onClick={() => handleSend(item.title)}
                              >
                                <div className="starter-card-title">{item.title}</div>
                                <div className="starter-card-sub">{item.sub}</div>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    messages.map((msg, idx) => {
                      let precedingQuestion = ''
                      if (msg.role === 'assistant') {
                        for (let j = idx - 1; j >= 0; j--) {
                          if (messages[j].role === 'user') {
                            precedingQuestion = messages[j].content
                            break
                          }
                        }
                      }
                      return (
                        <MessageCard
                          key={msg.id}
                          message={msg}
                          userQuestion={precedingQuestion}
                          onSendPrompt={(q) => handleSend(q)}
                          onFocusTables={handleFocusTables}
                          onNotify={notify}
                        />
                      )
                    })
                  )}
                </div>

                {/* 底部输入栏 */}
                <div className="composer-bar">
                  <div className="composer-inner">
                    <textarea
                      className="composer-input"
                      rows={1}
                      placeholder={
                        connected
                          ? '输入你的数据分析问题，例如：各学院平均 GPA 与消费总额对比... (Enter 发送，Shift+Enter 换行)'
                          : '请先点击右上角“连接数据源”启用内置沙箱或 MySQL...'
                      }
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault()
                          handleSend()
                        }
                      }}
                    />
                    {config && (
                      <span className="pill" style={{ flexShrink: 0 }}>
                        {config.query_mode}
                      </span>
                    )}
                    <button
                      type="button"
                      className="btn btn-primary"
                      disabled={isSending || !input.trim()}
                      onClick={() => handleSend()}
                    >
                      <Send size={14} />
                      {isSending ? '分析中' : '发送'}
                    </button>
                  </div>
                </div>
              </section>

              {splitGraphOpen && connected && (
                <>
                  <div
                    className={`split-resizer-vertical ${isResizingGraph ? 'active' : ''}`}
                    role="separator"
                    aria-orientation="vertical"
                    aria-label="调整 Schema 图谱宽度"
                    onPointerDown={handleStartResizeGraph}
                  />
                  <aside className="split-graph-pane" style={{ width: splitGraphWidth }}>
                    <SchemaGraphCanvas
                      graph={schemaGraph}
                      highlightedTables={highlightedTables}
                      selectedTable={selectedTable}
                      onSelectTable={setSelectedTable}
                      onClose={() => setSplitGraphOpen(false)}
                      compact
                    />
                  </aside>
                </>
              )}
            </>
          )}
        </main>
      </div>

      {/* 连接配置模态框 */}
      <ConnectionModal
        open={connectModalOpen}
        onClose={() => setConnectModalOpen(false)}
        providers={providers}
        initialConfig={config}
        initialMeta={connectionMeta}
        lastConnection={lastConnection}
        savedProfiles={savedProfiles}
        onProfilesChange={(profs, nextLast) => {
          setSavedProfiles(profs)
          if (nextLast) setLastConnection(nextLast)
        }}
        onConnected={handleConnected}
        onNotify={notify}
      />

      {/* 登录/注册模态框 */}
      {!authChecking && !authUser && (
        <AuthModal onAuthenticated={handleAuthenticated} onNotify={notify} />
      )}

      {/* 全局 Toast 通知 */}
      {toast && (
        <div
          style={{
            position: 'fixed',
            right: 20,
            bottom: 20,
            zIndex: 80,
            padding: '10px 14px',
            borderRadius: 'var(--radius-md)',
            background: toast.isError ? 'var(--danger)' : 'var(--bg-code)',
            color: '#ffffff',
            fontSize: 13,
            boxShadow: 'var(--shadow-lg)',
            maxWidth: 380,
          }}
        >
          {toast.msg}
        </div>
      )}
    </div>
  )
}

export default App
