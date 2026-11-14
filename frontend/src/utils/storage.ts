import type { ChatMessage, ConnectPayload } from '../types/api'

const LAST_CONNECTION_KEY = 'smartdb_last_connection'
const SAVED_PROFILES_KEY = 'smartdb_saved_connections'
const AUTO_CONNECT_DISABLED_KEY = 'smartdb_auto_connect_disabled'

const LEGACY_MESSAGES_KEY = 'smartdb_chat_messages'
const CHAT_SESSIONS_KEY = 'smartdb_chat_sessions'
const ACTIVE_SESSION_KEY = 'smartdb_active_session_id'

export interface SavedConnectionProfile {
  id: string
  name: string
  payload: ConnectPayload
  updatedAt: number
}

export interface ChatSession {
  id: string
  title: string
  updatedAt: number
  messages: ChatMessage[]
}

export function loadLastConnection(): ConnectPayload | null {
  try {
    const raw = localStorage.getItem(LAST_CONNECTION_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (parsed && typeof parsed === 'object' && typeof parsed.provider === 'string') {
      return parsed as ConnectPayload
    }
    return null
  } catch {
    return null
  }
}

export function saveLastConnection(payload: ConnectPayload): void {
  try {
    localStorage.setItem(LAST_CONNECTION_KEY, JSON.stringify(payload))
  } catch {
    // ignore storage errors
  }
}

export function isAutoConnectDisabled(): boolean {
  return localStorage.getItem(AUTO_CONNECT_DISABLED_KEY) === '1'
}

export function setAutoConnectDisabled(disabled: boolean): void {
  if (disabled) {
    localStorage.setItem(AUTO_CONNECT_DISABLED_KEY, '1')
  } else {
    localStorage.removeItem(AUTO_CONNECT_DISABLED_KEY)
  }
}

export function loadSavedProfiles(): SavedConnectionProfile[] {
  try {
    const raw = localStorage.getItem(SAVED_PROFILES_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.filter(
      (p) => p && typeof p.id === 'string' && typeof p.name === 'string' && p.payload,
    )
  } catch {
    return []
  }
}

export function saveConnectionProfile(
  name: string,
  payload: ConnectPayload,
  existingId?: string,
): SavedConnectionProfile[] {
  const profiles = loadSavedProfiles()
  const now = Date.now()
  const trimmedName = name.trim() || `${payload.db_type === 'sandbox' ? '沙箱' : payload.database || 'MySQL'} · ${payload.provider}`

  let next: SavedConnectionProfile[]
  if (existingId && profiles.some((p) => p.id === existingId)) {
    next = profiles.map((p) =>
      p.id === existingId ? { ...p, name: trimmedName, payload: { ...payload }, updatedAt: now } : p,
    )
  } else {
    const newProfile: SavedConnectionProfile = {
      id: `conn-${now}-${Math.random().toString(36).slice(2, 6)}`,
      name: trimmedName,
      payload: { ...payload },
      updatedAt: now,
    }
    next = [newProfile, ...profiles].slice(0, 20)
  }

  try {
    localStorage.setItem(SAVED_PROFILES_KEY, JSON.stringify(next))
  } catch {
    // ignore
  }
  return next
}

export function deleteConnectionProfile(id: string): SavedConnectionProfile[] {
  const next = loadSavedProfiles().filter((p) => p.id !== id)
  try {
    localStorage.setItem(SAVED_PROFILES_KEY, JSON.stringify(next))
  } catch {
    // ignore
  }
  return next
}

function sanitizeMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages
    .filter((m) => m && typeof m.id === 'string' && typeof m.role === 'string' && !m.isStreaming)
    .slice(-30)
    .map((m) => ({ ...m, isStreaming: false }))
}

export function deriveSessionTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find((m) => m.role === 'user' && m.content.trim())
  if (!firstUser) return '新对话'
  const text = firstUser.content.trim().replace(/\s+/g, ' ')
  return text.length > 22 ? `${text.slice(0, 22)}...` : text
}

export function loadChatSessions(): ChatSession[] {
  try {
    const raw = localStorage.getItem(CHAT_SESSIONS_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) {
        return parsed
          .filter((s) => s && typeof s.id === 'string' && Array.isArray(s.messages))
          .map((s) => ({
            id: s.id,
            title: typeof s.title === 'string' && s.title ? s.title : deriveSessionTitle(s.messages),
            updatedAt: typeof s.updatedAt === 'number' ? s.updatedAt : Date.now(),
            messages: s.messages.map((m: ChatMessage) => ({ ...m, isStreaming: false })),
          }))
      }
    }

    // 兼容迁移旧版单会话存储
    const legacyRaw = localStorage.getItem(LEGACY_MESSAGES_KEY)
    if (legacyRaw) {
      const legacyParsed = JSON.parse(legacyRaw)
      if (Array.isArray(legacyParsed) && legacyParsed.length > 0) {
        const migrated: ChatSession = {
          id: `sess-${Date.now()}`,
          title: deriveSessionTitle(legacyParsed),
          updatedAt: Date.now(),
          messages: sanitizeMessages(legacyParsed),
        }
        saveChatSessions([migrated])
        localStorage.removeItem(LEGACY_MESSAGES_KEY)
        return [migrated]
      }
    }
  } catch {
    // ignore parse errors
  }
  return []
}

export function saveChatSessions(sessions: ChatSession[]): void {
  const normalized = sessions
    .map((s) => ({
      ...s,
      messages: sanitizeMessages(s.messages),
    }))
    .filter((s) => s.messages.length > 0)
    .slice(0, 25)

  try {
    localStorage.setItem(CHAT_SESSIONS_KEY, JSON.stringify(normalized))
  } catch {
    // 超出配额时只保留最近 10 个会话，并裁剪较早会话的大体积 rows
    try {
      const trimmed = normalized.slice(0, 10).map((s, idx) =>
        idx >= 3
          ? {
              ...s,
              messages: s.messages.map((m) =>
                m.result
                  ? { ...m, result: { ...m.result, rows: m.result.rows.slice(0, 20) } }
                  : m,
              ),
            }
          : s,
      )
      localStorage.setItem(CHAT_SESSIONS_KEY, JSON.stringify(trimmed))
    } catch {
      // ignore
    }
  }
}

export function loadActiveSessionId(): string | null {
  return localStorage.getItem(ACTIVE_SESSION_KEY)
}

export function saveActiveSessionId(id: string | null): void {
  if (id) {
    localStorage.setItem(ACTIVE_SESSION_KEY, id)
  } else {
    localStorage.removeItem(ACTIVE_SESSION_KEY)
  }
}
