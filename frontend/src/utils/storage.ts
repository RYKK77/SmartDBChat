import type { ChatMessage, ConnectPayload } from '../types/api'

const AUTH_TOKEN_KEY = 'smartdb_auth_token'

// 旧版本地存储 Key（仅用于首次登录迁移至后端后立即清理，不再写入浏览器）
const LEGACY_LAST_CONNECTION_KEY = 'smartdb_last_connection'
const LEGACY_SAVED_PROFILES_KEY = 'smartdb_saved_connections'
const LEGACY_AUTO_CONNECT_DISABLED_KEY = 'smartdb_auto_connect_disabled'
const LEGACY_MESSAGES_KEY = 'smartdb_chat_messages'
const LEGACY_SESSIONS_KEY = 'smartdb_chat_sessions'
const LEGACY_ACTIVE_SESSION_KEY = 'smartdb_active_session_id'

export interface AuthUser {
  id: number
  username: string
  role: 'admin' | 'user' | string
}

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

export function getAuthToken(): string | null {
  return localStorage.getItem(AUTH_TOKEN_KEY)
}

export function setAuthToken(token: string | null): void {
  if (token) {
    localStorage.setItem(AUTH_TOKEN_KEY, token)
  } else {
    localStorage.removeItem(AUTH_TOKEN_KEY)
  }
}

export function deriveSessionTitle(messages: ChatMessage[]): string {
  const firstUser = messages.find((m) => m.role === 'user' && m.content.trim())
  if (!firstUser) return '新对话'
  const text = firstUser.content.trim().replace(/\s+/g, ' ')
  return text.length > 22 ? `${text.slice(0, 22)}...` : text
}

export function consumeLegacyBrowserData(): {
  lastConnection: ConnectPayload | null
  profiles: SavedConnectionProfile[]
  sessions: ChatSession[]
} {
  let lastConnection: ConnectPayload | null = null
  let profiles: SavedConnectionProfile[] = []
  let sessions: ChatSession[] = []

  try {
    const rawLast = localStorage.getItem(LEGACY_LAST_CONNECTION_KEY)
    if (rawLast) {
      const parsed = JSON.parse(rawLast)
      if (parsed && typeof parsed === 'object' && typeof parsed.provider === 'string') {
        lastConnection = parsed as ConnectPayload
      }
    }
  } catch {
    // ignore
  }

  try {
    const rawProfiles = localStorage.getItem(LEGACY_SAVED_PROFILES_KEY)
    if (rawProfiles) {
      const parsed = JSON.parse(rawProfiles)
      if (Array.isArray(parsed)) {
        profiles = parsed.filter((p) => p && typeof p.id === 'string' && p.payload)
      }
    }
  } catch {
    // ignore
  }

  try {
    const rawSessions = localStorage.getItem(LEGACY_SESSIONS_KEY)
    if (rawSessions) {
      const parsed = JSON.parse(rawSessions)
      if (Array.isArray(parsed)) {
        sessions = parsed.filter((s) => s && typeof s.id === 'string' && Array.isArray(s.messages))
      }
    } else {
      const rawLegacyMsgs = localStorage.getItem(LEGACY_MESSAGES_KEY)
      if (rawLegacyMsgs) {
        const parsedMsgs = JSON.parse(rawLegacyMsgs)
        if (Array.isArray(parsedMsgs) && parsedMsgs.length > 0) {
          sessions = [
            {
              id: `sess-${Date.now()}`,
              title: deriveSessionTitle(parsedMsgs),
              updatedAt: Date.now(),
              messages: parsedMsgs,
            },
          ]
        }
      }
    }
  } catch {
    // ignore
  }

  // 彻底清除浏览器中的敏感连接配置与历史对话缓存
  localStorage.removeItem(LEGACY_LAST_CONNECTION_KEY)
  localStorage.removeItem(LEGACY_SAVED_PROFILES_KEY)
  localStorage.removeItem(LEGACY_AUTO_CONNECT_DISABLED_KEY)
  localStorage.removeItem(LEGACY_MESSAGES_KEY)
  localStorage.removeItem(LEGACY_SESSIONS_KEY)
  localStorage.removeItem(LEGACY_ACTIVE_SESSION_KEY)

  return { lastConnection, profiles, sessions }
}
