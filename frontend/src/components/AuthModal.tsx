import { useState } from 'react'
import { Database, KeyRound, ShieldCheck, UserPlus } from 'lucide-react'
import { api } from '../services/api'
import type { AuthUser } from '../utils/storage'

interface AuthModalProps {
  onAuthenticated: (token: string, user: AuthUser) => void
  onNotify: (msg: string, isError?: boolean) => void
}

export function AuthModal({ onAuthenticated, onNotify }: AuthModalProps) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password) {
      onNotify('请输入用户名和密码', true)
      return
    }
    setSubmitting(true)
    try {
      const res =
        mode === 'register'
          ? await api.register(username.trim(), password)
          : await api.login(username.trim(), password)
      onAuthenticated(res.token, res.user)
      onNotify(
        mode === 'register'
          ? `注册成功，欢迎 ${res.user.username}（角色: ${res.user.role.toUpperCase()}）`
          : `欢迎回来，${res.user.username}`,
      )
    } catch (err: any) {
      onNotify(err.message || (mode === 'register' ? '注册失败' : '登录失败'), true)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="modal-backdrop">
      <div className="modal-card" style={{ maxWidth: 420 }}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div className="brand-icon" style={{ width: 32, height: 32 }}>
              <Database size={16} />
            </div>
            <div>
              <div style={{ fontWeight: 700, fontSize: 15 }}>
                SmartDBChat 账号{mode === 'login' ? '登录' : '注册'}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                连接配置（含 API Key）与历史对话安全持久化在服务端
              </div>
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div className="segmented">
              <button
                type="button"
                className={mode === 'login' ? 'active' : ''}
                onClick={() => setMode('login')}
              >
                <KeyRound size={13} style={{ marginRight: 5 }} />
                账号登录
              </button>
              <button
                type="button"
                className={mode === 'register' ? 'active' : ''}
                onClick={() => setMode('register')}
              >
                <UserPlus size={13} style={{ marginRight: 5 }} />
                新用户注册
              </button>
            </div>

            <div className="field-group">
              <label className="field-label">用户名</label>
              <input
                className="input"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="至少 2 个字符，如 admin"
                autoFocus
              />
            </div>

            <div className="field-group">
              <label className="field-label">密码</label>
              <input
                className="input"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="至少 4 个字符"
              />
            </div>

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                fontSize: 11.5,
                color: 'var(--text-muted)',
                background: 'var(--bg-subtle)',
                padding: '8px 10px',
                borderRadius: 'var(--radius-sm)',
              }}
            >
              <ShieldCheck size={14} color="var(--accent)" style={{ flexShrink: 0 }} />
              <span>
                首个注册账号自动授予管理员（ADMIN）权限；各账号的连接凭据与对话记录在服务端严格隔离。
              </span>
            </div>
          </div>

          <div className="modal-footer">
            <button
              type="submit"
              className="btn btn-primary"
              disabled={submitting}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              {submitting ? '正在验证...' : mode === 'login' ? '立即登录' : '注册并进入系统'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
