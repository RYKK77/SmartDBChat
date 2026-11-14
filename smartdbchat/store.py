import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional


def _hash_password(password: str, salt: Optional[str] = None) -> str:
    salt_hex = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        100_000,
    )
    return f"{salt_hex}${dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    if "$" not in stored_hash:
        return False
    salt_hex, _ = stored_hash.split("$", 1)
    try:
        candidate = _hash_password(password, salt=salt_hex)
    except ValueError:
        return False
    return hmac.compare_digest(candidate, stored_hash)


class AppStore:
    """基于 SQLite 的服务端用户认证、权限与连接/会话持久化存储。"""

    def __init__(self, db_path: str = "smartdb_app.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS user_connections (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id INTEGER PRIMARY KEY,
                    last_connection_json TEXT,
                    auto_connect_disabled INTEGER NOT NULL DEFAULT 0,
                    active_session_id TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS mcp_api_keys (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    api_key TEXT NOT NULL UNIQUE,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    call_count INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL,
                    last_used_at INTEGER,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS mcp_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    require_api_key INTEGER NOT NULL DEFAULT 1
                );

                INSERT OR IGNORE INTO mcp_settings (id, require_api_key) VALUES (1, 1);
                """
            )

    def has_any_users(self) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(1) AS cnt FROM users").fetchone()
            return bool(row and row["cnt"] > 0)

    def register_user(self, username: str, password: str) -> Dict[str, Any]:
        clean_user = username.strip()
        if len(clean_user) < 2:
            raise ValueError("用户名至少需要 2 个字符")
        if len(password) < 4:
            raise ValueError("密码至少需要 4 个字符")

        now = int(time.time())
        pw_hash = _hash_password(password)

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE username = ?", (clean_user,)
            ).fetchone()
            if existing:
                raise ValueError("该用户名已被注册")

            count_row = conn.execute("SELECT COUNT(1) AS cnt FROM users").fetchone()
            role = "admin" if (count_row and count_row["cnt"] == 0) else "user"

            cur = conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (clean_user, pw_hash, role, now),
            )
            user_id = int(cur.lastrowid)
            conn.execute(
                "INSERT OR IGNORE INTO user_preferences (user_id) VALUES (?)",
                (user_id,),
            )
            token = secrets.token_urlsafe(32)
            conn.execute(
                "INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, now + 30 * 86400),
            )
            return {
                "token": token,
                "user": {"id": user_id, "username": clean_user, "role": role},
            }

    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        clean_user = username.strip()
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, role FROM users WHERE username = ?",
                (clean_user,),
            ).fetchone()
            if not row or not _verify_password(password, row["password_hash"]):
                raise ValueError("用户名或密码错误")

            user_id = int(row["id"])
            token = secrets.token_urlsafe(32)
            conn.execute(
                "INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, now + 30 * 86400),
            )
            return {
                "token": token,
                "user": {
                    "id": user_id,
                    "username": row["username"],
                    "role": row["role"],
                },
            }

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        now = int(time.time())
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.role, t.expires_at
                FROM auth_tokens t
                JOIN users u ON u.id = t.user_id
                WHERE t.token = ?
                """,
                (token,),
            ).fetchone()
            if not row:
                return None
            if int(row["expires_at"]) < now:
                conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
                return None
            return {
                "id": int(row["id"]),
                "username": row["username"],
                "role": row["role"],
            }

    def revoke_token(self, token: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))

    def get_user_state(self, user_id: int) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            pref = conn.execute(
                "SELECT last_connection_json, auto_connect_disabled, active_session_id FROM user_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            last_connection = None
            auto_connect_disabled = False
            active_session_id = None
            if pref:
                if pref["last_connection_json"]:
                    try:
                        last_connection = json.loads(pref["last_connection_json"])
                    except Exception:
                        last_connection = None
                auto_connect_disabled = bool(pref["auto_connect_disabled"])
                active_session_id = pref["active_session_id"]

            conn_rows = conn.execute(
                "SELECT id, name, payload_json, updated_at FROM user_connections WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
            profiles: List[Dict[str, Any]] = []
            for r in conn_rows:
                try:
                    profiles.append(
                        {
                            "id": r["id"],
                            "name": r["name"],
                            "payload": json.loads(r["payload_json"]),
                            "updatedAt": int(r["updated_at"]),
                        }
                    )
                except Exception:
                    continue

            sess_rows = conn.execute(
                "SELECT id, title, messages_json, updated_at FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
            sessions: List[Dict[str, Any]] = []
            for r in sess_rows:
                try:
                    sessions.append(
                        {
                            "id": r["id"],
                            "title": r["title"],
                            "messages": json.loads(r["messages_json"]),
                            "updatedAt": int(r["updated_at"]),
                        }
                    )
                except Exception:
                    continue

            return {
                "last_connection": last_connection,
                "auto_connect_disabled": auto_connect_disabled,
                "active_session_id": active_session_id,
                "profiles": profiles,
                "sessions": sessions,
            }

    def update_preferences(
        self,
        user_id: int,
        last_connection: Optional[Dict[str, Any]] = None,
        update_last_connection: bool = False,
        auto_connect_disabled: Optional[bool] = None,
        active_session_id: Optional[str] = None,
        update_active_session: bool = False,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO user_preferences (user_id) VALUES (?)",
                (user_id,),
            )
            if update_last_connection:
                conn.execute(
                    "UPDATE user_preferences SET last_connection_json = ? WHERE user_id = ?",
                    (
                        json.dumps(last_connection, ensure_ascii=False)
                        if last_connection is not None
                        else None,
                        user_id,
                    ),
                )
            if auto_connect_disabled is not None:
                conn.execute(
                    "UPDATE user_preferences SET auto_connect_disabled = ? WHERE user_id = ?",
                    (1 if auto_connect_disabled else 0, user_id),
                )
            if update_active_session:
                conn.execute(
                    "UPDATE user_preferences SET active_session_id = ? WHERE user_id = ?",
                    (active_session_id, user_id),
                )

    def save_connection_profile(
        self,
        user_id: int,
        name: str,
        payload: Dict[str, Any],
        profile_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        now = int(time.time() * 1000)
        db_type = payload.get("db_type", "sandbox")
        db_name = payload.get("database") or "MySQL"
        provider = payload.get("provider") or "LLM"
        clean_name = name.strip() or f"{'沙箱' if db_type == 'sandbox' else db_name} · {provider}"

        with self._lock, self._connect() as conn:
            if profile_id:
                owner_row = conn.execute(
                    "SELECT user_id FROM user_connections WHERE id = ?", (profile_id,)
                ).fetchone()
                if owner_row and int(owner_row["user_id"]) != user_id:
                    raise PermissionError("无权修改其他用户的连接配置")
                if owner_row:
                    conn.execute(
                        "UPDATE user_connections SET name = ?, payload_json = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                        (
                            clean_name,
                            json.dumps(payload, ensure_ascii=False),
                            now,
                            profile_id,
                            user_id,
                        ),
                    )
                else:
                    conn.execute(
                        "INSERT INTO user_connections (id, user_id, name, payload_json, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            profile_id,
                            user_id,
                            clean_name,
                            json.dumps(payload, ensure_ascii=False),
                            now,
                        ),
                    )
            else:
                new_id = f"conn-{now}-{secrets.token_hex(3)}"
                conn.execute(
                    "INSERT INTO user_connections (id, user_id, name, payload_json, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        new_id,
                        user_id,
                        clean_name,
                        json.dumps(payload, ensure_ascii=False),
                        now,
                    ),
                )
        return self.get_user_state(user_id)["profiles"]

    def delete_connection_profile(self, user_id: int, profile_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            owner_row = conn.execute(
                "SELECT user_id FROM user_connections WHERE id = ?", (profile_id,)
            ).fetchone()
            if owner_row and int(owner_row["user_id"]) != user_id:
                raise PermissionError("无权删除其他用户的连接配置")
            conn.execute(
                "DELETE FROM user_connections WHERE id = ? AND user_id = ?",
                (profile_id, user_id),
            )
        return self.get_user_state(user_id)["profiles"]

    def save_chat_session(
        self,
        user_id: int,
        session_id: str,
        title: str,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        now = int(time.time() * 1000)
        clean_title = title.strip() or "新对话"
        with self._lock, self._connect() as conn:
            owner_row = conn.execute(
                "SELECT user_id FROM chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if owner_row and int(owner_row["user_id"]) != user_id:
                raise PermissionError("无权修改其他用户的对话记录")
            if owner_row:
                conn.execute(
                    "UPDATE chat_sessions SET title = ?, messages_json = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                    (
                        clean_title,
                        json.dumps(messages, ensure_ascii=False),
                        now,
                        session_id,
                        user_id,
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO chat_sessions (id, user_id, title, messages_json, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        user_id,
                        clean_title,
                        json.dumps(messages, ensure_ascii=False),
                        now,
                    ),
                )
        return self.get_user_state(user_id)["sessions"]

    def delete_chat_session(self, user_id: int, session_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            owner_row = conn.execute(
                "SELECT user_id FROM chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if owner_row and int(owner_row["user_id"]) != user_id:
                raise PermissionError("无权删除其他用户的对话记录")
            conn.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
        return self.get_user_state(user_id)["sessions"]

    def get_mcp_settings(self) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT require_api_key FROM mcp_settings WHERE id = 1"
            ).fetchone()
            return {"require_api_key": bool(row["require_api_key"]) if row else True}

    def update_mcp_settings(self, require_api_key: bool) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE mcp_settings SET require_api_key = ? WHERE id = 1",
                (1 if require_api_key else 0,),
            )
        return self.get_mcp_settings()

    def list_mcp_keys(self, user_id: int) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, api_key, is_active, call_count, created_at, last_used_at
                FROM mcp_api_keys
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "api_key": r["api_key"],
                    "is_active": bool(r["is_active"]),
                    "call_count": int(r["call_count"] or 0),
                    "created_at": int(r["created_at"]),
                    "last_used_at": int(r["last_used_at"]) if r["last_used_at"] else None,
                }
                for r in rows
            ]

    def create_mcp_key(self, user_id: int, name: str) -> Dict[str, Any]:
        clean_name = (name or "").strip() or "MCP Client Key"
        now = int(time.time() * 1000)
        key_id = f"mcpk-{now}-{secrets.token_hex(3)}"
        api_key = f"sdb_mcp_{secrets.token_urlsafe(24)}"
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO mcp_api_keys (id, user_id, name, api_key, is_active, call_count, created_at, last_used_at)
                VALUES (?, ?, ?, ?, 1, 0, ?, NULL)
                """,
                (key_id, user_id, clean_name, api_key, now),
            )
        return {
            "id": key_id,
            "name": clean_name,
            "api_key": api_key,
            "is_active": True,
            "call_count": 0,
            "created_at": now,
            "last_used_at": None,
        }

    def toggle_mcp_key(self, user_id: int, key_id: str, is_active: bool) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            owner_row = conn.execute(
                "SELECT user_id FROM mcp_api_keys WHERE id = ?", (key_id,)
            ).fetchone()
            if owner_row and int(owner_row["user_id"]) != user_id:
                raise PermissionError("无权修改其他用户的 MCP 密钥")
            conn.execute(
                "UPDATE mcp_api_keys SET is_active = ? WHERE id = ? AND user_id = ?",
                (1 if is_active else 0, key_id, user_id),
            )
        return self.list_mcp_keys(user_id)

    def delete_mcp_key(self, user_id: int, key_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            owner_row = conn.execute(
                "SELECT user_id FROM mcp_api_keys WHERE id = ?", (key_id,)
            ).fetchone()
            if owner_row and int(owner_row["user_id"]) != user_id:
                raise PermissionError("无权删除其他用户的 MCP 密钥")
            conn.execute(
                "DELETE FROM mcp_api_keys WHERE id = ? AND user_id = ?",
                (key_id, user_id),
            )
        return self.list_mcp_keys(user_id)

    def verify_and_touch_mcp_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        clean_key = (api_key or "").strip()
        if not clean_key:
            return None
        now = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, name, is_active FROM mcp_api_keys WHERE api_key = ?",
                (clean_key,),
            ).fetchone()
            if not row or not bool(row["is_active"]):
                return None
            conn.execute(
                "UPDATE mcp_api_keys SET call_count = call_count + 1, last_used_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            return {
                "id": row["id"],
                "user_id": int(row["user_id"]),
                "name": row["name"],
            }
