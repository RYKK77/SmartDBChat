import logging
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector

from smartdbchat.config import DatabaseConfig

logger = logging.getLogger(__name__)

_READ_ONLY_PREFIXES = ("SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN")
_FORBIDDEN_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "REPLACE",
    "GRANT",
    "REVOKE",
    "LOCK",
    "UNLOCK",
    "CALL",
)


class UnsafeSQLError(ValueError):
    """当生成的 SQL 不是安全的只读查询时抛出。"""


def validate_read_only_sql(sql: str) -> Tuple[bool, Optional[str]]:
    """校验 SQL 是否为单条只读查询语句（基于 sqlglot AST 解析）。"""
    from smartdbchat.sql_guard import validate_sql_ast

    guard_res = validate_sql_ast(sql, dialect="mysql")
    return guard_res.is_valid, guard_res.reason


def _normalize_value(value: Any) -> Any:
    """将 MySQL 返回的 Decimal / 日期类型规范化为标准 Python 基础类型。"""
    from datetime import date, datetime
    from decimal import Decimal

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{k: _normalize_value(v) for k, v in row.items()} for row in rows]


def extract_tables_from_sql(sql: str) -> List[str]:
    """从 SQL 语句中提取引用的表名（去重，优先使用 AST 解析）。"""
    from smartdbchat.sql_guard import validate_sql_ast

    guard_res = validate_sql_ast(sql, dialect="mysql")
    if guard_res.tables:
        return guard_res.tables
    from_pattern = r"(?:FROM|JOIN)\s+`?([A-Za-z0-9_]+)`?"
    tables = re.findall(from_pattern, sql, re.IGNORECASE)
    return list(dict.fromkeys(tables))


class DatabaseManager:
    """负责 MySQL 连接、Schema 提取与安全只读查询执行。"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.conn = None
        self.cursor = None

    def connect(self) -> None:
        self.conn = mysql.connector.connect(**self.config.to_mysql_kwargs())
        self.cursor = self.conn.cursor(dictionary=True)

    def close(self) -> None:
        try:
            if self.cursor is not None:
                self.cursor.close()
        finally:
            self.cursor = None
            if self.conn is not None:
                try:
                    self.conn.close()
                finally:
                    self.conn = None

    def _ensure_connected(self) -> None:
        if self.conn is None or self.cursor is None or not self.conn.is_connected():
            self.connect()

    def _drain_cursor(self) -> None:
        try:
            if self.cursor is not None and self.cursor.with_rows:
                self.cursor.fetchall()
        except Exception:
            pass

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """校验并执行只读 SQL 查询，返回规范化后的字典列表。"""
        is_valid, reason = validate_read_only_sql(sql)
        if not is_valid:
            raise UnsafeSQLError(reason or "不安全的 SQL 语句")

        self._ensure_connected()
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            return _normalize_rows(list(result)) if result else []
        except Exception:
            self._drain_cursor()
            try:
                self.connect()
            except Exception as conn_err:
                logger.warning("重置数据库连接失败: %s", conn_err)
            raise

    def get_schema_info(self, tables: Optional[List[str]] = None) -> str:
        """获取指定表或当前数据库全部表的建表语句。"""
        self._ensure_connected()
        schema_info: List[str] = []

        target_tables = tables
        if target_tables is None:
            self.cursor.execute(
                """
                SELECT TABLE_NAME
                FROM information_schema.TABLES
                WHERE TABLE_SCHEMA = %s
                """,
                (self.config.database,),
            )
            rows = self.cursor.fetchall()
            target_tables = [row["TABLE_NAME"] for row in rows]

        for table_name in target_tables:
            try:
                self.cursor.execute(
                    """
                    SELECT 1
                    FROM information_schema.TABLES
                    WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
                    """,
                    (self.config.database, table_name),
                )
                exists = self.cursor.fetchone()
                self._drain_cursor()

                if not exists:
                    continue

                self.cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
                create_table = self.cursor.fetchone()
                self._drain_cursor()

                if create_table and "Create Table" in create_table:
                    schema_info.append(create_table["Create Table"])
            except Exception as exc:
                logger.warning("获取表 %s 的 schema 失败: %s", table_name, exc)
                self._drain_cursor()

        return "\n\n".join(schema_info)

    def list_tables(self) -> List[str]:
        """列出当前数据库的全部表名。"""
        self._ensure_connected()
        self.cursor.execute(
            """
            SELECT TABLE_NAME
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = %s
            ORDER BY TABLE_NAME
            """,
            (self.config.database,),
        )
        rows = self.cursor.fetchall()
        self._drain_cursor()
        return [row["TABLE_NAME"] for row in rows]

    @staticmethod
    def _sanitize_identifier(identifier: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_]+", identifier or ""):
            raise ValueError(f"非法的数据库标识符: {identifier!r}")
        return identifier

    def explain_sql(self, sql: str) -> Dict[str, Any]:
        """校验只读安全性并执行 EXPLAIN 检查 SQL 语法与执行计划。"""
        is_valid, reason = validate_read_only_sql(sql)
        if not is_valid:
            raise UnsafeSQLError(reason or "不安全的 SQL 语句")

        cleaned = sql.strip().rstrip(";")
        self._ensure_connected()
        try:
            self.cursor.execute(f"EXPLAIN {cleaned}")
            plan_rows = _normalize_rows(list(self.cursor.fetchall() or []))
            return {"valid": True, "plan": plan_rows}
        except Exception:
            self._drain_cursor()
            raise

    def sample_table_rows(self, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """安全采样指定表的前 N 行数据。"""
        safe_table = self._sanitize_identifier(table_name)
        safe_limit = max(1, min(int(limit), 10))
        return self.execute_query(f"SELECT * FROM `{safe_table}` LIMIT {safe_limit}")

    def get_distinct_values(
        self, table_name: str, column_name: str, limit: int = 20
    ) -> List[Any]:
        """获取指定表列的去重非空示例值（用于 Value Linking）。"""
        safe_table = self._sanitize_identifier(table_name)
        safe_col = self._sanitize_identifier(column_name)
        safe_limit = max(1, min(int(limit), 50))
        rows = self.execute_query(
            f"SELECT DISTINCT `{safe_col}` AS val FROM `{safe_table}` "
            f"WHERE `{safe_col}` IS NOT NULL LIMIT {safe_limit}"
        )
        return [row["val"] for row in rows if "val" in row]

    def search_column_values(
        self, table_name: str, column_name: str, keyword: str, limit: int = 20
    ) -> List[Any]:
        """按关键字模糊检索指定表列的去重非空值（用于枚举值/名称精准对齐）。"""
        safe_table = self._sanitize_identifier(table_name)
        safe_col = self._sanitize_identifier(column_name)
        safe_limit = max(1, min(int(limit), 50))
        kw = (keyword or "").strip()
        if not kw:
            return self.get_distinct_values(safe_table, safe_col, limit=safe_limit)
        self._ensure_connected()
        try:
            self.cursor.execute(
                f"SELECT DISTINCT `{safe_col}` AS val FROM `{safe_table}` "
                f"WHERE `{safe_col}` IS NOT NULL AND CAST(`{safe_col}` AS CHAR) LIKE %s "
                f"LIMIT %s",
                (f"%{kw}%", safe_limit),
            )
            rows = _normalize_rows(list(self.cursor.fetchall() or []))
            return [row["val"] for row in rows if "val" in row]
        except Exception:
            self._drain_cursor()
            raise

    def get_specific_schema_info(self, tables: List[str]) -> str:
        """获取指定表的建表语句，未找到时返回提示文本。"""
        if not tables:
            return "未找到相关表的schema信息"
        schema = self.get_schema_info(tables)
        return schema if schema else "未找到相关表的schema信息"


def get_tables_by_module(
    module: str, sys_tables_db_path: str = "./sys_tables.db"
) -> List[str]:
    """从 sys_tables.db 获取指定业务模块的表名列表。"""
    if not module or not os.path.exists(sys_tables_db_path):
        return []
    try:
        conn = sqlite3.connect(sys_tables_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT table_name FROM system_tables WHERE module = ?",
            (module,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]
    except Exception as exc:
        logger.warning("获取模块 %s 的表信息失败: %s", module, exc)
        return []


def get_all_modules(sys_tables_db_path: str = "./sys_tables.db") -> List[str]:
    """获取 sys_tables.db 中所有业务模块名称。"""
    if not os.path.exists(sys_tables_db_path):
        return []
    try:
        conn = sqlite3.connect(sys_tables_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT module FROM system_tables ORDER BY module")
        modules = [row[0] for row in cursor.fetchall()]
        conn.close()
        return modules
    except Exception as exc:
        logger.warning("获取业务模块列表失败: %s", exc)
        return []
