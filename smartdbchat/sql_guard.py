from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

_READ_ONLY_ROOT_TYPES = (exp.Select, exp.Union, exp.Subquery)
_SHOW_DESCRIBE_PREFIXES = ("SHOW", "DESCRIBE", "DESC", "EXPLAIN")

_FORBIDDEN_AST_TYPES = tuple(
    t
    for t in (
        getattr(exp, "Insert", None),
        getattr(exp, "Update", None),
        getattr(exp, "Delete", None),
        getattr(exp, "Drop", None),
        getattr(exp, "Alter", None),
        getattr(exp, "Create", None),
        getattr(exp, "Merge", None),
        getattr(exp, "Grant", None),
        getattr(exp, "Revoke", None),
        getattr(exp, "Commit", None),
        getattr(exp, "Rollback", None),
        getattr(exp, "Transaction", None),
        getattr(exp, "Set", None),
        getattr(exp, "Use", None),
        getattr(exp, "TruncateTable", None),
        getattr(exp, "LoadData", None),
        getattr(exp, "Copy", None),
        getattr(exp, "Pragma", None),
        getattr(exp, "Attach", None),
        getattr(exp, "Detach", None),
    )
    if t is not None
)

_FORBIDDEN_FUNCTIONS = frozenset(
    {
        "SLEEP",
        "BENCHMARK",
        "LOAD_FILE",
        "SYS_EXEC",
        "SYS_EVAL",
        "PG_SLEEP",
        "WAITFOR",
        "XP_CMDSHELL",
    }
)

_FORBIDDEN_KEYWORDS_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|LOCK|UNLOCK|CALL)\b",
    re.IGNORECASE,
)

_SQL_KEYWORDS = frozenset(
    {
        "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON",
        "AS", "AND", "OR", "NOT", "IN", "BETWEEN", "LIKE", "IS", "NULL", "ORDER",
        "BY", "GROUP", "HAVING", "LIMIT", "OFFSET", "UNION", "ALL", "DISTINCT",
        "CASE", "WHEN", "THEN", "ELSE", "END", "CAST", "ASC", "DESC", "WITH",
    }
)

_SPECIAL_IDENT_CHARS = set("- .@#$%^&*()+={}[]|\\:;\"'<>?,/")


@dataclass
class SQLGuardResult:
    """AST 级 SQL 安全与 Schema 校验结果。"""

    is_valid: bool
    reason: Optional[str] = None
    sql: str = ""
    tables: List[str] = field(default_factory=list)
    unknown_tables: List[str] = field(default_factory=list)
    unknown_columns: List[str] = field(default_factory=list)
    limit_applied: Optional[int] = None


def quote_identifier_if_needed(identifier: str, quote_char: str = "`") -> str:
    """借鉴 QueryWeaver SQLIdentifierQuoter：对含特殊字符的标识符自动加引号。"""
    ident = (identifier or "").strip()
    if not ident:
        return ident
    if (ident.startswith("`") and ident.endswith("`")) or (
        ident.startswith('"') and ident.endswith('"')
    ):
        return ident
    if ident.upper() in _SQL_KEYWORDS:
        return ident
    if any(ch in _SPECIAL_IDENT_CHARS for ch in ident):
        return f"{quote_char}{ident}{quote_char}"
    return ident


def _extract_cte_names(tree: exp.Expression) -> Set[str]:
    cte_names: Set[str] = set()
    for cte in tree.find_all(exp.CTE):
        alias = cte.alias_or_name
        if alias:
            cte_names.add(alias.lower())
    return cte_names


def _extract_table_aliases(tree: exp.Expression) -> Dict[str, str]:
    """提取 alias -> real_table_name 映射（均为小写）。"""
    aliases: Dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        real_name = (table.name or "").strip()
        if not real_name:
            continue
        aliases[real_name.lower()] = real_name.lower()
        alias = (table.alias or "").strip()
        if alias:
            aliases[alias.lower()] = real_name.lower()
    return aliases


def _extract_select_aliases(tree: exp.Expression) -> Set[str]:
    aliases: Set[str] = set()
    for alias_node in tree.find_all(exp.Alias):
        alias = (alias_node.alias or "").strip()
        if alias:
            aliases.add(alias.lower())
    return aliases


def _apply_limit_clamp(
    tree: exp.Expression, max_limit: int, dialect: str
) -> Tuple[str, Optional[int]]:
    """对顶层 SELECT/UNION 注入或钳位 LIMIT，防止全表扫描拖垮系统。"""
    if max_limit <= 0 or not isinstance(tree, (exp.Select, exp.Union)):
        return tree.sql(dialect=dialect), None

    # 无 GROUP BY 的纯聚合查询（如 SELECT COUNT(*) FROM t）通常只返回 1 行，无需强加 LIMIT
    if isinstance(tree, exp.Select):
        has_group = tree.args.get("group") is not None
        select_exprs = tree.expressions or []
        is_pure_agg = (
            bool(select_exprs)
            and not has_group
            and all(
                any(True for _ in expr.find_all(exp.AggFunc))
                for expr in select_exprs
            )
        )
        if is_pure_agg:
            return tree.sql(dialect=dialect), None

    existing_limit = tree.args.get("limit")
    if existing_limit is not None:
        limit_expr = existing_limit.expression
        if isinstance(limit_expr, exp.Literal) and limit_expr.is_int:
            try:
                val = int(limit_expr.this)
                if val <= max_limit:
                    return tree.sql(dialect=dialect), None
            except ValueError:
                pass
    cloned = tree.copy()
    cloned.set("limit", exp.Limit(expression=exp.Literal.number(max_limit)))
    return cloned.sql(dialect=dialect), max_limit


def _strip_comments_for_fallback(sql: str) -> str:
    cleaned = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL).strip()
    return cleaned


def validate_sql_ast(
    sql: str,
    dialect: str = "mysql",
    allowed_tables: Optional[Iterable[str]] = None,
    table_columns: Optional[Dict[str, Iterable[str]]] = None,
    max_limit: Optional[int] = None,
) -> SQLGuardResult:
    """
    使用 sqlglot AST 进行只读 SQL 安全校验、Schema 一致性检查与 LIMIT 钳位。

    Args:
        sql: 待校验的 SQL 语句
        dialect: SQL 方言（默认 "mysql"，评测沙箱可用 "sqlite"）
        allowed_tables: 允许访问的表名集合（传入后将校验所有引用表是否合法）
        table_columns: 表名 -> 列名集合映射（传入后将校验显式/无歧义列名是否存在）
        max_limit: 可选的最大返回行数限制（自动注入或钳位 LIMIT）
    """
    if not sql or not sql.strip():
        return SQLGuardResult(is_valid=False, reason="SQL 语句为空")

    raw_stripped = _strip_comments_for_fallback(sql).rstrip(";").strip()
    if not raw_stripped:
        return SQLGuardResult(is_valid=False, reason="SQL 语句为空")

    # 检查是否是 SHOW / DESCRIBE / DESC / EXPLAIN 管理只读语句
    first_word_match = re.match(r"^\s*([A-Za-z]+)\b", raw_stripped)
    first_word = first_word_match.group(1).upper() if first_word_match else ""

    # 引号外分号快速拦截（防止某些方言解析器容忍空语句或多语句）
    without_strings = re.sub(r"'[^']*'|\"[^\"]*\"", "", raw_stripped)
    if ";" in without_strings:
        return SQLGuardResult(is_valid=False, reason="禁止执行多条 SQL 语句")

    try:
        expressions = sqlglot.parse(raw_stripped, read=dialect)
    except Exception as exc:
        # 对 MySQL 特有的 SHOW / DESCRIBE 语句做安全兜底
        if first_word in _SHOW_DESCRIBE_PREFIXES:
            if _FORBIDDEN_KEYWORDS_RE.search(without_strings):
                return SQLGuardResult(is_valid=False, reason="检测到潜在写操作或高危关键字")
            return SQLGuardResult(is_valid=True, sql=raw_stripped)
        return SQLGuardResult(is_valid=False, reason=f"SQL 语法解析失败: {exc}")

    expressions = [e for e in expressions if e is not None]
    if len(expressions) == 0:
        return SQLGuardResult(is_valid=False, reason="SQL 语句为空")
    if len(expressions) > 1:
        return SQLGuardResult(is_valid=False, reason="禁止执行多条 SQL 语句")

    tree = expressions[0]

    # 处理 SHOW / DESCRIBE / EXPLAIN 节点
    if isinstance(tree, (exp.Show, exp.Describe)) or (
        isinstance(tree, exp.Command) and first_word in _SHOW_DESCRIBE_PREFIXES
    ):
        if _FORBIDDEN_KEYWORDS_RE.search(without_strings):
            return SQLGuardResult(is_valid=False, reason="检测到潜在写操作或高危关键字")
        return SQLGuardResult(is_valid=True, sql=raw_stripped)

    if not isinstance(tree, _READ_ONLY_ROOT_TYPES):
        return SQLGuardResult(
            is_valid=False,
            reason=f"仅允许只读查询，禁止执行 {type(tree).__name__.upper()} 语句",
        )

    # 遍历整棵 AST，拦截任何子句中的写操作、加锁、文件导出或高危函数
    for node in tree.walk():
        if _FORBIDDEN_AST_TYPES and isinstance(node, _FORBIDDEN_AST_TYPES):
            return SQLGuardResult(
                is_valid=False,
                reason=f"检测到潜在写操作或高危 AST 节点: {type(node).__name__.upper()}",
            )
        if isinstance(node, exp.Command):
            return SQLGuardResult(
                is_valid=False,
                reason=f"禁止执行未知或高危命令: {node.name}",
            )
        if isinstance(node, exp.Lock):
            return SQLGuardResult(is_valid=False, reason="禁止执行带锁的 FOR UPDATE / SHARE 查询")
        if isinstance(node, exp.Into):
            return SQLGuardResult(is_valid=False, reason="禁止使用 INTO OUTFILE / DUMPFILE / 表写入")
        if isinstance(node, exp.Anonymous):
            func_name = (node.name or "").upper()
            if func_name in _FORBIDDEN_FUNCTIONS:
                return SQLGuardResult(
                    is_valid=False,
                    reason=f"禁止调用高危系统函数: {func_name}",
                )
        if isinstance(node, exp.Func):
            func_name = (node.sql_name() or "").upper()
            if func_name in _FORBIDDEN_FUNCTIONS:
                return SQLGuardResult(
                    is_valid=False,
                    reason=f"禁止调用高危系统函数: {func_name}",
                )

    if re.search(r"\bINTO\s+(?:OUTFILE|DUMPFILE)\b", without_strings, flags=re.IGNORECASE):
        return SQLGuardResult(is_valid=False, reason="禁止使用 INTO OUTFILE / DUMPFILE 导出文件")
    if re.search(r"\bFOR\s+UPDATE\b", without_strings, flags=re.IGNORECASE):
        return SQLGuardResult(is_valid=False, reason="禁止执行带锁的 FOR UPDATE 查询")

    # 提取引用表名（排除 CTE 别名）
    cte_names = _extract_cte_names(tree)
    raw_tables: List[str] = []
    seen_tables: Set[str] = set()
    for tbl in tree.find_all(exp.Table):
        name = (tbl.name or "").strip()
        if not name or name.lower() in cte_names:
            continue
        if name.lower() not in seen_tables:
            seen_tables.add(name.lower())
            raw_tables.append(name)

    # 表存在性校验
    unknown_tables: List[str] = []
    if allowed_tables is not None:
        allowed_lower = {t.lower() for t in allowed_tables}
        for tbl_name in raw_tables:
            if tbl_name.lower() not in allowed_lower:
                unknown_tables.append(tbl_name)
        if unknown_tables:
            return SQLGuardResult(
                is_valid=False,
                reason=f"引用了不存在或未授权的表: {', '.join(unknown_tables)}",
                tables=raw_tables,
                unknown_tables=unknown_tables,
            )

    # 列存在性校验（仅在提供 table_columns 且查询未使用未知上下文时校验）
    unknown_columns: List[str] = []
    if table_columns is not None and raw_tables:
        norm_schema = {
            t.lower(): {c.lower() for c in cols} for t, cols in table_columns.items()
        }
        table_aliases = _extract_table_aliases(tree)
        select_aliases = _extract_select_aliases(tree)
        # 收集所有引用到的合法表的列全集
        all_referenced_cols: Set[str] = set()
        for tbl_name in raw_tables:
            all_referenced_cols.update(norm_schema.get(tbl_name.lower(), set()))

        for col_node in tree.find_all(exp.Column):
            col_name = (col_node.name or "").strip()
            if not col_name or col_name == "*":
                continue
            col_lower = col_name.lower()
            tbl_ref = (col_node.table or "").strip().lower()

            if tbl_ref:
                if tbl_ref in cte_names:
                    continue
                real_tbl = table_aliases.get(tbl_ref, tbl_ref)
                if real_tbl in norm_schema and col_lower not in norm_schema[real_tbl]:
                    unknown_columns.append(f"{col_node.table}.{col_name}")
            else:
                # 无表前缀的列：可能是 SELECT 别名（如 ORDER BY avg_score）或 CTE 列
                if col_lower in select_aliases or cte_names:
                    continue
                if all_referenced_cols and col_lower not in all_referenced_cols:
                    unknown_columns.append(col_name)

        if unknown_columns:
            dedup_cols = list(dict.fromkeys(unknown_columns))
            return SQLGuardResult(
                is_valid=False,
                reason=f"引用了不存在的字段: {', '.join(dedup_cols)}",
                tables=raw_tables,
                unknown_columns=dedup_cols,
            )

    final_sql = raw_stripped
    limit_applied: Optional[int] = None
    if max_limit is not None and max_limit > 0:
        final_sql, limit_applied = _apply_limit_clamp(tree, max_limit, dialect=dialect)

    return SQLGuardResult(
        is_valid=True,
        sql=final_sql,
        tables=raw_tables,
        limit_applied=limit_applied,
    )
