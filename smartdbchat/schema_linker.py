from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class ColumnMeta:
    """数据库列元数据。"""

    name: str
    data_type: str
    comment: str = ""
    is_primary_key: bool = False
    is_nullable: bool = True
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class ForeignKeyMeta:
    """外键关系元数据（支持显式外键与隐式推断外键）。"""

    from_table: str
    from_column: str
    to_table: str
    to_column: str
    inferred: bool = False


@dataclass
class TableMeta:
    """数据库表元数据。"""

    name: str
    ddl: str
    comment: str = ""
    columns: List[ColumnMeta] = field(default_factory=list)
    foreign_keys: List[ForeignKeyMeta] = field(default_factory=list)
    row_count: Optional[int] = None


@dataclass
class SchemaGraph:
    """数据库 Schema 关系图。"""

    tables: Dict[str, TableMeta] = field(default_factory=dict)
    edges: List[ForeignKeyMeta] = field(default_factory=list)

    def neighbors(self, table_name: str) -> List[str]:
        """返回与指定表存在外键关联（无向）的相邻表名列表。"""
        if table_name not in self.tables:
            return []
        result: List[str] = []
        seen: Set[str] = set()
        for edge in self.edges:
            neighbor: Optional[str] = None
            if edge.from_table == table_name and edge.to_table != table_name:
                neighbor = edge.to_table
            elif edge.to_table == table_name and edge.from_table != table_name:
                neighbor = edge.from_table
            if neighbor and neighbor in self.tables and neighbor not in seen:
                seen.add(neighbor)
                result.append(neighbor)
        return result

    def shortest_path(self, src: str, dst: str) -> Optional[List[str]]:
        """使用 BFS 查找两张表之间的最短关联路径（用于发现 JOIN 桥接表）。"""
        if src not in self.tables or dst not in self.tables:
            return None
        if src == dst:
            return [src]

        visited: Set[str] = {src}
        queue: deque[List[str]] = deque([[src]])
        while queue:
            path = queue.popleft()
            curr = path[-1]
            for nxt in self.neighbors(curr):
                if nxt == dst:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(path + [nxt])
        return None

    def find_join_edges(self, src: str, dst: str) -> List[ForeignKeyMeta]:
        """借鉴 QueryWeaver 图遍历思路：返回两表最短路径上的完整外键 JOIN 链。"""
        path = self.shortest_path(src, dst)
        if not path or len(path) < 2:
            return []
        join_chain: List[ForeignKeyMeta] = []
        for a, b in zip(path, path[1:]):
            matched: Optional[ForeignKeyMeta] = None
            for edge in self.edges:
                if (edge.from_table == a and edge.to_table == b) or (
                    edge.from_table == b and edge.to_table == a
                ):
                    matched = edge
                    break
            if matched is not None:
                join_chain.append(matched)
        return join_chain

    def to_graph_summary(self, tables: Optional[List[str]] = None) -> str:
        """导出 QueryWeaver 风格的 Schema 图谱摘要（表节点 + 关联边）。"""
        target_set = set(tables) if tables is not None else set(self.tables.keys())
        if not target_set:
            return ""
        lines: List[str] = ["[Schema Graph]"]
        for tname in sorted(target_set):
            tmeta = self.tables.get(tname)
            if not tmeta:
                continue
            cols = ", ".join(c.name for c in tmeta.columns[:12])
            if len(tmeta.columns) > 12:
                cols += f", ... (+{len(tmeta.columns) - 12})"
            desc = f" — {tmeta.comment}" if tmeta.comment else ""
            lines.append(f"- Node({tname}){desc}: [{cols}]")

        seen_edges: Set[Tuple[str, str, str, str]] = set()
        for edge in self.edges:
            if edge.from_table in target_set and edge.to_table in target_set:
                key = (edge.from_table, edge.from_column, edge.to_table, edge.to_column)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                tag = "inferred" if edge.inferred else "fk"
                lines.append(
                    f"- Edge[{tag}]: {edge.from_table}.{edge.from_column} -> "
                    f"{edge.to_table}.{edge.to_column}"
                )
        return "\n".join(lines)


@dataclass
class LinkedSchema:
    """Schema Linking 的输出结果。"""

    selected_tables: List[str] = field(default_factory=list)
    selected_columns: Dict[str, List[str]] = field(default_factory=dict)
    ddl_context: str = ""
    join_hints: List[str] = field(default_factory=list)
    value_hints: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    table_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DDL Parser Helpers
# ---------------------------------------------------------------------------


def _unquote_identifier(ident: str) -> str:
    """去除标识符两端的反引号、双引号或方括号，并处理 schema.table 前缀。"""
    ident = ident.strip()
    if "." in ident:
        parts = [p.strip() for p in ident.split(".") if p.strip()]
        if parts:
            ident = parts[-1]
    if len(ident) >= 2:
        if (ident[0] == "`" and ident[-1] == "`") or (ident[0] == '"' and ident[-1] == '"'):
            return ident[1:-1]
        if ident[0] == "[" and ident[-1] == "]":
            return ident[1:-1]
    return ident


def _unescape_sql_string(text: str) -> str:
    return text.replace("\\'", "'").replace("''", "'").replace('\\"', '"')


def _strip_sql_comments(sql: str) -> str:
    """去除引号外的 -- 单行注释和 /* ... */ 块注释。"""
    out: List[str] = []
    i = 0
    n = len(sql)
    in_single = False
    in_double = False
    in_backtick = False

    while i < n:
        ch = sql[i]
        if in_single:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(sql[i + 1])
                i += 2
                continue
            if ch == "'" and i + 1 < n and sql[i + 1] == "'":
                out.append("'")
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
        elif in_double:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(sql[i + 1])
                i += 2
                continue
            if ch == '"':
                in_double = False
            i += 1
        elif in_backtick:
            out.append(ch)
            if ch == "`":
                in_backtick = False
            i += 1
        else:
            if ch == "'":
                in_single = True
                out.append(ch)
                i += 1
            elif ch == '"':
                in_double = True
                out.append(ch)
                i += 1
            elif ch == "`":
                in_backtick = True
                out.append(ch)
                i += 1
            elif ch == "-" and i + 1 < n and sql[i + 1] == "-":
                # 跳到行尾
                newline_idx = sql.find("\n", i)
                if newline_idx == -1:
                    break
                out.append("\n")
                i = newline_idx + 1
            elif ch == "/" and i + 1 < n and sql[i + 1] == "*":
                end_idx = sql.find("*/", i + 2)
                if end_idx == -1:
                    break
                out.append(" ")
                i = end_idx + 2
            else:
                out.append(ch)
                i += 1
    return "".join(out)


def _find_matching_paren(text: str, open_idx: int) -> int:
    """从 open_idx ('(') 出发查找匹配的 ')'，忽略引号内的括号。"""
    depth = 0
    i = open_idx
    n = len(text)
    in_single = False
    in_double = False
    in_backtick = False

    while i < n:
        ch = text[i]
        if in_single:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "'" and i + 1 < n and text[i + 1] == "'":
                i += 2
                continue
            if ch == "'":
                in_single = False
        elif in_double:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                in_double = False
        elif in_backtick:
            if ch == "`":
                in_backtick = False
        else:
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
            elif ch == "`":
                in_backtick = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _split_top_level_commas(text: str) -> List[str]:
    """按顶层逗号拆分 CREATE TABLE 内部定义，忽略括号与引号内的逗号。"""
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    in_single = False
    in_double = False
    in_backtick = False
    in_bracket = False
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if in_single:
            current.append(ch)
            if ch == "\\" and i + 1 < n:
                current.append(text[i + 1])
                i += 2
                continue
            if ch == "'" and i + 1 < n and text[i + 1] == "'":
                current.append("'")
                i += 2
                continue
            if ch == "'":
                in_single = False
        elif in_double:
            current.append(ch)
            if ch == "\\" and i + 1 < n:
                current.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_double = False
        elif in_backtick:
            current.append(ch)
            if ch == "`":
                in_backtick = False
        elif in_bracket:
            current.append(ch)
            if ch == "]":
                in_bracket = False
        else:
            if ch == "'":
                in_single = True
                current.append(ch)
            elif ch == '"':
                in_double = True
                current.append(ch)
            elif ch == "`":
                in_backtick = True
                current.append(ch)
            elif ch == "[":
                in_bracket = True
                current.append(ch)
            elif ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth = max(0, depth - 1)
                current.append(ch)
            elif ch == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 1
                continue
            else:
                current.append(ch)
        i += 1

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _mask_single_quoted_strings(text: str) -> str:
    """将单引号字符串替换为等长空格，便于安全检测关键字（如 PRIMARY KEY / NOT NULL）。"""
    out: List[str] = []
    i = 0
    n = len(text)
    in_single = False
    while i < n:
        ch = text[i]
        if in_single:
            out.append(" ")
            if ch == "\\" and i + 1 < n:
                out.append(" ")
                i += 2
                continue
            if ch == "'" and i + 1 < n and text[i + 1] == "'":
                out.append(" ")
                i += 2
                continue
            if ch == "'":
                in_single = False
        else:
            if ch == "'":
                in_single = True
                out.append(" ")
            else:
                out.append(ch)
        i += 1
    return "".join(out)


_IDENT_PATTERN = r"(?:`[^`]+`|\"[^\"]+\"|\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)"
_QUALIFIED_IDENT_PATTERN = rf"{_IDENT_PATTERN}(?:\s*\.\s*{_IDENT_PATTERN})?"
_CREATE_TABLE_RE = re.compile(
    rf"CREATE\s+(?:TEMPORARY\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?({_QUALIFIED_IDENT_PATTERN})\s*\(",
    re.IGNORECASE,
)
_TABLE_CONSTRAINT_PREFIX_RE = re.compile(
    r"^(?:CONSTRAINT|PRIMARY\s+KEY|UNIQUE|KEY|INDEX|FULLTEXT|SPATIAL|CHECK|FOREIGN\s+KEY)\b",
    re.IGNORECASE,
)
_TABLE_PK_RE = re.compile(r"PRIMARY\s+KEY\s*\(([^)]+)\)", re.IGNORECASE)
_TABLE_FK_RE = re.compile(
    rf"(?:CONSTRAINT\s+{_IDENT_PATTERN}\s+)?FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+({_QUALIFIED_IDENT_PATTERN})\s*(?:\(([^)]+)\))?",
    re.IGNORECASE,
)
_INLINE_FK_RE = re.compile(
    rf"\bREFERENCES\s+({_QUALIFIED_IDENT_PATTERN})\s*(?:\(([^)]+)\))?",
    re.IGNORECASE,
)
_COMMENT_RE = re.compile(
    r"\bCOMMENT\s*=?\s*'((?:[^'\\]|\\.|'')*)'",
    re.IGNORECASE,
)


def parse_ddl_to_table_meta(ddl: str) -> Optional[TableMeta]:
    """解析 MySQL 或 SQLite 的 CREATE TABLE DDL，返回 TableMeta；非 CREATE TABLE 返回 None。"""
    if not ddl or not ddl.strip():
        return None

    cleaned = _strip_sql_comments(ddl)
    match = _CREATE_TABLE_RE.search(cleaned)
    if not match:
        return None

    table_name = _unquote_identifier(match.group(1))
    open_paren_idx = match.end() - 1
    close_paren_idx = _find_matching_paren(cleaned, open_paren_idx)
    if close_paren_idx == -1:
        return None

    body = cleaned[open_paren_idx + 1 : close_paren_idx]
    table_options = cleaned[close_paren_idx + 1 :]

    table_comment = ""
    table_comment_match = _COMMENT_RE.search(table_options)
    if table_comment_match:
        table_comment = _unescape_sql_string(table_comment_match.group(1))
    else:
        header_comment_match = re.search(r"CREATE\s+(?:TEMPORARY\s+)?TABLE[^\n]*?--\s*(.+)$", ddl, re.IGNORECASE | re.MULTILINE)
        if header_comment_match:
            table_comment = header_comment_match.group(1).strip()

    inline_col_comments: Dict[str, str] = {}
    for raw_line in ddl.splitlines():
        line_match = re.match(
            rf"^\s*({_IDENT_PATTERN})\s+[^-\n]+--\s*(.+)$",
            raw_line,
        )
        if line_match:
            c_name = _unquote_identifier(line_match.group(1))
            if c_name and not _TABLE_CONSTRAINT_PREFIX_RE.match(c_name):
                inline_col_comments[c_name.lower()] = line_match.group(2).strip()

    columns: List[ColumnMeta] = []
    foreign_keys: List[ForeignKeyMeta] = []
    pk_columns: Set[str] = set()

    items = _split_top_level_commas(body)
    for item in items:
        stripped = item.strip()
        if not stripped:
            continue

        if _TABLE_CONSTRAINT_PREFIX_RE.match(stripped):
            # 表级 PRIMARY KEY (...)
            pk_match = _TABLE_PK_RE.search(stripped)
            if pk_match and not stripped.upper().startswith("FOREIGN"):
                for raw_col in pk_match.group(1).split(","):
                    col_id = _unquote_identifier(raw_col)
                    if col_id:
                        pk_columns.add(col_id)

            # 表级 FOREIGN KEY (...) REFERENCES ...
            fk_match = _TABLE_FK_RE.search(stripped)
            if fk_match:
                from_cols = [_unquote_identifier(c) for c in fk_match.group(1).split(",") if c.strip()]
                to_table = _unquote_identifier(fk_match.group(2))
                raw_to_cols = fk_match.group(3)
                if raw_to_cols:
                    to_cols = [_unquote_identifier(c) for c in raw_to_cols.split(",") if c.strip()]
                else:
                    to_cols = ["id"] * len(from_cols)

                for idx, from_col in enumerate(from_cols):
                    to_col = to_cols[idx] if idx < len(to_cols) else (to_cols[-1] if to_cols else "id")
                    foreign_keys.append(
                        ForeignKeyMeta(
                            from_table=table_name,
                            from_column=from_col,
                            to_table=to_table,
                            to_column=to_col,
                            inferred=False,
                        )
                    )
            continue

        # 列定义
        col_match = re.match(rf"^({_IDENT_PATTERN})\s+(.+)$", stripped, re.DOTALL)
        if not col_match:
            continue

        col_name = _unquote_identifier(col_match.group(1))
        rest = col_match.group(2).strip()

        # 提取数据类型（如 INT, VARCHAR(100), DECIMAL(10, 2), BIGINT UNSIGNED）
        type_match = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*(?:\s*\([^)]*\))?(?:\s+(?:UNSIGNED|SIGNED|VARYING|PRECISION))?)",
            rest,
            re.IGNORECASE,
        )
        data_type = re.sub(r"\s+", " ", type_match.group(1)).strip() if type_match else "TEXT"

        masked_rest = _mask_single_quoted_strings(rest)
        is_pk = bool(re.search(r"\bPRIMARY\s+KEY\b", masked_rest, re.IGNORECASE))
        is_not_null = bool(re.search(r"\bNOT\s+NULL\b", masked_rest, re.IGNORECASE))
        is_nullable = not (is_not_null or is_pk)

        col_comment = ""
        col_comment_match = _COMMENT_RE.search(rest)
        if col_comment_match:
            col_comment = _unescape_sql_string(col_comment_match.group(1))
        elif col_name.lower() in inline_col_comments:
            col_comment = inline_col_comments[col_name.lower()]

        inline_fk_match = _INLINE_FK_RE.search(masked_rest)
        if inline_fk_match:
            # 用原始 rest 同样位置或重新在 rest 中匹配以保留标识符大小写
            raw_fk_match = _INLINE_FK_RE.search(rest)
            if raw_fk_match:
                ref_table = _unquote_identifier(raw_fk_match.group(1))
                ref_col = _unquote_identifier(raw_fk_match.group(2)) if raw_fk_match.group(2) else "id"
                foreign_keys.append(
                    ForeignKeyMeta(
                        from_table=table_name,
                        from_column=col_name,
                        to_table=ref_table,
                        to_column=ref_col,
                        inferred=False,
                    )
                )

        if is_pk:
            pk_columns.add(col_name)

        columns.append(
            ColumnMeta(
                name=col_name,
                data_type=data_type,
                comment=col_comment,
                is_primary_key=is_pk,
                is_nullable=is_nullable,
            )
        )

    # 回填表级 PRIMARY KEY 定义到对应列
    if pk_columns:
        pk_lower = {p.lower() for p in pk_columns}
        for col in columns:
            if col.name.lower() in pk_lower:
                col.is_primary_key = True
                col.is_nullable = False

    return TableMeta(
        name=table_name,
        ddl=ddl.strip(),
        comment=table_comment,
        columns=columns,
        foreign_keys=foreign_keys,
    )


def _singular_variants(table_name: str) -> Set[str]:
    """生成表名对应的单数/词根候选，用于隐式外键推断。"""
    low = table_name.lower()
    variants = {low}
    if low.endswith("ies") and len(low) > 3:
        variants.add(low[:-3] + "y")
    if low.endswith(("ses", "xes", "zes", "ches", "shes")) and len(low) > 3:
        variants.add(low[:-2])
    if low.endswith("s") and not low.endswith("ss") and len(low) > 1:
        variants.add(low[:-1])
    return variants


def build_schema_graph(ddls: List[str], infer_implicit_fks: bool = True) -> SchemaGraph:
    """解析 DDL 列表构建 SchemaGraph，并可选地推断隐式外键（如 student_id -> students.id）。"""
    tables: Dict[str, TableMeta] = {}
    edges: List[ForeignKeyMeta] = []
    seen_edges: Set[Tuple[str, str, str, str]] = set()

    for ddl in ddls:
        meta = parse_ddl_to_table_meta(ddl)
        if meta is None:
            continue
        tables[meta.name] = meta
        for fk in meta.foreign_keys:
            key = (fk.from_table.lower(), fk.from_column.lower(), fk.to_table.lower(), fk.to_column.lower())
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append(fk)

    if infer_implicit_fks and tables:
        # 建立单数形式/原名 -> 规范表名的映射
        variant_to_table: Dict[str, str] = {}
        for t_name in tables:
            for var in _singular_variants(t_name):
                variant_to_table.setdefault(var, t_name)
        # 精确表名优先覆盖
        for t_name in tables:
            variant_to_table[t_name.lower()] = t_name

        explicit_from_cols: Set[Tuple[str, str]] = {
            (e.from_table.lower(), e.from_column.lower()) for e in edges
        }

        for t_name, t_meta in tables.items():
            for col in t_meta.columns:
                col_low = col.name.lower()
                if (t_name.lower(), col_low) in explicit_from_cols:
                    continue
                if not col_low.endswith("_id") or len(col_low) <= 3:
                    continue

                prefix = col_low[:-3]
                target_table = variant_to_table.get(prefix)
                if not target_table or target_table == t_name:
                    continue

                target_meta = tables[target_table]
                target_cols_by_low = {c.name.lower(): c for c in target_meta.columns}
                target_pks = [c.name for c in target_meta.columns if c.is_primary_key]

                target_col: Optional[str] = None
                if "id" in target_cols_by_low:
                    target_col = target_cols_by_low["id"].name
                elif col_low in target_cols_by_low:
                    target_col = target_cols_by_low[col_low].name
                elif len(target_pks) == 1:
                    target_col = target_pks[0]

                if not target_col:
                    continue

                edge_key = (t_name.lower(), col_low, target_table.lower(), target_col.lower())
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                inferred_fk = ForeignKeyMeta(
                    from_table=t_name,
                    from_column=col.name,
                    to_table=target_table,
                    to_column=target_col,
                    inferred=True,
                )
                edges.append(inferred_fk)
                t_meta.foreign_keys.append(inferred_fk)
                explicit_from_cols.add((t_name.lower(), col_low))

    return SchemaGraph(tables=tables, edges=edges)


# ---------------------------------------------------------------------------
# SchemaLinker
# ---------------------------------------------------------------------------

_EN_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "what",
        "which",
        "who",
        "how",
        "many",
        "much",
        "all",
        "show",
        "list",
        "find",
        "get",
        "query",
        "select",
        "count",
    }
)

_ZH_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "是",
        "在",
        "有",
        "和",
        "与",
        "及",
        "或",
        "吗",
        "呢",
        "啊",
        "请",
        "问",
        "查",
        "找",
        "看",
        "下",
        "个",
        "每",
        "各",
        "都",
        "把",
        "被",
        "给",
        "按",
        "中",
        "上",
    }
)


class SchemaLinker:
    """结合词法匹配、值链接、模块先验与图谱扩展的轻量级 Schema Linking 模块。"""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """中英文混合分词：英文支持 snake_case / camelCase 切分，中文生成 unigram + bigram。"""
        if not text:
            return []

        tokens: List[str] = []
        seen_order: List[str] = []

        # 1. 提取英文/数字标识符并按 snake_case & camelCase 切分
        for raw in re.findall(r"[A-Za-z0-9_]+", text):
            for snake_part in raw.split("_"):
                if not snake_part:
                    continue
                # camelCase / PascalCase / 数字边界切分
                camel_parts = re.findall(
                    r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+",
                    snake_part,
                )
                for part in camel_parts:
                    low = part.lower()
                    if low and low not in _EN_STOPWORDS:
                        tokens.append(low)
                        seen_order.append(low)

        # 2. 提取连续中文字符片段，生成 bigram 与 unigram
        for zh_seq in re.findall(r"[\u4e00-\u9fff]+", text):
            n = len(zh_seq)
            for i in range(n - 1):
                bigram = zh_seq[i : i + 2]
                tokens.append(bigram)
                seen_order.append(bigram)
            for ch in zh_seq:
                if ch not in _ZH_STOPWORDS:
                    tokens.append(ch)
                    seen_order.append(ch)

        return list(dict.fromkeys(seen_order))

    @staticmethod
    def _token_overlap_score(query_tokens: List[str], target_text: str) -> float:
        """计算查询词与目标文本（表名/列名/注释）的加权匹配分。"""
        if not query_tokens or not target_text:
            return 0.0
        target_tokens = set(SchemaLinker.tokenize(target_text))
        if not target_tokens:
            return 0.0

        score = 0.0
        for tok in query_tokens:
            if tok in target_tokens:
                if len(tok) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in tok):
                    # 中文 bigram 区分度高
                    score += 2.5
                elif len(tok) >= 2:
                    score += 2.0
                else:
                    # 中文单字或单字符
                    score += 0.6
        return score

    @staticmethod
    def _longest_common_cjk_len(a: str, b: str) -> int:
        """计算两个字符串的最长公共连续子串长度（用于中文未分词查询的值链接）。"""
        if not a or not b:
            return 0
        m, n = len(a), len(b)
        dp = [0] * (n + 1)
        best = 0
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if a[i - 1] == b[j - 1] and "\u4e00" <= a[i - 1] <= "\u9fff":
                    dp[j] = prev + 1
                    if dp[j] > best:
                        best = dp[j]
                else:
                    dp[j] = 0
                prev = temp
        return best

    @staticmethod
    def _match_value_samples(query: str, samples: List[Any]) -> List[Any]:
        """检查查询文本与列样本值之间是否存在值链接匹配。"""
        if not query or not samples:
            return []

        q_low = query.lower()
        q_en_words = {
            w.lower()
            for w in re.findall(r"[A-Za-z0-9_]{2,}", query)
            if w.lower() not in _EN_STOPWORDS
        }

        scored_matches: List[Tuple[int, Any]] = []
        seen_str: Set[str] = set()

        for val in samples:
            if val is None:
                continue
            val_str = str(val).strip()
            if not val_str:
                continue
            val_low = val_str.lower()
            if val_low in seen_str:
                continue

            match_strength = 0
            # 1) 样本值完整出现在查询中（长度 >= 2，避免单数字/单字误命中）
            if len(val_low) >= 2 and val_low in q_low:
                match_strength = len(val_low) * 2
            # 2) 中文连续公共子串匹配（如查询包含 "计算机" 命中样本 "计算机科学与技术"）
            elif any("\u4e00" <= c <= "\u9fff" for c in val_str):
                lcs_len = SchemaLinker._longest_common_cjk_len(query, val_str)
                if lcs_len >= 2:
                    match_strength = lcs_len
            # 3) 查询中的英文词与样本词精确重合
            elif q_en_words:
                val_en_words = {w.lower() for w in re.findall(r"[A-Za-z0-9_]{2,}", val_str)}
                overlap = q_en_words & val_en_words
                if overlap:
                    match_strength = max(len(w) for w in overlap)

            if match_strength > 0:
                seen_str.add(val_low)
                scored_matches.append((match_strength, val))

        if not scored_matches:
            return []

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        best_strength = scored_matches[0][0]
        # 若存在更长精确匹配（>=3 字或完整命中），过滤掉仅 2 字弱重合的噪音样本
        if best_strength >= 3:
            scored_matches = [item for item in scored_matches if item[0] >= 3]

        return [val for _, val in scored_matches]

    @staticmethod
    def _find_components(tables: List[str], graph: SchemaGraph) -> List[Set[str]]:
        """在当前选中表集合的诱导子图中计算连通分量。"""
        table_set = set(tables)
        adj: Dict[str, Set[str]] = {t: set() for t in tables}
        for edge in graph.edges:
            if edge.from_table in table_set and edge.to_table in table_set:
                adj[edge.from_table].add(edge.to_table)
                adj[edge.to_table].add(edge.from_table)

        visited: Set[str] = set()
        components: List[Set[str]] = []
        for t in tables:
            if t in visited:
                continue
            comp: Set[str] = set()
            q = deque([t])
            visited.add(t)
            while q:
                curr = q.popleft()
                comp.add(curr)
                for nxt in adj[curr]:
                    if nxt not in visited:
                        visited.add(nxt)
                        q.append(nxt)
            components.append(comp)
        return components

    def _expand_bridge_tables(
        self,
        selected_tables: List[str],
        graph: SchemaGraph,
        max_hops: int = 3,
    ) -> List[str]:
        """若选中表存在多个不连通分量且在 SchemaGraph 中最短路径 <= max_hops，自动补全桥接表。"""
        expanded = list(selected_tables)
        selected_set = set(expanded)

        while True:
            components = self._find_components(expanded, graph)
            if len(components) <= 1:
                break

            best_path: Optional[List[str]] = None
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    for src in components[i]:
                        for dst in components[j]:
                            path = graph.shortest_path(src, dst)
                            if path is not None and len(path) - 1 <= max_hops:
                                if best_path is None or len(path) < len(best_path):
                                    best_path = path

            if best_path is None:
                break

            added = False
            for node in best_path:
                if node not in selected_set:
                    selected_set.add(node)
                    expanded.append(node)
                    added = True
            if not added:
                break

        return expanded

    def link(
        self,
        query: str,
        graph: SchemaGraph,
        module_tables: Optional[List[str]] = None,
        top_k_tables: int = 5,
        max_columns_per_table: int = 12,
        value_samples: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        llm_pruner: Optional[Callable[[str, SchemaGraph, List[str]], List[str]]] = None,
    ) -> LinkedSchema:
        """执行 Schema Linking，返回精简后的表、列、JOIN 提示、值提示与格式化 DDL 上下文。"""
        if not graph.tables:
            return LinkedSchema()

        # 1. 模块先验过滤候选表池
        if module_tables:
            module_set = set(module_tables)
            candidates = [t for t in graph.tables if t in module_set]
            if not candidates:
                candidates = list(graph.tables.keys())
        else:
            candidates = list(graph.tables.keys())

        query_tokens = self.tokenize(query)
        q_low = query.lower()

        table_scores: Dict[str, float] = {}
        column_scores: Dict[str, Dict[str, float]] = {}
        value_hints: Dict[str, Dict[str, List[Any]]] = {}

        # 2. 对候选表和列进行多维度打分
        for t_name in candidates:
            t_meta = graph.tables[t_name]
            t_score = 0.0

            # a) 表名 + 表注释匹配
            t_score += 2.0 * self._token_overlap_score(query_tokens, t_name)
            t_score += 2.0 * self._token_overlap_score(query_tokens, t_meta.comment)
            if t_name.lower() in q_low and len(t_name) >= 2:
                t_score += 3.0

            # b) 列名 + 列注释匹配
            col_score_map: Dict[str, float] = {}
            col_score_list: List[float] = []
            for col in t_meta.columns:
                c_score = 0.0
                c_score += 1.5 * self._token_overlap_score(query_tokens, col.name)
                c_score += 1.5 * self._token_overlap_score(query_tokens, col.comment)
                if col.name.lower() in q_low and len(col.name) >= 2:
                    c_score += 1.5

                # c) 值链接匹配（同时支持传入的 value_samples 与 ColumnMeta.sample_values）
                samples_to_check: List[Any] = []
                if value_samples and t_name in value_samples and col.name in value_samples[t_name]:
                    samples_to_check.extend(value_samples[t_name][col.name])
                if col.sample_values:
                    samples_to_check.extend(col.sample_values)

                if samples_to_check:
                    matched_vals = self._match_value_samples(query, samples_to_check)
                    if matched_vals:
                        value_hints.setdefault(t_name, {})[col.name] = matched_vals[:5]
                        c_score += 6.0
                        t_score += 6.0

                col_score_map[col.name] = c_score
                if c_score > 0:
                    col_score_list.append(c_score)

            # 取表内最相关列得分（避免列多的表稀释或累加失真）
            if col_score_list:
                col_score_list.sort(reverse=True)
                t_score += col_score_list[0] + 0.5 * sum(col_score_list[1:4])

            table_scores[t_name] = round(t_score, 4)
            column_scores[t_name] = col_score_map

        # 3. 选择 Top-K 表
        sorted_candidates = sorted(candidates, key=lambda t: (-table_scores[t], t))
        positive_tables = [t for t in sorted_candidates if table_scores[t] > 0]
        if positive_tables:
            selected_tables = positive_tables[:top_k_tables]
        else:
            selected_tables = sorted_candidates[:top_k_tables]

        # 可选 LLM 精简回调
        if llm_pruner is not None and selected_tables:
            pruned = llm_pruner(query, graph, list(selected_tables))
            if pruned:
                valid_pruned: List[str] = []
                seen_p: Set[str] = set()
                for t in pruned:
                    if t in graph.tables and t not in seen_p:
                        seen_p.add(t)
                        valid_pruned.append(t)
                if valid_pruned:
                    selected_tables = valid_pruned

        # 4. 图谱扩展（自动补充 <= 3 跳的桥接表）
        selected_tables = self._expand_bridge_tables(selected_tables, graph, max_hops=3)

        # 为扩展进来的桥接表补全分数记录（若此前不在候选池中）
        for t_name in selected_tables:
            if t_name not in table_scores:
                table_scores[t_name] = 0.0
            if t_name not in column_scores:
                column_scores[t_name] = {c.name: 0.0 for c in graph.tables[t_name].columns}

        # 5. 生成 JOIN 提示
        selected_set = set(selected_tables)
        join_hints: List[str] = []
        seen_joins: Set[str] = set()
        join_cols_by_table: Dict[str, Set[str]] = {t: set() for t in selected_tables}

        for edge in graph.edges:
            if edge.from_table in selected_set and edge.to_table in selected_set:
                hint = f"{edge.from_table}.{edge.from_column} = {edge.to_table}.{edge.to_column}"
                if hint not in seen_joins:
                    seen_joins.add(hint)
                    join_hints.append(hint)
                join_cols_by_table[edge.from_table].add(edge.from_column)
                join_cols_by_table[edge.to_table].add(edge.to_column)

        # 6. 列裁剪：保留 PK、FK/Join 列、值命中列，以及高分列（上限 max_columns_per_table）
        selected_columns: Dict[str, List[str]] = {}
        for t_name in selected_tables:
            t_meta = graph.tables[t_name]
            fk_cols = {fk.from_column for fk in t_meta.foreign_keys} | join_cols_by_table.get(t_name, set())
            val_cols = set(value_hints.get(t_name, {}).keys())
            c_scores = column_scores.get(t_name, {})

            must_keep: List[str] = []
            optional_cols: List[Tuple[float, int, str]] = []

            for idx, col in enumerate(t_meta.columns):
                if col.is_primary_key or col.name in fk_cols or col.name in val_cols:
                    must_keep.append(col.name)
                else:
                    optional_cols.append((c_scores.get(col.name, 0.0), -idx, col.name))

            optional_cols.sort(reverse=True)
            chosen_set: Set[str] = set(must_keep[:max_columns_per_table])
            remaining_slots = max(0, max_columns_per_table - len(chosen_set))
            for _, _, col_name in optional_cols[:remaining_slots]:
                chosen_set.add(col_name)

            # 按表原始列顺序输出
            selected_columns[t_name] = [c.name for c in t_meta.columns if c.name in chosen_set]

        # 只保留选中表与选中列的 value_hints
        filtered_value_hints: Dict[str, Dict[str, List[Any]]] = {}
        for t_name in selected_tables:
            if t_name in value_hints:
                col_set = set(selected_columns[t_name])
                hits = {c: vals for c, vals in value_hints[t_name].items() if c in col_set}
                if hits:
                    filtered_value_hints[t_name] = hits

        ddl_context = self._format_ddl_context(
            graph=graph,
            selected_tables=selected_tables,
            selected_columns=selected_columns,
            join_hints=join_hints,
            value_hints=filtered_value_hints,
        )

        return LinkedSchema(
            selected_tables=selected_tables,
            selected_columns=selected_columns,
            ddl_context=ddl_context,
            join_hints=join_hints,
            value_hints=filtered_value_hints,
            table_scores=table_scores,
        )

    @staticmethod
    def _format_ddl_context(
        graph: SchemaGraph,
        selected_tables: List[str],
        selected_columns: Dict[str, List[str]],
        join_hints: List[str],
        value_hints: Dict[str, Dict[str, List[Any]]],
    ) -> str:
        """将裁剪后的 Schema 信息、JOIN 提示和值提示格式化为清晰的 DDL 上下文。"""
        sections: List[str] = []

        for t_name in selected_tables:
            t_meta = graph.tables[t_name]
            kept_cols = set(selected_columns.get(t_name, []))
            fk_map: Dict[str, ForeignKeyMeta] = {
                fk.from_column: fk for fk in t_meta.foreign_keys
            }

            header = f"-- Table: {t_meta.name}"
            if t_meta.comment:
                header += f" ({t_meta.comment})"

            col_lines: List[str] = []
            for col in t_meta.columns:
                if col.name not in kept_cols:
                    continue
                tags: List[str] = []
                if col.is_primary_key:
                    tags.append("PK")
                if not col.is_nullable and not col.is_primary_key:
                    tags.append("NOT NULL")
                if col.name in fk_map:
                    fk = fk_map[col.name]
                    tags.append(f"FK -> {fk.to_table}.{fk.to_column}")

                tag_str = f" [{' | '.join(tags)}]" if tags else ""
                comment_str = f" -- {col.comment}" if col.comment else ""
                col_lines.append(f"  {col.name} {col.data_type}{tag_str}{comment_str}")

            table_block = [header, f"CREATE TABLE {t_meta.name} ("]
            for i, line in enumerate(col_lines):
                comma = "," if i < len(col_lines) - 1 else ""
                # 若行末有 -- 注释，把逗号放在注释前面
                if " -- " in line:
                    main_part, cmt_part = line.split(" -- ", 1)
                    table_block.append(f"{main_part}{comma} -- {cmt_part}")
                else:
                    table_block.append(f"{line}{comma}")
            table_block.append(");")
            sections.append("\n".join(table_block))

        if join_hints:
            hint_lines = ["-- Join Hints:"] + [f"--   {h}" for h in join_hints]
            sections.append("\n".join(hint_lines))

        if value_hints:
            val_lines = ["-- Value Hints:"]
            for t_name, col_vals in value_hints.items():
                for c_name, vals in col_vals.items():
                    formatted_vals = ", ".join(repr(v) for v in vals)
                    val_lines.append(f"--   {t_name}.{c_name}: [{formatted_vals}]")
            sections.append("\n".join(val_lines))

        return "\n\n".join(sections)
