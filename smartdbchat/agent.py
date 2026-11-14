import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from smartdbchat.db import UnsafeSQLError, validate_read_only_sql
from smartdbchat.llm import clean_sql_response

logger = logging.getLogger(__name__)

_SQL_FALLBACK_START_RE = re.compile(
    r"^\s*(?:--|/\*.*?\*/\s*)*(?:SELECT|WITH|SHOW|DESCRIBE|DESC|EXPLAIN|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|SET|USE|CALL)\b",
    re.IGNORECASE | re.DOTALL,
)


def _extract_fallback_sql(text: str) -> str:
    """从模型未调用工具的纯文本回复中提取 SQL；若回复是自然语言闲聊/解释则返回空字符串。"""
    raw = (text or "").strip()
    if not raw:
        return ""
    sql_fence = re.findall(r"```sql\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
    if sql_fence:
        candidate = sql_fence[0].strip()
        if candidate:
            return candidate
    generic_fence = re.findall(r"```\s*(.*?)\s*```", raw, re.DOTALL)
    if generic_fence:
        candidate = generic_fence[0].strip()
        if _SQL_FALLBACK_START_RE.match(candidate):
            return candidate
    cleaned = clean_sql_response(raw)
    if _SQL_FALLBACK_START_RE.match(cleaned):
        return cleaned
    return ""

SUPPORTED_CHART_TYPES = frozenset(
    {
        "pie",
        "bar",
        "line",
        "scatter",
        "area",
        "heatmap",
        "radar",
        "funnel",
        "treemap",
    }
)


@runtime_checkable
class DatabaseBackend(Protocol):
    """数据库后端协议，定义 Agent 所需的只读探索与执行接口。"""

    def execute_query(self, sql: str) -> List[Dict[str, Any]]: ...

    def get_schema_info(self, tables: Optional[List[str]] = None) -> str: ...

    def explain_sql(self, sql: str) -> Dict[str, Any]: ...

    def sample_table_rows(self, table_name: str, limit: int = 3) -> List[Dict[str, Any]]: ...

    def get_distinct_values(
        self, table_name: str, column_name: str, limit: int = 20
    ) -> List[Any]: ...

    def list_tables(self) -> List[str]: ...


@dataclass
class AgentConfig:
    """Agent 运行配置。"""

    max_steps: int = 8
    max_sql_executions: int = 4
    enable_explain_check: bool = True
    temperature: float = 0.0


@dataclass
class AgentStep:
    """Agent 单步推理与工具调用记录。"""

    step: int
    thought: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    observation: str = ""
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    """Agent 执行结果。"""

    success: bool
    sql: str = ""
    rows: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[AgentStep] = field(default_factory=list)
    error: Optional[str] = None
    final_message: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)


AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_schema",
            "description": "Search available table DDLs, table names, and comments for matching tables and columns by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search for in table names, column names, or comments.",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_ddl",
            "description": "Fetch full CREATE TABLE DDL statements for specific tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of exact table names to fetch DDL for.",
                    }
                },
                "required": ["table_names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_table_rows",
            "description": "Return up to 5 sample rows from a table to understand data format and values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to sample.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of sample rows to return (1 to 5, default 3).",
                        "default": 3,
                    },
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_distinct_values",
            "description": "Return distinct non-null values of a column for accurate WHERE filtering.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table.",
                    },
                    "column_name": {
                        "type": "string",
                        "description": "Name of the column.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of distinct values to return (default 20).",
                        "default": 20,
                    },
                },
                "required": ["table_name", "column_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_column_values",
            "description": "Search distinct values in a column matching a keyword (case-insensitive substring). Ideal when a column has many values or when recovering from an empty WHERE filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table.",
                    },
                    "column_name": {
                        "type": "string",
                        "description": "Name of the column.",
                    },
                    "keyword": {
                        "type": "string",
                        "description": "Keyword to search for inside column values.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matching values to return (default 20).",
                        "default": 20,
                    },
                },
                "required": ["table_name", "column_name", "keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_join_path",
            "description": "Find the shortest foreign-key JOIN path and exact join columns between two tables in the schema graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_table": {
                        "type": "string",
                        "description": "Starting table name.",
                    },
                    "to_table": {
                        "type": "string",
                        "description": "Target table name.",
                    },
                },
                "required": ["from_table", "to_table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_sql",
            "description": "Validate read-only safety and run EXPLAIN on a SQL query to verify syntax, tables, and columns before execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The read-only SQL query to validate and explain.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Validate read-only safety and execute a SQL query, returning row count and preview rows (or an error message for self-healing).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The read-only SQL query to execute.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
]


def validate_chart_config(config: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """校验图表配置 JSON 是否包含有效的 charts 列表、受支持的图表类型及必需的数据字段。"""
    if config is None or not isinstance(config, dict):
        return False, "图表配置为空或不是合法的 JSON 对象"

    if "charts" not in config or not isinstance(config["charts"], list):
        return False, "图表配置缺少 charts 列表"

    for idx, chart in enumerate(config["charts"]):
        if not isinstance(chart, dict):
            return False, f"charts[{idx}] 不是字典对象"

        chart_type = chart.get("type")
        if chart_type not in SUPPORTED_CHART_TYPES:
            return False, f"charts[{idx}] 使用了不支持的图表类型: {chart_type}"

        data = chart.get("data")
        if not isinstance(data, dict):
            return False, f"charts[{idx}] 缺少 data 字典"

        if chart_type in {"pie", "bar", "area", "funnel", "treemap"}:
            if not isinstance(data.get("labels"), list) or not isinstance(data.get("values"), list):
                return False, f"charts[{idx}] ({chart_type}) 需要在 data 中提供 labels 和 values 列表"
        elif chart_type == "line":
            has_series = isinstance(data.get("series"), list) and len(data["series"]) > 0
            has_lv = isinstance(data.get("labels"), list) and isinstance(data.get("values"), list)
            if not (has_series or has_lv):
                return False, f"charts[{idx}] (line) 需要提供 series 列表或 labels/values 列表"
        elif chart_type == "scatter":
            has_xy = isinstance(data.get("x"), list) and isinstance(data.get("y"), list)
            has_lv = isinstance(data.get("labels"), list) and isinstance(data.get("values"), list)
            if not (has_xy or has_lv):
                return False, f"charts[{idx}] (scatter) 需要提供 x/y 列表或 labels/values 列表"
        elif chart_type == "heatmap":
            if not isinstance(data.get("z_values"), list):
                return False, f"charts[{idx}] (heatmap) 需要在 data 中提供 z_values 列表"
        elif chart_type == "radar":
            has_labels = isinstance(data.get("labels"), list)
            has_vals = isinstance(data.get("series"), list) or isinstance(data.get("values"), list)
            if not (has_labels and has_vals):
                return False, f"charts[{idx}] (radar) 需要提供 labels 及 series/values 列表"

    return True, None


class NL2SQLAgent:
    """基于工具调用 (Tool-Use) 的 NL2SQL 智能体。"""

    def __init__(
        self,
        db: DatabaseBackend,
        llm_client: Any,
        model: str,
        config: Optional[AgentConfig] = None,
    ):
        from smartdbchat.llm import LLMClient

        self.db = db
        self.llm_client = llm_client
        self._raw_client = llm_client.client if isinstance(llm_client, LLMClient) else llm_client
        self.model = model
        self.config = config or AgentConfig()

    def _build_system_prompt(self, schema_context: str = "") -> str:
        explain_instruction = (
            "3. Before running a complex query, you may call `explain_sql` to verify syntax, tables, and columns.\n"
            if self.config.enable_explain_check
            else ""
        )
        schema_section = (
            f"\n\nAvailable Schema Context:\n```sql\n{schema_context.strip()}\n```"
            if schema_context and schema_context.strip()
            else ""
        )
        return (
            "You are an expert MySQL NL2SQL data analysis agent.\n"
            "Your goal is to accurately answer the user's question by generating and executing read-only SQL queries.\n\n"
            "Workflow & Rules:\n"
            "1. Only generate read-only SELECT / WITH / SHOW / DESCRIBE / EXPLAIN queries. Never attempt data modification.\n"
            "2. If you are uncertain about table names, column names, join paths, or exact categorical filter values, "
            "use `search_schema`, `get_table_ddl`, `get_join_path`, `sample_table_rows`, `get_distinct_values`, or `search_column_values` to explore first.\n"
            f"{explain_instruction}"
            "4. You MUST execute your final SQL query using `execute_sql` and verify it succeeds before providing your final answer.\n"
            "5. If `execute_sql` or `explain_sql` returns an error, analyze the error message, fix your SQL, and retry."
            f"{schema_section}"
        )

    def _active_tools(self) -> List[Dict[str, Any]]:
        if self.config.enable_explain_check:
            return AGENT_TOOLS
        return [t for t in AGENT_TOOLS if t["function"]["name"] != "explain_sql"]

    def _tool_search_schema(self, keyword: str, schema_context: str = "") -> str:
        kw = (keyword or "").strip().lower()
        tables = self.db.list_tables()
        if not kw:
            return json.dumps({"tables": tables}, ensure_ascii=False)

        matched_tables = [t for t in tables if kw in t.lower()]
        full_schema = self.db.get_schema_info() or schema_context
        matched_blocks: List[str] = []

        if full_schema:
            blocks = [b.strip() for b in full_schema.split("\n\n") if b.strip()]
            for block in blocks:
                if kw in block.lower():
                    matched_blocks.append(block)

        return json.dumps(
            {
                "keyword": keyword,
                "matched_tables": matched_tables,
                "matched_ddl": "\n\n".join(matched_blocks) if matched_blocks else "",
            },
            ensure_ascii=False,
        )

    def _tool_get_table_ddl(self, table_names: Any) -> str:
        if isinstance(table_names, str):
            table_list = [table_names]
        elif isinstance(table_names, list):
            table_list = [str(t) for t in table_names if t]
        else:
            table_list = []

        if not table_list:
            return "未提供有效表名"
        ddl = self.db.get_schema_info(table_list)
        return ddl if ddl else "未找到相关表的 schema 信息"

    def _tool_sample_table_rows(self, table_name: str, limit: int = 3) -> str:
        effective_limit = max(1, min(int(limit or 3), 5))
        rows = self.db.sample_table_rows(table_name, limit=effective_limit)
        return json.dumps(
            {"table": table_name, "rows": rows},
            ensure_ascii=False,
            default=str,
        )

    def _tool_get_distinct_values(
        self, table_name: str, column_name: str, limit: int = 20
    ) -> str:
        effective_limit = max(1, min(int(limit or 20), 100))
        values = self.db.get_distinct_values(
            table_name, column_name, limit=effective_limit
        )
        non_null_values = [v for v in values if v is not None]
        return json.dumps(
            {
                "table": table_name,
                "column": column_name,
                "values": non_null_values,
            },
            ensure_ascii=False,
            default=str,
        )

    def _tool_search_column_values(
        self, table_name: str, column_name: str, keyword: str, limit: int = 20
    ) -> str:
        effective_limit = max(1, min(int(limit or 20), 50))
        search_fn = getattr(self.db, "search_column_values", None)
        if callable(search_fn):
            values = search_fn(table_name, column_name, keyword, limit=effective_limit)
        else:
            raw = self.db.get_distinct_values(table_name, column_name, limit=100)
            kw_lower = (keyword or "").strip().lower()
            values = [
                v for v in raw if v is not None and (not kw_lower or kw_lower in str(v).lower())
            ][:effective_limit]
        return json.dumps(
            {
                "table": table_name,
                "column": column_name,
                "keyword": keyword,
                "matches": [v for v in values if v is not None],
            },
            ensure_ascii=False,
            default=str,
        )

    def _tool_get_join_path(
        self, from_table: str, to_table: str, schema_context: str = ""
    ) -> str:
        from smartdbchat.schema_linker import build_schema_graph

        full_schema = self.db.get_schema_info() or schema_context
        ddls = [b.strip() for b in full_schema.split("\n\n") if b.strip()] if full_schema else []
        graph = build_schema_graph(ddls)
        path = graph.shortest_path(from_table, to_table)
        edges = graph.find_join_edges(from_table, to_table)
        return json.dumps(
            {
                "from_table": from_table,
                "to_table": to_table,
                "path": path or [],
                "join_edges": [
                    f"{e.from_table}.{e.from_column} = {e.to_table}.{e.to_column}"
                    for e in edges
                ],
            },
            ensure_ascii=False,
        )

    def _tool_explain_sql(self, sql: str) -> Tuple[str, Optional[str]]:
        if not self.config.enable_explain_check:
            return json.dumps({"skipped": True, "message": "EXPLAIN check disabled"}), None

        is_valid, reason = validate_read_only_sql(sql)
        if not is_valid:
            err = f"UnsafeSQLError: {reason}"
            return json.dumps({"error": err}, ensure_ascii=False), err

        try:
            plan = self.db.explain_sql(sql)
            return json.dumps({"valid": True, "explain": plan}, ensure_ascii=False, default=str), None
        except UnsafeSQLError as exc:
            err = f"UnsafeSQLError: {exc}"
            return json.dumps({"error": err}, ensure_ascii=False), err
        except Exception as exc:
            err = str(exc)
            return json.dumps({"error": err}, ensure_ascii=False), err

    def _tool_execute_sql(
        self, sql: str, sql_executions: int
    ) -> Tuple[str, Optional[str], bool, List[Dict[str, Any]]]:
        if sql_executions >= self.config.max_sql_executions:
            err = f"已达到最大 SQL 执行次数限制 ({self.config.max_sql_executions})"
            return json.dumps({"error": err}, ensure_ascii=False), err, False, []

        is_valid, reason = validate_read_only_sql(sql)
        if not is_valid:
            err = f"UnsafeSQLError: {reason}"
            return json.dumps({"error": err}, ensure_ascii=False), err, False, []

        try:
            rows = self.db.execute_query(sql)
            payload = {
                "success": True,
                "row_count": len(rows),
                "preview_rows": rows[:10],
            }
            return json.dumps(payload, ensure_ascii=False, default=str), None, True, rows
        except UnsafeSQLError as exc:
            err = f"UnsafeSQLError: {exc}"
            return json.dumps({"error": err}, ensure_ascii=False), err, True, []
        except Exception as exc:
            err = str(exc)
            return json.dumps({"error": err}, ensure_ascii=False), err, True, []

    def run(
        self,
        query: str,
        schema_context: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> AgentResult:
        """执行 Tool-Use Agent 循环并返回 AgentResult。"""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._build_system_prompt(schema_context)}
        ]
        if history:
            for msg in history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": query})

        steps: List[AgentStep] = []
        last_sql = ""
        last_rows: List[Dict[str, Any]] = []
        last_error: Optional[str] = None
        has_successful_execution = False
        sql_executions = 0
        tools = self._active_tools()

        for step_idx in range(1, self.config.max_steps + 1):
            if on_status is not None:
                on_status("thinking", f"Agent 第 {step_idx} 步推理中...")

            step_start = time.perf_counter()
            response = self._raw_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=self.config.temperature,
            )
            message = response.choices[0].message
            thought = (getattr(message, "content", None) or "").strip()
            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:
                duration_ms = (time.perf_counter() - step_start) * 1000
                if has_successful_execution:
                    messages.append({"role": "assistant", "content": thought})
                    steps.append(
                        AgentStep(
                            step=step_idx,
                            thought=thought,
                            duration_ms=duration_ms,
                        )
                    )
                    return AgentResult(
                        success=True,
                        sql=last_sql,
                        rows=last_rows,
                        steps=steps,
                        final_message=thought,
                        messages=messages,
                    )

                # 回退处理：仅当模型直接返回了 SQL 语句而未调用工具时才执行回退查询；
                # 若模型返回的是自然语言闲聊/澄清说明，则直接作为对话回复返回，避免把自然语言当成 SQL 校验报错。
                cleaned_sql = _extract_fallback_sql(thought)
                messages.append({"role": "assistant", "content": thought})
                if not cleaned_sql:
                    err_msg = last_error if (last_error or not thought) else None
                    if not thought and not err_msg:
                        err_msg = "Agent 未生成有效 SQL 查询"
                    steps.append(
                        AgentStep(
                            step=step_idx,
                            thought=thought,
                            error=err_msg,
                            duration_ms=duration_ms,
                        )
                    )
                    return AgentResult(
                        success=False,
                        sql=last_sql,
                        rows=last_rows,
                        steps=steps,
                        error=err_msg,
                        final_message=thought,
                        messages=messages,
                    )

                if on_status is not None:
                    on_status("executing", "正在执行回退 SQL 查询...")

                obs, err, counted, rows = self._tool_execute_sql(
                    cleaned_sql, sql_executions
                )
                if counted:
                    sql_executions += 1
                total_ms = (time.perf_counter() - step_start) * 1000
                steps.append(
                    AgentStep(
                        step=step_idx,
                        thought=thought,
                        tool_name="execute_sql_fallback",
                        tool_args={"sql": cleaned_sql},
                        observation=obs,
                        error=err,
                        duration_ms=total_ms,
                    )
                )
                if err is None:
                    return AgentResult(
                        success=True,
                        sql=cleaned_sql,
                        rows=rows,
                        steps=steps,
                        final_message=thought,
                        messages=messages,
                    )
                return AgentResult(
                    success=False,
                    sql=cleaned_sql,
                    rows=[],
                    steps=steps,
                    error=err,
                    final_message=thought,
                    messages=messages,
                )

            # 记录带 tool_calls 的 assistant 消息
            messages.append(
                {
                    "role": "assistant",
                    "content": thought,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                                if isinstance(tc.function.arguments, str)
                                else json.dumps(tc.function.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                tc_start = time.perf_counter()
                tool_name = tc.function.name
                raw_args = tc.function.arguments
                tool_args: Optional[Dict[str, Any]] = None
                observation = ""
                err: Optional[str] = None

                try:
                    if isinstance(raw_args, str):
                        tool_args = json.loads(raw_args) if raw_args.strip() else {}
                    elif isinstance(raw_args, dict):
                        tool_args = raw_args
                    else:
                        tool_args = {}
                except json.JSONDecodeError as exc:
                    err = f"工具参数 JSON 解析失败: {exc}"
                    observation = json.dumps({"error": err}, ensure_ascii=False)

                if err is None and tool_args is not None:
                    if on_status is not None:
                        on_status("tool", f"调用工具 {tool_name}...")
                    try:
                        if tool_name == "search_schema":
                            observation = self._tool_search_schema(
                                keyword=str(tool_args.get("keyword", "")),
                                schema_context=schema_context,
                            )
                        elif tool_name == "get_table_ddl":
                            observation = self._tool_get_table_ddl(
                                table_names=tool_args.get("table_names", [])
                            )
                        elif tool_name == "sample_table_rows":
                            observation = self._tool_sample_table_rows(
                                table_name=str(tool_args.get("table_name", "")),
                                limit=int(tool_args.get("limit", 3)),
                            )
                        elif tool_name == "get_distinct_values":
                            observation = self._tool_get_distinct_values(
                                table_name=str(tool_args.get("table_name", "")),
                                column_name=str(tool_args.get("column_name", "")),
                                limit=int(tool_args.get("limit", 20)),
                            )
                        elif tool_name == "search_column_values":
                            observation = self._tool_search_column_values(
                                table_name=str(tool_args.get("table_name", "")),
                                column_name=str(tool_args.get("column_name", "")),
                                keyword=str(tool_args.get("keyword", "")),
                                limit=int(tool_args.get("limit", 20)),
                            )
                        elif tool_name == "get_join_path":
                            observation = self._tool_get_join_path(
                                from_table=str(tool_args.get("from_table", "")),
                                to_table=str(tool_args.get("to_table", "")),
                                schema_context=schema_context,
                            )
                        elif tool_name == "explain_sql":
                            observation, err = self._tool_explain_sql(
                                sql=str(tool_args.get("sql", ""))
                            )
                        elif tool_name == "execute_sql":
                            sql_str = str(tool_args.get("sql", ""))
                            observation, err, counted, rows = self._tool_execute_sql(
                                sql=sql_str,
                                sql_executions=sql_executions,
                            )
                            if counted:
                                sql_executions += 1
                            if err is None:
                                last_sql = sql_str
                                last_rows = rows
                                has_successful_execution = True
                                last_error = None
                            else:
                                last_error = err
                        else:
                            err = f"未知工具: {tool_name}"
                            observation = json.dumps({"error": err}, ensure_ascii=False)
                    except Exception as exc:
                        err = str(exc)
                        observation = json.dumps({"error": err}, ensure_ascii=False)
                        last_error = err

                if err is not None:
                    last_error = err

                tc_duration_ms = (time.perf_counter() - tc_start) * 1000
                steps.append(
                    AgentStep(
                        step=step_idx,
                        thought=thought,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=observation,
                        error=err,
                        duration_ms=tc_duration_ms,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tool_name,
                        "content": observation,
                    }
                )

        # 达到 max_steps
        if has_successful_execution:
            return AgentResult(
                success=True,
                sql=last_sql,
                rows=last_rows,
                steps=steps,
                final_message="",
                messages=messages,
            )

        return AgentResult(
            success=False,
            sql=last_sql,
            rows=last_rows,
            steps=steps,
            error=last_error or f"达到最大步数限制 ({self.config.max_steps})，未能完成查询",
            final_message="",
            messages=messages,
        )
