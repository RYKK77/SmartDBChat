from dataclasses import asdict
from typing import Any, Dict, List, Optional

from smartdbchat.agent import (
    AgentConfig,
    DatabaseBackend,
    NL2SQLAgent,
    validate_chart_config,
)
from smartdbchat.cache import SemanticQueryCache
from smartdbchat.charts import extract_chart_config
from smartdbchat.db import (
    DatabaseManager,
    UnsafeSQLError,
    extract_tables_from_sql,
    get_tables_by_module,
)
from smartdbchat.llm import LLMClient
from smartdbchat.models import (
    CacheHit,
    ExecutionAttempt,
    QueryResult,
    QueryStatus,
    StatusCallback,
    StreamCallback,
)
from smartdbchat.prompts import (
    build_answer_prompt,
    build_retry_sql_prompt,
    build_sql_prompt,
)
from smartdbchat.schema_linker import (
    LinkedSchema,
    SchemaLinker,
    build_schema_graph,
)


def is_valid_query_result(result: Any) -> bool:
    """判断查询结果是否有效（非空字典列表）。"""
    if not result or not isinstance(result, list):
        return False
    return isinstance(result[0], dict) and len(result[0]) > 0


def _split_ddl_statements(raw_schema: str) -> List[str]:
    """将拼接的 DDL 文本拆分为独立的 CREATE TABLE 语句列表。"""
    if not raw_schema or not raw_schema.strip():
        return []
    blocks = [b.strip() for b in raw_schema.split("\n\n") if b.strip()]
    if len(blocks) > 1:
        return blocks
    # 若仅以分号分隔
    parts = [p.strip() for p in raw_schema.split(";") if "CREATE TABLE" in p.upper()]
    return parts if parts else blocks


class SmartDBEngine:
    """自然语言数据库查询编排引擎（支持 Schema Linking、Tool-Use Agent 与传统模式）。"""

    def __init__(
        self,
        db: DatabaseManager,
        llm: LLMClient,
        cache: Optional[SemanticQueryCache] = None,
        sys_tables_db_path: str = "./sys_tables.db",
        schema_linker: Optional[SchemaLinker] = None,
        agent_config: Optional[AgentConfig] = None,
        default_mode: str = "linked",
    ):
        self.db = db
        self.llm = llm
        self.cache = cache or SemanticQueryCache()
        self.sys_tables_db_path = sys_tables_db_path
        self.schema_linker = schema_linker or SchemaLinker()
        self.agent_config = agent_config or AgentConfig()
        self.default_mode = default_mode

    @staticmethod
    def _emit(
        callback: Optional[StatusCallback], level: str, message: str
    ) -> None:
        if callback is not None:
            callback(level, message)

    def _collect_value_samples(
        self, table_names: List[str], max_tables: int = 8
    ) -> Dict[str, Dict[str, List[Any]]]:
        """在后端支持时轻量采样文本/枚举列，用于 Value Linking。"""
        if not isinstance(self.db, DatabaseBackend):
            return {}
        samples: Dict[str, Dict[str, List[Any]]] = {}
        try:
            raw_schema = self.db.get_schema_info(table_names[:max_tables])
            graph = build_schema_graph(_split_ddl_statements(raw_schema))
            for tname, tmeta in graph.tables.items():
                t_samples: Dict[str, List[Any]] = {}
                for col in tmeta.columns:
                    upper_type = col.data_type.upper()
                    if any(k in upper_type for k in ("VARCHAR", "CHAR", "TEXT", "ENUM")):
                        try:
                            vals = self.db.get_distinct_values(tname, col.name, limit=15)
                            if vals:
                                t_samples[col.name] = vals
                        except Exception:
                            continue
                if t_samples:
                    samples[tname] = t_samples
        except Exception:
            return {}
        return samples

    def _resolve_schema(
        self,
        query: str,
        module: Optional[str],
        use_schema_linking: bool = True,
        on_status: Optional[StatusCallback] = None,
    ) -> tuple[str, bool, List[str]]:
        module_tables: Optional[List[str]] = None
        if module:
            relevant_tables = get_tables_by_module(
                module, self.sys_tables_db_path
            )
            if relevant_tables:
                module_tables = relevant_tables
                self._emit(
                    on_status,
                    "info",
                    f"根据业务模块 '{module}' 过滤表（共 {len(relevant_tables)} 张表）",
                )
            else:
                self._emit(
                    on_status,
                    "warning",
                    f"未找到业务模块 '{module}' 的相关表，将使用全库表结构",
                )

        raw_schema = self.db.get_schema_info(module_tables)
        if not raw_schema or not raw_schema.strip():
            return "", bool(module_tables), []

        if not use_schema_linking:
            return raw_schema, bool(module_tables), module_tables or []

        ddls = _split_ddl_statements(raw_schema)
        graph = build_schema_graph(ddls)
        if not graph.tables:
            return raw_schema, bool(module_tables), module_tables or []

        value_samples = self._collect_value_samples(list(graph.tables.keys()))
        linked: LinkedSchema = self.schema_linker.link(
            query=query,
            graph=graph,
            module_tables=module_tables,
            value_samples=value_samples or None,
        )
        if linked.selected_tables:
            self._emit(
                on_status,
                "info",
                f"Schema Linking 已锁定 {len(linked.selected_tables)} 张相关表: {', '.join(linked.selected_tables)}",
            )
            return linked.ddl_context, True, linked.selected_tables
        return raw_schema, bool(module_tables), module_tables or []

    def _run_agent_mode(
        self,
        query: str,
        schema_info: str,
        history: Optional[List[Dict[str, str]]],
        on_status: Optional[StatusCallback],
    ) -> tuple[str, List[Dict[str, Any]], List[ExecutionAttempt], List[Dict[str, Any]], Optional[str], QueryStatus]:
        """使用 Tool-Use Agent Harness 执行探索、校验与自愈查询。"""
        openai_client = getattr(self.llm, "client", None)
        model_name = getattr(getattr(self.llm, "config", None), "model", "gpt-4o")
        agent = NL2SQLAgent(
            db=self.db,
            llm_client=openai_client if openai_client is not None else self.llm,
            model=model_name,
            config=self.agent_config,
        )
        agent_res = agent.run(
            query=query,
            schema_context=schema_info,
            history=history,
            on_status=on_status,
        )
        step_dicts = [asdict(s) for s in agent_res.steps]
        attempts: List[ExecutionAttempt] = []
        exec_idx = 0
        for step in agent_res.steps:
            if step.tool_name == "execute_sql":
                exec_idx += 1
                sql_arg = (step.tool_args or {}).get("sql", "")
                attempts.append(
                    ExecutionAttempt(
                        attempt=exec_idx,
                        sql=sql_arg,
                        row_count=len(agent_res.rows) if step.error is None and sql_arg == agent_res.sql else 0,
                        error=step.error,
                    )
                )
        if not attempts and agent_res.sql:
            attempts.append(
                ExecutionAttempt(
                    attempt=1,
                    sql=agent_res.sql,
                    row_count=len(agent_res.rows),
                    error=agent_res.error,
                )
            )

        if agent_res.success and is_valid_query_result(agent_res.rows):
            return agent_res.sql, agent_res.rows, attempts, step_dicts, None, QueryStatus.SUCCESS

        if agent_res.error and "安全" in agent_res.error:
            status = QueryStatus.INVALID_SQL
        elif agent_res.error:
            status = QueryStatus.ERROR
        else:
            status = QueryStatus.EMPTY_RESULT
        return (
            agent_res.sql,
            agent_res.rows,
            attempts,
            step_dicts,
            agent_res.error or "查询未返回有效数据",
            status,
        )

    def run_query(
        self,
        query: str,
        module: Optional[str] = None,
        max_retries: int = 2,
        on_status: Optional[StatusCallback] = None,
        on_stream: Optional[StreamCallback] = None,
        mode: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> QueryResult:
        """执行完整的 NL2SQL -> 执行/重试 -> 结果分析 -> 图表提取 -> 缓存流程。

        mode:
          - "pipeline": 传统全量/模块 DDL 单轮模式
          - "linked": Schema Linking 裁剪 + 重试模式
          - "agent": Schema Linking + Tool-Use Agent Harness 模式
        """
        effective_mode = mode or self.default_mode
        use_schema_linking = effective_mode in ("linked", "agent")
        use_agent = effective_mode == "agent" and isinstance(self.db, DatabaseBackend)

        db_name = self.db.config.database
        attempts: List[ExecutionAttempt] = []
        sql_prompts: List[List[Dict[str, str]]] = []
        linked_tables: List[str] = []
        agent_steps: List[Dict[str, Any]] = []

        # 多轮追问存在上下文依赖时跳过单句缓存，避免指代消解错误命中
        cache_hit: Optional[CacheHit] = (
            None if history else self.cache.get_from_cache(query, db_name)
        )

        sql = ""
        rows: List[Dict[str, Any]] = []
        last_error: Optional[str] = None
        last_status = QueryStatus.EMPTY_RESULT
        used_cache = False

        if cache_hit is not None:
            sql = cache_hit.sql
            if cache_hit.match_type == "exact":
                self._emit(on_status, "success", "从缓存中获取到完全匹配的 SQL")
            else:
                self._emit(
                    on_status,
                    "success",
                    f"从缓存中获取到相似问题的 SQL（{cache_hit.match_type}, 相似度 {cache_hit.similarity:.2f}）: \"{cache_hit.matched_query}\"",
                )
            try:
                rows = self.db.execute_query(sql)
                attempts.append(ExecutionAttempt(attempt=0, sql=sql, row_count=len(rows)))
                if is_valid_query_result(rows):
                    used_cache = True
                else:
                    last_error = "缓存的 SQL 执行成功，但没有返回数据"
                    last_status = QueryStatus.EMPTY_RESULT
                    self._emit(on_status, "warning", "缓存的 SQL 返回为空，将尝试重新生成 SQL")
            except UnsafeSQLError as exc:
                last_error = str(exc)
                last_status = QueryStatus.INVALID_SQL
                attempts.append(ExecutionAttempt(attempt=0, sql=sql, error=last_error))
                self._emit(on_status, "warning", f"缓存 SQL 未通过安全校验: {last_error}")
            except Exception as exc:
                last_error = str(exc)
                last_status = QueryStatus.ERROR
                attempts.append(ExecutionAttempt(attempt=0, sql=sql, error=last_error))
                self._emit(on_status, "warning", f"缓存的 SQL 执行失败: {last_error}")

        if not used_cache:
            schema_info, is_filtered, linked_tables = self._resolve_schema(
                query=query,
                module=module,
                use_schema_linking=use_schema_linking,
                on_status=on_status,
            )
            if not schema_info.strip():
                return QueryResult(
                    status=QueryStatus.ERROR,
                    sql=sql,
                    rows=[],
                    cache_hit=cache_hit,
                    attempts=attempts,
                    error="当前数据库未找到可用的表结构信息，请检查数据库名或表权限",
                    sql_prompts=sql_prompts,
                    linked_tables=linked_tables,
                    mode=effective_mode,
                )

            if use_agent:
                (
                    sql,
                    rows,
                    agent_attempts,
                    agent_steps,
                    last_error,
                    last_status,
                ) = self._run_agent_mode(
                    query=query,
                    schema_info=schema_info,
                    history=history,
                    on_status=on_status,
                )
                attempts.extend(agent_attempts)
            else:
                # 传统 / Schema Linking 线性纠错流程
                if cache_hit is None:
                    prompt = build_sql_prompt(
                        query, schema_info, is_filtered=is_filtered
                    )
                    if history:
                        prompt = [prompt[0], *history, prompt[1]]
                    sql_prompts.append(prompt)
                    sql = self.llm.generate_sql(prompt)
                    try:
                        rows = self.db.execute_query(sql)
                        attempts.append(ExecutionAttempt(attempt=1, sql=sql, row_count=len(rows)))
                        if not is_valid_query_result(rows):
                            last_error = "查询执行成功，但没有返回数据"
                            last_status = QueryStatus.EMPTY_RESULT
                    except UnsafeSQLError as exc:
                        last_error = str(exc)
                        last_status = QueryStatus.INVALID_SQL
                        attempts.append(ExecutionAttempt(attempt=1, sql=sql, error=last_error))
                    except Exception as exc:
                        last_error = str(exc)
                        last_status = QueryStatus.ERROR
                        attempts.append(ExecutionAttempt(attempt=1, sql=sql, error=last_error))

                # 若仍无有效结果，进入纠错重试循环
                remaining = max_retries
                retry_idx = 1 if cache_hit is None else 0
                while not is_valid_query_result(rows) and remaining > 0:
                    retry_idx += 1
                    remaining -= 1
                    self._emit(
                        on_status,
                        "warning",
                        f"SQL 执行失败或返回为空，正在重试（第 {retry_idx} 次尝试，剩余重试次数：{remaining}）",
                    )
                    retry_prompt = build_retry_sql_prompt(
                        query=query,
                        schema_info=schema_info,
                        failed_sql=sql,
                        error_info=last_error,
                    )
                    sql_prompts.append(retry_prompt)
                    sql = self.llm.generate_sql(retry_prompt)
                    try:
                        rows = self.db.execute_query(sql)
                        attempts.append(
                            ExecutionAttempt(attempt=retry_idx, sql=sql, row_count=len(rows))
                        )
                        if is_valid_query_result(rows):
                            last_error = None
                            break
                        last_error = "查询执行成功，但没有返回数据"
                        last_status = QueryStatus.EMPTY_RESULT
                    except UnsafeSQLError as exc:
                        last_error = str(exc)
                        last_status = QueryStatus.INVALID_SQL
                        attempts.append(
                            ExecutionAttempt(attempt=retry_idx, sql=sql, error=last_error)
                        )
                    except Exception as exc:
                        last_error = str(exc)
                        last_status = QueryStatus.ERROR
                        attempts.append(
                            ExecutionAttempt(attempt=retry_idx, sql=sql, error=last_error)
                        )

        if not is_valid_query_result(rows):
            return QueryResult(
                status=last_status,
                sql=sql,
                rows=[],
                cache_hit=cache_hit,
                attempts=attempts,
                error=last_error or "查询未返回有效数据",
                sql_prompts=sql_prompts,
                linked_tables=linked_tables,
                agent_steps=agent_steps,
                mode=effective_mode,
            )

        # 生成自然语言分析与图表配置
        tables = extract_tables_from_sql(sql)
        specific_schema = self.db.get_specific_schema_info(tables)
        answer_prompts = build_answer_prompt(rows, query, sql, specific_schema)
        markdown_answer = self.llm.stream_analysis(answer_prompts, on_chunk=on_stream)
        chart_config = extract_chart_config(markdown_answer)
        if chart_config is not None:
            is_valid_chart, chart_err = validate_chart_config(chart_config)
            if not is_valid_chart:
                self._emit(on_status, "warning", f"图表配置校验未通过，已忽略: {chart_err}")
                chart_config = None

        # 仅当使用新生成的 SQL 成功且无多轮依赖时写入缓存
        if not used_cache and not history:
            self.cache.save_to_cache(query, db_name, sql)

        return QueryResult(
            status=QueryStatus.SUCCESS,
            sql=sql,
            rows=rows,
            markdown_answer=markdown_answer,
            chart_config=chart_config,
            cache_hit=cache_hit,
            attempts=attempts,
            sql_prompts=sql_prompts,
            answer_prompts=answer_prompts,
            linked_tables=linked_tables,
            agent_steps=agent_steps,
            mode=effective_mode,
        )
