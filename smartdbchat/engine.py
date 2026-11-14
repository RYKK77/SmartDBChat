from typing import Any, Dict, List, Optional

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


def is_valid_query_result(result: Any) -> bool:
    """判断查询结果是否有效（非空字典列表）。"""
    if not result or not isinstance(result, list):
        return False
    return isinstance(result[0], dict) and len(result[0]) > 0


class SmartDBEngine:
    """自然语言数据库查询编排引擎（无 UI 耦合）。"""

    def __init__(
        self,
        db: DatabaseManager,
        llm: LLMClient,
        cache: Optional[SemanticQueryCache] = None,
        sys_tables_db_path: str = "./sys_tables.db",
    ):
        self.db = db
        self.llm = llm
        self.cache = cache or SemanticQueryCache()
        self.sys_tables_db_path = sys_tables_db_path

    @staticmethod
    def _emit(
        callback: Optional[StatusCallback], level: str, message: str
    ) -> None:
        if callback is not None:
            callback(level, message)

    def _resolve_schema(
        self,
        module: Optional[str],
        on_status: Optional[StatusCallback] = None,
    ) -> tuple[str, bool]:
        if module:
            relevant_tables = get_tables_by_module(
                module, self.sys_tables_db_path
            )
            if relevant_tables:
                self._emit(
                    on_status,
                    "info",
                    f"根据业务模块 '{module}' 过滤表（共 {len(relevant_tables)} 张表）",
                )
                return self.db.get_schema_info(relevant_tables), True
            self._emit(
                on_status,
                "warning",
                f"未找到业务模块 '{module}' 的相关表，将使用全库表结构",
            )
        return self.db.get_schema_info(), False

    def run_query(
        self,
        query: str,
        module: Optional[str] = None,
        max_retries: int = 2,
        on_status: Optional[StatusCallback] = None,
        on_stream: Optional[StreamCallback] = None,
    ) -> QueryResult:
        """执行完整的 NL2SQL -> 执行/重试 -> 结果分析 -> 图表提取 -> 缓存流程。"""
        db_name = self.db.config.database
        attempts: List[ExecutionAttempt] = []
        sql_prompts: List[List[Dict[str, str]]] = []
        cache_hit: Optional[CacheHit] = self.cache.get_from_cache(query, db_name)

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
            schema_info, is_filtered = self._resolve_schema(module, on_status)
            if not schema_info.strip():
                return QueryResult(
                    status=QueryStatus.ERROR,
                    sql=sql,
                    rows=[],
                    cache_hit=cache_hit,
                    attempts=attempts,
                    error="当前数据库未找到可用的表结构信息，请检查数据库名或表权限",
                    sql_prompts=sql_prompts,
                )

            # 如果不是缓存失败触发的重试，先进行首次生成
            if cache_hit is None:
                prompt = build_sql_prompt(query, schema_info, is_filtered=is_filtered)
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
            )

        # 生成自然语言分析与图表配置
        tables = extract_tables_from_sql(sql)
        specific_schema = self.db.get_specific_schema_info(tables)
        answer_prompts = build_answer_prompt(rows, query, sql, specific_schema)
        markdown_answer = self.llm.stream_analysis(answer_prompts, on_chunk=on_stream)
        chart_config = extract_chart_config(markdown_answer)

        # 仅当使用新生成的 SQL 成功时写入缓存
        if not used_cache:
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
        )
