from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from smartdbchat.llm import clean_sql_response
from smartdbchat.sql_guard import validate_sql_ast


@dataclass
class CandidateResult:
    """单条候选 SQL 的生成与试跑结果。"""

    strategy: str
    sql: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    signature: Optional[str] = None
    is_valid: bool = False


@dataclass
class ScalingResult:
    """Test-Time Scaling 多候选共识投票结果。"""

    selected_sql: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[CandidateResult] = field(default_factory=list)
    consensus_votes: int = 0
    total_valid: int = 0
    prompts: List[List[Dict[str, str]]] = field(default_factory=list)
    error: Optional[str] = None


def _norm_cell(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, (int, float)):
        return f"{float(val):.6g}"
    return str(val).strip()


def compute_result_signature(rows: List[Dict[str, Any]]) -> str:
    """
    计算查询结果的规范化签名（忽略列别名差异，聚焦实际数据集合），
    用于不同候选 SQL 之间的执行一致性多数投票（Majority Vote）。
    """
    if not rows:
        return "EMPTY"
    row_sigs: List[str] = []
    for row in rows:
        vals = sorted(_norm_cell(v) for v in row.values())
        row_sigs.append("|".join(vals))
    row_sigs.sort()
    payload = json.dumps(row_sigs, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


_STRATEGIES: List[Tuple[str, float, str]] = [
    ("standard", 0.0, ""),
    (
        "cte_step_by_step",
        0.2,
        "\n提示：若涉及多表关联或复杂聚合，优先使用 WITH (CTE) 分步拆解逻辑，确保每一步过滤和分组口径准确。",
    ),
    (
        "join_strict",
        0.3,
        "\n提示：请严格核对所有表名、字段名及外键 JOIN 条件，避免笛卡尔积与重复计数（必要时使用 DISTINCT）。",
    ),
]


class TestTimeScaler:
    """
    借鉴 ReFoRCE 与 Agentar-Scale-SQL 的 Test-Time Scaling 框架：
    多策略并行/多路生成候选 SQL -> AST 安全校验 -> 只读试跑 -> 执行结果一致性多数投票。
    """

    __test__ = False

    def __init__(
        self,
        llm_client: Any,
        model: str,
        execute_fn: Callable[[str], List[Dict[str, Any]]],
        dialect: str = "mysql",
        allowed_tables: Optional[List[str]] = None,
        table_columns: Optional[Dict[str, List[str]]] = None,
        max_limit: Optional[int] = 1000,
    ):
        self.llm_client = llm_client
        self.model = model
        self.execute_fn = execute_fn
        self.dialect = dialect
        self.allowed_tables = allowed_tables
        self.table_columns = table_columns
        self.max_limit = max_limit

    def run(
        self,
        base_messages: List[Dict[str, str]],
        num_candidates: int = 3,
    ) -> ScalingResult:
        count = max(1, min(int(num_candidates), len(_STRATEGIES)))
        candidates: List[CandidateResult] = []
        prompts: List[List[Dict[str, str]]] = []

        for strategy_name, temp, extra_hint in _STRATEGIES[:count]:
            msgs = [dict(m) for m in base_messages]
            if extra_hint and msgs:
                msgs[-1]["content"] = msgs[-1]["content"] + extra_hint
            prompts.append(msgs)

            try:
                raw_resp = self.llm_client.chat_completion(
                    messages=msgs,
                    model=self.model,
                    temperature=temp,
                )
                sql = clean_sql_response(raw_resp)
            except Exception as exc:
                candidates.append(
                    CandidateResult(strategy=strategy_name, sql="", error=f"LLM 生成失败: {exc}")
                )
                continue

            guard = validate_sql_ast(
                sql,
                dialect=self.dialect,
                allowed_tables=self.allowed_tables,
                table_columns=self.table_columns,
                max_limit=self.max_limit,
            )
            if not guard.is_valid:
                candidates.append(
                    CandidateResult(
                        strategy=strategy_name,
                        sql=sql,
                        error=guard.reason or "SQL 安全/Schema 校验未通过",
                        is_valid=False,
                    )
                )
                continue

            exec_sql = guard.sql or sql
            try:
                rows = self.execute_fn(exec_sql)
                sig = compute_result_signature(rows)
                candidates.append(
                    CandidateResult(
                        strategy=strategy_name,
                        sql=exec_sql,
                        rows=rows,
                        signature=sig,
                        is_valid=True,
                    )
                )
            except Exception as exc:
                candidates.append(
                    CandidateResult(
                        strategy=strategy_name,
                        sql=exec_sql,
                        error=str(exc),
                        is_valid=False,
                    )
                )

        valid_candidates = [c for c in candidates if c.is_valid]
        if not valid_candidates:
            first_err = next((c.error for c in candidates if c.error), "所有候选 SQL 均执行失败")
            first_sql = next((c.sql for c in candidates if c.sql), "")
            return ScalingResult(
                selected_sql=first_sql,
                rows=[],
                candidates=candidates,
                consensus_votes=0,
                total_valid=0,
                prompts=prompts,
                error=first_err,
            )

        # 非空结果优先参与多数投票；若全为空，则在空结果中选择第一个
        non_empty = [c for c in valid_candidates if len(c.rows) > 0]
        pool = non_empty if non_empty else valid_candidates
        sig_counts = Counter(c.signature for c in pool if c.signature)

        best_candidate = max(
            pool,
            key=lambda c: (
                sig_counts.get(c.signature or "", 0),
                1 if len(c.rows) > 0 else 0,
                -len(c.sql),
            ),
        )
        votes = sig_counts.get(best_candidate.signature or "", 1)

        return ScalingResult(
            selected_sql=best_candidate.sql,
            rows=best_candidate.rows,
            candidates=candidates,
            consensus_votes=votes,
            total_valid=len(valid_candidates),
            prompts=prompts,
        )
