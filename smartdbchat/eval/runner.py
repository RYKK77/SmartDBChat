import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from smartdbchat.db import UnsafeSQLError, extract_tables_from_sql, validate_read_only_sql
from smartdbchat.eval.sandbox import SQLiteEvalBackend

EvalEngineFn = Callable[
    [Dict[str, Any]], Tuple[str, List[Dict[str, Any]], Dict[str, Any]]
]


@dataclass
class CaseEvalResult:
    """单条评测用例的执行与评分结果。"""

    case_id: str
    question: str
    category: str
    difficulty: str
    expected_behavior: str
    predicted_sql: str
    reference_sql: str
    is_valid_sql: bool
    is_blocked: bool
    execution_match: bool
    passed: bool
    attempts: int = 1
    steps: int = 1
    latency_ms: float = 0.0
    schema_recall: Optional[float] = None
    error: Optional[str] = None
    pred_row_count: int = 0
    ref_row_count: int = 0


@dataclass
class EvalSummary:
    """整体评测汇总指标与分组统计。"""

    total_cases: int
    non_safety_cases: int
    safety_cases: int
    execution_accuracy: float
    valid_sql_rate: float
    safety_block_rate: float
    avg_attempts: float
    avg_steps: float
    avg_latency_ms: float
    schema_recall: Optional[float]
    by_category: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_difficulty: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    results: List[CaseEvalResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, str):
        return value.strip()
    return str(value)


def normalize_result_set(rows: List[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    """将查询结果字典列表规范化为忽略列别名的值元组列表。"""
    if not rows:
        return []
    return [tuple(_normalize_cell(v) for v in row.values()) for row in rows]


def _cells_match(a: Any, b: Any, float_tol: float = 1e-4) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= float_tol
    return a == b


def _tuples_match(
    t1: Tuple[Any, ...], t2: Tuple[Any, ...], float_tol: float = 1e-4
) -> bool:
    if len(t1) != len(t2):
        return False
    return all(_cells_match(a, b, float_tol) for a, b in zip(t1, t2))


def _sort_key(row: Tuple[Any, ...]) -> Tuple[Tuple[int, Any], ...]:
    key_parts: List[Tuple[int, Any]] = []
    for cell in row:
        if cell is None:
            key_parts.append((0, ""))
        elif isinstance(cell, bool):
            key_parts.append((1, int(cell)))
        elif isinstance(cell, (int, float)):
            key_parts.append((2, round(float(cell), 3)))
        else:
            key_parts.append((3, str(cell)))
    return tuple(key_parts)


def results_match(
    pred_rows: List[Dict[str, Any]],
    ref_rows: List[Dict[str, Any]],
    order_sensitive: bool = False,
    float_tol: float = 1e-4,
) -> bool:
    """比较预测结果集与参考结果集是否等价（支持忽略列别名、浮点数容差、有序/无序比较）。"""
    pred_norm = normalize_result_set(pred_rows)
    ref_norm = normalize_result_set(ref_rows)

    if len(pred_norm) != len(ref_norm):
        return False
    if not pred_norm:
        return True

    if order_sensitive:
        return all(_tuples_match(p, r, float_tol) for p, r in zip(pred_norm, ref_norm))

    pred_sorted = sorted(pred_norm, key=_sort_key)
    ref_sorted = sorted(ref_norm, key=_sort_key)
    if all(_tuples_match(p, r, float_tol) for p, r in zip(pred_sorted, ref_sorted)):
        return True

    # 兜底：处理浮点四舍五入边界导致排序位置微调的多重集匹配
    used = [False] * len(ref_norm)
    for p in pred_norm:
        matched = False
        for idx, r in enumerate(ref_norm):
            if not used[idx] and _tuples_match(p, r, float_tol):
                used[idx] = True
                matched = True
                break
        if not matched:
            return False
    return True


def load_golden_cases(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """从 JSON 文件加载 Golden Cases。"""
    if path is None:
        default_path = (
            Path(__file__).resolve().parents[2] / "eval_data" / "golden_cases.json"
        )
        target = default_path
    else:
        target = Path(path)

    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Golden cases 文件格式错误，应为 JSON 数组: {target}")
    return data


def _compute_group_metrics(results: List[CaseEvalResult]) -> Dict[str, Any]:
    total = len(results)
    safety = [r for r in results if r.expected_behavior == "blocked"]
    non_safety = [r for r in results if r.expected_behavior != "blocked"]

    ex = (
        sum(1 for r in non_safety if r.execution_match) / len(non_safety)
        if non_safety
        else 0.0
    )
    valid_rate = (
        sum(1 for r in non_safety if r.is_valid_sql) / len(non_safety)
        if non_safety
        else 0.0
    )
    block_rate = (
        sum(1 for r in safety if r.is_blocked) / len(safety) if safety else 0.0
    )
    pass_rate = sum(1 for r in results if r.passed) / total if total else 0.0

    return {
        "total": total,
        "non_safety": len(non_safety),
        "safety": len(safety),
        "passed": sum(1 for r in results if r.passed),
        "pass_rate": round(pass_rate, 4),
        "execution_accuracy": round(ex, 4),
        "valid_sql_rate": round(valid_rate, 4),
        "safety_block_rate": round(block_rate, 4),
    }


class EvalRunner:
    """轻量级 NL2SQL 评测运行器。"""

    def __init__(self, backend: SQLiteEvalBackend):
        self.backend = backend

    def evaluate(
        self, engine_fn: EvalEngineFn, cases: List[Dict[str, Any]]
    ) -> EvalSummary:
        results: List[CaseEvalResult] = []

        for case in cases:
            case_id = str(case.get("id", ""))
            question = str(case.get("question", ""))
            category = str(case.get("category", "unknown"))
            difficulty = str(case.get("difficulty", "unknown"))
            expected_behavior = str(
                case.get(
                    "expected_behavior",
                    "blocked" if category == "safety" else "answer",
                )
            )
            ref_sql = str(case.get("reference_sql", ""))
            order_sensitive = bool(case.get("order_sensitive", False))
            expected_tables = [str(t).lower() for t in case.get("expected_tables", [])]

            t0 = time.perf_counter()
            sql = ""
            pred_rows: List[Dict[str, Any]] = []
            meta: Dict[str, Any] = {}
            raised_unsafe = False
            raised_error: Optional[str] = None

            try:
                sql, pred_rows, meta = engine_fn(case)
                meta = meta or {}
            except UnsafeSQLError as exc:
                raised_unsafe = True
                raised_error = str(exc)
            except Exception as exc:
                raised_error = str(exc)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latency_ms = float(meta.get("latency_ms", elapsed_ms))
            attempts = int(meta.get("attempts", 1))
            steps = int(meta.get("steps", attempts))
            error = meta.get("error") or raised_error

            schema_recall: Optional[float] = None
            linked_tables = meta.get("linked_tables")
            if linked_tables is not None and expected_tables:
                linked_set = {str(t).lower() for t in linked_tables}
                schema_recall = len(set(expected_tables) & linked_set) / len(
                    set(expected_tables)
                )

            if expected_behavior == "blocked":
                is_blocked = bool(meta.get("blocked", False)) or raised_unsafe
                results.append(
                    CaseEvalResult(
                        case_id=case_id,
                        question=question,
                        category=category,
                        difficulty=difficulty,
                        expected_behavior=expected_behavior,
                        predicted_sql=sql,
                        reference_sql=ref_sql,
                        is_valid_sql=False,
                        is_blocked=is_blocked,
                        execution_match=False,
                        passed=is_blocked,
                        attempts=attempts,
                        steps=steps,
                        latency_ms=latency_ms,
                        schema_recall=schema_recall,
                        error=error,
                        pred_row_count=len(pred_rows) if isinstance(pred_rows, list) else 0,
                        ref_row_count=0,
                    )
                )
                continue

            ref_rows = self.backend.execute_query(ref_sql)
            is_blocked = bool(meta.get("blocked", False)) or raised_unsafe
            sql_ok, _ = validate_read_only_sql(sql)
            is_valid_sql = (
                not is_blocked
                and error is None
                and sql_ok
                and isinstance(pred_rows, list)
            )
            exec_match = is_valid_sql and results_match(
                pred_rows, ref_rows, order_sensitive=order_sensitive
            )

            results.append(
                CaseEvalResult(
                    case_id=case_id,
                    question=question,
                    category=category,
                    difficulty=difficulty,
                    expected_behavior=expected_behavior,
                    predicted_sql=sql,
                    reference_sql=ref_sql,
                    is_valid_sql=is_valid_sql,
                    is_blocked=is_blocked,
                    execution_match=exec_match,
                    passed=exec_match,
                    attempts=attempts,
                    steps=steps,
                    latency_ms=latency_ms,
                    schema_recall=schema_recall,
                    error=error,
                    pred_row_count=len(pred_rows) if isinstance(pred_rows, list) else 0,
                    ref_row_count=len(ref_rows),
                )
            )

        non_safety_results = [r for r in results if r.expected_behavior != "blocked"]
        safety_results = [r for r in results if r.expected_behavior == "blocked"]

        execution_accuracy = (
            sum(1 for r in non_safety_results if r.execution_match)
            / len(non_safety_results)
            if non_safety_results
            else 0.0
        )
        valid_sql_rate = (
            sum(1 for r in non_safety_results if r.is_valid_sql)
            / len(non_safety_results)
            if non_safety_results
            else 0.0
        )
        safety_block_rate = (
            sum(1 for r in safety_results if r.is_blocked) / len(safety_results)
            if safety_results
            else 0.0
        )

        total = len(results)
        avg_attempts = sum(r.attempts for r in results) / total if total else 0.0
        avg_steps = sum(r.steps for r in results) / total if total else 0.0
        avg_latency_ms = sum(r.latency_ms for r in results) / total if total else 0.0

        recall_values = [r.schema_recall for r in results if r.schema_recall is not None]
        avg_schema_recall = (
            sum(recall_values) / len(recall_values) if recall_values else None
        )

        by_category: Dict[str, Dict[str, Any]] = {}
        for cat in sorted({r.category for r in results}):
            by_category[cat] = _compute_group_metrics(
                [r for r in results if r.category == cat]
            )

        by_difficulty: Dict[str, Dict[str, Any]] = {}
        for diff in sorted({r.difficulty for r in results}):
            by_difficulty[diff] = _compute_group_metrics(
                [r for r in results if r.difficulty == diff]
            )

        return EvalSummary(
            total_cases=total,
            non_safety_cases=len(non_safety_results),
            safety_cases=len(safety_results),
            execution_accuracy=round(execution_accuracy, 4),
            valid_sql_rate=round(valid_sql_rate, 4),
            safety_block_rate=round(safety_block_rate, 4),
            avg_attempts=round(avg_attempts, 2),
            avg_steps=round(avg_steps, 2),
            avg_latency_ms=round(avg_latency_ms, 2),
            schema_recall=(
                round(avg_schema_recall, 4) if avg_schema_recall is not None else None
            ),
            by_category=by_category,
            by_difficulty=by_difficulty,
            results=results,
        )


def format_markdown_report(summary: EvalSummary) -> str:
    """将 EvalSummary 格式化为可读的 Markdown 评测报告。"""
    lines: List[str] = [
        "# SmartDBChat Eval Report",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total Cases | {summary.total_cases} (non-safety: {summary.non_safety_cases}, safety: {summary.safety_cases}) |",
        f"| Execution Accuracy (EX) | {summary.execution_accuracy:.2%} |",
        f"| Valid SQL Rate | {summary.valid_sql_rate:.2%} |",
        f"| Safety Block Rate | {summary.safety_block_rate:.2%} |",
        f"| Avg Attempts | {summary.avg_attempts:.2f} |",
        f"| Avg Steps | {summary.avg_steps:.2f} |",
        f"| Avg Latency (ms) | {summary.avg_latency_ms:.2f} |",
    ]
    if summary.schema_recall is not None:
        lines.append(f"| Schema Recall | {summary.schema_recall:.2%} |")

    lines.extend(
        [
            "",
            "## Breakdown by Category",
            "",
            "| Category | Total | Passed | Pass Rate | EX | Valid SQL | Safety Block |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for cat, m in summary.by_category.items():
        lines.append(
            f"| {cat} | {m['total']} | {m['passed']} | {m['pass_rate']:.2%} | "
            f"{m['execution_accuracy']:.2%} | {m['valid_sql_rate']:.2%} | {m['safety_block_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Breakdown by Difficulty",
            "",
            "| Difficulty | Total | Passed | Pass Rate | EX | Valid SQL | Safety Block |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for diff, m in summary.by_difficulty.items():
        lines.append(
            f"| {diff} | {m['total']} | {m['passed']} | {m['pass_rate']:.2%} | "
            f"{m['execution_accuracy']:.2%} | {m['valid_sql_rate']:.2%} | {m['safety_block_rate']:.2%} |"
        )

    failed = [r for r in summary.results if not r.passed]
    if failed:
        lines.extend(["", "## Failed Cases", ""])
        for r in failed:
            lines.append(
                f"- **{r.case_id}** ({r.category}/{r.difficulty}): {r.question} "
                f"— error=`{r.error}`, pred_rows={r.pred_row_count}, ref_rows={r.ref_row_count}"
            )

    return "\n".join(lines) + "\n"


def build_reference_engine(backend: SQLiteEvalBackend) -> EvalEngineFn:
    """使用 golden reference_sql 在沙盒上执行的基线引擎（用于验证评测集与管线）。"""

    def _run(case: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        sql = str(case.get("reference_sql", ""))
        linked = extract_tables_from_sql(sql)
        try:
            rows = backend.execute_query(sql)
            return sql, rows, {"attempts": 1, "steps": 1, "linked_tables": linked}
        except UnsafeSQLError as exc:
            return (
                sql,
                [],
                {
                    "blocked": True,
                    "error": str(exc),
                    "attempts": 1,
                    "steps": 1,
                    "linked_tables": linked,
                },
            )

    return _run


def build_mock_engine(backend: SQLiteEvalBackend) -> EvalEngineFn:
    """用于无 API Key 冒烟测试的轻量规则 Mock 引擎。"""

    def _run(case: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        question = str(case.get("question", ""))
        if any(kw in question for kw in ("删除", "清空", "提高", "插入", "DROP", "DELETE", "UPDATE", "INSERT")):
            return (
                "",
                [],
                {"blocked": True, "attempts": 1, "steps": 1, "linked_tables": []},
            )
        if "多少名学生" in question:
            sql = "SELECT COUNT(*) AS cnt FROM students;"
        elif "平均 GPA" in question and "学院" not in question:
            sql = "SELECT ROUND(AVG(gpa), 4) AS avg_gpa FROM students;"
        else:
            sql = "SELECT COUNT(*) AS total FROM students;"
        rows = backend.execute_query(sql)
        return (
            sql,
            rows,
            {
                "attempts": 1,
                "steps": 1,
                "linked_tables": extract_tables_from_sql(sql),
            },
        )

    return _run


def build_openai_engine(backend: SQLiteEvalBackend) -> EvalEngineFn:
    """使用 OpenAI 兼容 LLMClient 驱动 SQLiteEvalBackend 的在线评测引擎。"""
    from smartdbchat.config import LLMConfig
    from smartdbchat.llm import LLMClient
    from smartdbchat.prompts import build_sql_prompt

    config = LLMConfig.from_env()
    if not config.api_key and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("未检测到 LLM API Key（请设置 OPENAI_API_KEY 或 LLM_API_KEY）")
    client = LLMClient(config)
    schema = backend.get_schema_info()

    def _run(case: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        question = str(case.get("question", ""))
        history = case.get("history") or []
        history_text = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history
        )
        full_query = f"{history_text}\nuser: {question}".strip() if history_text else question
        messages = build_sql_prompt(full_query, schema)
        sql = client.generate_sql(messages)
        linked = extract_tables_from_sql(sql)
        try:
            rows = backend.execute_query(sql)
            return sql, rows, {"attempts": 1, "steps": 1, "linked_tables": linked}
        except UnsafeSQLError as exc:
            return (
                sql,
                [],
                {
                    "blocked": True,
                    "error": str(exc),
                    "attempts": 1,
                    "steps": 1,
                    "linked_tables": linked,
                },
            )
        except Exception as exc:
            return (
                sql,
                [],
                {
                    "blocked": False,
                    "error": str(exc),
                    "attempts": 1,
                    "steps": 1,
                    "linked_tables": linked,
                },
            )

    return _run


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SmartDBChat Lightweight Eval Runner")
    parser.add_argument(
        "--mode",
        choices=["reference", "mock", "openai"],
        default="reference",
        help="评测引擎模式 (默认: reference)",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Golden cases JSON 路径 (默认: eval_data/golden_cases.json)",
    )
    parser.add_argument("--output-json", default=None, help="输出 JSON 报告路径")
    parser.add_argument("--output-md", default=None, help="输出 Markdown 报告路径")
    args = parser.parse_args(argv)

    cases = load_golden_cases(args.cases)
    with SQLiteEvalBackend() as backend:
        if args.mode == "reference":
            engine = build_reference_engine(backend)
        elif args.mode == "mock":
            engine = build_mock_engine(backend)
        else:
            engine = build_openai_engine(backend)

        runner = EvalRunner(backend)
        summary = runner.evaluate(engine, cases)

    report_md = format_markdown_report(summary)
    print(report_md)

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report_md, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
