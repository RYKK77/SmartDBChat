from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class PreflightStatus(str, Enum):
    ON_TOPIC = "on_topic"
    NEEDS_CLARIFICATION = "needs_clarification"
    OFF_TOPIC = "off_topic"
    REJECTED = "rejected"


@dataclass
class PreflightResult:
    """语义预检结果（借鉴 QueryWeaver RelevancyAgent 与可控 Agent Preflight 设计）。"""

    status: PreflightStatus
    reason: str = ""
    clarification_question: str = ""
    suggestions: List[str] = field(default_factory=list)

    @property
    def allow_execution(self) -> bool:
        return self.status == PreflightStatus.ON_TOPIC


class ErrorCategory(str, Enum):
    UNSAFE_SQL = "unsafe_sql"
    SYNTAX_ERROR = "syntax_error"
    SCHEMA_MISMATCH = "schema_mismatch"
    EMPTY_RESULT = "empty_result"
    EXECUTION_ERROR = "execution_error"


_DESTRUCTIVE_INTENT_PATTERNS = [
    re.compile(r"\b(DROP\s+TABLE|DROP\s+DATABASE|TRUNCATE\s+TABLE|DELETE\s+FROM)\b", re.I),
    re.compile(r"(删库|删除表|清空表|删除所有数据|删掉数据|篡改数据|写入数据|修改密码)", re.I),
    re.compile(r"(ignore\s+previous\s+instructions|忽略之前.*指令|越权|注入|system\s+prompt)", re.I),
]

_OFF_TOPIC_PATTERNS = [
    re.compile(
        r"^(你好|您好|嗨|哈喽|hi|hello|hey|早上好|下午好|晚上好|早安|晚安|在吗|在不在|有人吗|谢谢|感谢|thank\s*you|thanks|再见|拜拜|bye)[呀啊哇呢吗嘛哈喽~～！!。.,，？?\s]*$",
        re.I,
    ),
    re.compile(
        r"^(你是谁|你叫什么|你能做什么|你会干什么|你有什么功能|介绍一下你自己|能帮我做什么|怎么用)[呀啊呢吗？?!！。.\s]*$",
        re.I,
    ),
    re.compile(r"(写一首诗|讲个笑话|今天天气|翻译成英文|帮我写周报)", re.I),
]


def _default_suggestions(tables: Optional[List[str]] = None) -> List[str]:
    if tables:
        sample = ", ".join(tables[:4])
        return [
            f"请统计 {tables[0]} 表的总记录数",
            f"查看 {sample} 相关数据的分布情况",
        ]
    return [
        "统计各分类的数量分布",
        "查询排名前 10 的记录",
    ]


def _build_off_topic_reply(
    available_tables: Optional[List[str]] = None,
    database_description: str = "",
) -> str:
    """为日常问候或闲聊生成友好的引导式回复。"""
    desc = f"（{database_description}）" if database_description else ""
    if available_tables:
        table_preview = "、".join(f"`{t}`" for t in available_tables[:6])
        more = f" 等 {len(available_tables)} 张表" if len(available_tables) > 6 else ""
        return (
            f"你好！我是 **SmartDBChat** 智能数据分析助手{desc}。\n\n"
            f"当前数据库包含 {table_preview}{more}，您可以直接用自然语言向我提问，例如：\n"
            f"- 统计某张表的记录总数或各分类占比分布\n"
            f"- 查询核心指标排名前 10 的明细数据\n"
            f"- 分析不同维度之间的关联或趋势变化\n\n"
            f"您可以点击下方建议问题快速开始，或直接输入您想查询的数据需求！"
        )
    return (
        f"你好！我是 **SmartDBChat** 智能数据分析助手{desc}。\n\n"
        f"我可以将您的自然语言问题转化为安全的只读 SQL 查询，并自动生成数据表格与可视化图表。"
        f"请直接告诉我您想查询或分析的数据指标！"
    )


class PreflightChecker:
    """
    问题语义预检器：
    1. 确定性规则拦截恶意/破坏性请求与明显闲聊；
    2. 可选调用轻量 LLM 判断问题是否可问数、是否需要澄清反问。
    """

    def __init__(
        self,
        llm_client: Any = None,
        model: str = "",
        enable_llm_check: bool = False,
    ):
        self.llm_client = llm_client
        self.model = model
        self.enable_llm_check = enable_llm_check

    def check(
        self,
        question: str,
        available_tables: Optional[List[str]] = None,
        database_description: str = "",
        history: Optional[List[Dict[str, str]]] = None,
    ) -> PreflightResult:
        q = (question or "").strip()
        if not q or len(q) < 2:
            return PreflightResult(
                status=PreflightStatus.NEEDS_CLARIFICATION,
                reason="输入问题过短或为空，请提供更具体的查询需求。",
                clarification_question="您想查询哪张表或哪类指标的数据？",
                suggestions=_default_suggestions(available_tables),
            )

        for pat in _DESTRUCTIVE_INTENT_PATTERNS:
            if pat.search(q):
                return PreflightResult(
                    status=PreflightStatus.REJECTED,
                    reason="检测到潜在破坏性操作或越权/注入意图，系统仅支持只读数据查询。",
                    suggestions=_default_suggestions(available_tables),
                )

        for pat in _OFF_TOPIC_PATTERNS:
            if pat.search(q):
                return PreflightResult(
                    status=PreflightStatus.OFF_TOPIC,
                    reason="该问题属于日常闲聊或非数据库查询需求。",
                    clarification_question=_build_off_topic_reply(
                        available_tables=available_tables,
                        database_description=database_description,
                    ),
                    suggestions=_default_suggestions(available_tables),
                )

        if self.enable_llm_check and self.llm_client is not None and self.model:
            llm_res = self._llm_check(q, available_tables, database_description, history)
            if llm_res is not None:
                return llm_res

        return PreflightResult(status=PreflightStatus.ON_TOPIC, reason="语义预检通过")

    def _llm_check(
        self,
        question: str,
        available_tables: Optional[List[str]],
        database_description: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Optional[PreflightResult]:
        tables_str = ", ".join(available_tables or [])
        hist_str = ""
        if history:
            recent = history[-4:]
            hist_str = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in recent)

        prompt = (
            "You are a Preflight Relevancy Gate for a read-only NL2SQL system.\n"
            f"Database description: {database_description or 'N/A'}\n"
            f"Available tables: {tables_str or 'N/A'}\n"
            f"Recent history:\n{hist_str or 'None'}\n\n"
            f"User question: {question}\n\n"
            "Classify into one of: on_topic, needs_clarification, off_topic, rejected.\n"
            'Return strict JSON: {"status": "...", "reason": "...", "clarification_question": "...", "suggestions": ["..."]}'
        )
        try:
            raw = self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
            )
            cleaned = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.I | re.M).strip()
            data = json.loads(cleaned)
            status_str = str(data.get("status", "on_topic")).lower()
            status_map = {
                "on_topic": PreflightStatus.ON_TOPIC,
                "needs_clarification": PreflightStatus.NEEDS_CLARIFICATION,
                "off_topic": PreflightStatus.OFF_TOPIC,
                "rejected": PreflightStatus.REJECTED,
            }
            status = status_map.get(status_str, PreflightStatus.ON_TOPIC)
            return PreflightResult(
                status=status,
                reason=str(data.get("reason", "")),
                clarification_question=str(data.get("clarification_question", "")),
                suggestions=[str(s) for s in (data.get("suggestions") or [])[:3]],
            )
        except Exception:
            return None


def classify_error(error_msg: Optional[str], is_empty: bool = False) -> ErrorCategory:
    """对 SQL 校验或执行失败原因进行细粒度分类，用于有界纠错决策。"""
    if is_empty and not error_msg:
        return ErrorCategory.EMPTY_RESULT
    msg = (error_msg or "").lower()
    if any(
        kw in msg
        for kw in ("unknown column", "no such column", "doesn't exist", "no such table", "不存在", "未授权的表")
    ):
        return ErrorCategory.SCHEMA_MISMATCH
    if any(kw in msg for kw in ("syntax", "parse", "解析失败", "near ")):
        return ErrorCategory.SYNTAX_ERROR
    if any(
        kw in msg
        for kw in ("unsafe", "仅允许只读", "禁止", "写操作", "多条 sql", "outfile", "高危")
    ):
        return ErrorCategory.UNSAFE_SQL
    if is_empty:
        return ErrorCategory.EMPTY_RESULT
    return ErrorCategory.EXECUTION_ERROR


class HealingTracker:
    """
    有界纠错追踪器（借鉴 QueryWeaver HealerAgent 与可控 Agent 状态机）：
    - 拦截 UNSAFE_SQL，绝不把应拒绝的请求“硬修”成查询；
    - 记录已尝试 SQL 哈希，防止重复无进展循环；
    - 限制各类错误与总重试预算。
    """

    def __init__(self, max_total_attempts: int = 3, max_empty_retries: int = 1):
        self.max_total_attempts = max(1, max_total_attempts)
        self.max_empty_retries = max(0, max_empty_retries)
        self.seen_hashes: Set[str] = set()
        self.category_counts: Dict[ErrorCategory, int] = {}
        self.total_attempts: int = 0

    @staticmethod
    def _hash_sql(sql: str) -> str:
        norm = re.sub(r"\s+", " ", (sql or "").strip().lower()).rstrip(";")
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]

    def record_attempt(self, sql: str, category: ErrorCategory) -> None:
        self.total_attempts += 1
        if sql:
            self.seen_hashes.add(self._hash_sql(sql))
        self.category_counts[category] = self.category_counts.get(category, 0) + 1

    def should_retry(self, next_sql: str, category: ErrorCategory) -> Tuple[bool, str]:
        if category == ErrorCategory.UNSAFE_SQL:
            return False, "检测到不安全或越权 SQL，终止纠错"
        if self.total_attempts >= self.max_total_attempts:
            return False, f"已达到最大纠错预算 ({self.max_total_attempts})"
        if category == ErrorCategory.EMPTY_RESULT:
            if self.category_counts.get(ErrorCategory.EMPTY_RESULT, 0) > self.max_empty_retries:
                return False, "空结果重试已达上限，转入 Recovery 诊断"
        if next_sql and self._hash_sql(next_sql) in self.seen_hashes:
            return False, "生成的修复 SQL 与历史尝试重复，终止无进展循环"
        return True, ""


def build_recovery_summary(
    question: str,
    last_sql: str,
    last_category: ErrorCategory,
    last_error: Optional[str],
    available_tables: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """当纠错耗尽或结果为空时，生成可解释的 Recovery 诊断说明与后续建议问题。"""
    suggestions = _default_suggestions(available_tables)
    if last_category == ErrorCategory.EMPTY_RESULT:
        msg = (
            f"已完成只读查询，但未找到符合条件的数据。\n\n"
            f"**可能原因**：\n"
            f"- 过滤条件中的枚举值或名称与数据库实际存储不完全一致（例如缩写 vs 全称）\n"
            f"- 时间范围或数值阈值过窄\n\n"
            f"**建议操作**：可尝试放宽筛选条件，或先查看相关字段的枚举值分布。"
        )
        return msg, suggestions

    msg = (
        f"查询在多次有界修复后仍未能成功执行。\n\n"
        f"- **错误类型**：`{last_category.value}`\n"
        f"- **错误详情**：{last_error or '未知错误'}\n"
        f"- **最后尝试 SQL**：`{last_sql or '无'}`\n"
    )
    return msg, suggestions


def generate_follow_up_suggestions(
    question: str,
    sql: str,
    tables: Optional[List[str]] = None,
) -> List[str]:
    """借鉴 QueryWeaver FollowUpAgent：基于当前查询上下文生成下一步探索建议。"""
    q_lower = (question or "").lower()
    suggestions: List[str] = []
    if "top" not in q_lower and "前" not in q_lower and "limit" not in (sql or "").lower():
        suggestions.append("按核心指标降序排列，查看排名前 10 的明细")
    if "趋势" not in q_lower and "月" not in q_lower and "年" not in q_lower:
        suggestions.append("按时间维度（年/月/学期）统计变化趋势")
    if "占比" not in q_lower and "分布" not in q_lower:
        suggestions.append("按分类维度统计数量与占比分布")
    if tables and len(tables) >= 2:
        suggestions.append(f"进一步下钻分析 {', '.join(tables[:2])} 的关联异常项")
    return suggestions[:3]
