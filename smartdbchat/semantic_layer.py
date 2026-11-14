from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import yaml


def _tokenize(text: str) -> Set[str]:
    """中英文混合分词（英文按词、中文按 2-gram 与单字），用于语义层检索。"""
    if not text:
        return set()
    tokens: Set[str] = set()
    for word in re.findall(r"[A-Za-z0-9_]+", text.lower()):
        tokens.add(word)
        if "_" in word:
            tokens.update(p for p in word.split("_") if p)
    for seq in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(seq) == 1:
            tokens.add(seq)
        else:
            for i in range(len(seq) - 1):
                tokens.add(seq[i : i + 2])
            for ch in seq:
                tokens.add(ch)
    return tokens


@dataclass
class MetricDef:
    """业务指标定义（参考 WrenAI MDL 语义层）。"""

    name: str
    expression: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)


@dataclass
class GoldenExample:
    """经过验证的 Few-Shot Golden SQL 示例。"""

    question: str
    sql: str
    tables: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class SemanticContext:
    """针对当前问题检索出的结构化语义上下文。"""

    database_description: str = ""
    user_rules: List[str] = field(default_factory=list)
    matched_metrics: List[MetricDef] = field(default_factory=list)
    matched_glossary: Dict[str, str] = field(default_factory=dict)
    matched_enums: Dict[str, Dict[str, str]] = field(default_factory=dict)
    table_descriptions: Dict[str, str] = field(default_factory=dict)
    column_descriptions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    golden_examples: List[GoldenExample] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.database_description
            or self.user_rules
            or self.matched_metrics
            or self.matched_glossary
            or self.matched_enums
            or self.table_descriptions
            or self.column_descriptions
            or self.golden_examples
        )

    def to_prompt_text(self) -> str:
        """渲染为可直接注入 SQL 生成 Prompt 的结构化语义约束文本。"""
        if self.is_empty():
            return ""
        sections: List[str] = []

        if self.database_description:
            sections.append(f"【数据库业务背景】\n{self.database_description}")

        if self.user_rules:
            rules_lines = "\n".join(f"- {rule}" for rule in self.user_rules)
            sections.append(f"【全局业务规则（必须遵守）】\n{rules_lines}")

        if self.table_descriptions or self.column_descriptions:
            desc_lines: List[str] = []
            for tbl, desc in self.table_descriptions.items():
                desc_lines.append(f"- 表 `{tbl}`: {desc}")
            for tbl, cols in self.column_descriptions.items():
                for col, cdesc in cols.items():
                    desc_lines.append(f"- 字段 `{tbl}.{col}`: {cdesc}")
            sections.append("【表与字段语义注释】\n" + "\n".join(desc_lines))

        if self.matched_metrics:
            metric_lines = []
            for m in self.matched_metrics:
                line = f"- {m.name}: `{m.expression}`"
                if m.description:
                    line += f" ({m.description})"
                metric_lines.append(line)
            sections.append("【相关业务指标口径】\n" + "\n".join(metric_lines))

        if self.matched_glossary:
            gloss_lines = [f"- {term}: {defn}" for term, defn in self.matched_glossary.items()]
            sections.append("【业务术语映射】\n" + "\n".join(gloss_lines))

        if self.matched_enums:
            enum_lines = []
            for col_key, mapping in self.matched_enums.items():
                pairs = ", ".join(f"{k}={v}" for k, v in mapping.items())
                enum_lines.append(f"- `{col_key}`: {pairs}")
            sections.append("【字段枚举值说明】\n" + "\n".join(enum_lines))

        if self.golden_examples:
            ex_lines = []
            for idx, ex in enumerate(self.golden_examples, 1):
                ex_lines.append(f"示例 {idx}: Q: {ex.question}\nSQL: {ex.sql}")
            sections.append("【相似历史参考 SQL（Few-Shot）】\n" + "\n".join(ex_lines))

        return "\n\n".join(sections)


@dataclass
class SemanticLayer:
    """
    轻量级业务语义层（融合 WrenAI MDL、SQLBot 术语校准与 QueryWeaver user_rules）。
    支持从 YAML/Dict 加载，按问题动态检索相关指标、术语、枚举值与 Golden SQL。
    """

    database_description: str = ""
    user_rules: List[str] = field(default_factory=list)
    metrics: List[MetricDef] = field(default_factory=list)
    glossary: Dict[str, str] = field(default_factory=dict)
    table_aliases: Dict[str, List[str]] = field(default_factory=dict)
    table_descriptions: Dict[str, str] = field(default_factory=dict)
    column_descriptions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    column_enums: Dict[str, Dict[str, str]] = field(default_factory=dict)
    golden_examples: List[GoldenExample] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticLayer":
        if not data:
            return cls()
        metrics = [
            MetricDef(
                name=m["name"],
                expression=m["expression"],
                description=m.get("description", ""),
                aliases=list(m.get("aliases", []) or []),
                tables=list(m.get("tables", []) or []),
            )
            for m in (data.get("metrics") or [])
            if isinstance(m, dict) and m.get("name") and m.get("expression")
        ]
        examples = [
            GoldenExample(
                question=ex["question"],
                sql=ex["sql"],
                tables=list(ex.get("tables", []) or []),
                tags=list(ex.get("tags", []) or []),
            )
            for ex in (data.get("golden_examples") or [])
            if isinstance(ex, dict) and ex.get("question") and ex.get("sql")
        ]
        raw_enums = data.get("column_enums") or {}
        norm_enums: Dict[str, Dict[str, str]] = {}
        for k, v in raw_enums.items():
            if isinstance(v, dict):
                norm_enums[str(k)] = {str(ek): str(ev) for ek, ev in v.items()}

        return cls(
            database_description=str(data.get("database_description", "") or ""),
            user_rules=[str(r) for r in (data.get("user_rules") or [])],
            metrics=metrics,
            glossary={str(k): str(v) for k, v in (data.get("glossary") or {}).items()},
            table_aliases={
                str(k): [str(a) for a in (v or [])]
                for k, v in (data.get("table_aliases") or {}).items()
            },
            table_descriptions={
                str(k): str(v) for k, v in (data.get("table_descriptions") or {}).items()
            },
            column_descriptions={
                str(tbl): {str(col): str(desc) for col, desc in (cols or {}).items()}
                for tbl, cols in (data.get("column_descriptions") or {}).items()
                if isinstance(cols, dict)
            },
            column_enums=norm_enums,
            golden_examples=examples,
        )

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "SemanticLayer":
        parsed = yaml.safe_load(yaml_text) or {}
        if not isinstance(parsed, dict):
            raise ValueError("语义层 YAML 根节点必须是字典结构")
        return cls.from_dict(parsed)

    @classmethod
    def from_yaml_file(cls, path: str) -> "SemanticLayer":
        if not os.path.exists(path):
            raise FileNotFoundError(f"语义层配置文件不存在: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_yaml(f.read())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database_description": self.database_description,
            "user_rules": list(self.user_rules),
            "metrics": [
                {
                    "name": m.name,
                    "expression": m.expression,
                    "description": m.description,
                    "aliases": list(m.aliases),
                    "tables": list(m.tables),
                }
                for m in self.metrics
            ],
            "glossary": dict(self.glossary),
            "table_aliases": dict(self.table_aliases),
            "table_descriptions": dict(self.table_descriptions),
            "column_descriptions": dict(self.column_descriptions),
            "column_enums": dict(self.column_enums),
            "golden_examples": [
                {
                    "question": ex.question,
                    "sql": ex.sql,
                    "tables": list(ex.tables),
                    "tags": list(ex.tags),
                }
                for ex in self.golden_examples
            ],
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)

    def save_yaml_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_yaml())

    def add_golden_example(
        self,
        question: str,
        sql: str,
        tables: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        q_norm = (question or "").strip()
        s_norm = (sql or "").strip()
        if not q_norm or not s_norm:
            return
        for ex in self.golden_examples:
            if ex.question.strip().lower() == q_norm.lower():
                ex.sql = s_norm
                if tables:
                    ex.tables = list(tables)
                return
        self.golden_examples.append(
            GoldenExample(
                question=q_norm,
                sql=s_norm,
                tables=list(tables or []),
                tags=list(tags or []),
            )
        )

    def retrieve_context(
        self,
        question: str,
        selected_tables: Optional[List[str]] = None,
        top_k_examples: int = 3,
    ) -> SemanticContext:
        """按用户问题和候选表检索最相关的指标、术语、枚举值与 Golden SQL 示例。"""
        q_tokens = _tokenize(question)
        q_lower = (question or "").lower()
        selected_lower = {t.lower() for t in (selected_tables or [])}

        # 1. 匹配业务指标
        matched_metrics: List[MetricDef] = []
        for metric in self.metrics:
            names = [metric.name] + list(metric.aliases)
            direct_hit = any(n.lower() in q_lower for n in names if n)
            m_tokens = _tokenize(" ".join(names + [metric.description]))
            token_overlap = len(q_tokens & m_tokens) >= 2
            table_overlap = bool(selected_lower) and any(
                t.lower() in selected_lower for t in metric.tables
            )
            if direct_hit or token_overlap or (table_overlap and len(q_tokens & m_tokens) >= 1):
                matched_metrics.append(metric)

        # 2. 匹配业务术语
        matched_glossary: Dict[str, str] = {}
        for term, defn in self.glossary.items():
            if term.lower() in q_lower or len(q_tokens & _tokenize(term)) >= 2:
                matched_glossary[term] = defn

        # 3. 过滤选中表的描述与枚举值
        tbl_descs: Dict[str, str] = {}
        for tbl, desc in self.table_descriptions.items():
            if not selected_lower or tbl.lower() in selected_lower:
                tbl_descs[tbl] = desc

        col_descs: Dict[str, Dict[str, str]] = {}
        for tbl, cols in self.column_descriptions.items():
            if not selected_lower or tbl.lower() in selected_lower:
                col_descs[tbl] = dict(cols)

        matched_enums: Dict[str, Dict[str, str]] = {}
        for col_key, mapping in self.column_enums.items():
            tbl_part = col_key.split(".", 1)[0].lower() if "." in col_key else ""
            if not selected_lower or not tbl_part or tbl_part in selected_lower:
                matched_enums[col_key] = dict(mapping)

        # 4. Few-Shot Golden Examples 召回
        scored_examples: List[tuple[float, GoldenExample]] = []
        for ex in self.golden_examples:
            ex_tokens = _tokenize(ex.question + " " + " ".join(ex.tags))
            if not q_tokens or not ex_tokens:
                continue
            inter = len(q_tokens & ex_tokens)
            if inter == 0:
                continue
            jaccard = inter / len(q_tokens | ex_tokens)
            table_bonus = 0.0
            if selected_lower and ex.tables:
                overlap = sum(1 for t in ex.tables if t.lower() in selected_lower)
                table_bonus = 0.15 * (overlap / len(ex.tables))
            scored_examples.append((jaccard + table_bonus, ex))

        scored_examples.sort(key=lambda item: item[0], reverse=True)
        top_examples = [ex for score, ex in scored_examples[:top_k_examples] if score > 0.05]

        return SemanticContext(
            database_description=self.database_description,
            user_rules=list(self.user_rules),
            matched_metrics=matched_metrics,
            matched_glossary=matched_glossary,
            matched_enums=matched_enums,
            table_descriptions=tbl_descs,
            column_descriptions=col_descs,
            golden_examples=top_examples,
        )
