import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def clean_json_comments(json_str: str) -> str:
    """清洗 JSON 字符串中的单行/多行注释与多余空行。"""
    cleaned_str = re.sub(r"(?<!:)//.*?(\n|$)", r"\1", json_str)
    cleaned_str = re.sub(r"/\*.*?\*/", "", cleaned_str, flags=re.DOTALL)
    cleaned_str = re.sub(r"\s+$", "", cleaned_str, flags=re.MULTILINE)
    cleaned_str = re.sub(r"\n\s*\n", "\n", cleaned_str)
    return cleaned_str


def extract_chart_config(response_text: str) -> Optional[Dict[str, Any]]:
    """从 LLM 响应文本中提取并解析图表配置 JSON。"""
    cleaned_text = clean_json_comments(response_text)
    matches = re.findall(r"```(?:json)?\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE)
    if not matches:
        return None

    for block in matches:
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            if "charts" in parsed and isinstance(parsed["charts"], list):
                return parsed
            if "type" in parsed and "data" in parsed:
                return {"charts": [parsed]}
        elif isinstance(parsed, list):
            chart_items = [
                item for item in parsed if isinstance(item, dict) and "type" in item and "data" in item
            ]
            if chart_items:
                return {"charts": chart_items}

    logger.warning("图表配置 JSON 格式错误，无法解析")
    return None


def strip_chart_json_block(response_text: str) -> str:
    """移除 LLM 分析文本中用于图表渲染的 JSON 代码块，避免在界面直接暴露配置 JSON。"""
    patterns = [
        r'```(?:json)?\s*\{[\s\S]*?"charts"[\s\S]*?\}\s*```',
        r'```(?:json)?\s*\[\s*\{[\s\S]*?"type"[\s\S]*?"data"[\s\S]*?\}\s*\]\s*```',
        r'```(?:json)?\s*\{[\s\S]*?"type"[\s\S]*?"data"[\s\S]*?\}\s*```',
    ]
    cleaned = response_text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def recommend_chart_config(
    rows: List[Dict[str, Any]],
    title: str = "查询结果自动可视化",
) -> Optional[Dict[str, Any]]:
    """
    确定性图表推荐兜底（借鉴 WrenAI / SQLBot 自动图表推断）：
    当 LLM 未输出图表配置或格式校验失败时，根据结果集列类型自动生成 bar / line / pie 配置。
    """
    if not rows or not isinstance(rows, list) or not isinstance(rows[0], dict):
        return None
    if len(rows) < 2:
        return None

    sample = rows[:50]
    keys = list(sample[0].keys())
    if len(keys) < 2:
        return None

    def _is_number(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    def _looks_temporal(col_name: str, vals: List[Any]) -> bool:
        lower = col_name.lower()
        if any(tok in lower for tok in ("date", "time", "month", "year", "day", "semester", "日期", "时间", "月份", "年份", "学期")):
            return True
        str_vals = [str(v) for v in vals[:5] if v is not None]
        return any(re.match(r"^\d{4}[-/]\d{1,2}", s) for s in str_vals)

    numeric_cols: List[str] = []
    label_cols: List[str] = []
    for k in keys:
        non_null = [r.get(k) for r in sample if r.get(k) is not None]
        if not non_null:
            continue
        if all(_is_number(v) for v in non_null):
            # 跳过纯自增 id 列
            if k.lower() in ("id", "student_id", "course_id", "department_id"):
                continue
            numeric_cols.append(k)
        else:
            label_cols.append(k)

    if not numeric_cols:
        return None

    if not label_cols:
        label_col = keys[0]
        val_col = numeric_cols[0] if numeric_cols[0] != label_col else (numeric_cols[1] if len(numeric_cols) > 1 else None)
        if not val_col:
            return None
    else:
        label_col = label_cols[0]
        val_col = numeric_cols[0]

    labels = [str(r.get(label_col, "")) for r in sample]
    values = [float(r.get(val_col) or 0) for r in sample]
    col_vals = [r.get(label_col) for r in sample]

    if _looks_temporal(label_col, col_vals):
        chart_type = "line"
        data = {
            "labels": labels,
            "values": values,
            "show_markers": True,
            "x_axis_title": label_col,
            "y_axis_title": val_col,
        }
    elif len(sample) <= 6 and all(v >= 0 for v in values) and any(
        tok in val_col.lower() for tok in ("count", "cnt", "total", "sum", "amount", "占比", "数量", "总额")
    ):
        chart_type = "pie"
        data = {
            "labels": labels,
            "values": values,
            "hole": 0.4,
        }
    else:
        chart_type = "bar"
        data = {
            "labels": labels,
            "values": values,
            "orientation": "h" if len(sample) > 8 else "v",
            "show_values": True,
        }

    return {
        "charts": [
            {
                "type": chart_type,
                "title": f"{val_col} 按 {label_col} 分布",
                "description": f"基于 {len(sample)} 条查询结果自动推荐的{chart_type}可视化",
                "data": data,
            }
        ]
    }

