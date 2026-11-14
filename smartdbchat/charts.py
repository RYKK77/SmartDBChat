import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import plotly.express as px
import plotly.graph_objects as go

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
    matches = re.findall(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL)
    if not matches:
        return None

    try:
        parsed = json.loads(matches[0])
        if isinstance(parsed, dict):
            return parsed
        return None
    except json.JSONDecodeError as exc:
        logger.warning("图表配置 JSON 格式错误，无法解析: %s", exc)
        return None


def build_plotly_figure(chart: Dict[str, Any]) -> Optional[go.Figure]:
    """根据单个图表配置构建 Plotly Figure 对象。"""
    chart_type = chart.get("type")
    title = chart.get("title", "数据可视化")
    data = chart.get("data", {})
    labels = data.get("labels", [])
    values = data.get("values", [])

    if chart_type == "pie":
        return px.pie(
            names=labels,
            values=values,
            title=title,
            hole=data.get("hole", 0),
            color_discrete_sequence=data.get("colors"),
        )

    if chart_type == "treemap":
        parents = data.get("parents", [""] * len(labels))
        return px.treemap(
            names=labels,
            values=values,
            parents=parents,
            title=title,
            color_discrete_sequence=data.get("colors"),
        )

    if chart_type == "bar":
        orientation = data.get("orientation", "v")
        if orientation == "h":
            fig = px.bar(
                y=labels,
                x=values,
                title=title,
                orientation="h",
                color=data.get("color_by"),
                color_discrete_sequence=data.get("colors"),
                text_auto=data.get("show_values", True),
            )
        else:
            fig = px.bar(
                x=labels,
                y=values,
                title=title,
                color=data.get("color_by"),
                color_discrete_sequence=data.get("colors"),
                text_auto=data.get("show_values", True),
            )
        if data.get("is_grouped", False):
            fig.update_layout(barmode="group")
        elif data.get("is_stacked", False):
            fig.update_layout(barmode="stack")
        return fig

    if chart_type == "line":
        if "series" in data:
            fig = go.Figure()
            for series in data.get("series", []):
                fig.add_trace(
                    go.Scatter(
                        x=series.get("x", labels),
                        y=series.get("y", []),
                        mode="lines+markers"
                        if data.get("show_markers", True)
                        else "lines",
                        name=series.get("name", ""),
                        line=dict(width=series.get("line_width", 2)),
                    )
                )
            fig.update_layout(title=title)
        else:
            fig = px.line(
                x=labels,
                y=values,
                title=title,
                markers=data.get("show_markers", True),
                line_shape=data.get("line_shape", "linear"),
            )
        fig.update_layout(
            xaxis_title=data.get("x_axis_title", ""),
            yaxis_title=data.get("y_axis_title", ""),
        )
        return fig

    if chart_type == "scatter":
        return px.scatter(
            x=data.get("x", labels),
            y=data.get("y", values),
            size=data.get("sizes"),
            color=data.get("color_values"),
            title=title,
            hover_name=data.get("hover_names"),
            size_max=data.get("size_max", 20),
            opacity=data.get("opacity", 0.7),
        )

    if chart_type == "area":
        fig = px.area(
            x=labels,
            y=values,
            title=title,
            color_discrete_sequence=data.get("colors"),
        )
        if not data.get("is_stacked", True):
            fig.update_layout(groupnorm="fraction")
        return fig

    if chart_type == "heatmap":
        z_values = data.get("z_values", [])
        x_labels = data.get("x_labels", labels)
        y_labels = data.get("y_labels", [])
        fig = px.imshow(
            z_values,
            x=x_labels,
            y=y_labels,
            color_continuous_scale=data.get("color_scale", "Viridis"),
            title=title,
        )
        if data.get("show_values", True):
            fig.update_traces(text=z_values, texttemplate="%{text}")
        return fig

    if chart_type == "radar":
        categories = labels
        if "series" in data:
            fig = go.Figure()
            for series in data.get("series", []):
                fig.add_trace(
                    go.Scatterpolar(
                        r=series.get("values", []),
                        theta=categories,
                        fill="toself",
                        name=series.get("name", ""),
                    )
                )
        else:
            fig = go.Figure(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill="toself",
                )
            )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, data.get("max_value", None)],
                )
            ),
            title=title,
        )
        return fig

    if chart_type == "box":
        return px.box(
            x=data.get("group_by"),
            y=values,
            title=title,
            points=data.get("show_points", "outliers"),
        )

    if chart_type == "funnel":
        text_info = data.get("text_info", "value+percent initial")
        if text_info == "value+percent":
            text_info = "value+percent initial"
        fig = go.Figure(
            go.Funnel(
                y=labels,
                x=values,
                textinfo=text_info,
            )
        )
        fig.update_layout(title=title)
        return fig

    if chart_type == "waterfall":
        measure = data.get("measures", ["relative"] * len(labels))
        fig = go.Figure(
            go.Waterfall(
                name=title,
                orientation="v",
                measure=measure,
                x=labels,
                y=values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            )
        )
        fig.update_layout(title=title)
        return fig

    logger.warning("不支持的图表类型: %s", chart_type)
    return None


def strip_chart_json_block(response_text: str) -> str:
    """移除 LLM 分析文本末尾用于图表渲染的 ```json``` 代码块，避免在界面直接暴露配置 JSON。"""
    cleaned = re.sub(r"```json\s*\{[\s\S]*?\"charts\"[\s\S]*?\}\s*```", "", response_text)
    return cleaned.strip()


def build_figures(
    chart_config: Optional[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], go.Figure]]:
    """批量构建图表配置中的所有 Plotly Figure。"""
    if not chart_config or "charts" not in chart_config:
        return []

    figures: List[Tuple[Dict[str, Any], go.Figure]] = []
    for chart in chart_config.get("charts", []):
        try:
            fig = build_plotly_figure(chart)
            if fig is not None:
                figures.append((chart, fig))
        except Exception as exc:
            logger.warning("构建图表失败 (%s): %s", chart.get("type"), exc)
    return figures


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

