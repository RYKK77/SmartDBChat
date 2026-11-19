from smartdbchat.charts import (
    extract_chart_config,
    recommend_chart_config,
    strip_chart_json_block,
)


def test_extract_chart_config_strips_comments():
    text = """分析完成。
```json
{
  "charts": [
    {
      "type": "bar",
      "title": "各专业人数", // 注释
      "data": {
        "labels": ["CS", "Math"],
        "values": [120, 80] /* 多行注释 */
      }
    }
  ]
}
```"""
    config = extract_chart_config(text)
    assert config is not None
    assert len(config["charts"]) == 1
    assert config["charts"][0]["type"] == "bar"


def test_recommend_chart_config_types():
    # 时间序列 -> line
    time_rows = [
        {"month": "2026-01", "revenue": 100},
        {"month": "2026-02", "revenue": 150},
    ]
    line_cfg = recommend_chart_config(time_rows)
    assert line_cfg is not None
    assert line_cfg["charts"][0]["type"] == "line"

    # 少量计数占比 -> pie
    pie_rows = [
        {"department": "计算机", "student_count": 120},
        {"department": "数学", "student_count": 80},
    ]
    pie_cfg = recommend_chart_config(pie_rows)
    assert pie_cfg is not None
    assert pie_cfg["charts"][0]["type"] == "pie"

    # 一般指标对比 -> bar
    bar_rows = [
        {"course": "高数", "avg_gpa": 3.4},
        {"course": "线代", "avg_gpa": 3.6},
    ]
    bar_cfg = recommend_chart_config(bar_rows)
    assert bar_cfg is not None
    assert bar_cfg["charts"][0]["type"] == "bar"


def test_strip_chart_json_block_and_single_object_format():
    raw = '这是分析结论。\n\n```json\n{"charts": [{"type": "pie"}]}\n```'
    assert strip_chart_json_block(raw) == "这是分析结论。"

    single_raw = '结论文本\n```json\n{"type": "bar", "title": "T", "data": {"labels": ["A"], "values": [1]}}\n```'
    cfg = extract_chart_config(single_raw)
    assert cfg is not None
    assert len(cfg["charts"]) == 1
    assert strip_chart_json_block(single_raw) == "结论文本"


def test_extract_and_strip_array_chart_json_block():
    raw = (
        "二、可视化方案\n\n"
        "```json\n"
        '[{"type": "bar", "title": "GPA", "data": {"labels": ["CS"], "values": [3.8]}}]\n'
        "```"
    )
    config = extract_chart_config(raw)
    assert config is not None
    assert len(config["charts"]) == 1
    assert config["charts"][0]["type"] == "bar"
    assert strip_chart_json_block(raw) == "二、可视化方案"

