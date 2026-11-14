from smartdbchat.charts import build_figures, extract_chart_config


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


def test_build_figures_supports_graph_objects_charts():
    config = {
        "charts": [
            {
                "type": "line",
                "title": "多系列折线图",
                "data": {
                    "series": [
                        {"name": "A", "x": ["1月", "2月"], "y": [10, 20]},
                        {"name": "B", "x": ["1月", "2月"], "y": [15, 25]},
                    ]
                },
            },
            {
                "type": "radar",
                "title": "能力雷达图",
                "data": {"labels": ["速度", "准确", "成本"], "values": [80, 90, 70]},
            },
            {
                "type": "funnel",
                "title": "转化漏斗",
                "data": {"labels": ["访问", "下单"], "values": [100, 40]},
            },
            {
                "type": "waterfall",
                "title": "瀑布图",
                "data": {"labels": ["期初", "增长"], "values": [100, 20]},
            },
        ]
    }
    figures = build_figures(config)
    assert len(figures) == 4


def test_strip_chart_json_block_and_malformed_resilience():
    from smartdbchat.charts import strip_chart_json_block

    raw = '这是分析结论。\n\n```json\n{"charts": [{"type": "pie"}]}\n```'
    assert strip_chart_json_block(raw) == "这是分析结论。"

    # 即使其中一个图表配置异常，也不应影响其他图表或抛出异常
    bad_config = {
        "charts": [
            {"type": "pie", "data": {"labels": ["A"], "values": [1]}},
            {"type": "pie", "data": {"labels": ["A", "B"], "values": "invalid_not_list"}},
        ]
    }
    figs = build_figures(bad_config)
    assert len(figs) == 1


def test_extract_and_strip_array_chart_json_block():
    from smartdbchat.charts import strip_chart_json_block

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
