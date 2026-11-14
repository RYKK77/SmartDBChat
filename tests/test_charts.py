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
