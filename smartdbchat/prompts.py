from typing import Any, Dict, List, Optional


def build_sql_prompt(
    query: str, schema_info: str, is_filtered: bool = False
) -> List[Dict[str, str]]:
    """构建自然语言转 SQL 的提示词。"""
    module_hint = ""
    if is_filtered:
        module_hint = """
            注意：当前只提供了与用户查询意图相关的表结构信息，请只使用提供的表结构生成SQL，不要尝试使用不在schema中的表。
            """

    return [
        {
            "role": "system",
            "content": f"""你是一个SQL专家，负责将自然语言转换为SQL查询语句。
                请注意：
                1. 直接返回SQL语句，不要包含任何其他信息。
                2. 现在使用的是MySQL数据库，确保SQL语句语法正确
                3. 使用提供的schema信息构建查询
                4. 注意使用适当的表连接和条件
                5. 查询的表头都需要有中文注释，不要使用表别名
                6. 确保查询安全，避免SQL注入风险，仅生成只读查询（SELECT/WITH/SHOW）
                7. 检查表名和字段名是否正确，确保它们存在于schema中
                8. 考虑可能的数据类型转换问题
                9. 确保WHERE条件合理
                {module_hint}""",
        },
        {
            "role": "user",
            "content": f"""
                数据库Schema信息如下：
                {schema_info}

                请将以下问题转换为SQL查询语句：
                {query}
                """,
        },
    ]


def build_retry_sql_prompt(
    query: str,
    schema_info: str,
    failed_sql: Optional[str] = None,
    error_info: Optional[str] = None,
) -> List[Dict[str, str]]:
    """构建 SQL 执行失败或结果为空时的重试提示词。"""
    return [
        {
            "role": "system",
            "content": """你是一个SQL专家，负责将自然语言转换为SQL查询语句。
                之前生成的SQL查询执行失败或返回为空，请尝试生成一个更准确的SQL查询。
                请注意：
                1. 直接返回SQL语句，不要包含任何其他信息。
                2. 现在使用的是MySQL数据库，确保SQL语句语法正确
                3. 使用提供的schema信息构建查询
                4. 注意使用适当的表连接和条件
                5. 查询的表头都需要有中文注释，不要使用表别名
                6. 确保查询安全，避免SQL注入风险，仅生成只读查询（SELECT/WITH/SHOW）
                7. 检查表名和字段名是否正确，确保它们存在于schema中
                8. 考虑可能的数据类型转换问题
                9. 确保WHERE条件合理，不会过滤掉所有数据""",
        },
        {
            "role": "user",
            "content": f"""
                数据库Schema信息如下：
                {schema_info}
                请将以下问题转换为SQL查询语句：
                {query}
                上一次生成的SQL查询失败，详情如下：
                SQL语句: {failed_sql if failed_sql else "无"}
                错误信息: {error_info if error_info else "查询执行成功，但没有返回数据"}
                请分析上述错误，生成一个更准确的SQL查询。
                """,
        },
    ]


def build_answer_prompt(
    result: Any, query: str, sql: str, schema_info: str
) -> List[Dict[str, str]]:
    """构建查询结果解读与可视化配置生成的提示词。"""
    return [
        {
            "role": "system",
            "content": "你是一个SQL查询解读专家，负责将SQL查询结果转换为自然语言分析和可视化数据。你擅长数据分析并能够推荐最合适的可视化方式。",
        },
        {
            "role": "user",
            "content": f"""请帮我分析以下查询结果，提供专业的数据洞察：
                查询的问题：
                {query}
                查询用到的sql：
                {sql}
                数据库Schema信息如下：
                {schema_info}
                查询结果：
                {result}

                请提供【markdown格式】：
                1. 数据的专业分析和洞察（无需展示原始数据表格，因为数据已直接显示在界面上）
                2. 关键趋势、异常值或重要发现的解释

                另外，请分析数据特点，选择最合适的图表类型进行可视化，并提供以下JSON格式的可视化配置：【注意！！：1.json中不可含有任何注释，特别是value里面不能含有任何注释;2.若有多个图表，以json数组形式一并返回】

                ```json
                {{
                  "charts": [
                    {{
                      "type": "图表类型1",
                      "title": "图表标题1",
                      "description": "图表描述1",
                      "data": {{}}
                    }}
                  ]
                }}
                ```

                支持的图表类型和配置说明：

                1. 饼图 (pie)：适合展示占比关系
                ```json
                {{
                  "type": "pie",
                  "data": {{
                    "labels": ["类别1", "类别2", "类别3"],
                    "values": [30, 40, 50],
                    "hole": 0.4,
                    "colors": ["#ff0000", "#00ff00", "#0000ff"]
                  }}
                }}
                ```

                2. 柱状图 (bar)：适合分类数据比较
                ```json
                {{
                  "type": "bar",
                  "data": {{
                    "labels": ["类别1", "类别2", "类别3"],
                    "values": [30, 40, 50],
                    "orientation": "v",
                    "is_grouped": false,
                    "is_stacked": false,
                    "show_values": true,
                    "colors": ["#ff0000", "#00ff00", "#0000ff"]
                  }}
                }}
                ```

                3. 折线图 (line)：适合时间序列和趋势分析
                ```json
                {{
                  "type": "line",
                  "data": {{
                    "labels": ["2023-01", "2023-02", "2023-03"],
                    "values": [30, 40, 50],
                    "show_markers": true,
                    "line_shape": "linear",
                    "x_axis_title": "时间",
                    "y_axis_title": "销售额",
                    "series": [
                      {{
                        "name": "系列1",
                        "x": ["2023-01", "2023-02", "2023-03"],
                        "y": [30, 40, 50],
                        "line_width": 2
                      }}
                    ]
                  }}
                }}
                ```

                4. 散点图 (scatter)：适合探索两个变量的关系
                ```json
                {{
                  "type": "scatter",
                  "data": {{
                    "x": [10, 20, 30, 40],
                    "y": [5, 15, 25, 35],
                    "sizes": [10, 20, 30, 40],
                    "color_values": [1, 2, 3, 4],
                    "hover_names": ["点1", "点2", "点3", "点4"],
                    "opacity": 0.7
                  }}
                }}
                ```

                5. 面积图 (area)：展示趋势和累计关系
                ```json
                {{
                  "type": "area",
                  "data": {{
                    "labels": ["2023-01", "2023-02", "2023-03"],
                    "values": [30, 40, 50],
                    "is_stacked": true,
                    "colors": ["#ff0000", "#00ff00", "#0000ff"]
                  }}
                }}
                ```

                6. 热力图 (heatmap)：展示二维数据的强度变化
                ```json
                {{
                  "type": "heatmap",
                  "data": {{
                    "z_values": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    "x_labels": ["A", "B", "C"],
                    "y_labels": ["X", "Y", "Z"],
                    "color_scale": "Viridis",
                    "show_values": true
                  }}
                }}
                ```

                7. 雷达图 (radar)：多维度数据比较
                ```json
                {{
                  "type": "radar",
                  "data": {{
                    "labels": ["指标1", "指标2", "指标3", "指标4", "指标5"],
                    "values": [80, 65, 90, 75, 85],
                    "max_value": 100,
                    "series": [
                      {{
                        "name": "产品A",
                        "values": [80, 65, 90, 75, 85]
                      }}
                    ]
                  }}
                }}
                ```

                8. 漏斗图 (funnel)：展示流程转化率
                ```json
                {{
                  "type": "funnel",
                  "data": {{
                    "labels": ["访问", "注册", "下单", "支付", "复购"],
                    "values": [1000, 800, 600, 400, 200],
                    "text_info": "value+percent initial"
                  }}
                }}
                ```

                请根据数据特点选择最合适的图表类型，最多提供2个图表配置。确保JSON格式正确，可以被解析。如果数据不适合可视化，可以不提供图表配置。
                """,
        },
    ]
