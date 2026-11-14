from smartdbchat.semantic_layer import SemanticLayer


SAMPLE_YAML = """
database_description: 高校教务与校园消费分析数据库
user_rules:
  - 统计消费金额时仅计算 payment_status = '已支付' 的订单
  - 默认只统计在读学生（status = '在读'），除非用户明确要求包含已毕业学生
metrics:
  - name: 校园消费总额
    aliases: [总消费, GMV, 消费金额]
    expression: SUM(CASE WHEN payment_status = '已支付' THEN amount ELSE 0 END)
    description: 已支付校园订单的累计消费金额
    tables: [campus_orders]
glossary:
  高绩点学生: gpa >= 3.8
table_descriptions:
  students: 存储学生学籍、所属学院及 GPA 信息
column_descriptions:
  students:
    gpa: 平均学分绩点，范围 0.0 到 4.0
column_enums:
  campus_orders.payment_status:
    已支付: 有效完成支付的订单
    已退款: 已退款订单，不计入净消费
golden_examples:
  - question: 统计各学院学生的平均 GPA
    sql: SELECT d.name, AVG(s.gpa) AS avg_gpa FROM students s JOIN departments d ON s.department_id = d.id GROUP BY d.name
    tables: [students, departments]
    tags: [学院, GPA]
"""


def test_semantic_layer_yaml_roundtrip_and_retrieval():
    layer = SemanticLayer.from_yaml(SAMPLE_YAML)
    assert layer.database_description == "高校教务与校园消费分析数据库"
    assert len(layer.user_rules) == 2
    assert len(layer.metrics) == 1

    dumped = layer.to_yaml()
    reloaded = SemanticLayer.from_yaml(dumped)
    assert reloaded.metrics[0].name == "校园消费总额"

    ctx = reloaded.retrieve_context(
        question="统计高绩点学生的校园消费总额，按学院汇总",
        selected_tables=["students", "campus_orders"],
    )
    assert len(ctx.matched_metrics) == 1
    assert "高绩点学生" in ctx.matched_glossary
    assert "campus_orders.payment_status" in ctx.matched_enums
    assert "students" in ctx.table_descriptions
    assert len(ctx.golden_examples) == 1

    prompt_text = ctx.to_prompt_text()
    assert "全局业务规则" in prompt_text
    assert "校园消费总额" in prompt_text
    assert "Few-Shot" in prompt_text
