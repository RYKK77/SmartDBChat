from unittest.mock import MagicMock

from smartdbchat.scaling import TestTimeScaler, compute_result_signature


def test_result_signature_is_alias_and_order_independent():
    rows_a = [{"name": "CS", "cnt": 4}, {"name": "SE", "cnt": 3}]
    rows_b = [{"total": 3, "dept": "SE"}, {"total": 4, "dept": "CS"}]
    assert compute_result_signature(rows_a) == compute_result_signature(rows_b)


def test_test_time_scaler_majority_vote():
    llm = MagicMock()
    # 候选 1 与候选 3 返回相同数据（不同别名），候选 2 报错/返回空
    llm.chat_completion.side_effect = [
        "SELECT name, gpa FROM students WHERE gpa > 3.9",
        "SELECT unknown_col FROM students",
        "WITH s AS (SELECT name, gpa AS score FROM students WHERE gpa > 3.9) SELECT name, score FROM s",
    ]

    def fake_exec(sql: str):
        if "unknown_col" in sql:
            raise RuntimeError("Unknown column 'unknown_col'")
        return [{"name": "Alice Wang", "gpa": 3.95}, {"name": "李四", "gpa": 3.92}]

    scaler = TestTimeScaler(
        llm_client=llm,
        model="mock-model",
        execute_fn=fake_exec,
        dialect="mysql",
        allowed_tables=["students"],
        table_columns={"students": ["id", "name", "gpa"]},
    )
    res = scaler.run(
        base_messages=[{"role": "user", "content": "查询 GPA 大于 3.9 的学生"}],
        num_candidates=3,
    )
    assert res.total_valid == 2
    assert res.consensus_votes == 2
    assert len(res.rows) == 2
    assert "students" in res.selected_sql
