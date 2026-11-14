from unittest.mock import MagicMock

from smartdbchat.cache import SemanticQueryCache
from smartdbchat.config import CacheConfig, DatabaseConfig
from smartdbchat.db import UnsafeSQLError
from smartdbchat.engine import SmartDBEngine
from smartdbchat.models import QueryStatus


def _make_cache(tmp_path):
    cache = SemanticQueryCache(config=CacheConfig(db_path=str(tmp_path / "c.db")))
    cache.embedding_available = False
    return cache


def test_engine_cache_miss_then_hit(tmp_path):
    db = MagicMock()
    db.config = DatabaseConfig(database="school")
    db.get_schema_info.return_value = "CREATE TABLE students (id INT, name VARCHAR(50));"
    db.get_specific_schema_info.return_value = "CREATE TABLE students (id INT, name VARCHAR(50));"
    db.execute_query.return_value = [{"id": 1, "name": "Alice"}]

    llm = MagicMock()
    llm.generate_sql.return_value = "SELECT id, name FROM students"
    llm.stream_analysis.return_value = "共查询到 1 名学生。\n```json\n{\"charts\": []}\n```"

    cache = _make_cache(tmp_path)
    engine = SmartDBEngine(db=db, llm=llm, cache=cache)

    res1 = engine.run_query("查询学生")
    assert res1.status == QueryStatus.SUCCESS
    assert res1.sql == "SELECT id, name FROM students"
    assert len(res1.rows) == 1
    assert llm.generate_sql.call_count == 1

    # 第二次命中缓存，不应再调用 generate_sql
    res2 = engine.run_query("查询学生")
    assert res2.status == QueryStatus.SUCCESS
    assert res2.cache_hit is not None
    assert llm.generate_sql.call_count == 1


def test_engine_retries_on_error_and_succeeds(tmp_path):
    db = MagicMock()
    db.config = DatabaseConfig(database="school")
    db.get_schema_info.return_value = "CREATE TABLE students (id INT, name VARCHAR(50));"
    db.get_specific_schema_info.return_value = "CREATE TABLE students (id INT, name VARCHAR(50));"
    db.execute_query.side_effect = [
        RuntimeError("Unknown column 'bad_col'"),
        [{"id": 1, "name": "Bob"}],
    ]

    llm = MagicMock()
    llm.generate_sql.side_effect = [
        "SELECT bad_col FROM students",
        "SELECT id, name FROM students",
    ]
    llm.stream_analysis.return_value = "查询成功。"

    cache = _make_cache(tmp_path)
    engine = SmartDBEngine(db=db, llm=llm, cache=cache)

    res = engine.run_query("查询学生", max_retries=2)
    assert res.status == QueryStatus.SUCCESS
    assert res.sql == "SELECT id, name FROM students"
    assert len(res.attempts) == 2
    assert res.attempts[0].error == "Unknown column 'bad_col'"


def test_engine_blocks_unsafe_sql_and_returns_invalid_status(tmp_path):
    db = MagicMock()
    db.config = DatabaseConfig(database="school")
    db.get_schema_info.return_value = "CREATE TABLE students (id INT);"
    db.execute_query.side_effect = UnsafeSQLError("禁止执行 DROP 语句")

    llm = MagicMock()
    llm.generate_sql.return_value = "DROP TABLE students"

    cache = _make_cache(tmp_path)
    engine = SmartDBEngine(db=db, llm=llm, cache=cache)

    res = engine.run_query("删除学生表", max_retries=0)
    assert res.status == QueryStatus.INVALID_SQL
    assert not res.is_success
    assert "DROP" in (res.error or "")


def test_database_manager_implements_database_backend():
    from smartdbchat.agent import DatabaseBackend
    from smartdbchat.db import DatabaseManager

    mgr = DatabaseManager(DatabaseConfig(database="school"))
    assert isinstance(mgr, DatabaseBackend)


def test_engine_linked_mode_with_sqlite_sandbox(tmp_path):
    from smartdbchat.eval.sandbox import SQLiteEvalBackend

    backend = SQLiteEvalBackend()
    llm = MagicMock()
    llm.generate_sql.return_value = (
        "SELECT d.name, COUNT(s.id) AS cnt "
        "FROM departments d JOIN students s ON d.id = s.department_id "
        "WHERE s.status = '在读' GROUP BY d.id, d.name"
    )
    llm.stream_analysis.return_value = "统计完成。"

    cache = _make_cache(tmp_path)
    engine = SmartDBEngine(db=backend, llm=llm, cache=cache, default_mode="linked")
    res = engine.run_query("统计各学院在读学生人数")

    assert res.status == QueryStatus.SUCCESS
    assert res.mode == "linked"
    assert "students" in res.linked_tables
    assert "departments" in res.linked_tables
    assert len(res.rows) > 0


def test_engine_agent_mode_and_multiturn_bypasses_cache(tmp_path):
    import json
    from types import SimpleNamespace
    from smartdbchat.eval.sandbox import SQLiteEvalBackend

    backend = SQLiteEvalBackend()

    # 构造支持 tool_calls 的 fake OpenAI client
    tc1 = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="get_distinct_values",
            arguments=json.dumps({"table_name": "courses", "column_name": "course_type"}),
        ),
    )
    tc2 = SimpleNamespace(
        id="call_2",
        function=SimpleNamespace(
            name="execute_sql",
            arguments=json.dumps(
                {"sql": "SELECT title, credits FROM courses WHERE course_type = '必修'"}
            ),
        ),
    )
    responses = [
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="查看课程类型枚举", tool_calls=[tc1]))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="执行查询", tool_calls=[tc2]))]
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="已完成查询", tool_calls=None))]
        ),
    ]

    fake_openai = MagicMock()
    fake_openai.chat.completions.create.side_effect = responses

    llm = MagicMock()
    llm.client = fake_openai
    llm.config = SimpleNamespace(model="fake-model")
    # 返回非法图表类型，验证 validate_chart_config 会安全过滤
    llm.stream_analysis.return_value = (
        '共找到多门必修课。\n```json\n{"charts": [{"type": "unknown_type", "data": {}}]}\n```'
    )

    cache = _make_cache(tmp_path)
    engine = SmartDBEngine(db=backend, llm=llm, cache=cache, default_mode="agent")

    res = engine.run_query(
        "查询所有必修课程",
        history=[{"role": "user", "content": "先看看课程表"}],
    )
    assert res.status == QueryStatus.SUCCESS
    assert res.mode == "agent"
    assert len(res.agent_steps) >= 2
    assert res.chart_config is None
    # 多轮对话不应写入单句缓存
    assert cache.get_cache_stats().total_entries == 0

