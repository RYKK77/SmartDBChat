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
