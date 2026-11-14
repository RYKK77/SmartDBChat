import sqlite3

from smartdbchat.db import (
    extract_tables_from_sql,
    get_all_modules,
    get_tables_by_module,
    validate_read_only_sql,
)


def test_validate_read_only_sql_allows_select_and_with():
    ok, err = validate_read_only_sql("SELECT * FROM students WHERE id = 1;")
    assert ok is True
    assert err is None

    ok, err = validate_read_only_sql("WITH cte AS (SELECT 1) SELECT * FROM cte")
    assert ok is True
    assert err is None

    ok, err = validate_read_only_sql("SHOW TABLES")
    assert ok is True
    assert err is None


def test_validate_read_only_sql_blocks_mutations_and_multi_statements():
    for bad_sql in [
        "DROP TABLE students",
        "DELETE FROM students WHERE id = 1",
        "UPDATE students SET name = 'x'",
        "INSERT INTO students VALUES (1, 'a')",
        "SELECT 1; DROP TABLE students;",
        "SELECT * FROM students INTO OUTFILE '/tmp/out.csv'",
        "SELECT * FROM students FOR UPDATE",
        "   ",
    ]:
        ok, err = validate_read_only_sql(bad_sql)
        assert ok is False
        assert err is not None


def test_normalize_rows_converts_decimal_and_datetime():
    from datetime import date, datetime
    from decimal import Decimal
    from smartdbchat.db import _normalize_rows

    rows = [
        {
            "avg_score": Decimal("91.50"),
            "created": datetime(2026, 11, 13, 12, 0, 0),
            "day": date(2026, 11, 13),
            "name": "Alice",
        }
    ]
    normalized = _normalize_rows(rows)
    assert normalized == [
        {
            "avg_score": 91.5,
            "created": "2026-11-13T12:00:00",
            "day": "2026-11-13",
            "name": "Alice",
        }
    ]


def test_extract_tables_from_sql():
    sql = "SELECT s.name, c.title FROM `students` s JOIN enrollments e ON s.id = e.student_id JOIN courses c ON e.course_id = c.id"
    assert extract_tables_from_sql(sql) == ["students", "enrollments", "courses"]


def test_module_lookup_handles_missing_and_present_db(tmp_path):
    missing = str(tmp_path / "missing.db")
    assert get_all_modules(missing) == []
    assert get_tables_by_module("教务", missing) == []

    db_path = str(tmp_path / "sys_tables.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE system_tables (table_name TEXT, module TEXT, description TEXT)")
    conn.execute("INSERT INTO system_tables VALUES ('students', '教务', '学生表')")
    conn.execute("INSERT INTO system_tables VALUES ('courses', '教务', '课程表')")
    conn.execute("INSERT INTO system_tables VALUES ('orders', '财务', '订单表')")
    conn.commit()
    conn.close()

    assert get_all_modules(db_path) == ["教务", "财务"]
    assert get_tables_by_module("教务", db_path) == ["students", "courses"]
