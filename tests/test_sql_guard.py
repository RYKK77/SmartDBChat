from smartdbchat.sql_guard import quote_identifier_if_needed, validate_sql_ast


def test_ast_guard_allows_select_cte_and_union():
    res = validate_sql_ast(
        "WITH top_s AS (SELECT id, name FROM students WHERE gpa > 3.5) "
        "SELECT id, name FROM top_s ORDER BY id DESC"
    )
    assert res.is_valid
    assert res.tables == ["students"]

    res_union = validate_sql_ast(
        "SELECT name FROM students UNION ALL SELECT name FROM instructors"
    )
    assert res_union.is_valid
    assert set(res_union.tables) == {"students", "instructors"}


def test_ast_guard_blocks_mutations_functions_and_multi_statements():
    for bad_sql in [
        "DROP TABLE students",
        "DELETE FROM students WHERE id = 1",
        "SELECT * FROM students; DROP TABLE students;",
        "SELECT * FROM students INTO OUTFILE '/tmp/x.csv'",
        "SELECT * FROM students FOR UPDATE",
        "SELECT SLEEP(5)",
    ]:
        res = validate_sql_ast(bad_sql)
        assert not res.is_valid, f"Expected blocked: {bad_sql}"


def test_ast_guard_schema_validation_tables_and_columns():
    allowed = ["students", "departments"]
    schema_cols = {
        "students": ["id", "name", "department_id", "gpa"],
        "departments": ["id", "name"],
    }

    ok = validate_sql_ast(
        "SELECT s.name, d.name AS dept_name, AVG(s.gpa) AS avg_gpa "
        "FROM students s JOIN departments d ON s.department_id = d.id "
        "GROUP BY s.name, d.name ORDER BY avg_gpa DESC",
        allowed_tables=allowed,
        table_columns=schema_cols,
    )
    assert ok.is_valid

    bad_tbl = validate_sql_ast(
        "SELECT * FROM secret_users",
        allowed_tables=allowed,
        table_columns=schema_cols,
    )
    assert not bad_tbl.is_valid
    assert "secret_users" in bad_tbl.unknown_tables

    bad_col = validate_sql_ast(
        "SELECT s.non_existent_col FROM students s",
        allowed_tables=allowed,
        table_columns=schema_cols,
    )
    assert not bad_col.is_valid
    assert "s.non_existent_col" in bad_col.unknown_columns


def test_ast_guard_limit_clamping_and_identifier_quoting():
    res_inject = validate_sql_ast("SELECT id, name FROM students", max_limit=100)
    assert res_inject.is_valid
    assert res_inject.limit_applied == 100
    assert "LIMIT 100" in res_inject.sql

    res_clamp = validate_sql_ast("SELECT id FROM students LIMIT 5000", max_limit=100)
    assert res_clamp.is_valid
    assert res_clamp.limit_applied == 100
    assert "LIMIT 100" in res_clamp.sql

    res_keep = validate_sql_ast("SELECT id FROM students LIMIT 10", max_limit=100)
    assert res_keep.is_valid
    assert res_keep.limit_applied is None
    assert "LIMIT 10" in res_keep.sql

    res_agg = validate_sql_ast("SELECT COUNT(*) AS cnt FROM students", max_limit=100)
    assert res_agg.is_valid
    assert res_agg.limit_applied is None

    assert quote_identifier_if_needed("order-items") == "`order-items`"
    assert quote_identifier_if_needed("students") == "students"
