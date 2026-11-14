from smartdbchat.llm import clean_sql_response


def test_clean_sql_response_strips_markdown_fences():
    raw = """```sql
    SELECT name, avg_score
    FROM students;
    ```"""
    assert clean_sql_response(raw) == "SELECT name, avg_score\nFROM students;"


def test_clean_sql_response_handles_plain_and_inline():
    assert clean_sql_response("`SELECT 1`") == "SELECT 1"
    assert clean_sql_response("  SELECT 1\n\n  FROM dual  ") == "SELECT 1\nFROM dual"
