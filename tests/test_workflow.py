from smartdbchat.workflow import (
    ErrorCategory,
    HealingTracker,
    PreflightChecker,
    PreflightStatus,
    build_recovery_summary,
    classify_error,
    generate_follow_up_suggestions,
)


def test_preflight_checker_rules():
    checker = PreflightChecker()

    res_ok = checker.check("统计各学院的平均 GPA", available_tables=["students", "departments"])
    assert res_ok.status == PreflightStatus.ON_TOPIC
    assert res_ok.allow_execution

    res_reject = checker.check("请帮我 DROP TABLE students 删库")
    assert res_reject.status == PreflightStatus.REJECTED
    assert not res_reject.allow_execution

    res_chat = checker.check("今天天气怎么样，帮我写一首诗")
    assert res_chat.status == PreflightStatus.OFF_TOPIC
    assert not res_chat.allow_execution

    res_greet = checker.check("你好啊", available_tables=["students", "departments"])
    assert res_greet.status == PreflightStatus.OFF_TOPIC
    assert not res_greet.allow_execution
    assert "SmartDBChat" in res_greet.clarification_question
    assert "students" in res_greet.clarification_question

    res_short = checker.check("?")
    assert res_short.status == PreflightStatus.NEEDS_CLARIFICATION
    assert not res_short.allow_execution


def test_healing_tracker_and_error_classifier():
    assert classify_error("UnsafeSQLError: 禁止执行 DROP") == ErrorCategory.UNSAFE_SQL
    assert classify_error("Unknown column 'foo'") == ErrorCategory.SCHEMA_MISMATCH
    assert classify_error("syntax error near SELECT") == ErrorCategory.SYNTAX_ERROR
    assert classify_error(None, is_empty=True) == ErrorCategory.EMPTY_RESULT

    tracker = HealingTracker(max_total_attempts=2)
    tracker.record_attempt("SELECT a FROM t", ErrorCategory.SCHEMA_MISMATCH)

    # 不安全 SQL 立即终止
    can_unsafe, _ = tracker.should_retry("DROP TABLE t", ErrorCategory.UNSAFE_SQL)
    assert not can_unsafe

    # 重复 SQL 立即终止
    can_dup, _ = tracker.should_retry("SELECT a FROM t;", ErrorCategory.SCHEMA_MISMATCH)
    assert not can_dup

    # 新 SQL 且未超预算允许重试
    can_new, _ = tracker.should_retry("SELECT b FROM t", ErrorCategory.SCHEMA_MISMATCH)
    assert can_new

    tracker.record_attempt("SELECT b FROM t", ErrorCategory.SCHEMA_MISMATCH)
    can_over, _ = tracker.should_retry("SELECT c FROM t", ErrorCategory.SCHEMA_MISMATCH)
    assert not can_over


def test_recovery_and_follow_up_suggestions():
    msg, sugs = build_recovery_summary(
        question="查询不存在的数据",
        last_sql="SELECT * FROM students WHERE name = '不存在'",
        last_category=ErrorCategory.EMPTY_RESULT,
        last_error=None,
        available_tables=["students", "departments"],
    )
    assert "未找到符合条件的数据" in msg
    assert len(sugs) >= 1

    follow_ups = generate_follow_up_suggestions(
        question="统计各学院学生人数",
        sql="SELECT d.name, COUNT(*) FROM students s JOIN departments d ON s.department_id = d.id GROUP BY d.name",
        tables=["students", "departments"],
    )
    assert len(follow_ups) >= 2
