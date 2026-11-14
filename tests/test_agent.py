import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from smartdbchat.agent import (
    AGENT_TOOLS,
    AgentConfig,
    DatabaseBackend,
    NL2SQLAgent,
    validate_chart_config,
)


def _make_tool_call(call_id: str, name: str, arguments: Dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def _make_completion_response(
    content: Optional[str] = "",
    tool_calls: Optional[List[Any]] = None,
) -> Any:
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeDatabase:
    def __init__(self) -> None:
        self.executed_sqls: List[str] = []
        self.explained_sqls: List[str] = []
        self.sampled_calls: List[tuple[str, int]] = []
        self.distinct_calls: List[tuple[str, str, int]] = []
        self.fail_first_execute = False

    def list_tables(self) -> List[str]:
        return ["orders", "customers"]

    def get_schema_info(self, tables: Optional[List[str]] = None) -> str:
        ddls = {
            "orders": (
                "CREATE TABLE orders (\n"
                "  id INT PRIMARY KEY,\n"
                "  customer_id INT,\n"
                "  status VARCHAR(32) COMMENT '订单状态',\n"
                "  amount DECIMAL(10,2)\n"
                ");"
            ),
            "customers": (
                "CREATE TABLE customers (\n"
                "  id INT PRIMARY KEY,\n"
                "  name VARCHAR(64),\n"
                "  region VARCHAR(32)\n"
                ");"
            ),
        }
        target = tables if tables is not None else list(ddls.keys())
        return "\n\n".join(ddls[t] for t in target if t in ddls)

    def sample_table_rows(self, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        self.sampled_calls.append((table_name, limit))
        if table_name == "orders":
            return [{"id": 1, "customer_id": 10, "status": "completed", "amount": 99.5}][:limit]
        return []

    def get_distinct_values(
        self, table_name: str, column_name: str, limit: int = 20
    ) -> List[Any]:
        self.distinct_calls.append((table_name, column_name, limit))
        if table_name == "orders" and column_name == "status":
            return ["completed", "pending", "cancelled"][:limit]
        return []

    def explain_sql(self, sql: str) -> Dict[str, Any]:
        self.explained_sqls.append(sql)
        return {"id": 1, "select_type": "SIMPLE", "table": "orders", "type": "ALL"}

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        self.executed_sqls.append(sql)
        if self.fail_first_execute and len(self.executed_sqls) == 1:
            raise RuntimeError("Unknown column 'order_status' in 'where clause'")
        return [{"total_amount": 199.0, "order_count": 2}]


class FakeLLMClient:
    def __init__(self, responses: List[Any]) -> None:
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No more fake LLM responses configured")
        return self._responses.pop(0)


def test_database_backend_protocol_and_agent_tools():
    db = FakeDatabase()
    assert isinstance(db, DatabaseBackend)
    assert not isinstance(object(), DatabaseBackend)

    tool_names = [t["function"]["name"] for t in AGENT_TOOLS]
    assert tool_names == [
        "search_schema",
        "get_table_ddl",
        "sample_table_rows",
        "get_distinct_values",
        "explain_sql",
        "execute_sql",
    ]


def test_agent_multi_step_exploration_self_healing_and_success():
    db = FakeDatabase()
    db.fail_first_execute = True

    llm = FakeLLMClient(
        [
            # Step 1: search_schema + get_table_ddl
            _make_completion_response(
                content="先检索订单相关表结构",
                tool_calls=[
                    _make_tool_call("tc1", "search_schema", {"keyword": "order"}),
                    _make_tool_call("tc2", "get_table_ddl", {"table_names": ["orders"]}),
                ],
            ),
            # Step 2: sample_table_rows (limit=10 should clamp to 5) + get_distinct_values
            _make_completion_response(
                content="查看样例数据与状态枚举值",
                tool_calls=[
                    _make_tool_call("tc3", "sample_table_rows", {"table_name": "orders", "limit": 10}),
                    _make_tool_call(
                        "tc4",
                        "get_distinct_values",
                        {"table_name": "orders", "column_name": "status"},
                    ),
                ],
            ),
            # Step 3: explain_sql + bad execute_sql (triggers error for self-healing)
            _make_completion_response(
                content="验证并执行查询",
                tool_calls=[
                    _make_tool_call(
                        "tc5",
                        "explain_sql",
                        {"sql": "SELECT SUM(amount) AS total_amount FROM orders WHERE status = 'completed'"},
                    ),
                    _make_tool_call(
                        "tc6",
                        "execute_sql",
                        {"sql": "SELECT SUM(amount) AS total_amount FROM orders WHERE order_status = 'completed'"},
                    ),
                ],
            ),
            # Step 4: self-heal execute_sql
            _make_completion_response(
                content="修复列名 order_status -> status 后重新执行",
                tool_calls=[
                    _make_tool_call(
                        "tc7",
                        "execute_sql",
                        {"sql": "SELECT SUM(amount) AS total_amount, COUNT(*) AS order_count FROM orders WHERE status = 'completed'"},
                    ),
                ],
            ),
            # Step 5: final answer
            _make_completion_response(content="已完成查询，已完成订单总金额为 199.0。"),
        ]
    )

    statuses: List[tuple[str, str]] = []
    agent = NL2SQLAgent(db=db, llm_client=llm, model="gpt-4o-mini")
    result = agent.run(
        query="统计已完成订单的总金额",
        on_status=lambda stage, msg: statuses.append((stage, msg)),
    )

    assert result.success is True
    assert result.sql == "SELECT SUM(amount) AS total_amount, COUNT(*) AS order_count FROM orders WHERE status = 'completed'"
    assert result.rows == [{"total_amount": 199.0, "order_count": 2}]
    assert result.final_message == "已完成查询，已完成订单总金额为 199.0。"
    assert result.error is None

    # sample_table_rows clamped limit=10 to 5
    assert db.sampled_calls == [("orders", 5)]
    assert db.distinct_calls == [("orders", "status", 20)]
    assert len(db.explained_sqls) == 1
    assert len(db.executed_sqls) == 2

    # Check step error recorded on tc6 and cleared on final result
    failed_steps = [s for s in result.steps if s.error is not None]
    assert len(failed_steps) == 1
    assert "Unknown column" in (failed_steps[0].error or "")
    assert len(statuses) > 0


def test_agent_safety_blocks_unsafe_sql_and_allows_recovery():
    db = FakeDatabase()
    llm = FakeLLMClient(
        [
            # Unsafe explain_sql and execute_sql
            _make_completion_response(
                content="尝试执行危险语句",
                tool_calls=[
                    _make_tool_call("tc1", "explain_sql", {"sql": "DELETE FROM orders"}),
                    _make_tool_call("tc2", "execute_sql", {"sql": "DROP TABLE orders"}),
                ],
            ),
            # Self-heal to safe SELECT
            _make_completion_response(
                content="改为只读查询",
                tool_calls=[
                    _make_tool_call("tc3", "execute_sql", {"sql": "SELECT COUNT(*) AS cnt FROM orders"}),
                ],
            ),
            _make_completion_response(content="订单表查询完成。"),
        ]
    )

    agent = NL2SQLAgent(db=db, llm_client=llm, model="gpt-4o-mini")
    result = agent.run("查看订单数量")

    assert result.success is True
    assert result.sql == "SELECT COUNT(*) AS cnt FROM orders"
    # Unsafe queries never reached db.explain_sql or db.execute_query
    assert db.explained_sqls == []
    assert db.executed_sqls == ["SELECT COUNT(*) AS cnt FROM orders"]
    assert "UnsafeSQLError" in (result.steps[0].error or "")
    assert "UnsafeSQLError" in (result.steps[1].error or "")


def test_agent_plain_sql_fallback_and_unsafe_block():
    db = FakeDatabase()
    llm_safe = FakeLLMClient(
        [_make_completion_response(content="```sql\nSELECT id, status FROM orders\n```")]
    )
    agent = NL2SQLAgent(db=db, llm_client=llm_safe, model="gpt-4o-mini")
    res_ok = agent.run("查询订单")
    assert res_ok.success is True
    assert res_ok.sql == "SELECT id, status FROM orders"
    assert db.executed_sqls == ["SELECT id, status FROM orders"]

    db2 = FakeDatabase()
    llm_unsafe = FakeLLMClient(
        [_make_completion_response(content="```sql\nDROP TABLE orders;\n```")]
    )
    agent2 = NL2SQLAgent(db=db2, llm_client=llm_unsafe, model="gpt-4o-mini")
    res_bad = agent2.run("删除订单表")
    assert res_bad.success is False
    assert "UnsafeSQLError" in (res_bad.error or "")
    assert db2.executed_sqls == []


def test_agent_history_max_sql_executions_and_disable_explain():
    db = FakeDatabase()
    db.fail_first_execute = True

    llm = FakeLLMClient(
        [
            _make_completion_response(
                content="第一次执行",
                tool_calls=[
                    _make_tool_call("tc1", "execute_sql", {"sql": "SELECT bad FROM orders"}),
                ],
            ),
            _make_completion_response(
                content="第二次执行（应被 max_sql_executions=1 拦截）",
                tool_calls=[
                    _make_tool_call("tc2", "execute_sql", {"sql": "SELECT id FROM orders"}),
                ],
            ),
        ]
    )

    config = AgentConfig(max_steps=2, max_sql_executions=1, enable_explain_check=False)
    agent = NL2SQLAgent(db=db, llm_client=llm, model="gpt-4o-mini", config=config)

    history = [
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答"},
    ]
    result = agent.run("继续查询订单", schema_context="CREATE TABLE orders (id INT);", history=history)

    assert result.success is False
    assert len(db.executed_sqls) == 1
    assert "最大 SQL 执行次数限制" in (result.error or "")

    # Verify history was included in messages sent to LLM
    sent_messages = llm.calls[0]["messages"]
    assert sent_messages[1] == {"role": "user", "content": "上一轮问题"}
    assert sent_messages[2] == {"role": "assistant", "content": "上一轮回答"}
    assert sent_messages[3] == {"role": "user", "content": "继续查询订单"}

    # Verify explain_sql was excluded when enable_explain_check=False
    sent_tool_names = [t["function"]["name"] for t in llm.calls[0]["tools"]]
    assert "explain_sql" not in sent_tool_names


def test_validate_chart_config():
    assert validate_chart_config({"charts": []}) == (True, None)
    assert validate_chart_config(
        {"charts": [{"type": "pie", "title": "T", "data": {"labels": ["A"], "values": [1]}}]}
    ) == (True, None)
    assert validate_chart_config(
        {"charts": [{"type": "line", "data": {"series": [{"name": "s1", "y": [1, 2]}]}}]}
    ) == (True, None)
    assert validate_chart_config(
        {"charts": [{"type": "scatter", "data": {"x": [1, 2], "y": [3, 4]}}]}
    ) == (True, None)
    assert validate_chart_config(
        {"charts": [{"type": "heatmap", "data": {"z_values": [[1, 2], [3, 4]]}}]}
    ) == (True, None)
    assert validate_chart_config(
        {"charts": [{"type": "radar", "data": {"labels": ["A", "B"], "values": [10, 20]}}]}
    ) == (True, None)
    assert validate_chart_config(
        {"charts": [{"type": "treemap", "data": {"labels": ["A"], "values": [1]}}]}
    ) == (True, None)

    ok_none, err_none = validate_chart_config(None)
    assert ok_none is False and err_none is not None

    ok_type, err_type = validate_chart_config({"charts": [{"type": "unknown", "data": {}}]})
    assert ok_type is False and "不支持" in (err_type or "")

    ok_data, err_data = validate_chart_config({"charts": [{"type": "bar", "data": {"labels": ["A"]}}]})
    assert ok_data is False and "values" in (err_data or "")
