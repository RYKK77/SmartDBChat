import asyncio
import json
from unittest.mock import MagicMock

from smartdbchat.cache import SemanticQueryCache
from smartdbchat.config import CacheConfig, DatabaseConfig, LLMConfig
from smartdbchat.engine import SmartDBEngine
from smartdbchat.eval.sandbox import SQLiteEvalBackend
from smartdbchat.mcp_server import create_mcp_server
from smartdbchat.semantic_layer import SemanticLayer


def test_mcp_server_tools_and_execution(tmp_path):
    backend = SQLiteEvalBackend()
    llm = MagicMock()
    llm.config = LLMConfig(model="mock-model")
    llm.generate_sql.return_value = "SELECT name, gpa FROM students ORDER BY gpa DESC LIMIT 3"
    llm.stream_analysis.return_value = "排名前三的学生 GPA 均高于 3.85。"

    cache = SemanticQueryCache(CacheConfig(db_path=str(tmp_path / "mcp_cache.db")))
    sem = SemanticLayer(
        database_description="高校学生教务数据库",
        table_descriptions={"students": "学生学籍与 GPA 表"},
    )
    engine = SmartDBEngine(
        db=backend,
        llm=llm,
        cache=cache,
        semantic_layer=sem,
        dialect="sqlite",
    )
    server = create_mcp_server(engine)

    def _extract_text(res):
        if hasattr(res, "content") and res.content:
            return getattr(res.content[0], "text", str(res.content[0]))
        return json.dumps(res, ensure_ascii=False, default=str)

    async def _run():
        tools = await server.list_tools()
        tool_names = {t.name for t in tools}
        assert {
            "list_tables",
            "get_database_schema",
            "find_join_path",
            "sample_column_values",
            "validate_sql",
            "ask_database",
        }.issubset(tool_names)

        tbl_res = _extract_text(await server.call_tool("list_tables", {}))
        assert "students" in tbl_res

        join_res = _extract_text(
            await server.call_tool(
                "find_join_path", {"from_table": "departments", "to_table": "enrollments"}
            )
        )
        join_data = json.loads(join_res)
        assert len(join_data["path"]) == 3
        assert join_data["path"][0] == "departments"
        assert join_data["path"][-1] == "enrollments"
        assert len(join_data["join_conditions"]) == 2

        val_res = _extract_text(
            await server.call_tool(
                "sample_column_values",
                {"table_name": "departments", "column_name": "name", "keyword": "计算机"},
            )
        )
        assert "计算机科学与技术" in val_res

        guard_res = _extract_text(
            await server.call_tool("validate_sql", {"sql": "DROP TABLE students"})
        )
        assert '"is_valid": false' in guard_res.lower()

        ask_res = _extract_text(
            await server.call_tool(
                "ask_database", {"question": "查询 GPA 排名前 3 的学生", "mode": "linked"}
            )
        )
        assert "success" in ask_res

    asyncio.run(_run())
    backend.close()
