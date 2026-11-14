from unittest.mock import MagicMock

from smartdbchat.cache import SemanticQueryCache
from smartdbchat.config import CacheConfig, LLMConfig
from smartdbchat.engine import SmartDBEngine
from smartdbchat.eval.sandbox import SQLiteEvalBackend
from smartdbchat.models import QueryStatus
from smartdbchat.schema_linker import build_schema_graph
from smartdbchat.semantic_layer import SemanticLayer


def test_schema_graph_join_edges_and_summary():
    backend = SQLiteEvalBackend()
    ddl = backend.get_schema_info()
    blocks = [b.strip() for b in ddl.split("\n\n") if b.strip()]
    graph = build_schema_graph(blocks)

    edges = graph.find_join_edges("departments", "enrollments")
    assert len(edges) == 2

    summary = graph.to_graph_summary(["students", "departments"])
    assert "[Schema Graph]" in summary
    assert "Edge[fk]: students.department_id -> departments.id" in summary
    backend.close()


def test_engine_end_to_end_preflight_semantic_and_scaling(tmp_path):
    backend = SQLiteEvalBackend()
    llm = MagicMock()
    llm.config = LLMConfig(model="mock-model")

    sem = SemanticLayer(
        database_description="高校教务系统",
        user_rules=["仅统计在读学生"],
    )
    cache = SemanticQueryCache(CacheConfig(db_path=str(tmp_path / "int_cache.db")))
    engine = SmartDBEngine(
        db=backend,
        llm=llm,
        cache=cache,
        semantic_layer=sem,
        dialect="sqlite",
        enable_preflight=True,
        enable_ast_schema_check=True,
    )

    # 1. Preflight 拦截破坏性请求
    rej = engine.run_query("请帮我 DROP TABLE students 删库")
    assert rej.status == QueryStatus.REJECTED
    assert rej.preflight_status == "rejected"
    assert len(rej.follow_up_suggestions) > 0

    # 2. High-Accuracy Test-Time Scaling 模式端到端验证
    llm.chat_completion.side_effect = [
        "SELECT name, gpa FROM students WHERE status = '在读' ORDER BY gpa DESC LIMIT 3",
        "SELECT name, gpa FROM students WHERE status = '在读' ORDER BY gpa DESC LIMIT 3",
        "SELECT name, gpa FROM students WHERE status = '在读' ORDER BY gpa DESC LIMIT 3",
    ]
    llm.stream_analysis.return_value = "在读学生中 GPA 最高为 Alice Wang (3.95)。"

    res = engine.run_query(
        "查询在读学生中 GPA 最高的 3 名学生",
        mode="linked",
        high_accuracy_mode=True,
        num_candidates=3,
    )
    assert res.status == QueryStatus.SUCCESS
    assert res.scaling_info is not None
    assert res.scaling_info["consensus_votes"] == 3
    assert len(res.follow_up_suggestions) > 0
    assert any(t["stage"] == "semantic_layer" for t in res.workflow_trace)
    backend.close()
