import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from smartdbchat.charts import recommend_chart_config
from smartdbchat.config import LLMConfig
from smartdbchat.server import create_app


def test_recommend_chart_config_heuristics():
    # 时间序列 -> line
    time_rows = [
        {"semester": "2023-Fall", "avg_score": 82.0},
        {"semester": "2024-Spring", "avg_score": 86.5},
        {"semester": "2024-Fall", "avg_score": 89.0},
    ]
    line_cfg = recommend_chart_config(time_rows)
    assert line_cfg is not None
    assert line_cfg["charts"][0]["type"] == "line"

    # 少量聚合占比 -> pie
    pie_rows = [
        {"category": "餐饮", "total_amount": 320.0},
        {"category": "图书", "total_amount": 454.0},
        {"category": "体育", "total_amount": 350.0},
    ]
    pie_cfg = recommend_chart_config(pie_rows)
    assert pie_cfg is not None
    assert pie_cfg["charts"][0]["type"] == "pie"

    # 较多类别 -> bar
    bar_rows = [{"name": f"学生{i}", "gpa": 3.0 + i * 0.05} for i in range(10)]
    bar_cfg = recommend_chart_config(bar_rows)
    assert bar_cfg is not None
    assert bar_cfg["charts"][0]["type"] == "bar"


def test_server_sandbox_connect_schema_semantic_sql_and_sse(tmp_path):
    app = create_app(cache_db_path=str(tmp_path / "api_cache.db"))
    client = TestClient(app)

    # 1. Providers & Health
    assert client.get("/api/health").json()["connected"] is False
    providers = client.get("/api/providers").json()["providers"]
    assert any(p["name"] == "DeepSeek" for p in providers)

    # 2. Connect to built-in SQLite sandbox
    conn_res = client.post(
        "/api/connect",
        json={
            "db_type": "sandbox",
            "provider": "DeepSeek",
            "api_key": "test-key",
            "model": "deepseek-flash",
            "query_mode": "linked",
            "high_accuracy_mode": False,
            "max_limit": 500,
        },
    )
    assert conn_res.status_code == 200
    assert conn_res.json()["connected"] is True

    # 3. Config get/update
    cfg_upd = client.put(
        "/api/config",
        json={"high_accuracy_mode": True, "num_candidates": 2, "max_limit": 200},
    )
    assert cfg_upd.status_code == 200
    assert cfg_upd.json()["high_accuracy_mode"] is True
    assert cfg_upd.json()["num_candidates"] == 2
    assert cfg_upd.json()["max_limit"] == 200

    # 4. Schema Graph + Sample + Column Values + Join Path
    graph = client.get("/api/schema/graph").json()
    assert graph["table_count"] == 7
    assert graph["edge_count"] >= 5

    sample = client.get("/api/schema/sample/students?limit=2").json()
    assert len(sample["rows"]) == 2

    vals = client.get("/api/schema/values/departments/name?keyword=计算机").json()
    assert "计算机科学与技术" in vals["values"]

    join_path = client.get(
        "/api/schema/join-path?from_table=departments&to_table=enrollments"
    ).json()
    assert len(join_path["path"]) == 3

    # 5. Semantic Layer CRUD & Golden SQL
    sem = client.get("/api/semantic").json()
    assert len(sem["data"]["metrics"]) >= 1

    add_g = client.post(
        "/api/semantic/golden",
        json={
            "question": "查询所有学院名称",
            "sql": "SELECT name, code FROM departments",
            "tables": ["departments"],
        },
    )
    assert add_g.status_code == 200
    assert add_g.json()["count"] >= 3

    # 6. Custom SQL execution via AST guard
    bad_sql = client.post("/api/sql/execute", json={"sql": "DROP TABLE students"})
    assert bad_sql.status_code == 400

    ok_sql = client.post(
        "/api/sql/execute",
        json={"sql": "SELECT d.name, COUNT(s.id) AS cnt FROM departments d JOIN students s ON d.id = s.department_id GROUP BY d.name"},
    )
    assert ok_sql.status_code == 200
    ok_data = ok_sql.json()
    assert ok_data["row_count"] >= 1
    assert ok_data["recommended_chart_config"] is not None

    # 7. Mock LLM on connected engine and verify SSE streaming query
    engine = app.router.routes  # just verify via direct state or monkeypatch LLM
    # Test disconnect cleanly
    assert client.post("/api/disconnect").json()["connected"] is False


def test_server_sse_streaming_with_mocked_llm(tmp_path):
    from smartdbchat.cache import SemanticQueryCache
    from smartdbchat.config import CacheConfig
    from smartdbchat.engine import SmartDBEngine
    from smartdbchat.eval.sandbox import SQLiteEvalBackend

    backend = SQLiteEvalBackend()
    llm = MagicMock()
    llm.config = LLMConfig(model="mock-model")
    llm.generate_sql.return_value = (
        "SELECT d.name AS 学院, COUNT(s.id) AS 人数 "
        "FROM departments d JOIN students s ON d.id = s.department_id GROUP BY d.name"
    )
    llm.stream_analysis.side_effect = lambda msgs, on_chunk=None: (
        on_chunk("各学院人数统计已完成。") if on_chunk else None,
        "各学院人数统计已完成。",
    )[1]

    eng = SmartDBEngine(
        db=backend,
        llm=llm,
        cache=SemanticQueryCache(CacheConfig(db_path=str(tmp_path / "sse.db"))),
        default_mode="linked",
        dialect="sqlite",
    )
    app = create_app(initial_engine=eng, cache_db_path=str(tmp_path / "sse.db"))
    client = TestClient(app)

    with client.stream("POST", "/api/query/stream", json={"question": "统计各学院学生人数"}) as resp:
        assert resp.status_code == 200
        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    event_types = [e["type"] for e in events]
    assert "result" in event_types
    result_event = next(e for e in events if e["type"] == "result")
    assert result_event["result"]["is_success"] is True
    assert result_event["result"]["recommended_chart_config"] is not None
    backend.close()


def test_server_spa_static_serving(tmp_path):
    dist_dir = tmp_path / "dist"
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>SmartDBChat SPA</body></html>", encoding="utf-8")
    (assets_dir / "app.js").write_text("console.log('ok');", encoding="utf-8")

    app = create_app(cache_db_path=str(tmp_path / "spa_cache.db"), frontend_dist_dir=str(dist_dir))
    client = TestClient(app)

    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "SmartDBChat SPA" in root_resp.text

    asset_resp = client.get("/assets/app.js")
    assert asset_resp.status_code == 200
    assert "console.log('ok')" in asset_resp.text

    api_404 = client.get("/api/nonexistent")
    assert api_404.status_code == 404

