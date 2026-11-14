from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from smartdbchat.agent import AgentConfig
from smartdbchat.cache import SemanticQueryCache
from smartdbchat.charts import recommend_chart_config, strip_chart_json_block
from smartdbchat.config import (
    LLM_PROVIDERS,
    CacheConfig,
    DatabaseConfig,
    EmbeddingConfig,
    LLMConfig,
)
from smartdbchat.db import DatabaseManager, UnsafeSQLError, get_all_modules
from smartdbchat.engine import SmartDBEngine, _split_ddl_statements
from smartdbchat.eval.sandbox import SQLiteEvalBackend
from smartdbchat.llm import LLMClient
from smartdbchat.models import QueryResult, QueryStatus
from smartdbchat.schema_linker import build_schema_graph
from smartdbchat.semantic_layer import SemanticLayer
from smartdbchat.sql_guard import validate_sql_ast
from smartdbchat.store import AppStore


def _default_sandbox_semantic_layer() -> SemanticLayer:
    """为内置高校教务沙箱提供开箱即用的语义层示例配置。"""
    return SemanticLayer.from_dict(
        {
            "database_description": "高校教务与校园消费分析数据库（内置演示沙箱），包含学院、学生、教师、课程、选课成绩、奖学金与校园消费流水。",
            "user_rules": [
                "统计校园消费净金额时，仅统计 payment_status = '已支付' 的订单，排除 '已退款'。",
                "计算平均 GPA 或统计在读人数时，如用户未指定全校范围，优先标注学籍状态（在读/已毕业/休学）。",
            ],
            "metrics": [
                {
                    "name": "校园净消费总额",
                    "aliases": ["总消费", "消费总额", "GMV", "净消费"],
                    "expression": "SUM(CASE WHEN payment_status = '已支付' THEN amount ELSE 0 END)",
                    "description": "剔除退款后的校园消费累计金额（元）",
                    "tables": ["campus_orders"],
                },
                {
                    "name": "平均学分绩点",
                    "aliases": ["平均GPA", "GPA均值", "平均绩点"],
                    "expression": "ROUND(AVG(gpa), 2)",
                    "description": "学生平均 GPA（满分 4.0）",
                    "tables": ["students"],
                },
                {
                    "name": "课程通过率",
                    "aliases": ["及格率", "考试通过率"],
                    "expression": "ROUND(SUM(CASE WHEN score >= 60 THEN 1.0 ELSE 0 END) * 100.0 / COUNT(*), 2)",
                    "description": "选课成绩 >= 60 分的占比百分比",
                    "tables": ["enrollments"],
                },
            ],
            "glossary": {
                "高绩点学生": "gpa >= 3.8 的学生",
                "挂科/重修": "enrollments.score < 60 或 enrollments.status IN ('补考', '重修')",
                "计算机学院": "departments.name = '计算机科学与技术' 或 code = 'CS'",
            },
            "table_descriptions": {
                "departments": "学院基础信息表（含学院名称、编码、院长）",
                "students": "学生学籍与 GPA 表（含学号、性别、入学年份、状态、所属学院）",
                "instructors": "授课教师表（含姓名、职称、所属学院）",
                "courses": "课程目录表（含课程代码、学分、必修/选修、开课学院、授课教师）",
                "enrollments": "学生选课与成绩记录表（含学期、分数、修读状态）",
                "scholarships": "学生奖学金获奖记录表",
                "campus_orders": "校园消费订单流水表（含消费类别、金额、支付状态）",
            },
            "column_enums": {
                "students.status": {
                    "在读": "当前正常在读学生",
                    "已毕业": "已毕业校友",
                    "休学": "休学中",
                },
                "courses.course_type": {
                    "必修": "培养方案必修课",
                    "选修": "公共或专业选修课",
                },
                "campus_orders.payment_status": {
                    "已支付": "有效订单，计入消费金额",
                    "已退款": "已退单，不计入净消费",
                },
            },
            "golden_examples": [
                {
                    "question": "统计各学院在读学生人数与平均 GPA",
                    "sql": (
                        "SELECT d.name AS 学院名称, COUNT(s.id) AS 在读人数, ROUND(AVG(s.gpa), 2) AS 平均GPA "
                        "FROM departments d JOIN students s ON d.id = s.department_id "
                        "WHERE s.status = '在读' GROUP BY d.id, d.name ORDER BY 平均GPA DESC"
                    ),
                    "tables": ["departments", "students"],
                    "tags": ["学院", "GPA", "在读"],
                },
                {
                    "question": "统计各类别校园已支付消费总额",
                    "sql": (
                        "SELECT category AS 消费类别, COUNT(*) AS 订单数, ROUND(SUM(amount), 2) AS 消费总额 "
                        "FROM campus_orders WHERE payment_status = '已支付' "
                        "GROUP BY category ORDER BY 消费总额 DESC"
                    ),
                    "tables": ["campus_orders"],
                    "tags": ["消费", "类别"],
                },
            ],
        }
    )


class ConnectRequest(BaseModel):
    db_type: str = Field(default="sandbox", description="sandbox | mysql")
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = ""

    provider: str = "DeepSeek"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-flash"

    embedding_mode: str = "openai_compatible"
    embedding_url: str = "http://localhost:11434/v1"
    embedding_model: str = "nomic-embed-text"
    similarity_threshold: float = 0.85

    query_mode: str = "agent"
    agent_max_steps: int = 8
    max_retries: int = 2
    high_accuracy_mode: bool = False
    num_candidates: int = 3
    enable_preflight: bool = True
    enable_ast_schema_check: bool = True
    enable_explain_check: bool = True
    max_limit: int = 1000
    semantic_yaml_path: str = ""


class ConfigUpdateRequest(BaseModel):
    query_mode: Optional[str] = None
    agent_max_steps: Optional[int] = None
    max_retries: Optional[int] = None
    high_accuracy_mode: Optional[bool] = None
    num_candidates: Optional[int] = None
    enable_preflight: Optional[bool] = None
    enable_ast_schema_check: Optional[bool] = None
    enable_explain_check: Optional[bool] = None
    max_limit: Optional[int] = None
    similarity_threshold: Optional[float] = None
    selected_module: Optional[str] = None
    model: Optional[str] = None


class QueryRequest(BaseModel):
    question: str
    mode: Optional[str] = None
    module: Optional[str] = None
    max_retries: Optional[int] = None
    high_accuracy_mode: Optional[bool] = None
    num_candidates: Optional[int] = None
    history: Optional[List[Dict[str, str]]] = None


class ExecuteSQLRequest(BaseModel):
    sql: str


class SemanticUpdateRequest(BaseModel):
    yaml_text: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class GoldenExampleRequest(BaseModel):
    question: str
    sql: str
    tables: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class AuthCredentialsRequest(BaseModel):
    username: str
    password: str


class SaveConnectionProfileRequest(BaseModel):
    id: Optional[str] = None
    name: str = ""
    payload: Dict[str, Any]


class SaveChatSessionRequest(BaseModel):
    title: str = "新对话"
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class UpdatePreferencesRequest(BaseModel):
    last_connection: Optional[Dict[str, Any]] = None
    update_last_connection: bool = False
    auto_connect_disabled: Optional[bool] = None
    active_session_id: Optional[str] = None
    update_active_session: bool = False


class AppRuntimeState:
    """线程安全的服务端运行时状态管理。"""

    def __init__(
        self,
        cache_db_path: str = "semantic_cache.db",
        app_db_path: Optional[str] = None,
    ) -> None:
        self.lock = threading.Lock()
        self.engine: Optional[SmartDBEngine] = None
        self.db_type: str = "sandbox"
        self.connection_meta: Dict[str, Any] = {}
        self.cache = SemanticQueryCache(CacheConfig(db_path=cache_db_path), EmbeddingConfig())
        resolved_app_db = app_db_path or (
            "smartdb_app.db"
            if cache_db_path == "semantic_cache.db"
            else f"{os.path.splitext(cache_db_path)[0]}_app.db"
        )
        self.store = AppStore(db_path=resolved_app_db)
        self.semantic_layer: SemanticLayer = _default_sandbox_semantic_layer()
        self.semantic_yaml_path: str = ""
        self.selected_module: Optional[str] = None
        self.max_retries: int = 2
        self.high_accuracy_mode: bool = False
        self.num_candidates: int = 3


def _serialize_query_result(result: QueryResult, question: str = "") -> Dict[str, Any]:
    recommended = recommend_chart_config(result.rows, title=question) if result.rows else None
    return {
        "status": result.status.value,
        "is_success": result.is_success,
        "sql": result.sql,
        "rows": result.rows,
        "columns": list(result.rows[0].keys()) if result.rows else [],
        "row_count": len(result.rows),
        "markdown_answer": result.markdown_answer,
        "clean_markdown": strip_chart_json_block(result.markdown_answer),
        "chart_config": result.chart_config,
        "recommended_chart_config": recommended,
        "cache_hit": asdict(result.cache_hit) if result.cache_hit else None,
        "attempts": [asdict(a) for a in result.attempts],
        "error": result.error,
        "linked_tables": result.linked_tables,
        "agent_steps": result.agent_steps,
        "mode": result.mode,
        "preflight_status": result.preflight_status,
        "follow_up_suggestions": result.follow_up_suggestions,
        "workflow_trace": result.workflow_trace,
        "scaling_info": result.scaling_info,
    }


def create_app(
    initial_engine: Optional[SmartDBEngine] = None,
    cache_db_path: str = "semantic_cache.db",
    frontend_dist_dir: Optional[str] = None,
    app_db_path: Optional[str] = None,
    require_auth: bool = False,
) -> FastAPI:
    app = FastAPI(title="SmartDBChat API", version="0.3.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state = AppRuntimeState(cache_db_path=cache_db_path, app_db_path=app_db_path)
    if initial_engine is not None:
        state.engine = initial_engine
        if initial_engine.semantic_layer is not None:
            state.semantic_layer = initial_engine.semantic_layer
        state.connection_meta = {
            "db_type": "custom",
            "database": getattr(getattr(initial_engine.db, "config", None), "database", "smartdb"),
            "host": getattr(getattr(initial_engine.db, "config", None), "host", "local"),
            "model": getattr(getattr(initial_engine.llm, "config", None), "model", ""),
            "provider": "Custom",
            "embedding_available": initial_engine.cache.embedding_available,
        }

    def _extract_bearer_token(request: Request) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        return ""

    def _require_user(request: Request) -> Dict[str, Any]:
        user = getattr(request.state, "user", None)
        if user:
            return user
        token = _extract_bearer_token(request)
        verified = state.store.verify_token(token)
        if not verified:
            raise HTTPException(status_code=401, detail="未登录或登录已过期，请先登录")
        request.state.user = verified
        return verified

    _PUBLIC_API_PATHS = {
        "/api/health",
        "/api/providers",
        "/api/auth/register",
        "/api/auth/login",
    }

    @app.middleware("http")
    async def auth_guard_middleware(request: Request, call_next: Any) -> Any:
        path = request.url.path
        if request.method == "OPTIONS" or not path.startswith("/api/"):
            return await call_next(request)
        if path in _PUBLIC_API_PATHS:
            return await call_next(request)

        token = _extract_bearer_token(request)
        if token:
            user = state.store.verify_token(token)
            if not user:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "登录凭证无效或已过期，请重新登录"},
                )
            request.state.user = user
            return await call_next(request)

        if require_auth or path.startswith("/api/user/") or path.startswith("/api/auth/") or state.store.has_any_users():
            return JSONResponse(
                status_code=401,
                content={"detail": "未登录或缺少访问权限，请先登录"},
            )
        return await call_next(request)

    def _require_engine() -> SmartDBEngine:
        if state.engine is None:
            raise HTTPException(status_code=400, detail="请先连接数据库或启用内置沙箱")
        return state.engine

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "connected": state.engine is not None,
            "connection_meta": state.connection_meta,
        }

    @app.get("/api/providers")
    def list_providers() -> Dict[str, Any]:
        return {
            "providers": [
                {
                    "name": p.name,
                    "base_url": p.base_url,
                    "default_model": p.default_model,
                    "models": list(p.models),
                    "requires_api_key": p.requires_api_key,
                }
                for p in LLM_PROVIDERS.values()
            ]
        }

    @app.post("/api/auth/register")
    def register(req: AuthCredentialsRequest) -> Dict[str, Any]:
        try:
            return state.store.register_user(req.username, req.password)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/auth/login")
    def login(req: AuthCredentialsRequest) -> Dict[str, Any]:
        try:
            return state.store.login_user(req.username, req.password)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.post("/api/auth/logout")
    def logout(request: Request) -> Dict[str, Any]:
        _require_user(request)
        token = _extract_bearer_token(request)
        if token:
            state.store.revoke_token(token)
        return {"ok": True}

    @app.get("/api/auth/me")
    def get_me(request: Request) -> Dict[str, Any]:
        return {"user": _require_user(request)}

    @app.get("/api/user/state")
    def get_user_state(request: Request) -> Dict[str, Any]:
        user = _require_user(request)
        return state.store.get_user_state(user["id"])

    @app.put("/api/user/preferences")
    def update_user_preferences(req: UpdatePreferencesRequest, request: Request) -> Dict[str, Any]:
        user = _require_user(request)
        state.store.update_preferences(
            user_id=user["id"],
            last_connection=req.last_connection,
            update_last_connection=req.update_last_connection,
            auto_connect_disabled=req.auto_connect_disabled,
            active_session_id=req.active_session_id,
            update_active_session=req.update_active_session,
        )
        return {"ok": True}

    @app.post("/api/user/connections")
    def save_user_connection(req: SaveConnectionProfileRequest, request: Request) -> Dict[str, Any]:
        user = _require_user(request)
        try:
            profiles = state.store.save_connection_profile(
                user_id=user["id"],
                name=req.name,
                payload=req.payload,
                profile_id=req.id,
            )
            state.store.update_preferences(
                user_id=user["id"],
                last_connection=req.payload,
                update_last_connection=True,
            )
            return {"profiles": profiles}
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.delete("/api/user/connections/{profile_id}")
    def delete_user_connection(profile_id: str, request: Request) -> Dict[str, Any]:
        user = _require_user(request)
        try:
            profiles = state.store.delete_connection_profile(user["id"], profile_id)
            return {"profiles": profiles}
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.put("/api/user/sessions/{session_id}")
    def save_user_session(
        session_id: str, req: SaveChatSessionRequest, request: Request
    ) -> Dict[str, Any]:
        user = _require_user(request)
        try:
            sessions = state.store.save_chat_session(
                user_id=user["id"],
                session_id=session_id,
                title=req.title,
                messages=req.messages,
            )
            state.store.update_preferences(
                user_id=user["id"],
                active_session_id=session_id,
                update_active_session=True,
            )
            return {"sessions": sessions}
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.delete("/api/user/sessions/{session_id}")
    def delete_user_session(session_id: str, request: Request) -> Dict[str, Any]:
        user = _require_user(request)
        try:
            sessions = state.store.delete_chat_session(user["id"], session_id)
            return {"sessions": sessions}
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/api/connect")
    def connect(req: ConnectRequest, request: Request) -> Dict[str, Any]:
        with state.lock:
            if state.engine is not None and hasattr(state.engine.db, "close"):
                try:
                    state.engine.db.close()
                except Exception:
                    pass

            if req.db_type == "sandbox":
                db_backend: Any = SQLiteEvalBackend()
                dialect = "sqlite"
                db_label = "smartdb_eval (SQLite 沙箱)"
                host_label = "local-sandbox"
            else:
                if not req.database.strip():
                    raise HTTPException(status_code=400, detail="MySQL 模式下数据库名不能为空")
                db_cfg = DatabaseConfig(
                    host=req.host,
                    port=req.port,
                    user=req.user,
                    password=req.password,
                    database=req.database.strip(),
                )
                db_backend = DatabaseManager(db_cfg)
                try:
                    db_backend.connect()
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"MySQL 连接失败: {exc}") from exc
                dialect = "mysql"
                db_label = req.database.strip()
                host_label = f"{req.host}:{req.port}"

            sem_path = req.semantic_yaml_path.strip()
            if sem_path:
                if not os.path.exists(sem_path):
                    raise HTTPException(status_code=400, detail=f"语义层配置文件不存在: {sem_path}")
                state.semantic_layer = SemanticLayer.from_yaml_file(sem_path)
                state.semantic_yaml_path = sem_path
            elif req.db_type == "sandbox" and not state.semantic_layer.metrics:
                state.semantic_layer = _default_sandbox_semantic_layer()

            state.cache.similarity_threshold = req.similarity_threshold
            state.cache.embedding_config = EmbeddingConfig(
                base_url=req.embedding_url,
                model=req.embedding_model,
                api_key=req.api_key,
                mode=req.embedding_mode,
            )
            emb_ok = state.cache.test_embedding_availability()

            llm_cfg = LLMConfig(
                api_key=req.api_key or "ollama",
                base_url=req.base_url,
                model=req.model,
            )
            llm_client = LLMClient(llm_cfg)
            agent_cfg = AgentConfig(
                max_steps=max(2, min(req.agent_max_steps, 15)),
                enable_explain_check=req.enable_explain_check,
            )

            engine = SmartDBEngine(
                db=db_backend,
                llm=llm_client,
                cache=state.cache,
                agent_config=agent_cfg,
                default_mode=req.query_mode,
                semantic_layer=state.semantic_layer,
                enable_preflight=req.enable_preflight,
                enable_ast_guard=True,
                enable_ast_schema_check=req.enable_ast_schema_check,
                max_limit=req.max_limit if req.max_limit > 0 else None,
                dialect=dialect,
            )

            state.engine = engine
            state.db_type = req.db_type
            state.max_retries = req.max_retries
            state.high_accuracy_mode = req.high_accuracy_mode
            state.num_candidates = req.num_candidates
            state.selected_module = None
            state.connection_meta = {
                "db_type": req.db_type,
                "database": db_label,
                "host": host_label,
                "provider": req.provider,
                "model": req.model,
                "base_url": req.base_url,
                "dialect": dialect,
                "embedding_available": emb_ok,
                "has_api_key": bool(req.api_key.strip()),
            }
            user = getattr(request.state, "user", None)
            if user:
                req_dict = req.model_dump()
                state.store.update_preferences(
                    user_id=user["id"],
                    last_connection=req_dict,
                    update_last_connection=True,
                    auto_connect_disabled=False,
                )
                cur_state = state.store.get_user_state(user["id"])
                if not cur_state["profiles"]:
                    state.store.save_connection_profile(
                        user_id=user["id"], name="", payload=req_dict
                    )
            return {
                "connected": True,
                "connection_meta": state.connection_meta,
                "config": _current_config_payload(),
            }

    @app.post("/api/disconnect")
    def disconnect(request: Request) -> Dict[str, Any]:
        with state.lock:
            if state.engine is not None and hasattr(state.engine.db, "close"):
                try:
                    state.engine.db.close()
                except Exception:
                    pass
            state.engine = None
            state.connection_meta = {}
            user = getattr(request.state, "user", None)
            if user:
                state.store.update_preferences(
                    user_id=user["id"], auto_connect_disabled=True
                )
            return {"connected": False}

    def _current_config_payload() -> Dict[str, Any]:
        eng = state.engine
        return {
            "query_mode": eng.default_mode if eng else "agent",
            "agent_max_steps": eng.agent_config.max_steps if eng else 8,
            "enable_explain_check": eng.agent_config.enable_explain_check if eng else True,
            "max_retries": state.max_retries,
            "high_accuracy_mode": state.high_accuracy_mode,
            "num_candidates": state.num_candidates,
            "enable_preflight": eng.enable_preflight if eng else True,
            "enable_ast_schema_check": eng.enable_ast_schema_check if eng else True,
            "max_limit": eng.max_limit if (eng and eng.max_limit is not None) else 0,
            "similarity_threshold": state.cache.similarity_threshold,
            "selected_module": state.selected_module,
            "model": state.connection_meta.get("model", ""),
        }

    @app.get("/api/config")
    def get_config() -> Dict[str, Any]:
        return _current_config_payload()

    @app.put("/api/config")
    def update_config(req: ConfigUpdateRequest) -> Dict[str, Any]:
        eng = _require_engine()
        with state.lock:
            if req.query_mode is not None:
                eng.default_mode = req.query_mode
            if req.agent_max_steps is not None:
                eng.agent_config.max_steps = max(2, min(req.agent_max_steps, 15))
            if req.enable_explain_check is not None:
                eng.agent_config.enable_explain_check = req.enable_explain_check
            if req.max_retries is not None:
                state.max_retries = max(0, min(req.max_retries, 5))
            if req.high_accuracy_mode is not None:
                state.high_accuracy_mode = req.high_accuracy_mode
            if req.num_candidates is not None:
                state.num_candidates = max(1, min(req.num_candidates, 3))
            if req.enable_preflight is not None:
                eng.enable_preflight = req.enable_preflight
            if req.enable_ast_schema_check is not None:
                eng.enable_ast_schema_check = req.enable_ast_schema_check
            if req.max_limit is not None:
                eng.max_limit = req.max_limit if req.max_limit > 0 else None
            if req.similarity_threshold is not None:
                state.cache.similarity_threshold = req.similarity_threshold
            if req.selected_module is not None:
                state.selected_module = req.selected_module or None
            if req.model is not None and req.model.strip():
                eng.llm.config.model = req.model.strip()
                state.connection_meta["model"] = req.model.strip()
        return _current_config_payload()

    @app.get("/api/modules")
    def list_modules() -> Dict[str, Any]:
        eng = _require_engine()
        return {"modules": get_all_modules(eng.sys_tables_db_path)}

    @app.get("/api/schema/graph")
    def get_schema_graph() -> Dict[str, Any]:
        eng = _require_engine()
        raw_schema = eng.db.get_schema_info()
        ddls = _split_ddl_statements(raw_schema)
        graph = build_schema_graph(ddls)
        sem = eng.semantic_layer

        nodes: List[Dict[str, Any]] = []
        for tname, tmeta in graph.tables.items():
            sem_desc = (sem.table_descriptions.get(tname, "") if sem else "") or tmeta.comment
            col_descs = sem.column_descriptions.get(tname, {}) if sem else {}
            nodes.append(
                {
                    "name": tname,
                    "comment": sem_desc,
                    "ddl": tmeta.ddl,
                    "columns": [
                        {
                            "name": c.name,
                            "data_type": c.data_type,
                            "comment": col_descs.get(c.name, "") or c.comment,
                            "is_primary_key": c.is_primary_key,
                            "is_nullable": c.is_nullable,
                        }
                        for c in tmeta.columns
                    ],
                }
            )

        edges = [
            {
                "from_table": e.from_table,
                "from_column": e.from_column,
                "to_table": e.to_table,
                "to_column": e.to_column,
                "inferred": e.inferred,
            }
            for e in graph.edges
        ]
        return {"nodes": nodes, "edges": edges, "table_count": len(nodes), "edge_count": len(edges)}

    @app.get("/api/schema/sample/{table_name}")
    def sample_table(table_name: str, limit: int = 5) -> Dict[str, Any]:
        eng = _require_engine()
        try:
            rows = eng.db.sample_table_rows(table_name, limit=max(1, min(limit, 20)))
            return {"table": table_name, "rows": rows}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/schema/values/{table_name}/{column_name}")
    def explore_column_values(
        table_name: str, column_name: str, keyword: str = "", limit: int = 20
    ) -> Dict[str, Any]:
        eng = _require_engine()
        try:
            safe_limit = max(1, min(limit, 50))
            if keyword.strip() and hasattr(eng.db, "search_column_values"):
                vals = eng.db.search_column_values(
                    table_name, column_name, keyword.strip(), limit=safe_limit
                )
            else:
                vals = eng.db.get_distinct_values(table_name, column_name, limit=safe_limit)
            return {
                "table": table_name,
                "column": column_name,
                "keyword": keyword,
                "values": vals,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/schema/join-path")
    def find_join_path(from_table: str, to_table: str) -> Dict[str, Any]:
        eng = _require_engine()
        raw_schema = eng.db.get_schema_info()
        graph = build_schema_graph(_split_ddl_statements(raw_schema))
        path = graph.shortest_path(from_table, to_table) or []
        edges = graph.find_join_edges(from_table, to_table)
        return {
            "from_table": from_table,
            "to_table": to_table,
            "path": path,
            "join_conditions": [
                f"{e.from_table}.{e.from_column} = {e.to_table}.{e.to_column}" for e in edges
            ],
        }

    @app.get("/api/semantic")
    def get_semantic_layer() -> Dict[str, Any]:
        layer = state.engine.semantic_layer if (state.engine and state.engine.semantic_layer) else state.semantic_layer
        return {
            "data": layer.to_dict(),
            "yaml_text": layer.to_yaml(),
            "yaml_path": state.semantic_yaml_path,
        }

    @app.put("/api/semantic")
    def update_semantic_layer(req: SemanticUpdateRequest) -> Dict[str, Any]:
        try:
            if req.yaml_text is not None:
                layer = SemanticLayer.from_yaml(req.yaml_text)
            elif req.data is not None:
                layer = SemanticLayer.from_dict(req.data)
            else:
                raise HTTPException(status_code=400, detail="请提供 yaml_text 或 data")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"语义层解析失败: {exc}") from exc

        with state.lock:
            state.semantic_layer = layer
            if state.engine is not None:
                state.engine.semantic_layer = layer
            if state.semantic_yaml_path:
                try:
                    layer.save_yaml_file(state.semantic_yaml_path)
                except Exception:
                    pass

        return {
            "data": layer.to_dict(),
            "yaml_text": layer.to_yaml(),
            "yaml_path": state.semantic_yaml_path,
        }

    @app.post("/api/semantic/golden")
    def add_golden_sql(req: GoldenExampleRequest) -> Dict[str, Any]:
        layer = state.engine.semantic_layer if (state.engine and state.engine.semantic_layer) else state.semantic_layer
        layer.add_golden_example(
            question=req.question,
            sql=req.sql,
            tables=req.tables,
            tags=req.tags,
        )
        if state.semantic_yaml_path:
            try:
                layer.save_yaml_file(state.semantic_yaml_path)
            except Exception:
                pass
        return {"count": len(layer.golden_examples), "data": layer.to_dict()}

    @app.post("/api/sql/execute")
    def execute_custom_sql(req: ExecuteSQLRequest) -> Dict[str, Any]:
        eng = _require_engine()
        raw_schema = eng.db.get_schema_info()
        graph = build_schema_graph(_split_ddl_statements(raw_schema))
        allowed_tables = list(graph.tables.keys()) if graph.tables else None
        table_columns = (
            {t: [c.name for c in tm.columns] for t, tm in graph.tables.items()}
            if graph.tables
            else None
        )
        guard = validate_sql_ast(
            req.sql,
            dialect=eng.dialect,
            allowed_tables=allowed_tables if eng.enable_ast_schema_check else None,
            table_columns=table_columns if eng.enable_ast_schema_check else None,
            max_limit=eng.max_limit,
        )
        if not guard.is_valid:
            raise HTTPException(status_code=400, detail=guard.reason or "SQL 安全校验未通过")

        exec_sql = guard.sql or req.sql
        try:
            rows = eng.db.execute_query(exec_sql)
        except UnsafeSQLError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"SQL 执行失败: {exc}") from exc

        return {
            "sql": exec_sql,
            "rows": rows,
            "columns": list(rows[0].keys()) if rows else [],
            "row_count": len(rows),
            "tables": guard.tables,
            "recommended_chart_config": recommend_chart_config(rows, title="自定义 SQL 结果"),
        }

    @app.post("/api/query")
    def run_query_sync(req: QueryRequest) -> Dict[str, Any]:
        eng = _require_engine()
        res = eng.run_query(
            query=req.question,
            module=req.module if req.module is not None else state.selected_module,
            max_retries=req.max_retries if req.max_retries is not None else state.max_retries,
            mode=req.mode,
            history=req.history,
            high_accuracy_mode=(
                req.high_accuracy_mode
                if req.high_accuracy_mode is not None
                else state.high_accuracy_mode
            ),
            num_candidates=(
                req.num_candidates if req.num_candidates is not None else state.num_candidates
            ),
        )
        return _serialize_query_result(res, question=req.question)

    @app.post("/api/query/stream")
    async def run_query_stream(req: QueryRequest) -> StreamingResponse:
        eng = _require_engine()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        def _push_event(event_type: str, payload: Dict[str, Any]) -> None:
            data = json.dumps({"type": event_type, **payload}, ensure_ascii=False, default=str)
            loop.call_soon_threadsafe(queue.put_nowait, f"data: {data}\n\n")

        def _worker() -> None:
            try:
                res = eng.run_query(
                    query=req.question,
                    module=req.module if req.module is not None else state.selected_module,
                    max_retries=(
                        req.max_retries if req.max_retries is not None else state.max_retries
                    ),
                    on_status=lambda lvl, msg: _push_event(
                        "status", {"level": lvl, "message": msg}
                    ),
                    on_stream=lambda acc: _push_event(
                        "token", {"accumulated": strip_chart_json_block(acc)}
                    ),
                    mode=req.mode,
                    history=req.history,
                    high_accuracy_mode=(
                        req.high_accuracy_mode
                        if req.high_accuracy_mode is not None
                        else state.high_accuracy_mode
                    ),
                    num_candidates=(
                        req.num_candidates
                        if req.num_candidates is not None
                        else state.num_candidates
                    ),
                )
                _push_event("result", {"result": _serialize_query_result(res, question=req.question)})
            except Exception as exc:
                _push_event("error", {"message": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_worker, daemon=True).start()

        async def _event_generator():
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

        return StreamingResponse(_event_generator(), media_type="text/event-stream")

    @app.get("/api/cache/stats")
    def get_cache_stats() -> Dict[str, Any]:
        cache = state.engine.cache if state.engine else state.cache
        stats = cache.get_cache_stats()
        return {
            "total_entries": stats.total_entries,
            "total_hits": stats.total_hits,
            "top_queries": [{"query": q, "hits": h} for q, h in stats.top_queries],
        }

    @app.delete("/api/cache")
    def clear_cache() -> Dict[str, Any]:
        cache = state.engine.cache if state.engine else state.cache
        cache.clear_cache()
        return {"cleared": True}

    dist_path = Path(frontend_dist_dir) if frontend_dist_dir else (
        Path(__file__).resolve().parent.parent / "frontend" / "dist"
    )
    if dist_path.exists() and (dist_path / "index.html").exists():
        assets_dir = dist_path / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{full_path:path}")
        def serve_spa(full_path: str) -> Any:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API endpoint not found")
            candidate = dist_path / full_path
            if full_path and candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(dist_path / "index.html"))

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.getenv("SMARTDB_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SMARTDB_SERVER_PORT", "8000"))
    uvicorn.run("smartdbchat.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
