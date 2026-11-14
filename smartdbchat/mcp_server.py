from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

from smartdbchat.cache import SemanticQueryCache
from smartdbchat.config import CacheConfig, DatabaseConfig, LLMConfig
from smartdbchat.db import DatabaseManager
from smartdbchat.engine import SmartDBEngine, _split_ddl_statements
from smartdbchat.llm import LLMClient
from smartdbchat.schema_linker import build_schema_graph
from smartdbchat.semantic_layer import SemanticLayer
from smartdbchat.sql_guard import validate_sql_ast


def create_mcp_server(engine: SmartDBEngine, name: str = "SmartDBChat") -> MCPServer:
    """
    创建 SmartDBChat MCP Server（借鉴 QueryWeaver 与 SQLBot 的开放协议集成）。
    将图谱 Schema 理解、列值探索、AST 安全校验与自然语言问数能力暴露为标准 MCP 工具。
    """
    mcp = MCPServer(name)

    @mcp.tool()
    def list_tables() -> str:
        """列出当前数据库中所有可查询的表名及语义描述（若已配置语义层）。"""
        tables = engine.db.list_tables() if hasattr(engine.db, "list_tables") else []
        descs = engine.semantic_layer.table_descriptions if engine.semantic_layer else {}
        payload = [
            {"table": t, "description": descs.get(t, "")} for t in tables
        ]
        return json.dumps({"tables": payload}, ensure_ascii=False)

    @mcp.tool()
    def get_database_schema(
        tables: Optional[List[str]] = None,
        question: Optional[str] = None,
    ) -> str:
        """获取数据库 DDL、QueryWeaver 风格 Schema 图谱摘要及语义层上下文。"""
        if question:
            ddl, _, linked = engine._resolve_schema(
                query=question, module=None, use_schema_linking=True
            )
            target_tables = linked or tables
        else:
            ddl = engine.db.get_schema_info(tables)
            target_tables = tables

        graph = build_schema_graph(_split_ddl_statements(ddl))
        graph_summary = graph.to_graph_summary(target_tables) if graph.tables else ""
        sem_text = ""
        if engine.semantic_layer is not None and question:
            sem_text = engine.semantic_layer.retrieve_context(
                question, selected_tables=target_tables
            ).to_prompt_text()

        return json.dumps(
            {
                "tables": list(graph.tables.keys()) if graph.tables else (target_tables or []),
                "ddl": ddl,
                "schema_graph": graph_summary,
                "semantic_context": sem_text,
            },
            ensure_ascii=False,
        )

    @mcp.tool()
    def find_join_path(from_table: str, to_table: str) -> str:
        """基于 Schema 图谱查找两张表之间的最短外键 JOIN 路径。"""
        ddl = engine.db.get_schema_info()
        graph = build_schema_graph(_split_ddl_statements(ddl))
        path = graph.shortest_path(from_table, to_table) or []
        edges = graph.find_join_edges(from_table, to_table)
        return json.dumps(
            {
                "from_table": from_table,
                "to_table": to_table,
                "path": path,
                "join_conditions": [
                    f"{e.from_table}.{e.from_column} = {e.to_table}.{e.to_column}"
                    for e in edges
                ],
            },
            ensure_ascii=False,
        )

    @mcp.tool()
    def sample_column_values(
        table_name: str,
        column_name: str,
        keyword: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        """探索指定表列的真实去重取值（支持按关键字模糊筛选，解决枚举值不匹配）。"""
        safe_limit = max(1, min(int(limit or 20), 50))
        if keyword and hasattr(engine.db, "search_column_values"):
            values = engine.db.search_column_values(
                table_name, column_name, keyword, limit=safe_limit
            )
        else:
            raw = engine.db.get_distinct_values(table_name, column_name, limit=safe_limit)
            if keyword:
                kw_lower = keyword.strip().lower()
                values = [v for v in raw if v is not None and kw_lower in str(v).lower()]
            else:
                values = raw
        return json.dumps(
            {
                "table": table_name,
                "column": column_name,
                "keyword": keyword,
                "values": [v for v in values if v is not None],
            },
            ensure_ascii=False,
            default=str,
        )

    @mcp.tool()
    def validate_sql(sql: str) -> str:
        """使用 sqlglot AST 对 SQL 进行只读安全性与 Schema 一致性校验。"""
        ddl = engine.db.get_schema_info()
        graph = build_schema_graph(_split_ddl_statements(ddl))
        allowed_tables = list(graph.tables.keys()) if graph.tables else None
        table_columns = (
            {t: [c.name for c in tm.columns] for t, tm in graph.tables.items()}
            if graph.tables
            else None
        )
        res = validate_sql_ast(
            sql,
            dialect=engine.dialect,
            allowed_tables=allowed_tables,
            table_columns=table_columns,
            max_limit=engine.max_limit,
        )
        return json.dumps(
            {
                "is_valid": res.is_valid,
                "reason": res.reason,
                "sql": res.sql,
                "tables": res.tables,
                "unknown_tables": res.unknown_tables,
                "unknown_columns": res.unknown_columns,
            },
            ensure_ascii=False,
        )

    @mcp.tool()
    def ask_database(
        question: str,
        mode: Optional[str] = None,
        high_accuracy_mode: bool = False,
    ) -> str:
        """用自然语言提问数据库，返回生成的只读 SQL、查询结果、分析结论、图表配置与后续建议。"""
        result = engine.run_query(
            query=question,
            mode=mode,
            high_accuracy_mode=high_accuracy_mode,
        )
        payload: Dict[str, Any] = {
            "status": result.status.value,
            "sql": result.sql,
            "row_count": len(result.rows),
            "rows": result.rows[:100],
            "markdown_answer": result.markdown_answer,
            "chart_config": result.chart_config,
            "linked_tables": result.linked_tables,
            "follow_up_suggestions": result.follow_up_suggestions,
            "scaling_info": result.scaling_info,
            "error": result.error,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    return mcp


def build_engine_from_env() -> SmartDBEngine:
    """从环境变量构建 SmartDBEngine 实例，用于独立启动 MCP Server。"""
    db_cfg = DatabaseConfig(
        host=os.getenv("SMARTDB_HOST", "localhost"),
        port=int(os.getenv("SMARTDB_PORT", "3306")),
        user=os.getenv("SMARTDB_USER", "root"),
        password=os.getenv("SMARTDB_PASSWORD", ""),
        database=os.getenv("SMARTDB_DATABASE", ""),
    )
    llm_cfg = LLMConfig(
        api_key=os.getenv("SMARTDB_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        base_url=os.getenv("SMARTDB_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.getenv("SMARTDB_MODEL", "deepseek-flash"),
    )
    cache_cfg = CacheConfig(
        db_path=os.getenv("SMARTDB_CACHE_DB", "semantic_cache.db"),
    )

    semantic_layer: Optional[SemanticLayer] = None
    sem_path = os.getenv("SMARTDB_SEMANTIC_YAML", "")
    if sem_path and os.path.exists(sem_path):
        semantic_layer = SemanticLayer.from_yaml_file(sem_path)

    return SmartDBEngine(
        db=DatabaseManager(db_cfg),
        llm=LLMClient(llm_cfg),
        cache=SemanticQueryCache(cache_config=cache_cfg),
        semantic_layer=semantic_layer,
        enable_ast_schema_check=True,
        max_limit=int(os.getenv("SMARTDB_MAX_LIMIT", "1000")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartDBChat MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="MCP 传输协议（默认 stdio，适用于 Claude Desktop / Cursor）",
    )
    args = parser.parse_args()
    engine = build_engine_from_env()
    server = create_mcp_server(engine)
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
