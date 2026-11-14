import streamlit as st

from smartdbchat.cache import SemanticQueryCache
from smartdbchat.config import (
    LLM_PROVIDERS,
    DatabaseConfig,
    EmbeddingConfig,
    LLMConfig,
)
from smartdbchat.agent import AgentConfig
from smartdbchat.db import DatabaseManager, get_all_modules
from smartdbchat.engine import SmartDBEngine
from smartdbchat.llm import LLMClient


def _connect_engine(
    db_config: DatabaseConfig,
    llm_config: LLMConfig,
    embedding_config: EmbeddingConfig,
) -> SmartDBEngine | None:
    try:
        cache_system: SemanticQueryCache = st.session_state.cache_system
        cache_system.similarity_threshold = st.session_state.similarity_threshold
        cache_system.embedding_config = embedding_config
        if not cache_system.test_embedding_availability():
            st.warning("嵌入向量 API 不可用，已自动降级为文本/关键词相似度匹配")

        db_manager = DatabaseManager(db_config)
        db_manager.connect()

        llm_client = LLMClient(llm_config)
        agent_cfg = AgentConfig(max_steps=int(st.session_state.get("agent_max_steps", 8)))
        semantic_layer = None
        sem_path = (st.session_state.get("semantic_yaml_path") or "").strip()
        if sem_path:
            from smartdbchat.semantic_layer import SemanticLayer
            import os

            if os.path.exists(sem_path):
                semantic_layer = SemanticLayer.from_yaml_file(sem_path)
            else:
                st.warning(f"未找到语义层 YAML 文件: {sem_path}，已跳过加载")

        engine = SmartDBEngine(
            db=db_manager,
            llm=llm_client,
            cache=cache_system,
            agent_config=agent_cfg,
            default_mode=st.session_state.get("query_mode", "agent"),
            semantic_layer=semantic_layer,
            enable_preflight=bool(st.session_state.get("enable_preflight", True)),
            enable_ast_schema_check=bool(st.session_state.get("enable_ast_schema_check", True)),
            max_limit=int(st.session_state.get("max_limit", 1000)),
        )
        st.session_state.engine = engine
        st.session_state.connection_meta = {
            "host": db_config.host,
            "database": db_config.database,
            "model": llm_config.model,
            "base_url": llm_config.base_url,
        }
        return engine
    except Exception as exc:
        st.error(f"连接或初始化失败: {exc}")
        return None


def render_sidebar() -> None:
    """渲染侧边栏：连接配置、模型预设、缓存管理与查询参数。"""
    with st.sidebar:
        st.header("⚙️ 系统配置")

        if st.session_state.engine:
            meta = st.session_state.get("connection_meta", {})
            st.success(f"已连接：`{meta.get('database', '')}` @ `{meta.get('host', '')}`")
            if st.button("断开当前连接", width="stretch"):
                try:
                    st.session_state.engine.db.close()
                except Exception:
                    pass
                st.session_state.engine = None
                st.session_state.connection_meta = {}
                st.rerun()

        with st.expander("🗄️ 数据库连接", expanded=not bool(st.session_state.engine)):
            host = st.text_input("主机", value="localhost")
            port = st.number_input("端口", min_value=1, max_value=65535, value=3306, step=1)
            user = st.text_input("用户名", value="root")
            password = st.text_input("密码", type="password", value="")
            database = st.text_input("数据库名", value="")

        with st.expander("🤖 大语言模型 (LLM)", expanded=not bool(st.session_state.engine)):
            provider_names = list(LLM_PROVIDERS.keys())
            selected_provider = st.selectbox(
                "服务商预设",
                provider_names,
                index=0,
                help="选择主流服务商将自动填充最新 API 端点与推荐模型",
            )
            preset = LLM_PROVIDERS[selected_provider]

            api_key = st.text_input(
                "API 密钥",
                type="password",
                value="",
                help="本地 Ollama 可留空" if not preset.requires_api_key else "",
            )
            base_url = st.text_input(
                "Base URL",
                value=preset.base_url,
                help="OpenAI 兼容接口地址",
            )
            if preset.models:
                model_choice = st.selectbox(
                    "模型名称",
                    preset.models + ["自定义模型名称..."],
                    index=0,
                )
                if model_choice == "自定义模型名称...":
                    model_name = st.text_input("自定义模型", value=preset.default_model)
                else:
                    model_name = model_choice
            else:
                model_name = st.text_input("模型名称", value=preset.default_model)

        with st.expander("🧠 语义缓存与查询参数", expanded=False):
            embedding_mode_label = st.selectbox(
                "嵌入向量协议",
                ["OpenAI 兼容 (/embeddings)", "Ollama 旧版 (/api/embeddings)"],
                index=0,
            )
            embedding_mode = (
                "openai_compatible"
                if embedding_mode_label.startswith("OpenAI")
                else "ollama_legacy"
            )
            default_emb_url = (
                "http://localhost:11434/v1"
                if embedding_mode == "openai_compatible"
                else "http://localhost:11434"
            )
            embedding_url = st.text_input("嵌入服务地址", value=default_emb_url)
            embedding_model = st.text_input("嵌入模型", value="nomic-embed-text")
            st.session_state.similarity_threshold = st.slider(
                "缓存相似度阈值",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.similarity_threshold,
                step=0.05,
            )
            st.session_state.max_retries = st.number_input(
                "SQL 失败最大重试次数",
                min_value=0,
                max_value=5,
                value=st.session_state.get("max_retries", 2),
                step=1,
            )
            mode_options = {
                "agent": "Agent 智能探索 (Tool Use + Schema Linking)",
                "linked": "Schema Linking 增强 (裁剪 + 重试)",
                "pipeline": "传统单轮模式 (全量 DDL)",
            }
            mode_keys = list(mode_options.keys())
            current_mode = st.session_state.get("query_mode", "agent")
            st.session_state.query_mode = st.selectbox(
                "查询执行模式",
                mode_keys,
                index=mode_keys.index(current_mode) if current_mode in mode_keys else 0,
                format_func=lambda k: mode_options[k],
            )
            st.session_state.agent_max_steps = st.number_input(
                "Agent 最大探索步数",
                min_value=2,
                max_value=15,
                value=st.session_state.get("agent_max_steps", 8),
                step=1,
            )
            st.session_state.high_accuracy_mode = st.checkbox(
                "高精度模式 (Test-Time Scaling 多数投票)",
                value=st.session_state.get("high_accuracy_mode", False),
                help="开启后将生成多路候选 SQL 并基于执行结果一致性投票择优（非 Agent 模式下生效）",
            )
            st.session_state.enable_preflight = st.checkbox(
                "启用 Preflight 语义预检",
                value=st.session_state.get("enable_preflight", True),
            )
            st.session_state.enable_ast_schema_check = st.checkbox(
                "启用 AST Schema 一致性校验",
                value=st.session_state.get("enable_ast_schema_check", True),
            )
            st.session_state.semantic_yaml_path = st.text_input(
                "语义层配置 (YAML 路径，可选)",
                value=st.session_state.get("semantic_yaml_path", ""),
                placeholder="例如: ./semantic_layer.yaml",
            )

        if st.button("🔌 连接 / 更新配置", type="primary", width="stretch"):
            if preset.requires_api_key and not api_key:
                st.error("请输入 API 密钥")
            elif not database:
                st.error("请输入数据库名")
            else:
                db_cfg = DatabaseConfig(
                    host=host,
                    port=int(port),
                    user=user,
                    password=password,
                    database=database,
                )
                llm_cfg = LLMConfig(
                    api_key=api_key,
                    base_url=base_url,
                    model=model_name,
                )
                emb_cfg = EmbeddingConfig(
                    base_url=embedding_url,
                    model=embedding_model,
                    api_key=api_key,
                    mode=embedding_mode,
                )
                with st.spinner("正在连接数据库..."):
                    if _connect_engine(db_cfg, llm_cfg, emb_cfg):
                        st.success("数据库连接成功！")
                        st.rerun()

        if st.session_state.engine:
            modules = get_all_modules()
            if modules:
                st.subheader("🗂️ 业务模块过滤")
                selected = st.selectbox(
                    "选择业务模块",
                    ["全部"] + modules,
                    help="缩小表结构范围可提高 SQL 生成准确率",
                )
                st.session_state.selected_module = None if selected == "全部" else selected

        if st.session_state.cache_system:
            st.subheader("📦 缓存统计")
            stats = st.session_state.cache_system.get_cache_stats()
            col_a, col_b = st.columns(2)
            col_a.metric("缓存条目", stats.total_entries)
            col_b.metric("累计命中", stats.total_hits)
            if st.button("清空语义缓存", width="stretch"):
                st.session_state.cache_system.clear_cache()
                st.success("缓存已清空")
                st.rerun()
