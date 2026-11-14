import pandas as pd
import streamlit as st

from smartdbchat.cache import SemanticQueryCache
from smartdbchat.charts import build_figures
from smartdbchat.config import CacheConfig, DatabaseConfig, EmbeddingConfig, LLMConfig
from smartdbchat.db import DatabaseManager, get_all_modules
from smartdbchat.engine import SmartDBEngine
from smartdbchat.llm import LLMClient


def init_session_state() -> None:
    """初始化 Streamlit session state。"""
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "cache_system" not in st.session_state:
        st.session_state.cache_system = SemanticQueryCache(CacheConfig(), EmbeddingConfig())
    if "similarity_threshold" not in st.session_state:
        st.session_state.similarity_threshold = 0.85
    if "selected_module" not in st.session_state:
        st.session_state.selected_module = None


def create_engine(
    db_config: DatabaseConfig,
    llm_config: LLMConfig,
    embedding_config: EmbeddingConfig,
) -> SmartDBEngine | None:
    """创建并连接 SmartDBEngine 实例。"""
    try:
        cache_system: SemanticQueryCache = st.session_state.cache_system
        cache_system.similarity_threshold = st.session_state.similarity_threshold
        cache_system.embedding_config = embedding_config
        if not cache_system.test_embedding_availability():
            st.warning("嵌入向量 API 不可用，将使用备选文本相似度算法")

        db_manager = DatabaseManager(db_config)
        db_manager.connect()

        llm_client = LLMClient(llm_config)
        engine = SmartDBEngine(db=db_manager, llm=llm_client, cache=cache_system)
        st.session_state.engine = engine
        return engine
    except Exception as exc:
        st.error(f"连接或初始化失败: {exc}")
        return None


def render_sidebar() -> None:
    """渲染侧边栏配置与缓存管理面板。"""
    with st.sidebar:
        st.header("配置")

        st.subheader("数据库设置")
        host = st.text_input("主机", value="localhost")
        port = st.number_input("端口", value=3306, step=1)
        user = st.text_input("用户名", value="root")
        password = st.text_input("密码", type="password", value="")
        database = st.text_input("数据库名", value="")

        st.subheader("LLM 设置")
        api_key = st.text_input("API 密钥", type="password", value="")
        base_url = st.text_input(
            "基础 URL",
            value="https://open.bigmodel.cn/api/paas/v4/",
            help="兼容 OpenAI 接口标准的服务地址",
        )
        model_name = st.text_input("模型名称", value="glm-4-plus")

        st.subheader("缓存与向量设置")
        embedding_url = st.text_input(
            "Ollama 嵌入服务地址", value="http://localhost:11434"
        )
        similarity_threshold = st.slider(
            "相似度阈值",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.similarity_threshold,
            step=0.05,
            help="设置查询相似度匹配的阈值，值越高要求匹配越精确",
        )
        st.session_state.similarity_threshold = similarity_threshold

        if st.button("连接数据库"):
            if not api_key:
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
                emb_cfg = EmbeddingConfig(base_url=embedding_url)
                with st.spinner("正在连接数据库..."):
                    if create_engine(db_cfg, llm_cfg, emb_cfg):
                        st.success("数据库连接成功！")

        if st.session_state.engine:
            modules = get_all_modules()
            if modules:
                st.subheader("业务模块过滤")
                selected = st.selectbox(
                    "选择业务模块",
                    ["全部"] + modules,
                    help="选择业务模块可以减少 LLM 处理的表数量，提高生成 SQL 的准确性",
                )
                st.session_state.selected_module = (
                    None if selected == "全部" else selected
                )

        if st.session_state.cache_system:
            st.subheader("缓存管理")
            stats = st.session_state.cache_system.get_cache_stats()
            st.write(f"缓存条目数: {stats.total_entries}")
            st.write(f"缓存命中次数: {stats.total_hits}")
            if st.button("清空缓存"):
                st.session_state.cache_system.clear_cache()
                st.success("缓存已清空")


def render_status_message(level: str, message: str) -> None:
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def main() -> None:
    st.set_page_config(page_title="SmartDBChat - 自然语言数据库查询", layout="wide")
    init_session_state()

    st.title("SmartDBChat - 自然语言数据库查询")
    render_sidebar()

    engine: SmartDBEngine | None = st.session_state.engine
    if not engine:
        st.info("请先在侧边栏配置并连接数据库")
        return

    query = st.text_area("请输入你的问题", height=100)

    col1, _, col3 = st.columns([2, 2, 1])
    with col1:
        if st.session_state.selected_module:
            st.info(f"当前业务模块: {st.session_state.selected_module}")
        else:
            st.info("当前查询范围: 所有表")
    with col3:
        max_retries = st.number_input(
            "最大重试次数",
            min_value=0,
            max_value=5,
            value=2,
            help="如果 SQL 查询失败或返回为空，自动重试生成 SQL 的最大次数",
        )

    if st.button("查询") and query:
        with st.spinner("正在处理查询..."):
            answer_placeholder = None

            def on_stream(accumulated_text: str) -> None:
                nonlocal answer_placeholder
                if answer_placeholder is None:
                    st.write("### 答案：")
                    answer_placeholder = st.empty()
                answer_placeholder.markdown(accumulated_text)

            result = engine.run_query(
                query=query,
                module=st.session_state.selected_module,
                max_retries=int(max_retries),
                on_status=render_status_message,
                on_stream=on_stream,
            )

            if result.sql:
                st.code(result.sql, language="sql")

            if not result.is_success:
                st.error(f"查询未返回有效结果：{result.error or '未知错误'}")
            else:
                st.success(f"查询成功，返回 {len(result.rows)} 条记录")
                st.subheader("查询结果数据")
                st.dataframe(pd.DataFrame(result.rows), use_container_width=True)

                for chart_meta, fig in build_figures(result.chart_config):
                    st.subheader(chart_meta.get("title", "数据可视化"))
                    if chart_meta.get("description"):
                        st.write(chart_meta["description"])
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("调试信息（Prompts 与执行记录）", expanded=False):
                st.write("执行尝试记录:", [vars(a) for a in result.attempts])
                if result.sql_prompts:
                    st.write("SQL Prompts:", result.sql_prompts)
                if result.answer_prompts:
                    st.write("Answer Prompts:", result.answer_prompts)


if __name__ == "__main__":
    main()
