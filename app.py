import streamlit as st

from smartdbchat.cache import SemanticQueryCache
from smartdbchat.config import CacheConfig, EmbeddingConfig
from ui import apply_custom_styles, render_chat_interface, render_sidebar


def init_session_state() -> None:
    """初始化 Streamlit session state。"""
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "cache_system" not in st.session_state:
        st.session_state.cache_system = SemanticQueryCache(
            CacheConfig(), EmbeddingConfig()
        )
    if "similarity_threshold" not in st.session_state:
        st.session_state.similarity_threshold = 0.85
    if "selected_module" not in st.session_state:
        st.session_state.selected_module = None
    if "max_retries" not in st.session_state:
        st.session_state.max_retries = 2
    if "query_mode" not in st.session_state:
        st.session_state.query_mode = "agent"
    if "agent_max_steps" not in st.session_state:
        st.session_state.agent_max_steps = 8
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "connection_meta" not in st.session_state:
        st.session_state.connection_meta = {}


def main() -> None:
    st.set_page_config(
        page_title="SmartDBChat - 自然语言数据库查询",
        page_icon="💬",
        layout="wide",
    )
    apply_custom_styles()
    init_session_state()

    st.title("💬 SmartDBChat")
    st.caption("基于大语言模型的自然语言数据库查询、流式洞察与交互式可视化助手")

    render_sidebar()
    render_chat_interface(st.session_state.engine)


if __name__ == "__main__":
    main()
