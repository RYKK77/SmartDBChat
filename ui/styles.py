import streamlit as st


def apply_custom_styles() -> None:
    """注入轻量级自定义样式，优化聊天气泡、状态标签与卡片间距。"""
    st.markdown(
        """
        <style>
        .smartdb-banner {
            padding: 0.75rem 1rem;
            border-radius: 0.6rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background: rgba(128, 128, 128, 0.06);
            margin-bottom: 1rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            align-items: center;
            justify-content: space-between;
        }
        .smartdb-pill {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 500;
            border: 1px solid rgba(128, 128, 128, 0.3);
            background: rgba(128, 128, 128, 0.1);
        }
        .smartdb-pill-ok {
            border-color: rgba(34, 197, 94, 0.45);
            background: rgba(34, 197, 94, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
