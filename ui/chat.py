import time
from typing import Any, Dict

import pandas as pd
import streamlit as st

from smartdbchat.charts import build_figures, strip_chart_json_block
from smartdbchat.engine import SmartDBEngine
from smartdbchat.models import QueryResult

SAMPLE_QUESTIONS = [
    "查看当前数据库有哪些表以及各表的大致数据量",
    "统计核心维度的数据分布并生成可视化图表",
    "查询排名前 10 的记录并分析主要特征",
]


def _render_assistant_payload(
    result: QueryResult,
    elapsed_sec: float,
    msg_index: int,
) -> None:
    """渲染单条助手回复：分析与图表、原始数据表+CSV导出、SQL与调试详情。"""
    if not result.is_success:
        st.error(f"查询未返回有效结果：{result.error or '未知错误'}")
        if result.sql:
            st.code(result.sql, language="sql")
        if result.attempts:
            with st.expander("执行与重试记录", expanded=False):
                st.write([vars(a) for a in result.attempts])
        return

    badges = [f"返回 {len(result.rows)} 条记录", f"耗时 {elapsed_sec:.2f}s"]
    if result.cache_hit:
        badges.append(
            f"缓存命中 ({result.cache_hit.match_type}, 相似度 {result.cache_hit.similarity:.2f})"
        )
    st.caption(" · ".join(badges))

    tab_analysis, tab_data, tab_sql = st.tabs(
        ["📊 分析与图表", "📋 数据表格", "🧾 SQL 与调试"]
    )

    with tab_analysis:
        clean_md = strip_chart_json_block(result.markdown_answer)
        if clean_md:
            st.markdown(clean_md)
        figures = build_figures(result.chart_config)
        for chart_meta, fig in figures:
            st.subheader(chart_meta.get("title", "数据可视化"))
            if chart_meta.get("description"):
                st.caption(chart_meta["description"])
            st.plotly_chart(fig, width="stretch", key=f"chart_{msg_index}_{id(fig)}")

    with tab_data:
        df = pd.DataFrame(result.rows)
        st.dataframe(df, width="stretch")
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ 导出 CSV",
            data=csv_bytes,
            file_name="smartdbchat_result.csv",
            mime="text/csv",
            key=f"dl_{msg_index}",
        )

    with tab_sql:
        st.code(result.sql, language="sql")
        st.write("执行尝试记录:", [vars(a) for a in result.attempts])
        if result.sql_prompts:
            with st.expander("SQL 生成 Prompts", expanded=False):
                st.write(result.sql_prompts)
        if result.answer_prompts:
            with st.expander("分析与图表 Prompts", expanded=False):
                st.write(result.answer_prompts)


def _execute_and_append(engine: SmartDBEngine, prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_container = st.status("正在分析问题并生成 SQL...", expanded=False)
        stream_placeholder = st.empty()

        def on_status(level: str, message: str) -> None:
            status_container.write(f"[{level.upper()}] {message}")

        def on_stream(accumulated: str) -> None:
            stream_placeholder.markdown(strip_chart_json_block(accumulated))

        start = time.perf_counter()
        result = engine.run_query(
            query=prompt,
            module=st.session_state.selected_module,
            max_retries=int(st.session_state.get("max_retries", 2)),
            on_status=on_status,
            on_stream=on_stream,
        )
        elapsed = time.perf_counter() - start
        status_container.update(
            label="查询处理完成" if result.is_success else "查询未成功",
            state="complete" if result.is_success else "error",
            expanded=False,
        )
        stream_placeholder.empty()

        msg_index = len(st.session_state.messages)
        _render_assistant_payload(result, elapsed, msg_index)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "result": result,
            "elapsed_sec": elapsed,
        }
    )


def render_chat_interface(engine: SmartDBEngine | None) -> None:
    """渲染主聊天界面：状态栏、欢迎示例、多轮对话历史与底部输入框。"""
    if not engine:
        st.info("👈 请先在左侧边栏配置数据库连接与大模型 API 密钥")
        return

    meta: Dict[str, Any] = st.session_state.get("connection_meta", {})
    module_label = st.session_state.selected_module or "全部表"

    col_banner, col_clear = st.columns([6, 1])
    with col_banner:
        st.markdown(
            f"""
            <div class="smartdb-banner">
                <div>
                    <span class="smartdb-pill smartdb-pill-ok">● 已连接 {meta.get('database', '')}</span>
                    <span class="smartdb-pill">模型: {meta.get('model', '')}</span>
                    <span class="smartdb-pill">范围: {module_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_clear:
        if st.button("🗑️ 清空对话", width="stretch"):
            st.session_state.messages = []
            st.rerun()

    if not st.session_state.messages:
        st.markdown("#### 💡 试试用自然语言提问")
        cols = st.columns(len(SAMPLE_QUESTIONS))
        for idx, sample in enumerate(SAMPLE_QUESTIONS):
            if cols[idx].button(sample, key=f"sample_{idx}", width="stretch"):
                _execute_and_append(engine, sample)
                st.rerun()

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                _render_assistant_payload(
                    msg["result"],
                    msg.get("elapsed_sec", 0.0),
                    idx,
                )

    if prompt := st.chat_input("输入你的数据问题，例如：统计各专业学生平均分..."):
        _execute_and_append(engine, prompt)
