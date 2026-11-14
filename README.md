# SmartDBChat - 自然语言数据库查询与可视化系统

SmartDBChat 是一个基于大语言模型（LLM）的自然语言数据库查询与分析工具（NL2SQL）。用户无需编写 SQL，只需用自然语言提问，系统即可自动生成安全的只读 SQL、执行查询、纠错重试、流式解读数据洞察，并自动生成交互式 Plotly 可视化图表。

## 🌟 核心功能

- **自然语言转 SQL（NL2SQL）**：结合数据库 `SHOW CREATE TABLE` Schema 信息生成准确的 MySQL 查询语句，支持按业务模块过滤表范围。
- **只读 SQL 安全校验**：内置语句校验，仅允许 `SELECT` / `WITH` / `SHOW` / `DESCRIBE` / `EXPLAIN` 等只读查询，拦截写操作及多语句注入。
- **自动错误处理与重试**：当 SQL 执行报错或返回空结果时，自动携带失败 SQL 与错误详情重新生成并重试。
- **流式结果解读**：通过流式接口逐步输出专业数据分析、趋势与异常洞察。
- **智能多图表可视化**：支持饼图、柱状图、折线图（含多系列）、散点图、面积图、热力图、雷达图、箱线图、漏斗图、瀑布图、树状图等 Plotly 交互式图表。
- **多级语义查询缓存**：基于 SQLite 存储历史查询，支持精确哈希匹配、嵌入向量余弦相似度、TF-IDF 与关键词匹配兜底，并自动执行缓存容量上限管理。

## ✨ 支持的模型与接口

系统基于 OpenAI 兼容协议，侧边栏内置主流服务商预设（也支持任意自定义端点）：

| 服务商 | Base URL | 推荐模型示例 |
| :--- | :--- | :--- |
| **DeepSeek** | `https://api.deepseek.com/v1` | `deepseek-flash`、`deepseek-v4-pro` |
| **智谱 AI (BigModel)** | `https://open.bigmodel.cn/api/paas/v4/` | `glm-5.3`、`glm-5.3-flash`、`glm-4.5-flash`、`glm-4-plus` |
| **OpenAI** | `https://api.openai.com/v1` | `gpt-6.1-sol`、`gpt-6-astra`、`gpt-5-mini`、`gpt-4o` |
| **阿里云百炼 (Qwen)** | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus`、`qwen3.7-plus`、`qwen3.8-flash` |
| **Google Gemini** | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-3-flash`、`gemini-3.5-flash` |
| **OpenRouter** | `https://openrouter.ai/api/v1` | `anthropic/claude-sonnet-5`、`openai/gpt-6.1-sol` |
| **Ollama (本地)** | `http://localhost:11434/v1` | `qwen3:8b`、`deepseek-r1:8b`（支持免 API Key） |

语义缓存的嵌入向量服务同时支持 **OpenAI 兼容 `/embeddings`** 与 **Ollama 旧版 `/api/embeddings`** 两种协议；当向量服务不可用时会自动降级到文本/关键词匹配。

## 🧱 项目架构

```text
SmartDBChat/
├── app.py                   # Streamlit 前端入口与交互界面
├── smartdbchat/
│   ├── config.py            # 数据库、LLM 预设、Embedding 与缓存配置
│   ├── models.py            # 结构化查询结果与缓存命中数据模型
│   ├── prompts.py           # SQL 生成、纠错重试与结果分析提示词
│   ├── db.py                # MySQL 连接、只读 SQL 安全校验与 Schema 提取
│   ├── llm.py               # OpenAI 兼容客户端封装与流式响应处理
│   ├── cache.py             # SQLite 多级语义查询缓存（无 UI 耦合）
│   ├── charts.py            # 图表 JSON 解析与 Plotly Figure 构建器
│   └── engine.py            # SmartDBEngine 核心查询编排引擎
├── tests/                   # 单元测试套件
└── requirements.txt
```

## 📋 环境要求

- Python 3.10+
- MySQL 数据库
- 任意 OpenAI 兼容大模型 API 密钥（或本地 Ollama 服务）

## 🔧 安装与运行

1. 克隆仓库并安装依赖：

```bash
pip install -r requirements.txt
```

2. 启动 Streamlit 应用：

```bash
streamlit run app.py
```

3. 运行单元测试（可选）：

```bash
pytest -v
```

## 🚀 使用指南

1. **配置连接**：在左侧边栏填写 MySQL 连接信息，选择模型服务商预设（或自定义 Base URL 与模型名称）并填入 API Key，点击 **连接数据库**。
2. **输入问题**：在主界面输入自然语言问题，例如：
   - `"查询计算机科学专业平均分最高的前 5 名学生"`
   - `"统计各个学院的课程数量并按照数量降序排列"`
   - `"分析不同年级学生的选课情况"`
3. **查看结果**：系统会展示生成的 SQL、原始数据表、流式 AI 分析结论以及自动匹配的交互式图表。

## 📜 许可证

[MIT](LICENSE)