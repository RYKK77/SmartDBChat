# SmartDBChat - 自然语言数据库查询与可视化系统

SmartDBChat 是一个基于大语言模型（LLM）的自然语言数据库查询与分析工具（NL2SQL）。用户无需编写 SQL，只需用自然语言提问，系统即可自动生成安全的只读 SQL、执行查询、纠错重试、流式解读数据洞察，并自动生成交互式 Plotly 可视化图表。

## 🌟 核心功能

- **自然语言转 SQL（NL2SQL）与图谱 Schema Linking**：借鉴 QueryWeaver 图谱理解思路，构建表-列-外键 `SchemaGraph`，自动发现跨表最短 JOIN 路径与桥接表。
- **sqlglot AST 只读安全与 Schema 校验**：基于 `sqlglot` AST 解析拦截写操作、多语句注入、高危系统函数与 `INTO OUTFILE`，支持表/列存在性校验与自动 `LIMIT` 钳位。
- **轻量业务语义层（Semantic Layer）**：支持通过 YAML 配置数据库描述、全局业务规则（`user_rules`）、指标公式、业务术语、字段枚举说明与 Few-Shot Golden SQL 示例。
- **Preflight 语义预检与有界纠错/恢复**：在生成 SQL 前拦截越权/闲聊并支持澄清反问；执行失败时按错误类型分类实施有界纠错（防重复死循环），耗尽后输出可解释 Recovery 诊断与推荐追问。
- **列值探索与 Tool-Use Agent**：Agent 内置 `search_column_values`（关键字模糊探测列值）与 `get_join_path`（图谱关联路径查询）等工具，解决筛选值不匹配与多表关联难题。
- **Test-Time Scaling 高精度模式**：支持多策略生成候选 SQL、AST 校验与只读试跑，并基于执行结果签名进行多数投票（Majority Vote）择优。
- **MCP Server 开放集成**：内置标准 MCP Server（`python -m smartdbchat.mcp_server`），对外暴露 `list_tables`、`get_database_schema`、`find_join_path`、`sample_column_values`、`validate_sql` 与 `ask_database` 工具。
- **MCP 服务中心与 API Key 鉴权**：前端提供 MCP 服务中心页面，可查看工具目录、一键复制客户端接入配置、创建/启停/删除 MCP API Key（`sdb_mcp_` 前缀），并在在线调试台中直接调用 MCP 工具查看耗时与返回结果；可开启「强制 API Key 校验」限制 `/api/mcp/call` 的访问。
- **用户账号与服务端持久化**：支持注册/登录（密码加盐哈希 + Bearer Token），首个注册用户自动成为管理员；数据源连接配置、对话会话与偏好设置按用户保存在服务端 SQLite（`smartdb_app.db`），换浏览器登录也能恢复。
- **离线评测基线**：基于内置高校教务 SQLite 沙箱与 `eval_data/golden_cases.json` 黄金用例，统计执行准确率、安全拦截率等指标并输出 JSON / Markdown 报告。
- **流式结果解读与多视图分析工作台**：支持 **React + TypeScript + FastAPI (SSE)** 现代化前后端分离架构，提供分屏 `@xyflow/react` Schema 关系图谱与查询命中表高亮、Recharts 交互式 Chart Studio（支持智能推荐与图表类型/XY 轴动态切换）、可检索/排序/导出的数据表格、可编辑并受 AST 保护的 SQL 工作台（支持一键收藏 Golden SQL）、可视化 + YAML 语义层编辑器，以及追问建议芯片。

## ✨ 支持的模型与接口

系统基于 OpenAI 兼容协议，侧边栏/连接弹窗内置主流服务商预设（也支持任意自定义端点）：

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
├── frontend/                # React 19 + TypeScript + Vite 现代化前端工作台
│   ├── src/
│   │   ├── components/
│   │   │   ├── AuthModal.tsx           # 登录 / 注册弹窗
│   │   │   ├── ChartStudio.tsx         # KPI 指标条 + Recharts 多图表动态切换工作室
│   │   │   ├── ConnectionModal.tsx     # 沙箱/MySQL 数据源、模型预设与高级护栏配置弹窗
│   │   │   ├── DataTableView.tsx       # 搜索过滤、列排序、分页与 CSV 导出表格
│   │   │   ├── McpStudio.tsx           # MCP 服务中心：工具目录、接入配置、API Key 管理与在线调试台
│   │   │   ├── MessageCard.tsx         # 4 标签分析卡片 + SQL 工作台 + Golden SQL 收藏
│   │   │   ├── SchemaGraphCanvas.tsx   # @xyflow/react ER 关系图谱、采样、列值探索与 JOIN 路径
│   │   │   ├── SemanticLayerStudio.tsx # 业务语义层可视化与 YAML 双向编辑器
│   │   │   └── Sidebar.tsx             # 引擎热更新参数栏、缓存统计、会话列表与快速表目录
│   │   ├── services/api.ts             # REST & SSE 流式客户端（自动携带 Bearer Token）
│   │   ├── types/api.ts                # TypeScript 接口定义
│   │   └── utils/storage.ts            # 本地 Token / 会话缓存工具
├── smartdbchat/
│   ├── server.py            # FastAPI REST + SSE 后端服务（鉴权中间件、MCP 调试接口，支持托管 frontend/dist）
│   ├── store.py             # 用户、Token、连接配置、对话会话与 MCP API Key 的 SQLite 持久化
│   ├── config.py            # 数据库、LLM 预设、Embedding 与缓存配置
│   ├── models.py            # 结构化查询结果、工作流追踪与缓存命中数据模型
│   ├── prompts.py           # SQL 生成、纠错重试与结果分析提示词
│   ├── db.py                # MySQL 连接、Schema 提取与列值探索
│   ├── sql_guard.py         # sqlglot AST 只读安全、Schema 一致性校验与 LIMIT 钳位
│   ├── schema_linker.py     # SchemaGraph 图谱构建、最短 JOIN 路径与表/列裁剪
│   ├── semantic_layer.py    # YAML 语义层（指标/术语/user_rules/枚举/Few-Shot）
│   ├── workflow.py          # Preflight 预检、错误分类、有界纠错、Recovery 与追问建议
│   ├── scaling.py           # Test-Time Scaling 多候选生成与执行一致性投票
│   ├── agent.py             # Tool-Use NL2SQL Agent（含列值搜索与图谱路径工具）
│   ├── mcp_server.py        # 标准 MCP Server 暴露（stdio / streamable-http / sse）
│   ├── llm.py               # OpenAI 兼容客户端封装与流式响应处理
│   ├── cache.py             # SQLite 多级语义查询缓存（无 UI 耦合）
│   ├── charts.py            # 图表 JSON 解析、确定性图表推荐与 Plotly Figure 构建器
│   ├── engine.py            # SmartDBEngine 核心查询编排引擎
│   └── eval/
│       ├── sandbox.py       # 内置高校教务 SQLite 演示 / 评测沙箱
│       └── runner.py        # 黄金用例评测执行器与报告生成
├── eval_data/
│   └── golden_cases.json    # 评测黄金用例
├── ui/                      # Streamlit 界面组件（兼容保留）
├── app.py                   # 传统 Streamlit 前端入口（兼容保留）
├── tests/                   # 单元测试与端到端集成测试套件
├── pyproject.toml           # uv 项目与依赖配置
├── uv.lock                  # uv 依赖锁文件
└── requirements.txt         # 兼容传统 pip 的依赖列表
```

## 📋 环境要求

- Python 3.10+ & [uv](https://docs.astral.sh/uv/)
- Node.js 20+ & npm（用于构建或开发 `frontend/`）
- MySQL 数据库（或直接使用内置高校教务 SQLite 演示沙箱）
- 任意 OpenAI 兼容大模型 API 密钥（或本地 Ollama 服务）

## 🔧 安装与运行

1. 同步 Python 依赖：

```bash
uv sync
```

2. 构建 React 前端生产包并启动单端口一体化服务（FastAPI 同时提供 API 与前端页面）：

```bash
cd frontend && npm install && npm run build && cd ..
uv run smartdbchat-server
# 默认访问 http://127.0.0.1:8000
# 可通过 SMARTDB_SERVER_HOST / SMARTDB_SERVER_PORT 修改监听地址与端口
```

> 首次打开页面时注册的第一个账号会成为管理员。系统中一旦存在用户，所有 `/api/*` 接口（健康检查、模型预设与登录注册除外）都需要登录后携带 Token 访问。

3. 前后端分离开发模式（热重载）：

```bash
# 终端 1：启动 FastAPI 后端
uv run uvicorn smartdbchat.server:app --reload --port 8000

# 终端 2：启动 Vite 前端（自动代理 /api 到 8000 端口）
cd frontend && npm run dev
```

4. 启动传统 Streamlit 界面（兼容保留）：

```bash
uv run streamlit run app.py
```

5. 独立启动 MCP Server（供 Cursor / Claude Desktop 等 MCP 客户端接入）：

```bash
# 使用内置沙箱
SMARTDB_DB_TYPE=sandbox SMARTDB_API_KEY=sk-xxx uv run python -m smartdbchat.mcp_server

# 连接 MySQL，并切换为 HTTP 传输（可选 stdio / streamable-http / sse，默认 stdio）
SMARTDB_HOST=localhost SMARTDB_USER=root SMARTDB_PASSWORD=*** SMARTDB_DATABASE=mydb \
SMARTDB_API_KEY=sk-xxx uv run python -m smartdbchat.mcp_server --transport streamable-http
```

| 环境变量 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `SMARTDB_DB_TYPE` | `mysql` 或 `sandbox` | `mysql` |
| `SMARTDB_HOST` / `SMARTDB_PORT` / `SMARTDB_USER` / `SMARTDB_PASSWORD` / `SMARTDB_DATABASE` | MySQL 连接信息 | `localhost` / `3306` / `root` / 空 / 空 |
| `SMARTDB_API_KEY` | LLM API Key（未设置时读取 `OPENAI_API_KEY`） | 空 |
| `SMARTDB_BASE_URL` / `SMARTDB_MODEL` | LLM 端点与模型 | `https://api.deepseek.com/v1` / `deepseek-flash` |
| `SMARTDB_SEMANTIC_YAML` | 语义层 YAML 文件路径 | 空 |
| `SMARTDB_CACHE_DB` | 语义缓存 SQLite 路径 | `semantic_cache.db` |
| `SMARTDB_MAX_LIMIT` | 查询结果行数上限 | `1000` |

在 Cursor 中接入的 `mcp.json` 示例（前端「MCP 服务中心」页面也可一键复制按本机 Python 解释器与项目路径生成的配置）：

```json
{
  "mcpServers": {
    "smartdbchat": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/SmartDBChat", "python", "-m", "smartdbchat.mcp_server"],
      "env": { "SMARTDB_DB_TYPE": "sandbox", "SMARTDB_API_KEY": "sk-xxx" }
    }
  }
}
```

6. 运行离线评测（基于内置沙箱，无需外部数据库）：

```bash
# reference：用黄金 SQL 自评，验证评测链路；mock：使用模拟引擎
uv run python -m smartdbchat.eval.runner --mode reference \
  --output-json reports/eval.json --output-md reports/eval.md

# openai：调用真实大模型生成 SQL 并评测（API Key 按 SMARTDB_* > LLM_* > OPENAI_* 优先级读取）
SMARTDB_API_KEY=sk-xxx SMARTDB_BASE_URL=https://api.deepseek.com/v1 SMARTDB_MODEL=deepseek-flash \
uv run python -m smartdbchat.eval.runner --mode openai
```

7. 运行完整测试套件：

```bash
uv run pytest -v
```

## 🚀 使用指南

1. **登录账号**：首次访问时注册账号（第一个账号为管理员），之后的连接配置与对话记录会自动保存在服务端。
2. **配置连接**：点击右上角 **连接数据源**，可一键启用内置高校教务 SQLite 沙箱或填写 MySQL 连接信息；选择模型服务商预设并填入 API Key（本地 Ollama 可免密）。
3. **探索 Schema 与语义层**：在右侧分屏图谱或顶部 **Schema 图谱** 页面查看表间关系、探测列值与最短 JOIN 路径；在 **语义层中心** 维护指标口径与业务规则。
4. **对话分析与调优**：输入自然语言问题，查看流式 AI 解读、交互式 Chart Studio、可排序导出的数据表，或在 **SQL 工作台** 微调 SQL 后重跑并一键收藏为 Golden SQL。
5. **对外开放 MCP**：在 **MCP 服务中心** 复制客户端接入配置、创建 API Key，并在在线调试台中试调各个 MCP 工具。

## 📜 许可证

[MIT](LICENSE)