# SmartDBChat - 自然语言数据库查询系统

SmartDBChat 是一个基于大语言模型的自然语言数据库查询工具，它允许用户使用自然语言而非 SQL 来查询数据库，大大降低了数据分析的门槛。系统会自动将自然语言转换为 SQL 查询，执行查询并以易于理解的方式展示结果。

## 🌟 核心功能

- **自然语言转 SQL**：将用户的问题自动转换为准确的 SQL 查询语句
- **智能结果解析**：将查询结果转换为易懂的自然语言描述和可视化图表
- **语义查询缓存**：基于向量相似度匹配历史查询，提高响应速度
- **自动错误处理**：智能检测并修复失败的查询，支持多次重试
- **流式响应输出**：使用流式 API 逐步显示查询结果
- **数据可视化**：自动生成适合的图表（饼图、柱状图等）展示查询结果

## ✨ 技术特点

- **统一 LLM 接口**：基于 OpenAI 接口标准，支持多种大语言模型
- **模型灵活配置**：通过简单配置 `base_url` 和 `model` 参数即可切换不同 AI 服务提供商
- **多级语义匹配算法**：支持嵌入向量相似度、TF-IDF 和关键词匹配等多种相似度算法
- **缓存优化**：自动管理缓存大小，优先保留常用查询
- **完善的错误处理**：详细的错误反馈和智能重试机制
- **友好的用户界面**：基于 Streamlit 的简洁直观界面

## 📋 前提条件

- Python 3.8+
- MySQL 数据库
- 支持 OpenAI 接口的 API 密钥

## 🔧 安装步骤

1. 克隆仓库或下载源代码

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
```bash
streamlit run app.py
```

## 🚀 使用指南

### 1. 配置连接

在侧边栏中填写以下信息：
- 数据库设置（主机、端口、用户名、密码、数据库名）
- LLM 设置：
  - API 密钥
  - 基础 URL（默认为 OpenAI，可更改为其他兼容服务）
  - 模型名称（如 gpt-4-turbo, glm-4-plus 等）
- 缓存相似度阈值（可选）

点击"连接数据库"按钮建立连接。

### 2. 提问查询

在主界面的文本框中输入自然语言问题，例如：
- "查询所有学生的姓名和平均成绩"
- "哪个专业的学生人数最多？"
- "统计各个年级的学生性别比例"

设置最大重试次数（如果需要），然后点击"查询"按钮。

### 3. 查看结果

系统会显示：
- 生成的 SQL 查询语句
- 自然语言回答
- 相关的图表（如果适用）
- 可展开的原始数据视图

### 4. 缓存管理

在侧边栏可以：
- 查看缓存统计信息
- 调整相似度阈值
- 清空缓存

## 📝 示例查询

```
查询计算机科学专业平均分最高的前 5 名学生
```

```
统计各个学院的课程数量并按照数量降序排列
```

```
分析不同年级学生的选课情况
```

## 🔍 主要特性详解

### 模型配置灵活性

系统使用统一的 OpenAI 接口标准，支持多种模型：

- **OpenAI**：设置 base_url 和 gpt-4 系列模型
- **智谱 AI**：使用**默认** base_url 为智谱 AI 接口地址，model 为 glm-4-plus
- **Azure OpenAI**：设置 base_url 为 Azure 部署端点
- **其他兼容服务**：任何兼容 OpenAI 接口的服务均可通过配置接入

### 语义查询缓存

系统会存储之前执行过的查询及其 SQL，当用户提出语义相似的问题时，可直接使用缓存的 SQL 而无需重新生成，显著提升响应速度。缓存系统支持：

- 嵌入向量相似度匹配（首选）
- TF-IDF 文本相似度计算（备用）
- 关键词匹配（最后备选）

### 自动错误处理与重试

当生成的 SQL 执行失败或返回空结果时，系统会：

1. 分析错误原因
2. 重新生成更准确的 SQL
3. 最多重试用户设定的次数
4. 提供详细的错误和重试信息

### 智能可视化

系统会根据查询结果的数据特点，自动决定适合的图表类型：

- 饼图：适合展示占比关系
- 柱状图：适合数据对比关系
- 表格：适合详细数据展示

## 📊 技术架构

- **前端**：Streamlit
- **后端处理**：Python
- **数据库连接**：MySQL Connector
- **LLM 接口**：OpenAI 兼容接口
- **数据可视化**：Plotly
- **缓存系统**：SQLite + 向量相似度

## 🔧 模型配置示例

```python
# OpenAI 配置
llm_config = {
    "api_key": "your-openai-api-key",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4-turbo"
}

# 智谱 AI 配置
llm_config = {
    "api_key": "your-zhipu-api-key",
    "base_url": "https://open.bigmodel.cn/api/paas/v4",
    "model": "glm-4-plus"
}

# Azure OpenAI 配置
llm_config = {
    "api_key": "your-azure-api-key",
    "base_url": "https://your-resource-name.openai.azure.com/openai/deployments/your-deployment-name",
    "model": "gpt-4"
}
```

## ⚠️ 注意事项

- 首次查询可能需要较长时间，因为系统需要生成 SQL
- 查询结果的质量取决于数据库结构和模型对结构的理解
- 确保 API 密钥有足够的额度用于自然语言处理
- 缓存系统默认存储最多 1000 条查询记录
- 不同模型的能力和特点可能导致 SQL 生成质量有所差异

## 📜 许可证

[MIT](LICENSE)

## 🤝 贡献

欢迎提交 issues 和 pull requests，帮助改进这个项目！

---

希望 SmartDBChat 能够帮助您更便捷地探索和分析数据！