import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class LLMProviderPreset:
    """主流 OpenAI 兼容模型服务商预设。"""

    name: str
    base_url: str
    default_model: str
    models: List[str] = field(default_factory=list)
    requires_api_key: bool = True


LLM_PROVIDERS: Dict[str, LLMProviderPreset] = {
    "DeepSeek": LLMProviderPreset(
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-flash",
        models=["deepseek-flash", "deepseek-v4-pro"],
    ),
    "智谱 AI (BigModel)": LLMProviderPreset(
        name="智谱 AI (BigModel)",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        default_model="glm-5.3-flash",
        models=["glm-5.3", "glm-5.3-flash", "glm-4.5-flash", "glm-4-plus"],
    ),
    "OpenAI": LLMProviderPreset(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-6.1-sol",
        models=["gpt-6.1-sol", "gpt-6-astra", "gpt-6-luna", "gpt-5-mini", "gpt-4o"],
    ),
    "阿里云百炼 (Qwen)": LLMProviderPreset(
        name="阿里云百炼 (Qwen)",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        models=["qwen-plus", "qwen3.7-plus", "qwen3.8-flash"],
    ),
    "Google Gemini": LLMProviderPreset(
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-3-flash",
        models=["gemini-3-flash", "gemini-3.5-flash"],
    ),
    "OpenRouter": LLMProviderPreset(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="anthropic/claude-sonnet-5",
        models=[
            "anthropic/claude-sonnet-5",
            "openai/gpt-6.1-sol",
            "deepseek/deepseek-v4-pro",
        ],
    ),
    "Ollama (本地)": LLMProviderPreset(
        name="Ollama (本地)",
        base_url="http://localhost:11434/v1",
        default_model="qwen3:8b",
        models=["qwen3:8b", "deepseek-r1:8b", "llama3.3:8b"],
        requires_api_key=False,
    ),
    "自定义 (Custom)": LLMProviderPreset(
        name="自定义 (Custom)",
        base_url="https://api.openai.com/v1",
        default_model="",
        models=[],
    ),
}


@dataclass
class DatabaseConfig:
    """MySQL 数据库连接配置。"""

    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = ""

    def to_mysql_kwargs(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": int(self.port),
            "user": self.user,
            "password": self.password,
            "database": self.database,
        }


@dataclass
class LLMConfig:
    """OpenAI 兼容大语言模型配置。"""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-flash"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量读取配置，按 SMARTDB_* > LLM_* > OPENAI_* 的优先级取值。"""

        def _first(*names: str) -> str:
            for name in names:
                value = os.getenv(name, "").strip()
                if value:
                    return value
            return ""

        defaults = cls()
        return cls(
            api_key=_first("SMARTDB_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"),
            base_url=_first("SMARTDB_BASE_URL", "LLM_BASE_URL", "OPENAI_BASE_URL")
            or defaults.base_url,
            model=_first("SMARTDB_MODEL", "LLM_MODEL") or defaults.model,
        )


@dataclass
class EmbeddingConfig:
    """嵌入向量服务配置，支持 OpenAI 兼容 /v1/embeddings 与本地 Ollama。"""

    base_url: str = "http://localhost:11434/v1"
    model: str = "nomic-embed-text"
    api_key: str = ""
    mode: str = "openai_compatible"  # "openai_compatible" | "ollama_legacy"
    timeout_seconds: float = 5.0


@dataclass
class CacheConfig:
    """语义查询缓存配置。"""

    db_path: str = "semantic_cache.db"
    similarity_threshold: float = 0.85
    max_entries: int = 1000
    expiry_days: int = 7
    sys_tables_db_path: str = "./sys_tables.db"
