from dataclasses import dataclass
from typing import Any, Dict


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
    base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    model: str = "glm-4-plus"


@dataclass
class EmbeddingConfig:
    """本地或远程嵌入向量服务配置（默认 Ollama）。"""

    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    timeout_seconds: float = 5.0


@dataclass
class CacheConfig:
    """语义查询缓存配置。"""

    db_path: str = "semantic_cache.db"
    similarity_threshold: float = 0.85
    max_entries: int = 1000
    expiry_days: int = 7
    sys_tables_db_path: str = "./sys_tables.db"
