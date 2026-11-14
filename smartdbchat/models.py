from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class QueryStatus(str, Enum):
    SUCCESS = "success"
    EMPTY_RESULT = "empty_result"
    INVALID_SQL = "invalid_sql"
    ERROR = "error"


@dataclass
class CacheHit:
    sql: str
    matched_query: str
    similarity: float
    match_type: str  # "exact", "embedding", "tfidf", "keyword"


@dataclass
class CacheStats:
    total_entries: int
    total_hits: int
    top_queries: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class ExecutionAttempt:
    attempt: int
    sql: str
    row_count: int = 0
    error: Optional[str] = None


@dataclass
class QueryResult:
    status: QueryStatus
    sql: str = ""
    rows: List[Dict[str, Any]] = field(default_factory=list)
    markdown_answer: str = ""
    chart_config: Optional[Dict[str, Any]] = None
    cache_hit: Optional[CacheHit] = None
    attempts: List[ExecutionAttempt] = field(default_factory=list)
    error: Optional[str] = None
    sql_prompts: List[List[Dict[str, str]]] = field(default_factory=list)
    answer_prompts: Optional[List[Dict[str, str]]] = None

    @property
    def is_success(self) -> bool:
        return self.status == QueryStatus.SUCCESS and len(self.rows) > 0


StatusCallback = Callable[[str, str], None]
StreamCallback = Callable[[str], None]
