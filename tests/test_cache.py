from smartdbchat.cache import SemanticQueryCache
from smartdbchat.config import CacheConfig, EmbeddingConfig


def test_cache_exact_match_stats_and_clear(tmp_path):
    db_file = str(tmp_path / "cache.db")
    cache = SemanticQueryCache(
        config=CacheConfig(db_path=db_file, similarity_threshold=0.5, max_entries=3),
        embedding_config=EmbeddingConfig(base_url="http://127.0.0.1:9999", timeout_seconds=0.1),
    )
    cache.embedding_available = False

    assert cache.get_from_cache("查询所有学生", "school") is None

    cache.save_to_cache("查询所有学生", "school", "SELECT * FROM students")
    hit = cache.get_from_cache("查询所有学生", "school")
    assert hit is not None
    assert hit.sql == "SELECT * FROM students"
    assert hit.match_type == "exact"

    stats = cache.get_cache_stats()
    assert stats.total_entries == 1
    assert stats.total_hits == 2

    # 触发 max_entries=3 淘汰
    cache.save_to_cache("q2", "school", "SELECT 2")
    cache.save_to_cache("q3", "school", "SELECT 3")
    cache.save_to_cache("q4", "school", "SELECT 4")
    assert cache.get_cache_stats().total_entries == 3

    cache.clear_cache()
    assert cache.get_cache_stats().total_entries == 0


def test_cache_keyword_fallback(tmp_path):
    db_file = str(tmp_path / "cache.db")
    cache = SemanticQueryCache(
        config=CacheConfig(db_path=db_file, similarity_threshold=0.4),
    )
    cache.embedding_available = False
    cache.save_to_cache("students average score report", "school", "SELECT AVG(score) FROM students")

    hit = cache.get_from_cache("students average score", "school")
    assert hit is not None
    assert hit.sql == "SELECT AVG(score) FROM students"
    assert hit.match_type in ("tfidf", "keyword")


def test_embedding_supports_openai_compatible_and_ollama_legacy(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from smartdbchat.config import LLM_PROVIDERS

    assert "DeepSeek" in LLM_PROVIDERS
    assert LLM_PROVIDERS["DeepSeek"].base_url == "https://api.deepseek.com/v1"

    mock_post = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    mock_post.return_value = mock_resp
    monkeypatch.setattr("smartdbchat.cache.requests.post", mock_post)

    cache = SemanticQueryCache(
        config=CacheConfig(db_path=str(tmp_path / "emb1.db")),
        embedding_config=EmbeddingConfig(
            base_url="https://api.openai.com/v1",
            model="text-embedding-3-small",
            api_key="sk-test",
            mode="openai_compatible",
        ),
    )
    vec = cache._get_embedding("hello")
    assert vec == [0.1, 0.2, 0.3]
    called_url = mock_post.call_args[0][0]
    assert called_url == "https://api.openai.com/v1/embeddings"

    mock_resp.json.return_value = {"embedding": [0.4, 0.5]}
    cache.embedding_config = EmbeddingConfig(
        base_url="http://localhost:11434",
        model="nomic-embed-text",
        mode="ollama_legacy",
    )
    vec2 = cache._get_embedding("hello")
    assert vec2 == [0.4, 0.5]
    assert mock_post.call_args[0][0] == "http://localhost:11434/api/embeddings"

