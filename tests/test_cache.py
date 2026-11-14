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
