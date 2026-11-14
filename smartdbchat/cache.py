import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import requests

from smartdbchat.config import CacheConfig, EmbeddingConfig
from smartdbchat.models import CacheHit, CacheStats

logger = logging.getLogger(__name__)


class SemanticQueryCache:
    """基于 SQLite 与多级语义相似度的查询缓存系统（无 UI 耦合）。"""

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
    ):
        self.config = config or CacheConfig()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.embedding_available: Optional[bool] = None
        self._init_cache_db()

    @property
    def similarity_threshold(self) -> float:
        return self.config.similarity_threshold

    @similarity_threshold.setter
    def similarity_threshold(self, value: float) -> None:
        self.config.similarity_threshold = value

    def _init_cache_db(self) -> None:
        conn = sqlite3.connect(self.config.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    query_hash TEXT PRIMARY KEY,
                    query_text TEXT,
                    query_embedding TEXT,
                    db_name TEXT,
                    sql TEXT,
                    created_at TIMESTAMP,
                    accessed_at TIMESTAMP,
                    access_count INTEGER
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def test_embedding_availability(self) -> bool:
        """测试本地或远程嵌入向量服务是否可用。"""
        embedding = self._get_embedding("测试")
        self.embedding_available = embedding is not None
        if not self.embedding_available:
            logger.info("嵌入向量 API 不可用，将使用备选文本相似度算法")
        return self.embedding_available

    @staticmethod
    def _generate_hash(query: str, db_name: str) -> str:
        hash_input = f"{query}:{db_name}"
        return hashlib.md5(hash_input.encode("utf-8")).hexdigest()

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        try:
            base = self.embedding_config.base_url.rstrip("/")
            if self.embedding_config.mode == "ollama_legacy":
                response = requests.post(
                    f"{base}/api/embeddings",
                    json={
                        "model": self.embedding_config.model,
                        "prompt": text,
                    },
                    timeout=self.embedding_config.timeout_seconds,
                )
                if response.status_code == 200:
                    return response.json().get("embedding")
            else:
                headers = {"Content-Type": "application/json"}
                if self.embedding_config.api_key:
                    headers["Authorization"] = f"Bearer {self.embedding_config.api_key}"
                response = requests.post(
                    f"{base}/embeddings",
                    headers=headers,
                    json={
                        "model": self.embedding_config.model,
                        "input": text,
                    },
                    timeout=self.embedding_config.timeout_seconds,
                )
                if response.status_code == 200:
                    items = response.json().get("data", [])
                    if items:
                        return items[0].get("embedding")

            logger.warning("嵌入向量 API 请求失败，状态码: %s", response.status_code)
            return None
        except Exception as exc:
            logger.debug("获取嵌入向量失败: %s", exc)
            return None

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        try:
            a = np.asarray(vec1, dtype=float)
            b = np.asarray(vec2, dtype=float)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)
        except Exception as exc:
            logger.warning("计算余弦相似度失败: %s", exc)
            return 0.0

    def _find_similar_query(
        self, query: str, db_name: str
    ) -> Optional[Tuple[str, float, str]]:
        """查找语义相似的缓存条目，返回 (query_hash, similarity, match_type)。"""
        if self.embedding_available is None:
            self.test_embedding_availability()

        if self.embedding_available:
            query_embedding = self._get_embedding(query)
            if query_embedding:
                conn = sqlite3.connect(self.config.db_path)
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT query_hash, query_embedding FROM query_cache WHERE db_name = ?",
                        (db_name,),
                    )
                    cached_queries = cursor.fetchall()
                finally:
                    conn.close()

                max_similarity = 0.0
                most_similar_hash: Optional[str] = None
                for query_hash, embedding_json in cached_queries:
                    if not embedding_json:
                        continue
                    try:
                        cached_embedding = json.loads(embedding_json)
                        similarity = self._cosine_similarity(
                            query_embedding, cached_embedding
                        )
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_hash = query_hash
                    except Exception:
                        continue

                if most_similar_hash and max_similarity >= self.similarity_threshold:
                    return most_similar_hash, max_similarity, "embedding"
                return None

        return self._find_similar_query_fallback(query, db_name)

    def _find_similar_query_fallback(
        self, query: str, db_name: str
    ) -> Optional[Tuple[str, float, str]]:
        conn = sqlite3.connect(self.config.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT query_hash, query_text FROM query_cache WHERE db_name = ?",
                (db_name,),
            )
            cached_queries = cursor.fetchall()
        finally:
            conn.close()

        if not cached_queries:
            return None

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            all_texts = [query] + [q[1] for q in cached_queries]
            query_hashes = [q[0] for q in cached_queries]

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

            if similarities.size > 0:
                max_idx = int(similarities[0].argmax())
                max_similarity = float(similarities[0][max_idx])
                if max_similarity >= self.similarity_threshold:
                    return query_hashes[max_idx], max_similarity, "tfidf"
            return None
        except Exception as exc:
            logger.debug("TF-IDF 相似度计算不可用，降级为关键词匹配: %s", exc)
            return self._keyword_matching_fallback(query, cached_queries)

    def _keyword_matching_fallback(
        self, query: str, cached_queries: List[Tuple[str, str]]
    ) -> Optional[Tuple[str, float, str]]:
        stopwords = {
            "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
            "the", "a", "an", "of", "to", "in", "for", "and", "is", "on", "that", "by",
        }

        def extract_keywords(text: str) -> List[str]:
            words = re.findall(r"\w+|[\u4e00-\u9fff]", text.lower())
            return [w for w in words if w not in stopwords and len(w) > 1]

        query_keywords = set(extract_keywords(query))
        if not query_keywords:
            return None

        max_similarity = 0.0
        most_similar_hash: Optional[str] = None
        for query_hash, query_text in cached_queries:
            cached_keywords = set(extract_keywords(query_text))
            if not cached_keywords:
                continue
            common = query_keywords & cached_keywords
            if not common:
                continue
            similarity = len(common) / len(query_keywords | cached_keywords)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_hash = query_hash

        if most_similar_hash and max_similarity >= self.similarity_threshold:
            return most_similar_hash, max_similarity, "keyword"
        return None

    def get_from_cache(self, query: str, db_name: str) -> Optional[CacheHit]:
        """从缓存中获取精确匹配或语义相似的 SQL。"""
        query_hash = self._generate_hash(query, db_name)
        conn = sqlite3.connect(self.config.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sql, query_text, created_at FROM query_cache WHERE query_hash = ?",
                (query_hash,),
            )
            result = cursor.fetchone()
            similarity = 1.0
            match_type = "exact"

            if not result:
                similar = self._find_similar_query(query, db_name)
                if similar:
                    similar_hash, similarity, match_type = similar
                    cursor.execute(
                        "SELECT sql, query_text, created_at FROM query_cache WHERE query_hash = ?",
                        (similar_hash,),
                    )
                    result = cursor.fetchone()
                    query_hash = similar_hash

            if not result:
                return None

            sql, cached_query, created_at = result
            if created_at and self.is_cache_expired(created_at):
                cursor.execute(
                    "DELETE FROM query_cache WHERE query_hash = ?",
                    (query_hash,),
                )
                conn.commit()
                return None

            cursor.execute(
                "UPDATE query_cache SET accessed_at = ?, access_count = access_count + 1 WHERE query_hash = ?",
                (datetime.now().isoformat(), query_hash),
            )
            conn.commit()
            return CacheHit(
                sql=sql,
                matched_query=cached_query,
                similarity=similarity,
                match_type=match_type,
            )
        finally:
            conn.close()

    def save_to_cache(self, query: str, db_name: str, sql: str) -> None:
        """保存有效 SQL 查询到缓存。"""
        query_hash = self._generate_hash(query, db_name)
        query_embedding = None
        if self.embedding_available:
            query_embedding = self._get_embedding(query)
        query_embedding_json = json.dumps(query_embedding) if query_embedding else None

        now = datetime.now().isoformat()
        conn = sqlite3.connect(self.config.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO query_cache
                (query_hash, query_text, query_embedding, db_name, sql, created_at, accessed_at, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_hash,
                    query,
                    query_embedding_json,
                    db_name,
                    sql,
                    now,
                    now,
                    1,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        self.enforce_cache_size_limit(self.config.max_entries)

    def clear_cache(self) -> None:
        conn = sqlite3.connect(self.config.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM query_cache")
            conn.commit()
        finally:
            conn.close()

    def get_cache_stats(self) -> CacheStats:
        conn = sqlite3.connect(self.config.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM query_cache")
            total_entries = int(cursor.fetchone()[0])

            cursor.execute("SELECT SUM(access_count) FROM query_cache")
            total_hits = int(cursor.fetchone()[0] or 0)

            cursor.execute(
                "SELECT query_text, access_count FROM query_cache ORDER BY access_count DESC LIMIT 5"
            )
            top_queries = [(row[0], int(row[1])) for row in cursor.fetchall()]
            return CacheStats(
                total_entries=total_entries,
                total_hits=total_hits,
                top_queries=top_queries,
            )
        finally:
            conn.close()

    def is_cache_expired(
        self, created_at: datetime | str, expiry_days: Optional[int] = None
    ) -> bool:
        days = expiry_days if expiry_days is not None else self.config.expiry_days
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        return datetime.now() > created_at + timedelta(days=days)

    def enforce_cache_size_limit(self, max_entries: Optional[int] = None) -> None:
        limit = max_entries if max_entries is not None else self.config.max_entries
        conn = sqlite3.connect(self.config.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM query_cache")
            count = int(cursor.fetchone()[0])
            if count > limit:
                entries_to_delete = count - limit
                cursor.execute(
                    """
                    DELETE FROM query_cache
                    WHERE query_hash IN (
                        SELECT query_hash FROM query_cache
                        ORDER BY access_count ASC, accessed_at ASC
                        LIMIT ?
                    )
                    """,
                    (entries_to_delete,),
                )
                conn.commit()
        finally:
            conn.close()
