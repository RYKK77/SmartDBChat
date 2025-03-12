import streamlit as st
import mysql.connector
from openai import OpenAI
from typing import List, Dict
import json
import pandas as pd
import plotly.express as px
import sqlite3
import hashlib
from datetime import datetime, timedelta


class SemanticQueryCache:
    def __init__(self, cache_db_path="semantic_cache.db", similarity_threshold=0.85):
        """初始化语义查询缓存系统"""
        self.cache_db_path = cache_db_path
        self.similarity_threshold = similarity_threshold
        self.client = None  # 将在需要时初始化
        self.embedding_available = True  # 标记嵌入API是否可用
        self._init_cache_db()

    def set_api_client(self, client):
        """设置API客户端"""
        self.client = client
        # 测试嵌入API是否可用
        self._test_embedding_availability()

    def _test_embedding_availability(self):
        """测试嵌入API是否可用"""
        try:
            test_embedding = self._get_embedding("测试")
            self.embedding_available = test_embedding is not None
            if not self.embedding_available:
                st.warning("嵌入向量API不可用，将使用备选文本相似度算法")
        except Exception as e:
            self.embedding_available = False
            st.warning(f"嵌入向量API测试失败: {str(e)}，将使用备选文本相似度算法")

    def _init_cache_db(self):
        """初始化缓存数据库"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        # 修改缓存表结构，移除不需要的字段，只保留SQL相关信息
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash TEXT PRIMARY KEY,
            query_text TEXT,
            query_embedding TEXT,  -- 存储为JSON字符串
            db_name TEXT,
            sql TEXT,
            created_at TIMESTAMP,
            accessed_at TIMESTAMP,
            access_count INTEGER
        )
        ''')

        conn.commit()
        conn.close()

    def _generate_hash(self, query, db_name):
        """为查询生成唯一哈希值"""
        hash_input = f"{query}:{db_name}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _get_embedding(self, text):
        """获取文本的嵌入向量"""
        if not self.client:
            st.warning("API客户端未设置，无法获取嵌入向量")
            return None

        try:
            response = self.client.embeddings.create(
                model="embedding-2",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.warning(f"获取嵌入向量失败: {str(e)}")
            return None

    def _find_similar_query(self, query, db_name):
        """查找语义相似的查询"""
        # 如果嵌入API可用，使用嵌入向量计算相似度
        if self.embedding_available and self.client:
            # 获取当前查询的嵌入向量
            query_embedding = self._get_embedding(query)
            if query_embedding:
                # 从数据库获取所有查询及其嵌入向量
                conn = sqlite3.connect(self.cache_db_path)
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT query_hash, query_text, query_embedding FROM query_cache WHERE db_name = ?",
                    (db_name,)
                )
                cached_queries = cursor.fetchall()
                conn.close()

                if not cached_queries:
                    return None

                # 计算相似度并找出最相似的查询
                max_similarity = 0
                most_similar_hash = None

                for query_hash, query_text, embedding_json in cached_queries:
                    if not embedding_json:
                        continue

                    try:
                        cached_embedding = json.loads(embedding_json)
                        similarity = self._cosine_similarity(query_embedding, cached_embedding)

                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_hash = query_hash
                    except Exception as e:
                        continue

                # 如果相似度超过阈值，返回最相似查询的哈希值
                if max_similarity >= self.similarity_threshold:
                    st.info(f"找到相似问题，相似度: {max_similarity:.2f}")
                    return most_similar_hash

                return None

        # 如果嵌入API不可用或获取失败，使用备选方案
        return self._find_similar_query_fallback(query, db_name)

    def _cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        try:
            import numpy as np
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except Exception as e:
            st.warning(f"计算余弦相似度失败: {str(e)}")
            return 0

    def _find_similar_query_fallback(self, query, db_name):
        """备选方案：使用文本相似度计算查找相似查询"""
        try:
            # 从数据库获取所有查询
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT query_hash, query_text FROM query_cache WHERE db_name = ?",
                (db_name,)
            )
            cached_queries = cursor.fetchall()
            conn.close()

            if not cached_queries:
                return None

            try:
                # 尝试使用TF-IDF计算相似度
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                # 准备文本列表
                all_texts = [query] + [q[1] for q in cached_queries]
                query_hashes = [None] + [q[0] for q in cached_queries]

                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

                # 找出最相似的查询
                if similarities.size > 0:
                    max_idx = similarities[0].argmax()
                    max_similarity = similarities[0][max_idx]

                    if max_similarity >= self.similarity_threshold:
                        st.info(f"使用文本相似度找到相似问题，相似度: {max_similarity:.2f}")
                        return query_hashes[max_idx + 1]
            except Exception as e:
                st.warning(f"文本相似度计算失败: {str(e)}")
                # 如果TF-IDF失败，尝试使用关键词匹配
                return self._keyword_matching_fallback(query, cached_queries)

            return None
        except Exception as e:
            st.warning(f"备选相似度计算失败: {str(e)}")
            return None

    def _keyword_matching_fallback(self, query, cached_queries):
        """最后的备选方案：使用简单的关键词匹配"""
        import re

        # 简单的分词
        def extract_keywords(text):
            # 简单分词（英文按空格，中文按字符）
            words = re.findall(r'\w+|[\u4e00-\u9fff]', text.lower())
            # 简单的停用词过滤
            stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', 'the', 'a', 'an', 'of', 'to', 'in',
                         'for', 'and', 'is', 'on', 'that', 'by'}
            return [w for w in words if w not in stopwords and len(w) > 1]

        query_keywords = set(extract_keywords(query))
        if not query_keywords:
            return None

        # 计算关键词匹配度
        max_similarity = 0
        most_similar_hash = None

        for query_hash, query_text in cached_queries:
            cached_keywords = set(extract_keywords(query_text))

            # 计算关键词重叠度
            if not cached_keywords:
                continue

            common_keywords = query_keywords & cached_keywords
            if not common_keywords:
                continue

            # 计算Jaccard相似度
            similarity = len(common_keywords) / len(query_keywords | cached_keywords)

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_hash = query_hash

        # 如果相似度超过阈值，返回最相似查询的哈希值
        if max_similarity >= self.similarity_threshold:
            st.info(f"使用关键词匹配找到相似问题，相似度: {max_similarity:.2f}")
            return most_similar_hash

        return None

    def get_from_cache(self, query, db_name):
        """从缓存中获取查询结果，支持语义相似匹配"""
        # 首先尝试精确匹配
        query_hash = self._generate_hash(query, db_name)

        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT sql, query_text FROM query_cache WHERE query_hash = ?",
            (query_hash,)
        )
        result = cursor.fetchone()

        # 如果没有精确匹配，尝试查找语义相似的查询
        if not result:
            similar_hash = self._find_similar_query(query, db_name)
            if similar_hash:
                cursor.execute(
                    "SELECT sql, query_text FROM query_cache WHERE query_hash = ?",
                    (similar_hash,)
                )
                result = cursor.fetchone()
                query_hash = similar_hash  # 更新哈希值以便更新访问统计

        if result:
            # 更新访问时间和计数
            cursor.execute(
                "UPDATE query_cache SET accessed_at = ?, access_count = access_count + 1 WHERE query_hash = ?",
                (datetime.now(), query_hash)
            )
            conn.commit()

            sql, cached_query = result

            # 显示缓存命中信息
            if cached_query == query:
                st.success("✅ 从缓存中获取到完全匹配的SQL")
            else:
                st.success(f"✅ 从缓存中获取到相似问题的SQL: \"{cached_query}\"")

            st.code(f"缓存的SQL查询:\n{sql}", language="sql")

            conn.close()
            return sql  # 只返回SQL语句

        conn.close()
        return None

    def save_to_cache(self, query, db_name, sql):
        """保存SQL查询到缓存，包括嵌入向量"""
        query_hash = self._generate_hash(query, db_name)

        # 获取查询的嵌入向量
        query_embedding = None
        if self.embedding_available:
            query_embedding = self._get_embedding(query)
        query_embedding_json = json.dumps(query_embedding) if query_embedding else None

        conn = sqlite3.connect(self.cache_db_path)
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
                datetime.now(),
                datetime.now(),
                1
            )
        )

        conn.commit()
        conn.close()

        # 强制执行缓存大小限制
        self.enforce_cache_size_limit()

    def clear_cache(self):
        """清空缓存"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM query_cache")
        conn.commit()
        conn.close()
        return "缓存已清空"

    def get_cache_stats(self):
        """获取缓存统计信息"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM query_cache")
        total_entries = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(access_count) FROM query_cache")
        total_hits = cursor.fetchone()[0] or 0

        cursor.execute("SELECT query_text, access_count FROM query_cache ORDER BY access_count DESC LIMIT 5")
        top_queries = cursor.fetchall()

        conn.close()

        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "top_queries": top_queries
        }

    def is_cache_expired(self, created_at, expiry_days=7):
        """检查缓存是否过期"""
        # 将字符串转换为datetime对象
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        # 计算过期时间
        expiry_date = created_at + timedelta(days=expiry_days)
        return datetime.now() > expiry_date

    def enforce_cache_size_limit(self, max_entries=1000):
        """限制缓存条目数量，删除最不常用的条目"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()

        # 获取当前缓存条目数
        cursor.execute("SELECT COUNT(*) FROM query_cache")
        count = cursor.fetchone()[0]

        # 如果超过限制，删除最不常用的条目
        if count > max_entries:
            entries_to_delete = count - max_entries
            cursor.execute(
                "DELETE FROM query_cache WHERE query_hash IN (SELECT query_hash FROM query_cache ORDER BY access_count, accessed_at LIMIT ?)",
                (entries_to_delete,)
            )
            conn.commit()

        conn.close()




class NLDatabaseQuery:
    def __init__(self, db_config: Dict, api_key: str, cache_system: SemanticQueryCache = None):
        # 初始化数据库连接和OpenAI客户端
        self.db_config = db_config
        # 修改为OpenAI客户端初始化
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )
        self.conn = None
        self.cursor = None
        self.connect_db()

        # 添加缓存系统
        self.cache = cache_system if cache_system else SemanticQueryCache()
        # 设置API客户端到缓存系统
        self.cache.set_api_client(self.client)

    def connect_db(self):
        # 建立数据库连接
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            self.cursor = self.conn.cursor(dictionary=True)
        except Exception as e:
            st.error(f"数据库连接失败: {str(e)}")
            raise

    def extract_tables_from_sql(self, sql: str) -> List[str]:
        """从SQL语句中提取表名"""
        import re
        # 匹配FROM和JOIN后面的表名
        from_pattern = r'(?:FROM|JOIN)\s+`?(\w+)`?'
        tables = re.findall(from_pattern, sql, re.IGNORECASE)
        # 去重
        return list(set(tables))

    def get_schema_info(self) -> str:
        # 获取数据库schema信息
        schema_info = []
        self.cursor.execute("""
            SELECT TABLE_NAME, TABLE_COMMENT 
            FROM information_schema.TABLES 
            WHERE TABLE_SCHEMA = %s
        """, (self.db_config['database'],))

        tables = self.cursor.fetchall()
        for table in tables:
            table_name = table['TABLE_NAME']
            table_comment = table['TABLE_COMMENT']

            # 获取表的列信息
            self.cursor.execute("""
                SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_COMMENT, IS_NULLABLE, COLUMN_KEY
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """, (self.db_config['database'], table_name))

            columns = self.cursor.fetchall()
            columns_info = []
            for col in columns:
                col_info = (f"字段名: {col['COLUMN_NAME']}, "
                            f"类型: {col['COLUMN_TYPE']}, "
                            f"说明: {col['COLUMN_COMMENT']}, "
                            f"是否可空: {col['IS_NULLABLE']}, "
                            f"键类型: {col['COLUMN_KEY']}")
                columns_info.append(col_info)

            table_info = f"表名: {table_name}\n表说明: {table_comment}\n字段信息:\n" + "\n".join(columns_info)
            schema_info.append(table_info)

        return "\n\n".join(schema_info)

    def get_specific_schema_info(self, tables: List[str]) -> str:
        """获取指定表的schema信息"""
        if not tables:
            return "未找到相关表的schema信息"

        schema_info = []
        for table_name in tables:
            # 获取表信息
            self.cursor.execute("""
                SELECT TABLE_NAME, TABLE_COMMENT 
                FROM information_schema.TABLES 
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """, (self.db_config['database'], table_name))

            table = self.cursor.fetchone()
            if not table:
                continue

            table_comment = table['TABLE_COMMENT']

            # 获取表的列信息
            self.cursor.execute("""
                SELECT COLUMN_NAME, COLUMN_TYPE, COLUMN_COMMENT, IS_NULLABLE, COLUMN_KEY
                FROM information_schema.COLUMNS 
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """, (self.db_config['database'], table_name))

            columns = self.cursor.fetchall()
            columns_info = []
            for col in columns:
                col_info = (f"字段名: {col['COLUMN_NAME']}, "
                            f"类型: {col['COLUMN_TYPE']}, "
                            f"说明: {col['COLUMN_COMMENT']}, "
                            f"是否可空: {col['IS_NULLABLE']}, "
                            f"键类型: {col['COLUMN_KEY']}")
                columns_info.append(col_info)

            table_info = f"表名: {table_name}\n表说明: {table_comment}\n字段信息:\n" + "\n".join(columns_info)
            schema_info.append(table_info)

        return "\n\n".join(schema_info) if schema_info else "未找到相关表的schema信息"

    def generate_sql_prompt(self, query: str, schema_info: str) -> List[Dict]:
        return [
            {"role": "system", "content": """你是一个SQL专家，负责将自然语言转换为SQL查询语句。
                请注意：
                1. 直接返回SQL语句，不要包含任何其他信息。
                2. 现在使用的是mysql数据库，确保SQL语句语法正确
                3. 使用提供的schema信息构建查询
                4. 注意使用适当的表连接和条件
                5. 查询的表头都需要有中文注释，不要使用表别名
                6. 确保查询安全，避免SQL注入风险"""},
            {"role": "user", "content": f"""
                数据库Schema信息如下：
                {schema_info}

                请将以下问题转换为SQL查询语句：
                {query}
                """}
        ]

    def generate_answer_prompt(self, result: str, query: str, sql: str, schema_info: str) -> List[Dict]:
        return [
            {"role": "system", "content": """你是一个SQL查询解读专家，负责将SQL查询结果转换为自然语言和可视化数据。"""},
            {"role": "user", "content": f"""请帮我分析并格式化以下查询结果，以易于理解的方式展示：
                查询的问题：
                {query}
                查询用到的sql： 
                {sql}
                数据库Schema信息如下： 
                {schema_info}
                查询结果：
                {result}

                请提供【markdown格式】：
                1. 表格形式的数据展示，不要有```markdown ```这样的标识符
                2. 数据的简要分析

                另外，请分析数据是否适合以饼图或柱状图展示，如果适合，请提供以下JSON格式的可视化配置：

                ```json
                {{
            "charts": [
                    {{
            "type": "pie",  
                      "title": "图表标题",
                      "description": "图表描述",
                      "data": {{
            "labels": ["标签1", "标签2", ...],  // 分类标签
                        "values": [值1, 值2, ...],  // 对应的数值
                      }}
                    }}
                  ]
                }}
                ```

                注意：
                1. 只有当数据适合用饼图或柱状图展示时才提供JSON配置
                2. 饼图适合展示占比关系，柱状图适合数据对比关系
                3. 确保JSON格式正确，可以被解析
                """}
        ]

    # 清理大模型返回的SQL文本，去除代码块标记和其他格式
    def clean_sql_response(self, response_text: str):
        patterns = [
            r'```sql\s*(.*?)\s*```',  # SQL代码块
            r'```\s*(.*?)\s*```',  # 普通代码块
            r'`(.*?)`'  # 内联代码块
        ]

        cleaned_text = response_text
        for pattern in patterns:
            import re
            matches = re.findall(pattern, cleaned_text, re.DOTALL)
            if matches:
                # 使用第一个匹配到的SQL语句
                cleaned_text = matches[0]
                break

        # 去除多余的空行和首尾空白
        cleaned_text = '\n'.join(line.strip() for line in cleaned_text.splitlines() if line.strip())

        return cleaned_text

    def extract_chart_config(self, response_text: str):
        """从响应文本中提取图表配置JSON"""
        import re
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, response_text, re.DOTALL)

        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                st.warning("图表配置JSON格式错误，无法解析")

        return None

    def is_valid_query_result(self, result):
        """判断查询结果是否有效且值得缓存"""
        # 检查结果是否为空
        if not result:
            return False

        # 检查结果是否为列表且包含至少一条记录
        if not isinstance(result, list) or len(result) == 0:
            return False

        # 检查第一条记录是否为字典且包含数据
        if not isinstance(result[0], dict) or len(result[0]) == 0:
            return False

        # 结果有效
        return True

    def execute_query(self, sql: str) -> List[Dict]:
        """执行SQL查询，增加错误处理"""
        try:
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            if result:
                st.success(f"查询成功，返回 {len(result)} 条记录")
            else:
                st.warning("查询执行成功，但没有返回数据")
            return result
        except Exception as e:
            st.error(f"查询执行错误: {str(e)}")
            return []

    def natural_language_query(self, query: str, max_retries: int = 2) -> tuple:
        """执行自然语言查询，支持缓存，只缓存成功的查询，支持自动重试"""
        # 首先尝试从缓存中获取SQL（支持语义相似性匹配）
        cached_sql = self.cache.get_from_cache(
            query, self.db_config['database']
        )

        error_info = None  # 用于存储错误信息
        failed_sql = None  # 用于存储失败的SQL

        if cached_sql is not None:
            # 使用缓存的SQL执行查询
            sql = cached_sql
            # 执行SQL查询
            try:
                result = self.execute_query(sql)

                # 如果缓存的SQL执行失败或返回为空，则尝试重新生成SQL
                if not self.is_valid_query_result(result) and max_retries > 0:
                    st.warning(f"缓存的SQL执行失败或返回为空，将尝试重新生成SQL（剩余重试次数：{max_retries}）")
                    failed_sql = sql
                    return self._retry_query_generation(query, max_retries, failed_sql, error_info)
            except Exception as e:
                error_info = str(e)
                st.error(f"缓存的SQL执行错误: {error_info}")
                if max_retries > 0:
                    st.warning(f"将尝试重新生成SQL（剩余重试次数：{max_retries}）")
                    failed_sql = sql
                    return self._retry_query_generation(query, max_retries, failed_sql, error_info)
                result = []
        else:
            # 如果缓存中没有，则生成新的SQL
            schema_info = self.get_schema_info()

            # 生成并发送prompt到OpenAI
            messages = self.generate_sql_prompt(query, schema_info)
            response = self.client.chat.completions.create(
                model="glm-4-plus",
                messages=messages
            )
            # 获取生成的SQL
            sql = response.choices[0].message.content.strip()
            sql = self.clean_sql_response(sql)
            st.code(f"生成的SQL查询:\n{sql}", language="sql")

            # 执行SQL查询
            try:
                result = self.execute_query(sql)

                # 如果SQL执行失败或返回为空，则尝试重新生成SQL
                if not self.is_valid_query_result(result) and max_retries > 0:
                    st.warning(f"生成的SQL执行失败或返回为空，将尝试重新生成（剩余重试次数：{max_retries}）")
                    failed_sql = sql
                    return self._retry_query_generation(query, max_retries, failed_sql, error_info)
            except Exception as e:
                error_info = str(e)
                st.error(f"SQL执行错误: {error_info}")
                if max_retries > 0:
                    st.warning(f"将尝试重新生成SQL（剩余重试次数：{max_retries}）")
                    failed_sql = sql
                    return self._retry_query_generation(query, max_retries, failed_sql, error_info)
                result = []

        # 检查查询是否成功并返回有效结果
        query_successful = self.is_valid_query_result(result)

        # 提取SQL中使用的表，并获取相关的schema信息
        tables = self.extract_tables_from_sql(sql)
        specific_schema_info = self.get_specific_schema_info(tables)

        # 生成回答，只传递相关的schema信息
        messages1 = self.generate_answer_prompt(result, query, sql, specific_schema_info)
        with st.expander("prompt信息：", expanded=False):
            st.write(messages1)


        st.write("### 答案：")
        # 创建一个空的容器用于显示流式输出
        answer_container = st.empty()
        response_text = ""
        
        # 使用流式传输方式获取回答
        stream = self.client.chat.completions.create(
            model="glm-4-plus",
            messages=messages1,
            stream=True
        )
        
        # 逐步接收并显示回答
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
                answer_container.markdown(response_text)
        
        # 提取图表配置
        chart_config = self.extract_chart_config(response_text)

        # 只有在查询成功且SQL是新生成的（不是从缓存获取的）时才将SQL保存到缓存
        if query_successful and cached_sql is None:
            self.cache.save_to_cache(
                query,
                self.db_config['database'],
                sql
            )
            st.success("✅ 查询成功，SQL已保存到缓存")
        elif not query_successful:
            st.warning("⚠️ 查询未返回有效结果，此查询不会被缓存")

        # 返回文本回答和图表配置
        return response_text, chart_config, result

    def _retry_query_generation(self, query: str, max_retries: int, failed_sql: str = None,
                                error_info: str = None) -> tuple:
        """重试生成SQL查询，包含上一次失败的SQL和错误信息"""
        # 获取schema信息
        schema_info = self.get_schema_info()

        # 添加更详细的错误信息到提示中，包括上一次失败的SQL和错误信息
        retry_messages = [
            {"role": "system", "content": """你是一个SQL专家，负责将自然语言转换为SQL查询语句。
                之前生成的SQL查询执行失败或返回为空，请尝试生成一个更准确的SQL查询。
                请注意：
                1. 直接返回SQL语句，不要包含任何其他信息。
                2. 现在使用的是mysql数据库，确保SQL语句语法正确
                3. 使用提供的schema信息构建查询
                4. 注意使用适当的表连接和条件
                5. 查询的表头都需要有中文注释，不要使用表别名
                6. 确保查询安全，避免SQL注入风险
                7. 检查表名和字段名是否正确，确保它们存在于schema中
                8. 考虑可能的数据类型转换问题
                9. 确保WHERE条件合理，不会过滤掉所有数据"""},
            {"role": "user", "content": f"""
                数据库Schema信息如下：
                {schema_info}

                请将以下问题转换为SQL查询语句：
                {query}

                上一次生成的SQL查询失败，详情如下：
                SQL语句: {failed_sql if failed_sql else "无"}
                错误信息: {error_info if error_info else "查询执行成功，但没有返回数据"}

                请分析上述错误，生成一个更准确的SQL查询。
                """}
        ]

        # 生成新的SQL
        response = self.client.chat.completions.create(
            model="glm-4-plus",
            messages=retry_messages
        )

        # 获取生成的SQL
        sql = response.choices[0].message.content.strip()
        sql = self.clean_sql_response(sql)
        st.code(f"重试生成的SQL查询:\n{sql}", language="sql")

        # 执行SQL查询
        try:
            result = self.execute_query(sql)

            # 如果仍然失败且还有重试次数，继续重试
            if not self.is_valid_query_result(result) and max_retries > 1:
                st.warning(f"重试生成的SQL执行仍然失败或返回为空，将继续重试（剩余重试次数：{max_retries - 1}）")
                return self._retry_query_generation(query, max_retries - 1, sql, None)
        except Exception as e:
            error_info = str(e)
            st.error(f"重试SQL执行错误: {error_info}")
            if max_retries > 1:
                st.warning(f"将继续重试（剩余重试次数：{max_retries - 1}）")
                return self._retry_query_generation(query, max_retries - 1, sql, error_info)
            result = []

        # 检查查询是否成功并返回有效结果
        query_successful = self.is_valid_query_result(result)

        # 生成回答
        schema_info = self.get_schema_info()
        messages1 = self.generate_answer_prompt(result, query, sql, schema_info)
        with st.expander("prompt信息：", expanded=False):
            st.write(messages1)
        response1 = self.client.chat.completions.create(
            model="glm-4-plus",
            messages=messages1
        )

        response_text = response1.choices[0].message.content.strip()
        chart_config = self.extract_chart_config(response_text)

        # 只有在查询成功时才将SQL保存到缓存
        if query_successful:
            self.cache.save_to_cache(
                query,
                self.db_config['database'],
                sql
            )
            st.success("✅ 重试查询成功，SQL已保存到缓存")
        else:
            st.warning("⚠️ 所有重试都未返回有效结果，此查询不会被缓存")

        return response_text, chart_config, result
    def display_result(self, markdown_text, chart_config=None, query_result=None):
        # 显示文本回答
#         content = f"""
# ### 答案：
# {markdown_text}
#         """
#         st.write(content)

        # 如果有图表配置，显示图表
        if chart_config and 'charts' in chart_config and query_result:
            self.display_charts(chart_config, query_result)

    def display_charts(self, chart_config, query_result):
        """显示图表"""
        # 转换查询结果为DataFrame以便处理
        df = pd.DataFrame(query_result)

        for chart in chart_config.get('charts', []):
            chart_type = chart.get('type')
            title = chart.get('title', '数据可视化')

            st.subheader(title)
            if chart.get('description'):
                st.write(chart.get('description'))

            # 使用配置中的数据
            data = chart.get('data', {})
            labels = data.get('labels', [])
            values = data.get('values', [])

            if chart_type == 'pie':
                fig = px.pie(
                    names=labels,
                    values=values,
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == 'treemap':
                parents = data.get('parents', [''] * len(labels))
                fig = px.treemap(
                    names=labels,
                    values=values,
                    parents=parents,
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)


def init_session_state():
    """初始化session state变量"""
    if 'query_system' not in st.session_state:
        st.session_state.query_system = None
    if 'cache_system' not in st.session_state:
        st.session_state.cache_system = SemanticQueryCache()
    if 'similarity_threshold' not in st.session_state:
        st.session_state.similarity_threshold = 0.85


def create_query_system(db_config: Dict, api_key: str):
    """创建查询系统实例"""
    try:
        # 使用全局缓存系统
        cache_system = st.session_state.cache_system
        # 更新相似度阈值
        cache_system.similarity_threshold = st.session_state.similarity_threshold

        # 创建查询系统
        query_system = NLDatabaseQuery(db_config, api_key, cache_system)
        st.session_state.query_system = query_system
        return query_system
    except Exception as e:
        st.error(f"创建查询系统失败: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="SmartDBChat - 自然语言数据库查询", layout="wide")

    # 初始化session state
    init_session_state()

    st.title("SmartDBChat - 自然语言数据库查询")

    # 侧边栏配置
    with st.sidebar:
        st.header("配置")

        # 数据库配置
        st.subheader("数据库设置")
        host = st.text_input("主机", value="localhost")
        port = st.number_input("端口", value=3306)
        user = st.text_input("用户名", value="root")
        password = st.text_input("密码", type="password", value="")
        database = st.text_input("数据库名", value="")

        # API配置
        st.subheader("API设置")
        api_key = st.text_input("智谱AI API密钥", type="password", value="")

        # 缓存配置
        st.subheader("缓存设置")
        similarity_threshold = st.slider(
            "相似度阈值",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.similarity_threshold,
            step=0.05,
            help="设置查询相似度匹配的阈值，值越高要求匹配越精确"
        )
        st.session_state.similarity_threshold = similarity_threshold

        # 连接按钮
        if st.button("连接数据库"):
            db_config = {
                'host': host,
                'port': int(port),
                'user': user,
                'password': password,
                'database': database
            }

            if not api_key:
                st.error("请输入智谱AI API密钥")
            else:
                with st.spinner("正在连接数据库..."):
                    query_system = create_query_system(db_config, api_key)
                    if query_system:
                        st.success("数据库连接成功！")

        # 缓存管理
        if st.session_state.cache_system:
            st.subheader("缓存管理")

            cache_stats = st.session_state.cache_system.get_cache_stats()
            st.write(f"缓存条目数: {cache_stats['total_entries']}")
            st.write(f"缓存命中次数: {cache_stats['total_hits']}")

            if st.button("清空缓存"):
                message = st.session_state.cache_system.clear_cache()
                st.success(message)


    # 主界面
    if st.session_state.query_system:
        # 查询输入
        query = st.text_area("请输入你的问题", height=100)

        # 添加重试次数选项
        col1, col2 = st.columns([3, 1])
        with col2:
            max_retries = st.number_input("最大重试次数", min_value=0, max_value=5, value=2,
                                          help="如果SQL查询失败或返回为空，自动重试生成SQL的最大次数")

        if st.button("查询") and query:
            with st.spinner("正在处理查询..."):
                try:
                    markdown_text, chart_config, query_result = st.session_state.query_system.natural_language_query(
                        query, max_retries=max_retries)

                    # 显示结果
                    st.session_state.query_system.display_result(markdown_text, chart_config, query_result)

                    # 显示原始数据
                    with st.expander("查看原始数据"):
                        st.write(query_result)
                except Exception as e:
                    st.error(f"查询处理失败: {str(e)}")
    else:
        st.info("请先在侧边栏配置并连接数据库")


if __name__ == "__main__":
    main()
