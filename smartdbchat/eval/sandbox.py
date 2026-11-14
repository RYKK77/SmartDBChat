import sqlite3
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from smartdbchat.db import UnsafeSQLError, _normalize_rows, validate_read_only_sql

_DDL_STATEMENTS: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS departments ( -- 学院表: 存储学校各学院基本信息
        id INTEGER PRIMARY KEY,          -- 学院ID
        name TEXT NOT NULL UNIQUE,       -- 学院名称
        code TEXT NOT NULL UNIQUE,       -- 学院编码
        dean TEXT NOT NULL               -- 院长姓名
    );
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS students ( -- 学生表: 存储学生学籍与学业概况
        id INTEGER PRIMARY KEY,          -- 学生ID
        student_no TEXT NOT NULL UNIQUE, -- 学号
        name TEXT NOT NULL,              -- 学生姓名
        gender TEXT NOT NULL,            -- 性别 ('男', '女')
        enrollment_year INTEGER NOT NULL,-- 入学年份
        status TEXT NOT NULL CHECK (status IN ('在读', '已毕业', '休学')), -- 学籍状态
        department_id INTEGER NOT NULL,  -- 所属学院ID
        gpa REAL NOT NULL,               -- 平均学分绩点 (0.0 - 4.0)
        FOREIGN KEY (department_id) REFERENCES departments(id)
    );
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS instructors ( -- 教师表: 存储授课教师信息
        id INTEGER PRIMARY KEY,          -- 教师ID
        name TEXT NOT NULL,              -- 教师姓名
        title TEXT NOT NULL CHECK (title IN ('教授', '副教授', '讲师')), -- 职称
        department_id INTEGER NOT NULL,  -- 所属学院ID
        FOREIGN KEY (department_id) REFERENCES departments(id)
    );
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS courses ( -- 课程表: 存储开设课程信息
        id INTEGER PRIMARY KEY,          -- 课程ID
        course_code TEXT NOT NULL UNIQUE,-- 课程代码
        title TEXT NOT NULL,             -- 课程名称
        credits REAL NOT NULL,           -- 学分
        course_type TEXT NOT NULL CHECK (course_type IN ('必修', '选修')), -- 课程类型
        department_id INTEGER NOT NULL,  -- 开课学院ID
        instructor_id INTEGER NOT NULL,  -- 授课教师ID
        FOREIGN KEY (department_id) REFERENCES departments(id),
        FOREIGN KEY (instructor_id) REFERENCES instructors(id)
    );
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS enrollments ( -- 选课成绩表: 存储学生选课及成绩记录
        id INTEGER PRIMARY KEY,          -- 选课记录ID
        student_id INTEGER NOT NULL,     -- 学生ID
        course_id INTEGER NOT NULL,      -- 课程ID
        semester TEXT NOT NULL,          -- 学期 (如 '2024-Spring')
        score REAL NOT NULL,             -- 成绩 (0 - 100)
        status TEXT NOT NULL CHECK (status IN ('正常', '补考', '重修')), -- 修读状态
        FOREIGN KEY (student_id) REFERENCES students(id),
        FOREIGN KEY (course_id) REFERENCES courses(id)
    );
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS scholarships ( -- 奖学金表: 存储学生获奖学金记录
        id INTEGER PRIMARY KEY,          -- 奖学金记录ID
        student_id INTEGER NOT NULL,     -- 学生ID
        award_name TEXT NOT NULL,        -- 奖项名称
        amount REAL NOT NULL,            -- 奖金金额 (元)
        award_year INTEGER NOT NULL,     -- 获奖年份
        FOREIGN KEY (student_id) REFERENCES students(id)
    );
    """.strip(),
    """
    CREATE TABLE IF NOT EXISTS campus_orders ( -- 校园消费订单表: 存储学生在校园内的消费流水
        id INTEGER PRIMARY KEY,          -- 订单ID
        student_id INTEGER NOT NULL,     -- 学生ID
        category TEXT NOT NULL CHECK (category IN ('餐饮', '图书', '体育', '文创')), -- 消费类别
        amount REAL NOT NULL,            -- 消费金额 (元)
        order_date TEXT NOT NULL,        -- 订单日期 (YYYY-MM-DD)
        payment_status TEXT NOT NULL CHECK (payment_status IN ('已支付', '已退款')), -- 支付状态
        FOREIGN KEY (student_id) REFERENCES students(id)
    );
    """.strip(),
]

_DEPARTMENTS_DATA = [
    (1, "计算机科学与技术", "CS", "张明远"),
    (2, "软件工程", "SE", "李华清"),
    (3, "数据科学与人工智能", "AI", "王志强"),
    (4, "经济与管理", "EM", "陈思远"),
    (5, "外国语", "FL", "赵雅婷"),
]

_INSTRUCTORS_DATA = [
    (1, "刘伟", "教授", 1),
    (2, "周敏", "副教授", 1),
    (3, "陈刚", "讲师", 1),
    (4, "杨帆", "教授", 2),
    (5, "吴静", "副教授", 2),
    (6, "郑涛", "教授", 3),
    (7, "孙丽", "讲师", 3),
    (8, "黄磊", "教授", 4),
    (9, "徐芳", "副教授", 4),
    (10, "林娜", "讲师", 5),
]

_STUDENTS_DATA = [
    (1, "2021001", "张三", "男", 2021, "已毕业", 1, 3.85),
    (2, "2021002", "李四", "女", 2021, "已毕业", 1, 3.92),
    (3, "2022001", "王五", "男", 2022, "在读", 1, 3.65),
    (4, "2022002", "赵六", "女", 2022, "在读", 1, 3.78),
    (5, "2022003", "陈晨", "女", 2022, "在读", 2, 3.55),
    (6, "2022004", "刘洋", "男", 2022, "休学", 2, 2.85),
    (7, "2023001", "Alice Wang", "女", 2023, "在读", 3, 3.95),
    (8, "2023002", "Bob Chen", "男", 2023, "在读", 3, 3.40),
    (9, "2023003", "孙悦", "女", 2023, "在读", 4, 3.72),
    (10, "2023004", "周杰", "男", 2023, "在读", 4, 3.15),
    (11, "2024001", "吴敏", "女", 2024, "在读", 1, 3.88),
    (12, "2024002", "郑凯", "男", 2024, "在读", 2, 3.60),
    (13, "2024003", "林晓", "女", 2024, "在读", 3, 3.81),
    (14, "2024004", "黄俊", "男", 2024, "休学", 4, 2.70),
    (15, "2024005", "Emily Zhang", "女", 2024, "在读", 5, 3.90),
]

_COURSES_DATA = [
    (1, "CS101", "数据结构与算法", 4.0, "必修", 1, 1),
    (2, "CS102", "数据库系统原理", 3.5, "必修", 1, 2),
    (3, "CS201", "分布式系统", 3.0, "选修", 1, 3),
    (4, "SE101", "软件工程导论", 3.0, "必修", 2, 4),
    (5, "SE201", "敏捷开发与测试", 2.5, "选修", 2, 5),
    (6, "AI101", "机器学习基础", 4.0, "必修", 3, 6),
    (7, "AI201", "自然语言处理", 3.0, "选修", 3, 7),
    (8, "EM101", "微观经济学", 3.0, "必修", 4, 8),
    (9, "EM201", "商业数据分析", 2.5, "选修", 4, 9),
    (10, "FL101", "Academic Writing", 2.0, "必修", 5, 10),
]

_ENROLLMENTS_DATA = [
    (1, 1, 1, "2022-Fall", 92.0, "正常"),
    (2, 1, 2, "2023-Spring", 88.5, "正常"),
    (3, 2, 1, "2022-Fall", 95.0, "正常"),
    (4, 2, 2, "2023-Spring", 94.0, "正常"),
    (5, 2, 3, "2023-Fall", 90.0, "正常"),
    (6, 3, 1, "2023-Fall", 84.0, "正常"),
    (7, 3, 2, "2024-Spring", 79.5, "正常"),
    (8, 3, 6, "2024-Fall", 86.0, "正常"),
    (9, 4, 1, "2023-Fall", 91.0, "正常"),
    (10, 4, 2, "2024-Spring", 87.0, "正常"),
    (11, 4, 3, "2024-Fall", 89.0, "正常"),
    (12, 5, 4, "2023-Fall", 82.0, "正常"),
    (13, 5, 5, "2024-Spring", 85.0, "正常"),
    (14, 6, 4, "2023-Fall", 58.0, "补考"),
    (15, 6, 5, "2024-Spring", 52.0, "重修"),
    (16, 7, 6, "2024-Spring", 96.5, "正常"),
    (17, 7, 7, "2024-Fall", 94.0, "正常"),
    (18, 7, 1, "2024-Fall", 93.0, "正常"),
    (19, 8, 6, "2024-Spring", 74.0, "正常"),
    (20, 8, 7, "2024-Fall", 61.0, "补考"),
    (21, 9, 8, "2024-Spring", 88.0, "正常"),
    (22, 9, 9, "2024-Fall", 91.5, "正常"),
    (23, 10, 8, "2024-Spring", 64.0, "补考"),
    (24, 11, 1, "2024-Fall", 90.5, "正常"),
    (25, 11, 10, "2024-Fall", 87.0, "正常"),
    (26, 12, 4, "2024-Fall", 83.0, "正常"),
    (27, 13, 6, "2024-Fall", 89.0, "正常"),
    (28, 14, 8, "2024-Fall", 55.0, "重修"),
    (29, 15, 10, "2024-Fall", 97.0, "正常"),
    (30, 15, 7, "2024-Fall", 88.0, "正常"),
]

_SCHOLARSHIPS_DATA = [
    (1, 1, "国家奖学金", 8000.0, 2023),
    (2, 2, "国家奖学金", 8000.0, 2023),
    (3, 2, "学业一等奖学金", 5000.0, 2024),
    (4, 4, "学业二等奖学金", 3000.0, 2024),
    (5, 7, "国家奖学金", 8000.0, 2024),
    (6, 7, "科研创新奖学金", 4000.0, 2024),
    (7, 9, "学业一等奖学金", 5000.0, 2024),
    (8, 11, "新生优秀奖学金", 3000.0, 2024),
    (9, 13, "学业二等奖学金", 3000.0, 2024),
    (10, 15, "学业一等奖学金", 5000.0, 2024),
]

_CAMPUS_ORDERS_DATA = [
    (1, 1, "餐饮", 28.5, "2024-09-02", "已支付"),
    (2, 1, "图书", 86.0, "2024-09-10", "已支付"),
    (3, 2, "文创", 120.0, "2024-09-12", "已支付"),
    (4, 3, "餐饮", 35.0, "2024-09-15", "已支付"),
    (5, 3, "体育", 150.0, "2024-09-20", "已支付"),
    (6, 3, "图书", 64.0, "2024-10-03", "已退款"),
    (7, 4, "餐饮", 22.0, "2024-10-05", "已支付"),
    (8, 4, "图书", 95.0, "2024-10-08", "已支付"),
    (9, 5, "餐饮", 41.5, "2024-10-10", "已支付"),
    (10, 5, "文创", 78.0, "2024-10-12", "已退款"),
    (11, 7, "图书", 145.0, "2024-10-15", "已支付"),
    (12, 7, "餐饮", 32.0, "2024-10-18", "已支付"),
    (13, 7, "体育", 200.0, "2024-10-20", "已支付"),
    (14, 8, "餐饮", 45.0, "2024-10-22", "已支付"),
    (15, 9, "文创", 110.0, "2024-10-25", "已支付"),
    (16, 10, "体育", 90.0, "2024-10-28", "已退款"),
    (17, 11, "餐饮", 26.0, "2024-11-01", "已支付"),
    (18, 11, "图书", 112.0, "2024-11-03", "已支付"),
    (19, 12, "餐饮", 38.0, "2024-11-05", "已支付"),
    (20, 13, "文创", 65.0, "2024-11-07", "已支付"),
    (21, 15, "图书", 130.0, "2024-11-09", "已支付"),
    (22, 15, "体育", 160.0, "2024-11-11", "已支付"),
]


class SQLiteEvalBackend:
    """基于 SQLite 的轻量级评测沙盒数据库，兼容 DatabaseManager 的主要只读接口。"""

    def __init__(self, db_path: str = ":memory:", seed_data: bool = True):
        self.db_path = db_path
        self.config = SimpleNamespace(database="smartdb_eval")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        if seed_data:
            self._init_schema_and_seed()

    def _init_schema_and_seed(self) -> None:
        cursor = self.conn.cursor()
        for ddl in _DDL_STATEMENTS:
            cursor.execute(ddl)

        existing = cursor.execute("SELECT COUNT(*) FROM departments").fetchone()[0]
        if existing == 0:
            cursor.executemany(
                "INSERT INTO departments (id, name, code, dean) VALUES (?, ?, ?, ?)",
                _DEPARTMENTS_DATA,
            )
            cursor.executemany(
                "INSERT INTO instructors (id, name, title, department_id) VALUES (?, ?, ?, ?)",
                _INSTRUCTORS_DATA,
            )
            cursor.executemany(
                "INSERT INTO students (id, student_no, name, gender, enrollment_year, status, department_id, gpa) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                _STUDENTS_DATA,
            )
            cursor.executemany(
                "INSERT INTO courses (id, course_code, title, credits, course_type, department_id, instructor_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                _COURSES_DATA,
            )
            cursor.executemany(
                "INSERT INTO enrollments (id, student_id, course_id, semester, score, status) VALUES (?, ?, ?, ?, ?, ?)",
                _ENROLLMENTS_DATA,
            )
            cursor.executemany(
                "INSERT INTO scholarships (id, student_id, award_name, amount, award_year) VALUES (?, ?, ?, ?, ?)",
                _SCHOLARSHIPS_DATA,
            )
            cursor.executemany(
                "INSERT INTO campus_orders (id, student_id, category, amount, order_date, payment_status) VALUES (?, ?, ?, ?, ?, ?)",
                _CAMPUS_ORDERS_DATA,
            )
        self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None  # type: ignore[assignment]

    def __enter__(self) -> "SQLiteEvalBackend":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def list_tables(self) -> List[str]:
        """列出数据库中所有用户表名。"""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        )
        return [row[0] for row in cursor.fetchall()]

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """校验并执行只读 SQL 查询，返回规范化后的字典列表。"""
        is_valid, reason = validate_read_only_sql(sql)
        if not is_valid:
            raise UnsafeSQLError(reason or "不安全的 SQL 语句")

        cursor = self.conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return _normalize_rows(rows)

    def explain_sql(self, sql: str) -> List[Dict[str, Any]]:
        """使用 EXPLAIN QUERY PLAN 分析只读 SQL 的执行计划。"""
        is_valid, reason = validate_read_only_sql(sql)
        if not is_valid:
            raise UnsafeSQLError(reason or "不安全的 SQL 语句")

        cleaned = sql.strip().rstrip(";")
        cursor = self.conn.execute(f"EXPLAIN QUERY PLAN {cleaned}")
        rows = [dict(row) for row in cursor.fetchall()]
        return _normalize_rows(rows)

    def get_schema_info(self, tables: Optional[List[str]] = None) -> str:
        """获取指定表或全部表的建表 DDL（保留中文表/列注释）。"""
        all_tables = self.list_tables()
        target_tables = all_tables if tables is None else [t for t in tables if t in all_tables]
        if not target_tables:
            return ""

        ddls: List[str] = []
        for table_name in target_tables:
            row = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone()
            if row and row[0]:
                ddls.append(row[0])
        return "\n\n".join(ddls)

    def get_specific_schema_info(self, tables: List[str]) -> str:
        """获取指定表的建表语句，未找到时返回提示文本。"""
        if not tables:
            return "未找到相关表的schema信息"
        schema = self.get_schema_info(tables)
        return schema if schema else "未找到相关表的schema信息"

    def sample_table_rows(self, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """采样指定表的前 N 行数据。"""
        if table_name not in self.list_tables():
            raise ValueError(f"未知表名: {table_name}")
        cursor = self.conn.execute(
            f'SELECT * FROM "{table_name}" LIMIT ?',
            (int(limit),),
        )
        return _normalize_rows([dict(row) for row in cursor.fetchall()])

    def get_distinct_values(
        self, table_name: str, column_name: str, limit: int = 20
    ) -> List[Any]:
        """获取指定表某列的去重非空值列表（用于 value linking）。"""
        if table_name not in self.list_tables():
            raise ValueError(f"未知表名: {table_name}")

        col_rows = self.conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        valid_columns = {row["name"] for row in col_rows}
        if column_name not in valid_columns:
            raise ValueError(f"表 {table_name} 中不存在列: {column_name}")

        cursor = self.conn.execute(
            f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
            f'WHERE "{column_name}" IS NOT NULL ORDER BY "{column_name}" LIMIT ?',
            (int(limit),),
        )
        return [row[0] for row in cursor.fetchall()]
