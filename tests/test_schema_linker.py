from smartdbchat.schema_linker import (
    ColumnMeta,
    ForeignKeyMeta,
    LinkedSchema,
    SchemaGraph,
    SchemaLinker,
    TableMeta,
    build_schema_graph,
    parse_ddl_to_table_meta,
)

MYSQL_STUDENTS_DDL = """
CREATE TABLE IF NOT EXISTS `students` (
  `id` INT NOT NULL AUTO_INCREMENT COMMENT '学生ID',
  `student_no` VARCHAR(32) NOT NULL COMMENT '学号',
  `name` VARCHAR(64) NOT NULL COMMENT '学生姓名',
  `gpa` DECIMAL(3, 2) DEFAULT NULL COMMENT '平均绩点',
  `department_id` INT NOT NULL COMMENT '所属院系ID',
  `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  PRIMARY KEY (`id`),
  CONSTRAINT `fk_students_dept` FOREIGN KEY (`department_id`) REFERENCES `departments` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='学生信息表';
"""

MYSQL_DEPARTMENTS_DDL = """
CREATE TABLE `departments` (
  `id` INT NOT NULL AUTO_INCREMENT COMMENT '院系ID',
  `name` VARCHAR(100) NOT NULL COMMENT '院系名称',
  `building` VARCHAR(64) COMMENT '办公楼',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB COMMENT='院系表';
"""

SQLITE_COURSES_DDL = """
-- 课程表
CREATE TABLE courses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  course_code TEXT NOT NULL,
  title TEXT NOT NULL,
  credits REAL DEFAULT 3.0,
  department_id INTEGER REFERENCES departments(id)
);
"""

SQLITE_ENROLLMENTS_DDL = """
CREATE TABLE enrollments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id INTEGER NOT NULL,
  course_id INTEGER NOT NULL,
  score REAL,
  semester TEXT
);
"""

MYSQL_CATEGORIES_DDL = """
CREATE TABLE `categories` (
  `id` INT NOT NULL,
  `name` VARCHAR(50) NOT NULL,
  PRIMARY KEY (`id`)
);
"""

MYSQL_PRODUCTS_DDL = """
CREATE TABLE `products` (
  `id` INT NOT NULL,
  `category_id` INT NOT NULL,
  `price` DECIMAL(10, 2) NOT NULL,
  PRIMARY KEY (`id`)
);
"""


def test_parse_mysql_ddl():
    meta = parse_ddl_to_table_meta(MYSQL_STUDENTS_DDL)
    assert meta is not None
    assert meta.name == "students"
    assert meta.comment == "学生信息表"

    col_map = {c.name: c for c in meta.columns}
    assert set(col_map.keys()) == {"id", "student_no", "name", "gpa", "department_id", "created_at"}

    assert col_map["id"].is_primary_key is True
    assert col_map["id"].is_nullable is False
    assert col_map["id"].comment == "学生ID"

    assert col_map["gpa"].data_type.upper() == "DECIMAL(3, 2)"
    assert col_map["gpa"].is_nullable is True
    assert col_map["gpa"].comment == "平均绩点"

    assert len(meta.foreign_keys) == 1
    fk = meta.foreign_keys[0]
    assert fk.from_table == "students"
    assert fk.from_column == "department_id"
    assert fk.to_table == "departments"
    assert fk.to_column == "id"
    assert fk.inferred is False


def test_parse_sqlite_ddl_and_non_create_returns_none():
    meta = parse_ddl_to_table_meta(SQLITE_COURSES_DDL)
    assert meta is not None
    assert meta.name == "courses"

    col_map = {c.name: c for c in meta.columns}
    assert col_map["id"].is_primary_key is True
    assert col_map["id"].is_nullable is False
    assert col_map["title"].is_nullable is False
    assert col_map["credits"].is_nullable is True

    assert len(meta.foreign_keys) == 1
    assert meta.foreign_keys[0].from_column == "department_id"
    assert meta.foreign_keys[0].to_table == "departments"
    assert meta.foreign_keys[0].to_column == "id"

    assert parse_ddl_to_table_meta("") is None
    assert parse_ddl_to_table_meta("SELECT * FROM students;") is None


def test_build_schema_graph_with_implicit_fks():
    ddls = [
        MYSQL_STUDENTS_DDL,
        MYSQL_DEPARTMENTS_DDL,
        SQLITE_COURSES_DDL,
        SQLITE_ENROLLMENTS_DDL,
        MYSQL_CATEGORIES_DDL,
        MYSQL_PRODUCTS_DDL,
    ]
    graph = build_schema_graph(ddls, infer_implicit_fks=True)

    assert set(graph.tables.keys()) == {
        "students",
        "departments",
        "courses",
        "enrollments",
        "categories",
        "products",
    }

    edge_tuples = {(e.from_table, e.from_column, e.to_table, e.to_column, e.inferred) for e in graph.edges}
    # 显式外键不应被重复推断
    assert ("students", "department_id", "departments", "id", False) in edge_tuples
    assert ("courses", "department_id", "departments", "id", False) in edge_tuples
    # 隐式推断的外键
    assert ("enrollments", "student_id", "students", "id", True) in edge_tuples
    assert ("enrollments", "course_id", "courses", "id", True) in edge_tuples
    assert ("products", "category_id", "categories", "id", True) in edge_tuples

    # 禁用隐式推断
    graph_explicit = build_schema_graph(ddls, infer_implicit_fks=False)
    assert all(not e.inferred for e in graph_explicit.edges)
    assert len(graph_explicit.edges) == 2


def test_schema_graph_neighbors_and_shortest_path():
    graph = build_schema_graph(
        [MYSQL_STUDENTS_DDL, MYSQL_DEPARTMENTS_DDL, SQLITE_COURSES_DDL, SQLITE_ENROLLMENTS_DDL],
        infer_implicit_fks=True,
    )

    assert set(graph.neighbors("enrollments")) == {"students", "courses"}
    assert "enrollments" in graph.neighbors("students")
    assert graph.neighbors("nonexistent") == []

    assert graph.shortest_path("students", "students") == ["students"]
    assert graph.shortest_path("students", "nonexistent") is None

    path = graph.shortest_path("students", "courses")
    assert path is not None
    # students -> enrollments -> courses 或 students -> departments -> courses 均为 2 跳
    assert len(path) == 3
    assert path[0] == "students" and path[-1] == "courses"


def test_schema_linker_tokenize_english_and_chinese():
    en_tokens = SchemaLinker.tokenize("student_name camelCaseField UserID2")
    assert "student" in en_tokens
    assert "name" in en_tokens
    assert "camel" in en_tokens
    assert "case" in en_tokens
    assert "field" in en_tokens
    assert "user" in en_tokens
    assert "id" in en_tokens

    zh_tokens = SchemaLinker.tokenize("计算机科学与技术")
    assert "计算" in zh_tokens
    assert "算机" in zh_tokens
    assert "科学" in zh_tokens
    assert "技术" in zh_tokens
    assert "计" in zh_tokens


def test_link_with_value_matching_and_bridge_expansion():
    ddls = [
        MYSQL_STUDENTS_DDL,
        MYSQL_DEPARTMENTS_DDL,
        SQLITE_COURSES_DDL,
        SQLITE_ENROLLMENTS_DDL,
        MYSQL_CATEGORIES_DDL,
        MYSQL_PRODUCTS_DDL,
    ]
    graph = build_schema_graph(ddls, infer_implicit_fks=True)
    linker = SchemaLinker()

    value_samples = {
        "departments": {
            "name": ["计算机科学与技术", "软件工程", "数学学院"],
        }
    }

    # 查询同时涉及院系样本值（计算机）、学生（student）和课程（courses/enrollments）
    result = linker.link(
        query="查询计算机学院学生选修课程的成绩",
        graph=graph,
        top_k_tables=3,
        max_columns_per_table=10,
        value_samples=value_samples,
    )

    assert isinstance(result, LinkedSchema)
    assert "departments" in result.selected_tables
    assert "students" in result.selected_tables
    assert "departments" in result.value_hints
    assert "计算机科学与技术" in result.value_hints["departments"]["name"]

    # 无关表不应排在前面
    assert result.table_scores["departments"] > result.table_scores["products"]
    assert result.table_scores["students"] > result.table_scores["categories"]

    # 检查 ddl_context 包含关键信息
    assert "CREATE TABLE departments" in result.ddl_context
    assert "-- Value Hints:" in result.ddl_context
    assert "计算机科学与技术" in result.ddl_context


def test_link_bridge_table_auto_included_for_disconnected_components():
    # 只保留 students, courses, enrollments（去除 departments 以确保唯一路径经过 enrollments）
    graph = build_schema_graph(
        [
            "CREATE TABLE students (id INT PRIMARY KEY, name VARCHAR(50));",
            "CREATE TABLE courses (id INT PRIMARY KEY, title VARCHAR(50));",
            SQLITE_ENROLLMENTS_DDL,
        ],
        infer_implicit_fks=True,
    )
    linker = SchemaLinker()

    # top_k_tables=2 会先选出 students 和 courses（得分最高），enrollments 作为桥接表应被自动补全
    result = linker.link(
        query="List students and courses title",
        graph=graph,
        top_k_tables=2,
    )

    assert "students" in result.selected_tables
    assert "courses" in result.selected_tables
    assert "enrollments" in result.selected_tables
    assert any("enrollments.student_id = students.id" in h for h in result.join_hints)
    assert any("enrollments.course_id = courses.id" in h for h in result.join_hints)
    assert "-- Join Hints:" in result.ddl_context


def test_link_does_not_bridge_beyond_three_hops():
    # 链式 5 张表：t1 - t2 - t3 - t4 - t5（t1 到 t5 为 4 跳 > 3 跳）
    tables = {
        f"t{i}": TableMeta(
            name=f"t{i}",
            ddl=f"CREATE TABLE t{i} (id INT PRIMARY KEY);",
            columns=[ColumnMeta(name="id", data_type="INT", is_primary_key=True)],
        )
        for i in range(1, 6)
    }
    edges = [
        ForeignKeyMeta("t1", "id", "t2", "id"),
        ForeignKeyMeta("t2", "id", "t3", "id"),
        ForeignKeyMeta("t3", "id", "t4", "id"),
        ForeignKeyMeta("t4", "id", "t5", "id"),
    ]
    graph = SchemaGraph(tables=tables, edges=edges)
    linker = SchemaLinker()

    result = linker.link(query="t1 and t5", graph=graph, top_k_tables=2)
    assert set(result.selected_tables) == {"t1", "t5"}


def test_link_module_filter_and_column_pruning_and_llm_pruner():
    # 构造一个列数较多的表来测试列裁剪
    cols_ddl = ",\n  ".join([f"`extra_col_{i}` VARCHAR(20)" for i in range(15)])
    wide_table_ddl = f"""
    CREATE TABLE `orders` (
      `id` INT NOT NULL COMMENT '订单ID',
      `user_id` INT NOT NULL COMMENT '用户ID',
      `total_amount` DECIMAL(10, 2) COMMENT '订单总金额',
      `status` VARCHAR(20) COMMENT '订单状态',
      {cols_ddl},
      PRIMARY KEY (`id`)
    ) COMMENT='订单表';
    """
    users_ddl = """
    CREATE TABLE `users` (
      `id` INT NOT NULL,
      `username` VARCHAR(50),
      PRIMARY KEY (`id`)
    ) COMMENT='用户表';
    """
    graph = build_schema_graph([wide_table_ddl, users_ddl, MYSQL_PRODUCTS_DDL], infer_implicit_fks=True)
    linker = SchemaLinker()

    # 模块过滤只允许 orders 和 users
    result = linker.link(
        query="查询订单总金额",
        graph=graph,
        module_tables=["orders", "users"],
        top_k_tables=5,
        max_columns_per_table=4,
    )

    assert set(result.selected_tables).issubset({"orders", "users"})
    assert "orders" in result.selected_columns
    kept_order_cols = result.selected_columns["orders"]
    assert len(kept_order_cols) <= 4
    # PK, FK, 和查询命中的 total_amount 必须保留
    assert "id" in kept_order_cols
    assert "user_id" in kept_order_cols
    assert "total_amount" in kept_order_cols

    # 测试 llm_pruner 回调
    called_with = {}

    def fake_pruner(q, g, candidates):
        called_with["candidates"] = candidates
        return ["orders"]

    pruned_result = linker.link(
        query="查询订单和用户",
        graph=graph,
        top_k_tables=5,
        llm_pruner=fake_pruner,
    )
    assert "candidates" in called_with
    assert pruned_result.selected_tables == ["orders"]


def test_link_empty_graph():
    linker = SchemaLinker()
    empty_graph = SchemaGraph()
    res = linker.link("any query", empty_graph)
    assert res.selected_tables == []
    assert res.selected_columns == {}
    assert res.ddl_context == ""
    assert res.join_hints == []
    assert res.value_hints == {}
