# Cognitive Coding Agent — Vibe Coding 主 Prompt

## 项目概述

你是一个自主编码 Agent（主 Agent），负责完整实现 **Cognitive Coding Agent** 项目——一个基于认知科学的"四维分离记忆架构 (Quad-Memory Architecture)"的 Python 库。整个过程无人工参与，你需要自主完成所有编码、测试、质量检查工作。

## 环境信息

| 项目 | 值 |
|------|------|
| Python 版本 | 3.12 |
| 虚拟环境 | conda 环境 `coding_agent`（已创建） |
| 操作系统 | Windows |
| 外部服务 | Qdrant (本地 Docker, localhost:6333)、Neo4j (本地 Docker, bolt://localhost:7687) |
| Embedding API | DashScope (阿里云) |
| 项目根目录 | 当前工作目录 |

## 环境变量 (.env)

```env
# Embedding 配置
DASHSCOPE_API_KEY="sk-72bb477990a54a689be28098f702cd18"
DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL="text-embedding-v3"

# Qdrant 本地配置
QDRANT_HOST="localhost"
QDRANT_PORT=6333
QDRANT_API_KEY=""

# Neo4j 本地配置
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="coding-agent-password"
NEO4J_DATABASE="neo4j"
```

## Docker Compose 配置

在项目根目录创建 `docker-compose.yml`，包含以下服务：

```yaml
version: "3.8"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

  neo4j:
    image: neo4j:5-community
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/coding-agent-password
      NEO4J_PLUGINS: '["apoc"]'
    volumes:
      - neo4j_data:/data
    restart: unless-stopped

volumes:
  qdrant_data:
  neo4j_data:
```

启动命令：`docker-compose up -d`

---

## 你的角色与工作模式

### 主 Agent 职责

1. **进度跟踪**：按照下方任务列表顺序执行，跟踪每个任务的完成状态
2. **子 Agent 调度**：为每个模块生成子 Agent 任务，子 Agent 负责具体编码和测试
3. **质量把关**：每个模块完成后运行 pytest、mypy、ruff 确保通过
4. **集成验证**：在 Checkpoint 节点运行全量测试

### 子 Agent 划分（按模块）

| 子 Agent | 负责模块 | 文件 |
|----------|----------|------|
| Agent-Infrastructure | 项目结构 + 基础数据结构 + 抽象基类 | `pyproject.toml`, `coding_agents/memory/base.py` |
| Agent-Storage | 三个存储后端 | `storage/document_store.py`, `storage/qdrant_store.py`, `storage/neo4j_store.py` |
| Agent-Embedding | 嵌入服务 | `coding_agents/memory/embedding.py` |
| Agent-WorkingMemory | 工作记忆 | `coding_agents/memory/types/working.py` |
| Agent-EpisodicMemory | 情景记忆 | `coding_agents/memory/types/episodic.py` |
| Agent-SemanticMemory | 语义记忆 | `coding_agents/memory/types/semantic.py` |
| Agent-PerceptualMemory | 感知记忆 | `coding_agents/memory/types/perceptual.py` |
| Agent-Manager | 记忆管理器 | `coding_agents/memory/manager.py` |
| Agent-RAG | RAG 管道 + 文档处理器 | `coding_agents/memory/rag/pipeline.py`, `rag/document.py` |
| Agent-Context | 上下文构建器 | `coding_agents/context/builder.py` |
| Agent-Integration | 集成装配 + 端到端测试 | `coding_agents/memory/factory.py`, `tests/integration/` |

### 并行策略

- Wave 0-1（基础设施）：串行执行
- Wave 2（存储后端 + 属性测试）：Agent-Storage 可并行处理三个存储后端
- Wave 3-4（嵌入 + 四种记忆）：Agent-Embedding 先行，四种记忆类型可并行
- Wave 6+（Manager → RAG → Context → Integration）：串行执行，有依赖关系

---

## 质量要求（强制）

### 代码规范

1. **类型注解**：所有函数签名必须有完整的 type hints，通过 `mypy --strict` 检查
2. **代码风格**：通过 `ruff check` 和 `ruff format --check`，使用默认规则
3. **异步设计**：所有 I/O 操作使用 `async/await`
4. **文档字符串**：所有公开类和方法必须有 docstring

### 测试要求

1. **单元测试**：每个模块必须有对应的 `tests/unit/test_<module>.py`
2. **属性测试**：使用 Hypothesis 库，放在 `tests/properties/test_<module>.py`
3. **集成测试**：连接 Docker Compose 真实服务，放在 `tests/integration/test_<module>.py`
4. **Mock 策略**：单元测试和属性测试全部 mock 外部服务（Qdrant、Neo4j、DashScope）
5. **覆盖率**：核心逻辑覆盖率 > 80%

### 验证命令

每个模块完成后执行：

```bash
# 激活环境
conda activate coding_agent

# 类型检查
mypy coding_agents/ --strict

# 代码风格
ruff check coding_agents/ tests/
ruff format --check coding_agents/ tests/

# 单元测试 + 属性测试
pytest tests/unit/ tests/properties/ -v --tb=short

# 集成测试（需要 Docker 服务运行）
pytest tests/integration/ -v --tb=short -m integration
```

---

## 项目结构

```
coding-agent/
├── pyproject.toml                    # 项目配置和依赖
├── docker-compose.yml                # 本地服务编排
├── .env                              # 环境变量（不提交 git）
├── .env.example                      # 环境变量模板
├── prompt.md                         # 本文件
├── coding_agents/
│   ├── __init__.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── base.py                   # MemoryItem, MemoryConfig, BaseMemory, ScoringMixin
│   │   ├── manager.py               # MemoryManager
│   │   ├── embedding.py             # EmbeddingService + 三个后端
│   │   ├── factory.py               # 工厂函数，组装所有组件
│   │   ├── types/
│   │   │   ├── __init__.py
│   │   │   ├── working.py           # WorkingMemory
│   │   │   ├── episodic.py          # EpisodicMemory
│   │   │   ├── semantic.py          # SemanticMemory
│   │   │   └── perceptual.py        # PerceptualMemory
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── qdrant_store.py      # QdrantVectorStore
│   │   │   ├── neo4j_store.py       # Neo4jGraphStore
│   │   │   └── document_store.py    # SQLiteDocumentStore
│   │   └── rag/
│   │       ├── __init__.py
│   │       ├── pipeline.py          # RAGPipeline
│   │       └── document.py          # DocumentProcessor
│   └── context/
│       ├── __init__.py
│       └── builder.py               # ContextBuilder
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # 共享 fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   ├── test_embedding.py
│   │   ├── test_working_memory.py
│   │   ├── test_episodic_memory.py
│   │   ├── test_semantic_memory.py
│   │   ├── test_perceptual_memory.py
│   │   ├── test_manager.py
│   │   ├── test_document_processor.py
│   │   ├── test_rag_pipeline.py
│   │   ├── test_context_builder.py
│   │   └── test_storage/
│   │       ├── __init__.py
│   │       ├── test_sqlite_store.py
│   │       ├── test_qdrant_store.py
│   │       └── test_neo4j_store.py
│   ├── properties/
│   │   ├── __init__.py
│   │   ├── test_memory_item.py       # Property 1: 序列化往返
│   │   ├── test_scoring.py           # Property 3, 8: 评分公式 + 时间衰减
│   │   ├── test_working_memory.py    # Property 2: TTL 生命周期
│   │   ├── test_embedding.py         # Property 9, 10: 嵌入不变量
│   │   ├── test_semantic_memory.py   # Property 5: 级联删除
│   │   ├── test_perceptual_memory.py # Property 6, 7: 模态路由
│   │   ├── test_manager.py           # Property 11, 12, 13: 路由/隔离/去重
│   │   ├── test_retrieval.py         # Property 4: 结果排序
│   │   ├── test_document.py          # Property 16, 17: 分块往返
│   │   ├── test_context.py           # Property 14, 15: 预算/压缩
│   │   └── test_persistence.py       # Property 18: 持久化往返
│   └── integration/
│       ├── __init__.py
│       ├── conftest.py               # Docker 服务连接 fixtures
│       ├── test_qdrant_integration.py
│       ├── test_neo4j_integration.py
│       ├── test_memory_lifecycle.py
│       └── test_rag_e2e.py
└── py.typed                          # PEP 561 marker
```

---

## 任务执行顺序

按照依赖图的 Wave 顺序执行：

### Phase 1: 基础设施（Wave 0-1）
1. 创建项目结构、`pyproject.toml`、`docker-compose.yml`、`.env.example`
2. 实现 `MemoryItem`、`MemoryConfig` 数据模型
3. 实现 `BaseMemory` 抽象基类和 `ScoringMixin`
4. 编写属性测试：序列化往返、评分公式、时间衰减

### Phase 2: 存储后端（Wave 2）
5. 实现 `SQLiteDocumentStore`
6. 实现 `QdrantVectorStore`
7. 实现 `Neo4jGraphStore`
8. 编写存储后端单元测试

### Phase 3: 嵌入服务（Wave 3）
9. 实现 `EmbeddingService` + 三个后端 + 降级链
10. 编写嵌入属性测试

### Phase 4: 四种记忆类型（Wave 4-5）
11. 实现 `WorkingMemory` + TTL 属性测试
12. 实现 `EpisodicMemory` + 持久化属性测试
13. 实现 `SemanticMemory` + 级联删除属性测试
14. 实现 `PerceptualMemory` + 模态路由属性测试

### Phase 5: 记忆管理器（Wave 6-7）
15. 实现 `MemoryManager` + 路由/隔离/去重属性测试

### Phase 6: RAG 管道（Wave 7-8）
16. 实现 `DocumentProcessor` + 分块属性测试
17. 实现 `RAGPipeline`

### Phase 7: 上下文构建器（Wave 9-10）
18. 实现 `ContextBuilder` + 预算/压缩属性测试

### Phase 8: 集成（Wave 11-12）
19. 实现 `factory.py` 组装所有组件
20. 编写集成测试（连接真实 Docker 服务）
21. 全量验证：pytest + mypy + ruff

---

## 关键技术规格

### 评分公式

| 记忆类型 | 公式 |
|----------|------|
| Working Memory | `(tfidf_similarity × time_decay) × (0.8 + importance × 0.4)` |
| Episodic Memory | `(vector_similarity × 0.8 + time_recency × 0.2) × (0.8 + importance × 0.4)` |
| Semantic Memory | `(vector_similarity × 0.7 + graph_similarity × 0.3) × (0.8 + importance × 0.4)` |
| Perceptual Memory | `(vector_similarity × 0.8 + time_recency × 0.2) × (0.8 + importance × 0.4)` |

**时间衰减函数**：`time_decay = exp(-decay_rate × elapsed_seconds)`
- 默认 `decay_rate = ln(2) / 86400`，24小时后衰减50%

### MemoryItem 数据结构

```python
@dataclass
class MemoryItem:
    id: str                    # UUID v4
    content: str               # 最大 100,000 字符
    metadata: Dict[str, Any]   # 可扩展元数据
    importance: float          # 0.0 ~ 1.0，默认 0.5
    created_at: datetime       # UTC 创建时间
    last_accessed_at: datetime # UTC 最后访问时间
    memory_type: str           # "working" | "episodic" | "semantic" | "perceptual"
```

### 嵌入服务降级链

```
DashScope API → LocalTransformer (all-MiniLM-L6-v2) → TF-IDF
```

- DashScope 模型：`text-embedding-v3`
- 本地模型：`sentence-transformers/all-MiniLM-L6-v2`（384维）
- TF-IDF：scikit-learn TfidfVectorizer（兜底方案）
- 所有输出向量归一化为单位长度（L2 norm = 1.0）

### 存储后端接口

每个存储后端必须实现对应的抽象接口，支持替换（substitutability）：
- `QdrantVectorStore`：store, search, delete, health_check, create_collection
- `Neo4jGraphStore`：create_node, create_relationship, get_neighbors, delete_node, delete_relationship, health_check
- `SQLiteDocumentStore`：store, get, search, delete, list_all

### 错误处理策略

- 嵌入服务不可用 → 按优先级降级
- Qdrant 不可用 → 回退到 SQLite 全文搜索
- Neo4j 不可用 → 语义记忆仅使用向量模式
- 单个子系统故障 → 不影响其他子系统（故障隔离）

---

## pyproject.toml 依赖参考

```toml
[project]
name = "cognitive-coding-agent"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "qdrant-client>=1.9.0",
    "neo4j>=5.20.0",
    "aiosqlite>=0.20.0",
    "scikit-learn>=1.4.0",
    "numpy>=1.26.0",
    "tiktoken>=0.7.0",
    "pyyaml>=6.0.0",
    "openai>=1.30.0",          # DashScope 兼容 OpenAI SDK
    "sentence-transformers>=3.0.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "hypothesis>=6.100.0",
    "mypy>=1.10.0",
    "ruff>=0.4.0",
]

[tool.mypy]
strict = true
python_version = "3.12"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "integration: marks tests requiring Docker services",
]
```

---

## 正确性属性（Property-Based Testing）

以下 18 个属性必须通过 Hypothesis 验证（`@settings(max_examples=100)`）：

1. **MemoryItem 序列化往返**：serialize → deserialize 等价于原始对象
2. **Working Memory TTL 生命周期**：TTL 前可检索，TTL 后不可检索
3. **评分公式正确性**：计算结果匹配公式且在 [0, 1] 范围内
4. **检索结果排序**：按 Relevance_Score 降序，相同分数按 importance 降序
5. **级联删除完整性**：删除节点后无孤立关系和向量
6. **感知记忆模态路由**：存储到正确集合，同模态检索不跨集合
7. **跨模态检索合并**：搜索所有集合并按分数合并
8. **时间衰减指数模型**：单调递减且在 (0, 1] 范围内
9. **嵌入输出不变量**：固定维度 + L2 范数 = 1.0
10. **嵌入确定性**：相同输入产生相同输出，批量等价于逐个
11. **Memory Manager 路由**：按 memory_type 路由到正确子系统
12. **子系统故障隔离**：单个故障不影响其他子系统
13. **跨记忆去重**：相同 ID 只保留最高分条目
14. **Context Select 预算约束**：选择不超过 token 预算
15. **Context Compress 限制**：压缩后不超过 token 限制，保留高重要性项
16. **文档分块往返**：chunk → reassemble 等价于原始内容
17. **分块尺寸约束**：每块不超过 chunk_size，重叠率符合配置
18. **情景记忆持久化往返**：shutdown → startup 后状态等价

---

## 执行指令

开始执行时：
1. 先创建 `docker-compose.yml` 和 `.env.example`
2. 运行 `docker-compose up -d` 启动本地服务
3. 按 Phase 顺序逐步实现每个模块
4. 每个模块完成后立即运行 mypy + ruff + pytest 验证
5. 如果验证失败，立即修复后再继续下一个模块
6. 所有模块完成后运行全量集成测试

**开始吧。**
