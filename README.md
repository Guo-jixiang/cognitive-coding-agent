# 🧠 Cognitive Coding Agent

基于认知科学的**四维分离记忆架构 (Quad-Memory Architecture)** 的智能编码 Agent。采用 SubAgent 编排模式，通过 Reflection 范式实现自主推理、代码编写和任务执行。

## ✨ 核心特性

- **四维记忆系统**：工作记忆（TTL）、情景记忆（SQLite+Qdrant）、语义记忆（Neo4j+Qdrant）、感知记忆（多模态）
- **SubAgent 并行编排**：Orchestrator 拆解任务 → 消息队列分发 → 8 种专业 SubAgent 并行执行
- **三种推理范式**：ReAct（思考-行动-观察）、Plan-and-Solve（规划-执行）、Reflection（执行-反思-优化，默认）
- **工具隔离 & 上下文隔离**：每个 SubAgent 拥有独立的工具注册表和消息列表
- **多 LLM Provider 支持**：小米 MiMo、DashScope（通义千问）、DeepSeek、OpenAI 等，通过环境变量切换
- **双接口**：CLI 交互式 REPL + FastAPI HTTP 服务（支持 SSE 流式）

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      Interface Layer                               │
│   ┌─────────────────┐            ┌────────────────────────────┐  │
│   │  FastAPI Server  │            │        CLI REPL            │  │
│   │ /chat /ingest    │            │  --paradigm /clear /quit   │  │
│   └────────┬────────┘            └──────────────┬─────────────┘  │
├────────────┼─────────────────────────────────────┼────────────────┤
│            ▼                                     ▼                │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │                   Agent Engine                            │   │
│   └──────────────────────────┬───────────────────────────────┘   │
│                              ▼                                    │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │                    Orchestrator                            │   │
│   │   Task Decomposition → Dispatcher → Reflection            │   │
│   └──────┬───────────────────────────────────────────────────┘   │
│          ▼                                                        │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │              SubAgent Dispatcher (asyncio.Queue)           │   │
│   │   Wave 1: [analyzer]  →  Wave 2: [planner]               │   │
│   │   Wave 3: [coder]     →  Wave 4: [reviewer, tester] ∥    │   │
│   └──────────────────────────────────────────────────────────┘   │
│          ▼                                                        │
│   ┌──────────────────────────────────────────────────────────┐   │
│   │  SubAgents (Isolated Context & Tools)                     │   │
│   │  Planner│Coder│Reviewer│Tester│Analyzer│Debugger│Exec│Res │   │
│   └──────────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│                      Memory Layer                                  │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐      │
│   │ Working  │ │ Episodic │ │ Semantic │ │  Perceptual  │      │
│   │ (Dict)   │ │(SQLite+Q)│ │(Neo4j+Q) │ │ (Multimodal) │      │
│   └──────────┘ └──────────┘ └──────────┘ └──────────────┘      │
├──────────────────────────────────────────────────────────────────┤
│   Embedding: DashScope → LocalTransformer → TF-IDF (降级链)      │
├──────────────────────────────────────────────────────────────────┤
│   LLM: Xiaomi MiMo / DashScope / DeepSeek / OpenAI (可切换)     │
└──────────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
coding-agent/
├── coding_agents/
│   ├── api/                          # FastAPI HTTP 接口
│   │   ├── main.py                   # 应用入口 + lifespan 管理
│   │   ├── routes/
│   │   │   ├── chat.py              # POST /chat (SSE 流式)
│   │   │   ├── memory.py           # GET /memory/search, POST /memory/store
│   │   │   ├── ingest.py           # POST /ingest (文档摄入)
│   │   │   └── health.py           # GET /health (健康检查)
│   │   └── schemas/
│   │       └── models.py           # Pydantic 请求/响应模型
│   ├── cli/                          # 命令行交互界面
│   │   ├── main.py                   # click 入口 (--paradigm)
│   │   └── repl.py                   # 交互式 REPL 循环
│   ├── context/                      # 上下文工程
│   │   └── builder.py               # GSSC 管道 (Gather-Select-Structure-Compress)
│   ├── core/                         # 核心推理引擎
│   │   ├── engine.py                 # AgentEngine (中央执行引擎)
│   │   ├── agents/                   # SubAgent 框架
│   │   │   ├── base.py              # BaseSubAgent, SubAgentResult, SubAgentConfig
│   │   │   ├── messages.py          # TaskMessage, ResultMessage (消息协议)
│   │   │   ├── dispatcher.py        # SubAgentDispatcher (并行调度)
│   │   │   ├── orchestrator.py      # Orchestrator (任务编排)
│   │   │   ├── planner.py           # PlannerAgent (规划，无工具)
│   │   │   ├── coder.py             # CoderAgent (编码)
│   │   │   ├── reviewer.py          # ReviewerAgent (评审)
│   │   │   ├── tester.py            # TesterAgent (测试)
│   │   │   ├── analyzer.py          # AnalyzerAgent (分析)
│   │   │   ├── debugger.py          # DebuggerAgent (调试)
│   │   │   ├── executor.py          # ExecutorAgent (执行)
│   │   │   └── researcher.py        # ResearcherAgent (检索)
│   │   ├── paradigms/                # 推理范式
│   │   │   ├── base.py              # BaseParadigm, ParadigmResult
│   │   │   ├── react.py             # ReAct (Thought→Action→Observation)
│   │   │   ├── plan_and_solve.py    # Plan-and-Solve (规划→执行)
│   │   │   └── reflection.py        # Reflection (执行→反思→优化)
│   │   └── actions/                  # 工具动作
│   │       ├── base.py              # BaseAction, ActionSchema, ActionResult
│   │       ├── registry.py          # ActionRegistry (工具注册表)
│   │       ├── read_file.py         # 读取文件
│   │       ├── write_file.py        # 写入文件
│   │       ├── execute_command.py   # 执行命令
│   │       ├── search_code.py       # 搜索代码
│   │       └── list_directory.py    # 列出目录
│   ├── llm/                          # LLM 客户端
│   │   └── client.py                # 多 Provider 异步客户端
│   └── memory/                       # 四维记忆系统
│       ├── base.py                   # MemoryItem, BaseMemory, ScoringMixin
│       ├── manager.py               # MemoryManager (统一调度)
│       ├── embedding.py             # EmbeddingService (降级链)
│       ├── factory.py               # 工厂函数 (组装系统)
│       ├── types/
│       │   ├── working.py           # WorkingMemory (TTL, 纯内存)
│       │   ├── episodic.py          # EpisodicMemory (SQLite+Qdrant)
│       │   ├── semantic.py          # SemanticMemory (Neo4j+Qdrant)
│       │   └── perceptual.py        # PerceptualMemory (多模态)
│       ├── storage/
│       │   ├── document_store.py    # SQLiteDocumentStore (FTS5)
│       │   ├── qdrant_store.py      # QdrantVectorStore
│       │   └── neo4j_store.py       # Neo4jGraphStore
│       └── rag/
│           ├── document.py          # DocumentProcessor (多格式解析)
│           └── pipeline.py          # RAGPipeline (摄入+检索)
├── tests/                            # 测试套件
│   ├── unit/                         # 单元测试 (mock 外部服务)
│   ├── properties/                   # 属性测试 (Hypothesis)
│   └── integration/                  # 集成测试 (真实 Docker 服务)
├── docker-compose.yml                # Qdrant + Neo4j 本地服务
├── pyproject.toml                    # 项目配置和依赖
├── .env.example                      # 环境变量模板
└── prompt.md                         # Vibe Coding 参考文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/Guo-jixiang/cognitive-coding-agent.git
cd cognitive-coding-agent

# 创建 conda 环境
conda create -n coding_agent python=3.12 -y
conda activate coding_agent

# 安装依赖
pip install -e ".[dev]"
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

`.env` 关键配置：
```env
# LLM (选择一个 Provider)
LLM_API_KEY=your-api-key
LLM_MODEL_ID=your-model-name
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_TIMEOUT=60

# Embedding 配置
DASHSCOPE_API_KEY=your-dashscope-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=text-embedding-v3

# 本地服务 (可选，用于持久化记忆)
QDRANT_HOST=localhost
QDRANT_PORT=6333
NEO4J_URI=bolt://localhost:7687
```

### 3. 启动本地服务（可选）

```bash
docker-compose up -d
```

### 4. 启动 CLI

```bash
python -m coding_agents.cli.main
```

### 5. 启动 FastAPI 服务

```bash
uvicorn coding_agents.api.main:app --reload --port 8000
```

## 💬 CLI 使用示例

```
🧠 Cognitive Coding Agent REPL
   Paradigm: reflection | Type /quit to exit

You> 帮我写一个快速排序函数
  Thinking...
  [Plan] Decomposed into 3 sub-tasks: [analyzer], [coder], [reviewer]
  [Observation] [analyzer] ✓: 分析了项目结构
  [Observation] [coder] ✓: 编写了快速排序实现
  [Observation] [reviewer] ✓: 代码审查通过
  [Reflection] Synthesized final answer from sub-agent results.

Agent> 已为你编写了快速排序函数...

You> /paradigm react
  ✓ Paradigm switched to: react

You> /memory
  Memory Statistics:
    working: active
    episodic: active

You> /clear
  ✓ Conversation history cleared.

You> /quit
  Goodbye! 👋
```

## 🌐 API 使用示例

```bash
# 对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "写一个冒泡排序", "paradigm": "reflection"}'

# 流式对话 (SSE)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "解释递归", "stream": true}'

# 搜索记忆
curl "http://localhost:8000/memory/search?query=排序算法&top_k=5"

# 存储记忆
curl -X POST http://localhost:8000/memory/store \
  -H "Content-Type: application/json" \
  -d '{"content": "用户偏好Python", "memory_type": "episodic", "importance": 0.8}'

# 摄入文档
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./README.md", "chunk_size": 1000}'

# 健康检查
curl http://localhost:8000/health

# Swagger 文档
# 浏览器打开: http://localhost:8000/docs
```

## 🤖 SubAgent 工具权限

| SubAgent | 工具 | 职责 |
|----------|------|------|
| Planner | _(无)_ | 任务规划和分解 |
| Coder | read_file, write_file, list_directory | 代码编写和修改 |
| Reviewer | read_file, list_directory | 代码审查和反馈 |
| Tester | read_file, execute_command | 运行测试 |
| Analyzer | read_file, search_code, list_directory | 代码分析 |
| Debugger | read_file, search_code, execute_command | 问题排查 |
| Executor | execute_command, read_file | 命令执行 |
| Researcher | search_code, read_file, list_directory | 代码检索 |

## 📊 记忆系统评分公式

| 记忆类型 | 公式 |
|----------|------|
| Working | `(tfidf_similarity × time_decay) × (0.8 + importance × 0.4)` |
| Episodic | `(vector_sim × 0.8 + time_recency × 0.2) × (0.8 + importance × 0.4)` |
| Semantic | `(vector_sim × 0.7 + graph_sim × 0.3) × (0.8 + importance × 0.4)` |
| Perceptual | `(vector_sim × 0.8 + time_recency × 0.2) × (0.8 + importance × 0.4)` |

时间衰减：`exp(-ln(2)/86400 × elapsed_seconds)` → 24小时衰减50%

## 🧪 测试

```bash
# 单元测试 + 属性测试
pytest tests/unit/ tests/properties/ -v

# 类型检查
mypy coding_agents --strict

# 代码风格
ruff check coding_agents

# 集成测试 (需要 Docker 服务)
pytest tests/integration/ -v -m integration
```

## 📋 CLI 特殊命令

| 命令 | 说明 |
|------|------|
| `/quit` | 退出 REPL |
| `/clear` | 清除对话历史 |
| `/paradigm <name>` | 切换推理范式 (react / plan_and_solve / reflection) |
| `/memory` | 显示记忆系统状态 |

## 🔧 技术栈

- **语言**: Python 3.12+
- **LLM SDK**: OpenAI (AsyncOpenAI)
- **向量数据库**: Qdrant
- **图数据库**: Neo4j
- **文档存储**: SQLite (aiosqlite + FTS5)
- **Embedding**: DashScope / sentence-transformers / TF-IDF
- **Web 框架**: FastAPI + Uvicorn + SSE-Starlette
- **CLI**: Click
- **测试**: pytest + Hypothesis + pytest-asyncio
- **类型检查**: mypy (strict mode)
- **代码风格**: Ruff

## 📄 License

MIT
