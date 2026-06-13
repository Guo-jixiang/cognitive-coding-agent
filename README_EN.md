# Cognitive Coding Agent

English | [中文](README.md)

> An intelligent coding agent with Quad-Memory Architecture based on cognitive science. Features SubAgent parallel orchestration, three reasoning paradigms (ReAct / Plan-and-Solve / Reflection), and dual CLI + FastAPI interfaces.

## Key Features

- **Quad-Memory System**: Working Memory (TTL), Episodic Memory (SQLite+Qdrant), Semantic Memory (Neo4j+Qdrant), Perceptual Memory (multimodal)
- **SubAgent Parallel Orchestration**: Orchestrator decomposes tasks -> message queue dispatch -> 8 specialized SubAgents execute in parallel
- **Three Reasoning Paradigms**: ReAct (Thought-Action-Observation), Plan-and-Solve (Plan-Execute), Reflection (Execute-Reflect-Refine, default)
- **Tool & Context Isolation**: Each SubAgent has its own isolated tool registry and message list
- **Multi-Provider LLM Support**: Xiaomi MiMo, DashScope (Qwen), DeepSeek, OpenAI, and more — switchable via environment variables
- **Dual Interface**: CLI interactive REPL + FastAPI HTTP service (SSE streaming)

## System Architecture

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
│   Embedding: DashScope → LocalTransformer → TF-IDF (fallback)    │
├──────────────────────────────────────────────────────────────────┤
│   LLM: Xiaomi MiMo / DashScope / DeepSeek / OpenAI (switchable) │
└──────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
coding-agent/
├── coding_agents/
│   ├── api/                          # FastAPI HTTP interface
│   │   ├── main.py                   # App entry + lifespan management
│   │   ├── routes/
│   │   │   ├── chat.py              # POST /chat (SSE streaming)
│   │   │   ├── memory.py           # GET /memory/search, POST /memory/store
│   │   │   ├── ingest.py           # POST /ingest (document ingestion)
│   │   │   └── health.py           # GET /health (health check)
│   │   └── schemas/
│   │       └── models.py           # Pydantic request/response models
│   ├── cli/                          # Command-line interface
│   │   ├── main.py                   # Click entry point (--paradigm)
│   │   └── repl.py                   # Interactive REPL loop
│   ├── context/                      # Context engineering
│   │   └── builder.py               # GSSC pipeline (Gather-Select-Structure-Compress)
│   ├── core/                         # Core reasoning engine
│   │   ├── engine.py                 # AgentEngine (central execution engine)
│   │   ├── agents/                   # SubAgent framework
│   │   │   ├── base.py              # BaseSubAgent, SubAgentResult, SubAgentConfig
│   │   │   ├── messages.py          # TaskMessage, ResultMessage (message protocol)
│   │   │   ├── dispatcher.py        # SubAgentDispatcher (parallel dispatch)
│   │   │   ├── orchestrator.py      # Orchestrator (task orchestration)
│   │   │   ├── planner.py           # PlannerAgent (planning, no tools)
│   │   │   ├── coder.py             # CoderAgent (code generation)
│   │   │   ├── reviewer.py          # ReviewerAgent (code review)
│   │   │   ├── tester.py            # TesterAgent (testing)
│   │   │   ├── analyzer.py          # AnalyzerAgent (analysis)
│   │   │   ├── debugger.py          # DebuggerAgent (debugging)
│   │   │   ├── executor.py          # ExecutorAgent (execution)
│   │   │   └── researcher.py        # ResearcherAgent (research)
│   │   ├── paradigms/                # Reasoning paradigms
│   │   │   ├── base.py              # BaseParadigm, ParadigmResult
│   │   │   ├── react.py             # ReAct (Thought→Action→Observation)
│   │   │   ├── plan_and_solve.py    # Plan-and-Solve (Plan→Execute)
│   │   │   └── reflection.py        # Reflection (Execute→Reflect→Refine)
│   │   └── actions/                  # Tool actions
│   │       ├── base.py              # BaseAction, ActionSchema, ActionResult
│   │       ├── registry.py          # ActionRegistry (tool registry)
│   │       ├── read_file.py         # Read file
│   │       ├── write_file.py        # Write file
│   │       ├── execute_command.py   # Execute command
│   │       ├── search_code.py       # Search code
│   │       └── list_directory.py    # List directory
│   ├── llm/                          # LLM client
│   │   └── client.py                # Multi-provider async client
│   └── memory/                       # Quad-Memory system
│       ├── base.py                   # MemoryItem, BaseMemory, ScoringMixin
│       ├── manager.py               # MemoryManager (unified coordination)
│       ├── embedding.py             # EmbeddingService (fallback chain)
│       ├── factory.py               # Factory functions (system assembly)
│       ├── types/
│       │   ├── working.py           # WorkingMemory (TTL, in-memory)
│       │   ├── episodic.py          # EpisodicMemory (SQLite+Qdrant)
│       │   ├── semantic.py          # SemanticMemory (Neo4j+Qdrant)
│       │   └── perceptual.py        # PerceptualMemory (multimodal)
│       ├── storage/
│       │   ├── document_store.py    # SQLiteDocumentStore (FTS5)
│       │   ├── qdrant_store.py      # QdrantVectorStore
│       │   └── neo4j_store.py       # Neo4jGraphStore
│       └── rag/
│           ├── document.py          # DocumentProcessor (multi-format parsing)
│           └── pipeline.py          # RAGPipeline (ingestion + retrieval)
├── tests/                            # Test suite (298+ unit tests)
│   ├── conftest.py                   # Shared fixtures
│   ├── unit/                         # Unit tests (mocked external services)
│   │   ├── test_actions/             # Tool action tests
│   │   │   ├── test_registry.py     # ActionRegistry: register, lookup, overwrite
│   │   │   ├── test_read_file.py    # ReadFileAction: read, missing, permissions, encoding
│   │   │   ├── test_write_file.py   # WriteFileAction: write, create dirs, overwrite
│   │   │   ├── test_execute_command.py # ExecuteCommandAction: exec, timeout, stderr
│   │   │   ├── test_search_code.py  # SearchCodeAction: regex, glob, recursive
│   │   │   └── test_list_directory.py # ListDirectoryAction: list, recursive, sorted
│   │   ├── test_paradigms/           # Reasoning paradigm tests
│   │   │   ├── test_paradigm_factory.py # ParadigmFactory: create 3 paradigms, errors
│   │   │   ├── test_react.py        # ReAct: parse LLM output, tool desc, loop exec
│   │   │   ├── test_plan_and_solve.py # PlanAndSolve: parse plan, extract action, exec
│   │   │   └── test_reflection.py   # Reflection: LGTM detection, reflect loop, max iter
│   │   ├── test_engine.py            # AgentEngine: lifecycle, delegation, memory storage
│   │   ├── test_orchestrator.py      # Orchestrator: simple msg, JSON parse, reflect
│   │   ├── test_dispatcher.py        # SubAgentDispatcher: parallel, dep waves, errors
│   │   ├── test_llm_client.py        # LLMClient: construction, auto-detect, invoke
│   │   ├── test_context_builder.py   # ContextBuilder: GSSC pipeline, select, compress
│   │   ├── test_memory_manager.py    # MemoryManager: init, fault isolation, dedup
│   │   ├── test_messages.py          # TaskMessage, ResultMessage, SubAgentResult
│   │   ├── test_rag_document.py      # DocumentProcessor: 5 formats, chunk, reassemble
│   │   ├── test_rag_pipeline.py      # RAGPipeline: ingest, query, dir batch, fallback
│   │   ├── test_working_memory.py    # WorkingMemory: TTL, TF-IDF, keyword fallback
│   │   ├── test_episodic_memory.py   # EpisodicMemory: SQLite+Qdrant, persistence
│   │   ├── test_semantic_memory.py   # SemanticMemory: vector+graph hybrid retrieval
│   │   ├── test_perceptual_memory.py # PerceptualMemory: multimodal, cross-modal
│   │   └── test_storage/             # Storage backend tests
│   │       ├── test_sqlite_store.py  # SQLiteDocumentStore: CRUD, FTS5
│   │       ├── test_qdrant_store.py  # QdrantVectorStore: collection, upsert, search
│   │       └── test_neo4j_store.py   # Neo4jGraphStore: nodes, relationships, traversal
│   ├── properties/                   # Property-based tests (Hypothesis, 100 examples/test)
│   │   ├── test_memory_item.py       # MemoryItem serialization round-trip
│   │   ├── test_scoring.py           # Scoring formula correctness, time decay monotonicity
│   │   └── test_embedding.py         # Embedding dimension, L2 normalization, determinism
│   └── integration/                  # Integration tests (real Docker services)
├── docker-compose.yml                # Qdrant + Neo4j local services
├── pyproject.toml                    # Project config and dependencies
├── .env.example                      # Environment variable template
├── LICENSE                           # MIT License
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Guo-jixiang/cognitive-coding-agent.git
cd cognitive-coding-agent

# Create conda environment
conda create -n coding_agent python=3.12 -y
conda activate coding_agent

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and fill in your API key
```

Key `.env` configuration:
```env
# LLM (choose a provider)
LLM_API_KEY=your-api-key
LLM_MODEL_ID=your-model-name
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_TIMEOUT=60

# Embedding configuration
DASHSCOPE_API_KEY=your-dashscope-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=text-embedding-v3   # default embedding model

# Local services (optional, for persistent memory)
QDRANT_HOST=localhost
QDRANT_PORT=6333
NEO4J_URI=bolt://localhost:7687
```

### 3. Start Local Services (Optional)

```bash
docker-compose up -d
```

### 4. Start CLI

```bash
python -m coding_agents.cli.main
```

### 5. Start FastAPI Server

```bash
uvicorn coding_agents.api.main:app --reload --port 8000
```

## CLI Usage Example

```
Cognitive Coding Agent REPL
   Paradigm: reflection | Type /quit to exit

You> Write a quicksort function
  Thinking...
  [Plan] Decomposed into 3 sub-tasks: [analyzer], [coder], [reviewer]
  [Observation] [analyzer] : Analyzed project structure
  [Observation] [coder] : Wrote quicksort implementation
  [Observation] [reviewer] : Code review passed
  [Reflection] Synthesized final answer from sub-agent results.

Agent> Here is the quicksort function...

You> /paradigm react
  Paradigm switched to: react

You> /memory
  Memory Statistics:
    working: active
    episodic: active

You> /clear
  Conversation history cleared.

You> /quit
  Goodbye!
```

## API Usage Example

```bash
# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a bubble sort", "paradigm": "reflection"}'

# Streaming chat (SSE)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain recursion", "stream": true}'

# Search memory
curl "http://localhost:8000/memory/search?query=sorting+algorithm&top_k=5"

# Store memory
curl -X POST http://localhost:8000/memory/store \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers Python", "memory_type": "episodic", "importance": 0.8}'

# Ingest document
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "./README.md", "chunk_size": 1000}'

# Health check
curl http://localhost:8000/health

# Swagger docs
# Open in browser: http://localhost:8000/docs
```

## SubAgent Tool Permissions

| SubAgent | Tools | Responsibility |
|----------|-------|----------------|
| Planner | _(none)_ | Task planning and decomposition |
| Coder | read_file, write_file, list_directory | Code generation and modification |
| Reviewer | read_file, list_directory | Code review and feedback |
| Tester | read_file, execute_command | Running tests |
| Analyzer | read_file, search_code, list_directory | Code analysis |
| Debugger | read_file, search_code, execute_command | Issue diagnosis |
| Executor | execute_command, read_file | Command execution |
| Researcher | search_code, read_file, list_directory | Code research |

## Memory Scoring Formulas

| Memory Type | Formula |
|-------------|---------|
| Working | `(tfidf_similarity * time_decay) * (0.8 + importance * 0.4)` |
| Episodic | `(vector_sim * 0.8 + time_recency * 0.2) * (0.8 + importance * 0.4)` |
| Semantic | `(vector_sim * 0.7 + graph_sim * 0.3) * (0.8 + importance * 0.4)` |
| Perceptual | `(vector_sim * 0.8 + time_recency * 0.2) * (0.8 + importance * 0.4)` |

Time decay: `exp(-ln(2)/86400 * elapsed_seconds)` -> 50% decay in 24 hours

## Testing

The project uses a layered testing strategy covering the full chain from low-level tools to top-level orchestration:

```
Test Pyramid
─────────────────────────────
│  Integration (Docker)      |  <- Qdrant / Neo4j real services
│  Property (Hypothesis)     |  <- Scoring formulas, Embedding invariants
│  Unit (298+ tests)         |  <- All modules, mocked isolation
─────────────────────────────
```

### Test Coverage

| Layer | Test File | Covered Module | Test Points |
|-------|-----------|----------------|-------------|
| **Tool Actions** | `test_actions/` | 5 Actions + Registry | CRUD, error handling, edge cases, regex/timeout |
| **Reasoning Paradigms** | `test_paradigms/` | ReAct / PlanAndSolve / Reflection | LLM output parsing, loop control, LGTM detection |
| **Core Engine** | `test_engine.py` | AgentEngine | Lifecycle, paradigm exec, memory storage, error tolerance |
| **Orchestration** | `test_orchestrator.py` / `test_dispatcher.py` | Orchestrator / Dispatcher | Task decomposition, parallel dispatch, dep waves, reflection |
| **LLM Client** | `test_llm_client.py` | LLMClient | Provider auto-detect, message formatting, invoke/close |
| **Context Builder** | `test_context_builder.py` | ContextBuilder | GSSC 4-stage pipeline, token budget, high-importance preservation |
| **Memory Manager** | `test_memory_manager.py` | MemoryManager | Initialization, fault isolation, dedup, cross-memory search |
| **RAG Pipeline** | `test_rag_document.py` / `test_rag_pipeline.py` | DocumentProcessor / RAGPipeline | 5-format parsing, chunk round-trip, batch ingestion, fallback |
| **Memory Types** | `test_*_memory.py` | 4 memory implementations | TTL, vector+graph hybrid, multimodal, persistence |
| **Storage Backends** | `test_storage/` | SQLite / Qdrant / Neo4j | CRUD, FTS5 full-text search, health check |
| **Property Tests** | `properties/` | MemoryItem / Scoring / Embedding | Serialization round-trip, formula correctness, L2 normalization |

### Running Tests

```bash
# All unit tests + property tests
pytest tests/unit/ tests/properties/ -v

# Only new tests (tools/paradigms/engine/orchestration/LLM/context/RAG)
pytest tests/unit/test_actions/ tests/unit/test_paradigms/ \
       tests/unit/test_engine.py tests/unit/test_orchestrator.py \
       tests/unit/test_dispatcher.py tests/unit/test_llm_client.py \
       tests/unit/test_context_builder.py tests/unit/test_memory_manager.py \
       tests/unit/test_messages.py tests/unit/test_rag_document.py \
       tests/unit/test_rag_pipeline.py -v

# Single module test
pytest tests/unit/test_actions/test_read_file.py -v

# Type checking
mypy coding_agents --strict

# Code style
ruff check coding_agents

# Integration tests (requires Docker services)
pytest tests/integration/ -v -m integration
```

## CLI Special Commands

| Command | Description |
|---------|-------------|
| `/quit` | Exit REPL |
| `/clear` | Clear conversation history |
| `/paradigm <name>` | Switch reasoning paradigm (react / plan_and_solve / reflection) |
| `/memory` | Show memory system status |

## Tech Stack

- **Language**: Python 3.12+
- **LLM SDK**: OpenAI (AsyncOpenAI)
- **Vector Database**: Qdrant
- **Graph Database**: Neo4j
- **Document Store**: SQLite (aiosqlite + FTS5)
- **Embedding**: DashScope / sentence-transformers / TF-IDF
- **Web Framework**: FastAPI + Uvicorn + SSE-Starlette
- **CLI**: Click
- **Testing**: pytest + Hypothesis + pytest-asyncio
- **Type Checking**: mypy (strict mode)
- **Code Style**: Ruff

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
