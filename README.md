## RAG Workbench (FastAPI + Streamlit)

一个可运行的 RAG 项目骨架，包含：

- **前端**：Streamlit 工作台（登录/注册、上传文档、文档列表、聊天）
- **后端**：FastAPI（认证、文档入库/列出、对话接口）
- **LLM/RAG**：Chroma 向量库 +（可选）BM25 + RRF 重排 +（可选）最终压缩
- **评测**：中文 RAG 召回评测集（10k queries）脚本（Recall@K / MRR@K）

### 项目结构

- `backend/`：FastAPI 后端
- `web/`：Streamlit 前端
- `llm/`：检索、重排、切分、回答生成
- `test/`：评测脚本与数据集
- `project_config.py`：集中配置（通过环境变量覆盖）
- `main.py`：后端启动入口（导出 `app`）
- `test_main.http`：接口手工测试样例
- `API_SPEC.md`：接口规范/前后端对齐说明

---

## 环境准备（Windows / PowerShell）

建议使用 Python 3.10+ 的虚拟环境。

### 安装依赖（最小可运行集合）

```bash
pip install fastapi uvicorn[standard] sqlalchemy aiomysql pydantic
pip install streamlit requests
pip install python-dotenv
pip install langchain-openai langchain-community langchain-text-splitters chromadb
pip install langchain-chroma
```

> 说明：项目里既有代码使用 `langchain_community.vectorstores.Chroma`，也有代码使用 `langchain_chroma.Chroma`。推荐安装 `langchain-chroma` 来避免 LangChain 的弃用警告。

### 必要环境变量

至少需要：

- **`OPENAI_API_KEY`**：用于 `OpenAIEmbeddings` / `ChatOpenAI`

可选（有默认值）：

- **`API_BASE_URL`**：前端调用后端的地址（默认 `http://127.0.0.1:8000/api`）
- **`ASYNC_DATABASE_URL`**：后端 MySQL 连接（默认 `mysql+aiomysql://root:123456@localhost:3306/rag?charset=utf8mb4`）
- **`EMBEDDING_MODEL`**：默认 `text-embedding-3-small`
- **`LLM_MODEL`**：默认 `gpt-4o`
- **`RERANK_MODEL_NAME`**：默认 `BAAI/bge-reranker-v2-m3`

你可以把 OpenAI Key 写在 `llm/.env`（项目里会读取它），例如：

```ini
OPENAI_API_KEY=your_key_here
```

---

## 启动后端（FastAPI）

在项目根目录运行：

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

健康检查：

- `GET http://127.0.0.1:8000/`
- Swagger：`GET http://127.0.0.1:8000/docs`

### 数据库建表

后端启动时会在 lifespan 中调用 `backend/app/core/database.py:init_db()`。

> 注意：当前 `init_db()` 是用 `Base.metadata.create_all()` 建表；确保所有模型都在同一个 `Base.metadata` 下，否则会出现外键找不到表的错误（例如 `NoReferencedTableError`）。

---

## 启动前端（Streamlit）

在项目根目录运行：

```bash
streamlit run web/app.py
```

前端会通过 `web/services/api_client.py` 请求后端：

- `POST {API_BASE_URL}/auth/login`
- `GET  {API_BASE_URL}/auth/profile`（Header：`Authorization: Bearer <token>`）
- `POST {API_BASE_URL}/vector-store/upload`（上传文件）
- `POST {API_BASE_URL}/chat/completion`

---

## RAG / 向量库持久化与缓存

### 持久化

- **Chroma 向量库落盘目录**：`./chroma_db`
- **上传去重**：`md5.txt`（路径由 `MD5_PATH` 控制）

### Python 内置缓存（lru_cache）

- `backend/app/api/routes/vector_store.py:_get_kb_service()` 会缓存每个 `collection_name` 的 `KnowledgeBaseServce`（最多 256 个）
- `backend/app/crued/chat.py:_build_rag_service()` 会按 `(top_k, collection_name)` 缓存 RAG service（最多 16 组）

`lru_cache` **没有 TTL**：缓存会一直存在直到进程重启或被 LRU 淘汰。

---

## 评测：中文 RAG 召回基准（10k queries）

数据集说明见：`test/rag_recall_benchmark_cn_10k/README.md`

常用文件：

- `corpus.jsonl`：语料（约 2500）
- `queries.jsonl`：查询（10000）
- `qrels.tsv`：标注
- `evaluate_recall.py`：评测脚本（项目检索器 -> runs -> Recall/MRR）

运行：

```bash
python test/rag_recall_benchmark_cn_10k/evaluate_recall.py
```

输出：

- **Recall@K**：TopK 内“是否命中至少一个正例”的比例
- **MRR@K**：TopK 内“第一个命中正例”的倒数排名平均值

> 重要：如果评测结果全是 0，通常是 **Chroma collection 的语料**与 `qrels.tsv` 的 `doc_id` **不对齐**（不是“模型不行”）。

---

## 常见问题（Troubleshooting）

### 1) `Connection refused (WinError 10061)`

前端连不上后端：确认后端在 `127.0.0.1:8000` 运行，且 `API_BASE_URL` 指向正确。

### 2) `OPENAI_API_KEY` 缺失

确保环境变量或 `llm/.env` 中配置了 `OPENAI_API_KEY`。

### 3) `Could not import chromadb`

安装依赖：

```bash
pip install chromadb
```

### 4) `LangChainDeprecationWarning: Chroma`

安装并使用新包：

```bash
pip install -U langchain-chroma
```

并将导入改为：

```python
from langchain_chroma import Chroma
```

---

## 接口与手工测试

- 规范文档：`API_SPEC.md`
- 手工请求：`test_main.http`
