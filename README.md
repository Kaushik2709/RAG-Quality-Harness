# ⚡ Production RAG Evaluation & Observability Platform (LangGraph & LangChain)

A production-grade RAG (Retrieval-Augmented Generation) system powered natively by **LangChain** and **LangGraph**, built to empirically prove that **retrieval quality and hallucination detection — not just retrieval itself — are the core challenge in RAG**.

Equipped with an OpenTelemetry tracing harness (`gen_ai.*` semantic conventions), deterministic retrieval metrics (Recall@K, Precision@K, MRR, nDCG), LLM-as-Judge groundedness evaluation, automated regression CI quality gates, and a Next.js analytics dashboard.

---

## 🛠️ Tech Stack

| Category | Technologies & Libraries |
| :--- | :--- |
| **Orchestration & AI** | ![LangGraph](https://img.shields.io/badge/LangGraph-0055FF?style=for-the-badge&logo=graphviz&logoColor=white) ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white) ![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| **LLMs & Embeddings** | ![Google Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white) ![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black) |
| **Retrieval & Reranking** | ![Qdrant](https://img.shields.io/badge/Qdrant_Vector_DB-DC2626?style=for-the-badge&logo=qdrant&logoColor=white) ![BM25](https://img.shields.io/badge/Hybrid_Search-BM25-blue?style=for-the-badge) ![Cross-Encoder](https://img.shields.io/badge/Reranker-Cross--Encoder-orange?style=for-the-badge) |
| **Backend & Observability** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white) ![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-000000?style=for-the-badge&logo=opentelemetry&logoColor=white) ![Pydantic](https://img.shields.io/badge/Pydantic_v2-E92063?style=for-the-badge&logo=pydantic&logoColor=white) |
| **Analytics Dashboard** | ![Next.js](https://img.shields.io/badge/Next.js_14-000000?style=for-the-badge&logo=nextdotjs&logoColor=white) ![React](https://img.shields.io/badge/React_18-61DAFB?style=for-the-badge&logo=react&logoColor=black) ![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white) ![Recharts](https://img.shields.io/badge/Recharts-22B5BF?style=for-the-badge) |
| **Infra & Quality Gates** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white) ![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white) |

---

## 🏛️ LangGraph StateGraph Architecture

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                              User Request                              │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                LangGraph StateGraph Workflow (compile_rag_graph)       │
 ├───────────────────┬───────────────────┬────────────────────────────────┤
 │ 1. retrieve_node  │ 2. rerank_node    │ 3. generate_node               │
 │ (EnsembleRetriever│ (ContextualCompr. │ (ChatPromptTemplate +          │
 │  Dense + BM25)    │  CrossEncoder)    │  ChatGoogleGenerativeAI)       │
 └─────────┬─────────┴─────────┬─────────┴─────────┬──────────────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                    OpenTelemetry Tracing Layer                        │
 │  - GenAI Semantic Conventions                                          │
 │  - Latency per stage (ms) | Token usage | Cost calculation ($ USD)     │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   Next.js + Recharts Dashboard                         │
 │     - Trace Explorer (per-stage breakdown & total cost)                │
 │     - Eval Trends (Recall@K, MRR, Faithfulness over runs)               │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 🔗 LangChain & LangGraph Ecosystem Integration

- **Document Loaders**: `langchain_community.document_loaders` (`TextLoader`, `PyPDFLoader`, `BSHTMLLoader`)
- **Text Splitters**: `langchain_text_splitters` (`RecursiveCharacterTextSplitter`, `CharacterTextSplitter`)
- **Embeddings**: `langchain_google_genai.GoogleGenerativeAIEmbeddings` (model: `models/text-embedding-004`)
- **Vector Storage**: `langchain_qdrant` (`QdrantVectorStore` with `InMemoryVectorStore` fallback)
- **Hybrid Retrievers**: LangChain `EnsembleRetriever` combining dense vector search with sparse `BM25Retriever`
- **Rerankers**: LangChain `ContextualCompressionRetriever` with Cross-Encoder compression
- **Prompts**: `langchain_core.prompts.ChatPromptTemplate`
- **LLM Models**: `langchain_google_genai.ChatGoogleGenerativeAI` (model: `gemini-1.5-flash`)
- **Orchestration**: LangGraph `StateGraph` (`src/pipeline/graph.py`)

---

## 🎯 WORKED EXAMPLE: Catching a Retrieval & Groundedness Regression

### Step 1: Run Baseline Evaluation
Run the regression harness against the golden dataset:
```bash
python -m src.evaluation.regression_runner --accept-baseline
```
**Baseline Results:**
- `Recall@5`: **0.9000**
- `MRR`: **0.9500**
- `Mean Faithfulness`: **0.9500**
- Status: **PASS** ✅

---

### Step 2: Introduce a Sub-optimal Configuration Change
Simulate a developer reducing the chunk size to **40 characters** in `src/config.py`:

Edit `src/config.py`:
```python
CHUNK_SIZE = 40  # Regression test: tiny chunk size
```

---

### Step 3: Execute Regression Quality Gate
Run the regression harness:
```bash
python -m src.evaluation.regression_runner
```

**Console Output Flagging the Quality Breach:**
```text
=======================================================
[BASELINE COMPARISON] Regression Test Results vs Baseline
=======================================================
 [FAIL]  recall_at_5               | Current: 0.6000 | Baseline: 0.9000 | Diff: -0.3000 | Status: FAIL (RECALL REGRESSION)
 [FAIL]  mean_faithfulness         | Current: 0.7200 | Baseline: 0.9500 | Diff: -0.2300 | Status: FAIL (FAITHFULNESS BREACH)

[FAILURE] REGRESSION FAILURE DETECTED: Quality gates breached! Exit Code 1
```

---

## 🚀 Quick Start Guide

### 1. Installation & Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables (Gemini / Google API Key)
Create `.env` file in root:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/text-embedding-004
```

### 3. Run API Server
```bash
uvicorn src.api.main:app --reload --port 8000
```
Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run Evaluation Suite
```bash
python -m src.evaluation.regression_runner
```

### 5. Run Dashboard
```bash
cd dashboard
npm install
npm run dev
```
Dashboard URL: [http://localhost:3000](http://localhost:3000)
