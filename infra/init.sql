-- Initialize PostgreSQL database schema for RAG Tracing and Evaluation

CREATE TABLE IF NOT EXISTS traces (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(100) NOT NULL,
    stage VARCHAR(50) NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    cost_usd DOUBLE PRECISION DEFAULT 0.0,
    model_name VARCHAR(100) DEFAULT 'n/a',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_traces_query_id ON traces(query_id);
CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);

CREATE TABLE IF NOT EXISTS eval_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    recall_at_k DOUBLE PRECISION,
    precision_at_k DOUBLE PRECISION,
    mrr DOUBLE PRECISION,
    ndcg_at_k DOUBLE PRECISION,
    faithfulness DOUBLE PRECISION,
    relevancy DOUBLE PRECISION,
    context_precision DOUBLE PRECISION,
    status VARCHAR(50) NOT NULL
);
