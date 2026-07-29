from typing import List, TypedDict, Dict, Any, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from src.retrieval.vector_store import get_vector_store
from src.retrieval.retriever import build_retriever
from src.retrieval.reranker import build_reranked_retriever
from src.generation.prompt_builder import build_chat_prompt_template
from src.generation.generator import get_chat_model
from src.tracing.tracer import trace_stage
from src.config import settings

class RAGState(TypedDict):
    query: str
    query_id: str
    documents: List[Document]
    reranked_documents: List[Document]
    generation: str

def retrieve_node(state: RAGState) -> Dict[str, Any]:
    query_id = state.get("query_id", "q-default")
    query = state["query"]

    with trace_stage("retrieve", query_id=query_id, model_name=settings.EMBEDDING_MODEL) as span:
        vstore = get_vector_store()
        retriever = build_retriever(vectorstore=vstore, documents=state.get("documents", []), top_k=settings.TOP_K_RETRIEVAL)
        docs = retriever.invoke(query)
        span.set_tokens(tokens_in=len(query.split()), tokens_out=len(docs))

    return {"documents": docs}

def rerank_node(state: RAGState) -> Dict[str, Any]:
    query_id = state.get("query_id", "q-default")
    query = state["query"]
    input_docs = state.get("documents", [])

    with trace_stage("rerank", query_id=query_id, model_name=settings.RERANKER_MODEL) as span:
        base_retriever = build_retriever(documents=input_docs, top_k=settings.TOP_K_RETRIEVAL)
        reranked_retriever = build_reranked_retriever(base_retriever, top_n=settings.TOP_K_RERANK)
        reranked_docs = reranked_retriever.invoke(query)
        span.set_tokens(tokens_in=sum(len(d.page_content.split()) for d in reranked_docs), tokens_out=len(reranked_docs))

    return {"reranked_documents": reranked_docs}

def generate_node(state: RAGState) -> Dict[str, Any]:
    query_id = state.get("query_id", "q-default")
    query = state["query"]
    reranked = state.get("reranked_documents", [])

    context_str = "\n\n".join([f"[Source: {d.metadata.get('source', 'doc')}]\n{d.page_content}" for d in reranked])
    if not context_str.strip():
        context_str = "No relevant context documents were retrieved."

    prompt_template = build_chat_prompt_template()
    chat_model = get_chat_model()

    # LangChain LCEL Chain
    chain = prompt_template | chat_model

    with trace_stage("generate", query_id=query_id, model_name=getattr(chat_model, "model_name", "langchain-llm")) as span:
        ai_message = chain.invoke({"query": query, "context": context_str})
        if isinstance(ai_message.content, list):
            answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in ai_message.content])
        else:
            answer = str(ai_message.content)
        span.set_tokens(tokens_in=len(context_str.split()), tokens_out=len(answer.split()))

    return {"generation": answer}

def compile_rag_graph():
    """
    Compiles and returns the LangGraph StateGraph pipeline.
    """
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("generate", generate_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", END)

    return builder.compile()
