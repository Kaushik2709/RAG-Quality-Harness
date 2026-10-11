'use client';

import React, { useState } from 'react';
import { 
  Sparkles, 
  Send, 
  Sliders, 
  ShieldCheck, 
  ShieldAlert, 
  Clock, 
  Copy, 
  Check, 
  Layers, 
  FileText, 
  ChevronDown, 
  ChevronUp,
  Cpu,
  RefreshCw,
  Zap,
  Bookmark
} from 'lucide-react';

interface ChunkResponse {
  chunk_id: string;
  doc_id: string;
  text: string;
}

interface QueryResponseData {
  query_id: string;
  query: string;
  answer: string;
  retrieved_context: ChunkResponse[];
  total_latency_ms: number;
  guardrail_status?: any;
  guardrail_blocked?: boolean;
}

export default function PlaygroundPage() {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(4);
  const [topKRerank, setTopKRerank] = useState(3);
  const [useHybrid, setUseHybrid] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResponseData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copiedAnswer, setCopiedAnswer] = useState(false);
  const [expandedChunk, setExpandedChunk] = useState<number | null>(0);

  const samplePrompts = [
    'What is semantic chunking and when should you use it?',
    'Explain the difference between dense and sparse vector retrieval.',
    'How do hallucination guardrails evaluate groundedness in RAG?',
    'What are the key trade-offs between latency and reranker accuracy?',
  ];

  const handleExecuteQuery = async (queryText?: string) => {
    const textToQuery = queryText || query;
    if (!textToQuery.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: textToQuery,
          top_k: topK,
          top_k_rerank: topKRerank,
          use_hybrid: useHybrid,
        }),
      });

      if (!res.ok) {
        const errDetail = await res.text();
        throw new Error(`Query failed: ${res.status} ${errDetail}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Failed to query RAG backend.');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedAnswer(true);
    setTimeout(() => setCopiedAnswer(false), 2000);
  };

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Sparkles className="h-3.5 w-3.5 animate-pulse text-cyan-400" />
              Interactive Execution Engine
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white mt-1">
            RAG Query <span className="gradient-text-cyan-indigo">Sandbox & Inspector</span>
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Trigger real-time LangGraph queries, inspect reranked context chunks, and verify guardrail decisions.
          </p>
        </div>
      </div>

      {/* Main Grid: Controls + Playground */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Query Input & Settings (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          
          {/* Query Card */}
          <div className="glass-card p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-gray-400 flex items-center gap-2">
                <span>Prompt / Query String</span>
              </label>
              <span className="text-xs font-mono text-cyan-400">LangGraph Pipeline</span>
            </div>

            <div className="relative">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask any question to test retrieval, reranking, and synthesis..."
                rows={4}
                className="w-full bg-[#0a0e1a]/80 border border-white/[0.1] rounded-xl p-4 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/50 font-sans transition resize-none"
              />
              
              <div className="absolute bottom-3 right-3 flex items-center gap-2">
                <button
                  onClick={() => handleExecuteQuery()}
                  disabled={loading || !query.trim()}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-600 via-indigo-600 to-violet-600 hover:from-cyan-500 hover:to-indigo-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-cyan-950/50 border border-cyan-400/30 transition-all hover:scale-[1.02] active:scale-[0.98]"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                      <span>Executing Graph...</span>
                    </>
                  ) : (
                    <>
                      <Send className="h-3.5 w-3.5" />
                      <span>Run Query</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Quick Sample Prompts */}
            <div className="space-y-2 pt-2">
              <span className="text-xs text-gray-500 flex items-center gap-1.5 font-medium">
                <Bookmark className="h-3 w-3 text-cyan-400" />
                Quick Prompts:
              </span>
              <div className="flex flex-wrap gap-2">
                {samplePrompts.map((p, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setQuery(p);
                      handleExecuteQuery(p);
                    }}
                    className="text-left text-xs px-3 py-1.5 rounded-lg bg-white/[0.03] hover:bg-white/[0.08] text-gray-300 hover:text-white border border-white/[0.06] hover:border-cyan-500/30 transition"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Hyperparameter Tuner */}
          <div className="glass-card p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-5">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Sliders className="h-4 w-4 text-cyan-400" />
                <span>Runtime Pipeline Configuration</span>
              </h3>
              <span className="text-[11px] font-mono text-gray-400">Dynamic State Overrides</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              
              {/* Top-K Retrieval */}
              <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs text-gray-300 font-medium">Top-K Retrieve</label>
                  <span className="font-mono text-cyan-400 font-bold text-xs">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="15"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full accent-cyan-500 cursor-pointer"
                />
                <span className="text-[10px] text-gray-500 block">Dense vector candidates</span>
              </div>

              {/* Top-K Rerank */}
              <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-xs text-gray-300 font-medium">Top-K Rerank</label>
                  <span className="font-mono text-indigo-400 font-bold text-xs">{topKRerank}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="8"
                  value={topKRerank}
                  onChange={(e) => setTopKRerank(Number(e.target.value))}
                  className="w-full accent-indigo-500 cursor-pointer"
                />
                <span className="text-[10px] text-gray-500 block">Cross-encoder depth</span>
              </div>

              {/* Hybrid Search Toggle */}
              <div className="p-3.5 rounded-xl bg-white/[0.02] border border-white/[0.05] flex flex-col justify-between">
                <div className="flex items-center justify-between">
                  <label className="text-xs text-gray-300 font-medium">Hybrid Search</label>
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold ${
                    useHybrid ? 'bg-emerald-500/20 text-emerald-400' : 'bg-gray-800 text-gray-400'
                  }`}>
                    {useHybrid ? 'ON' : 'OFF'}
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setUseHybrid(!useHybrid)}
                  className={`mt-2 w-full py-1.5 rounded-lg text-xs font-semibold transition border ${
                    useHybrid
                      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                      : 'bg-white/[0.04] text-gray-400 border-white/[0.08]'
                  }`}
                >
                  {useHybrid ? 'Dense + BM25' : 'Dense Only'}
                </button>
              </div>

            </div>
          </div>

        </div>

        {/* Right Column: Execution Result & Context Inspector (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          
          {error && (
            <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs space-y-1">
              <span className="font-bold flex items-center gap-1.5">
                <ShieldAlert className="h-4 w-4" /> Pipeline Error
              </span>
              <p className="font-mono">{error}</p>
            </div>
          )}

          {result ? (
            <div className="space-y-5 animate-in fade-in duration-300">
              
              {/* Answer Card */}
              <div className="glass-card p-6 rounded-2xl border border-white/[0.08] shadow-2xl space-y-4">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                  <div className="flex items-center gap-2">
                    <Zap className="h-4 w-4 text-cyan-400" />
                    <span className="text-xs font-bold uppercase text-white tracking-wide">
                      Synthesized Response
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {result.total_latency_ms} ms
                    </span>
                    <button
                      onClick={() => copyToClipboard(result.answer)}
                      className="p-1.5 rounded-lg hover:bg-white/[0.08] text-gray-400 hover:text-white transition"
                      title="Copy response"
                    >
                      {copiedAnswer ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                </div>

                <div className="text-sm text-gray-200 leading-relaxed font-sans whitespace-pre-wrap max-h-80 overflow-y-auto pr-1">
                  {result.answer || 'No answer generated.'}
                </div>

                {/* Guardrail Status Banner */}
                <div className={`p-3 rounded-xl flex items-center justify-between text-xs border ${
                  result.guardrail_blocked 
                    ? 'bg-rose-950/40 border-rose-500/30 text-rose-300' 
                    : 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300'
                }`}>
                  <div className="flex items-center gap-2">
                    {result.guardrail_blocked ? (
                      <ShieldAlert className="h-4 w-4 text-rose-400" />
                    ) : (
                      <ShieldCheck className="h-4 w-4 text-emerald-400" />
                    )}
                    <span className="font-semibold">
                      {result.guardrail_blocked ? 'Guardrail Blocked' : 'Guardrail Passed: Grounded & Safe'}
                    </span>
                  </div>
                  <span className="font-mono text-[10px] opacity-75">
                    {result.query_id ? result.query_id.substring(0, 12) : ''}
                  </span>
                </div>
              </div>

              {/* Retrieved Context Chunks Explorer */}
              <div className="glass-card p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold uppercase text-white tracking-wide flex items-center gap-2">
                    <Layers className="h-4 w-4 text-indigo-400" />
                    <span>Retrieved Context ({result.retrieved_context?.length || 0} chunks)</span>
                  </h3>
                  <span className="text-[10px] font-mono text-gray-400">Post-Reranker Order</span>
                </div>

                <div className="space-y-2.5 max-h-96 overflow-y-auto pr-1">
                  {result.retrieved_context && result.retrieved_context.length > 0 ? (
                    result.retrieved_context.map((c, idx) => {
                      const isExpanded = expandedChunk === idx;
                      return (
                        <div
                          key={idx}
                          className="rounded-xl bg-white/[0.02] border border-white/[0.06] overflow-hidden transition hover:border-indigo-500/30"
                        >
                          <button
                            onClick={() => setExpandedChunk(isExpanded ? null : idx)}
                            className="w-full p-3 text-left flex items-center justify-between text-xs font-medium hover:bg-white/[0.02] transition"
                          >
                            <div className="flex items-center gap-2">
                              <span className="h-5 w-5 rounded-full bg-indigo-500/20 text-indigo-300 flex items-center justify-center font-mono text-[10px] font-bold">
                                #{idx + 1}
                              </span>
                              <span className="font-mono text-gray-300 text-[11px] truncate max-w-[180px]">
                                {c.doc_id}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-[10px] font-mono text-gray-500">
                                {c.chunk_id}
                              </span>
                              {isExpanded ? <ChevronUp className="h-3.5 w-3.5 text-gray-400" /> : <ChevronDown className="h-3.5 w-3.5 text-gray-400" />}
                            </div>
                          </button>
                          
                          {isExpanded && (
                            <div className="p-3 bg-black/30 border-t border-white/[0.04] text-xs text-gray-300 font-mono leading-relaxed whitespace-pre-wrap">
                              {c.text}
                            </div>
                          )}
                        </div>
                      );
                    })
                  ) : (
                    <p className="text-xs text-gray-500 italic p-3 text-center">
                      No chunks retrieved for this query.
                    </p>
                  )}
                </div>
              </div>

            </div>
          ) : (
            <div className="glass-card p-12 rounded-2xl border border-white/[0.08] shadow-xl text-center space-y-4">
              <div className="h-12 w-12 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 mx-auto">
                <Cpu className="h-6 w-6" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-white">Execution Inspector Idle</h3>
                <p className="text-xs text-gray-400 mt-1 max-w-xs mx-auto">
                  Type a prompt or choose a sample prompt on the left to fire the LangGraph pipeline and see real-time execution telemetry.
                </p>
              </div>
            </div>
          )}

        </div>

      </div>

    </div>
  );
}
