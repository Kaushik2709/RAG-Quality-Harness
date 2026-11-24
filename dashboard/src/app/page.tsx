'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { 
  Activity, 
  ArrowUpRight, 
  Clock, 
  DollarSign, 
  Layers, 
  ShieldCheck, 
  Sparkles, 
  Zap, 
  Database, 
  CheckCircle2, 
  TrendingUp, 
  Cpu,
  ChevronRight,
  Filter
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export default function OverviewPage() {
  const [traces, setTraces] = useState<any[]>([]);
  const [evalReports, setEvalReports] = useState<any[]>([]);
  const [runtimeStatus, setRuntimeStatus] = useState<any>(null);
  const [readiness, setReadiness] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [tracesRes, evalsRes, runtimeRes, readyRes] = await Promise.allSettled([
          fetch(`${API_BASE}/traces?limit=50`, { cache: 'no-store' }),
          fetch(`${API_BASE}/evals/reports`, { cache: 'no-store' }),
          fetch(`${API_BASE}/runtime/status`, { cache: 'no-store' }),
          fetch(`${API_BASE}/ready`, { cache: 'no-store' }),
        ]);

        if (tracesRes.status === 'fulfilled' && tracesRes.value.ok) {
          const tData = await tracesRes.value.json();
          setTraces(Array.isArray(tData) ? tData : []);
        }

        if (evalsRes.status === 'fulfilled' && evalsRes.value.ok) {
          const eData = await evalsRes.value.json();
          setEvalReports(Array.isArray(eData) ? eData : []);
        }

        if (runtimeRes.status === 'fulfilled' && runtimeRes.value.ok) {
          setRuntimeStatus(await runtimeRes.value.json());
        }

        if (readyRes.status === 'fulfilled') {
          // /ready returns 200 (ready/degraded) or 503 (not ready); both carry a body.
          try {
            setReadiness(await readyRes.value.json());
          } catch {
            setReadiness(null);
          }
        }
      } catch (err) {
        console.error('Failed to load telemetry', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const totalCost = traces.reduce((acc, t) => acc + (t.cost_usd || 0), 0);
  const avgLatency = traces.length
    ? (traces.reduce((acc, t) => acc + (t.latency_ms || 0), 0) / traces.length).toFixed(1)
    : '0';

  // Compute stage breakdown
  const stages = ['retrieve', 'rerank', 'generate'];
  const stageStats = stages.map((st) => {
    const stageTraces = traces.filter((t) => t.stage === st);
    const count = stageTraces.length;
    const avg = count ? Math.round(stageTraces.reduce((acc, t) => acc + (t.latency_ms || 0), 0) / count) : 0;
    const cost = stageTraces.reduce((acc, t) => acc + (t.cost_usd || 0), 0);
    return { name: st.toUpperCase(), avgLatency: avg, count, cost: Number(cost.toFixed(5)) };
  });

  const latestEval = evalReports.length > 0 ? evalReports[evalReports.length - 1] : null;
  const formatMetric = (value: number | null | undefined) =>
    value == null ? 'n/a' : `${(value * 100).toFixed(1)}%`;
  const runtimeConfig = runtimeStatus?.config || {};
  const indexStatus = readiness?.components?.index || runtimeStatus?.readiness?.components?.index || null;
  const telemetryStatus = runtimeStatus?.telemetry || runtimeStatus?.readiness?.components?.telemetry || null;
  const readinessStatus = readiness?.status || runtimeStatus?.readiness?.status || 'unavailable';
  const authStatus = readiness?.components?.authentication?.status || 'unknown';
  const modelState = readiness?.components?.models?.models || {};
  const degradedModels = Object.entries(modelState)
    .filter(([, value]: [string, any]) => value?.state && value.state !== 'ready')
    .map(([name, value]: [string, any]) => `${name}: ${value.state}`);

  return (
    <div className="space-y-10">
      
      {/* Hero Header HUD */}
      <div className="relative overflow-hidden rounded-2xl glass-card p-8 border border-white/[0.08] shadow-2xl">
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-96 h-96 bg-gradient-to-br from-cyan-500/10 via-indigo-500/10 to-transparent rounded-full blur-3xl pointer-events-none" />
        
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative z-10">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                <span className={`h-1.5 w-1.5 rounded-full ${readiness?.ready ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                {readiness?.ready ? `Runtime ${readinessStatus}` : 'Runtime status unavailable'}
              </span>
              <span className="text-xs text-gray-400 font-mono">
                Telemetry: {telemetryStatus?.status || 'unknown'}
              </span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
              Pipeline Health & <span className="gradient-text-cyan-indigo">Evaluation Matrix</span>
            </h1>
            <p className="text-gray-400 text-sm max-w-2xl leading-relaxed">
              Real-time distributed tracing, stage latency profiling, token economics, and LLM-as-a-judge quality regression benchmarks.
            </p>
          </div>

          {/* Quick Action Buttons */}
          <div className="flex flex-wrap sm:flex-nowrap gap-3">
            <Link
              href="/playground"
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 via-indigo-600 to-violet-600 hover:from-cyan-500 hover:to-indigo-500 text-white text-sm font-semibold shadow-lg shadow-cyan-950/50 border border-cyan-400/30 transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <Sparkles className="h-4 w-4" />
              <span>Launch Playground</span>
            </Link>
            <Link
              href="/evals"
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] text-gray-200 text-sm font-semibold border border-white/[0.08] transition hover:border-white/20"
            >
              <Activity className="h-4 w-4 text-emerald-400" />
              <span>Eval Benchmarks</span>
            </Link>
            <Link
              href="/ingest"
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] text-gray-200 text-sm font-semibold border border-white/[0.08] transition hover:border-white/20"
            >
              <Database className="h-4 w-4 text-cyan-400" />
              <span>Ingest Docs</span>
            </Link>
          </div>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        
        {/* Metric 1: Total Spans */}
        <div className="glass-card glass-card-interactive p-6 rounded-2xl relative overflow-hidden group">
          <div className="flex items-center justify-between text-gray-400 mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider">Recorded Spans</span>
            <div className="h-8 w-8 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400 border border-indigo-500/20">
              <Layers className="h-4 w-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <div className="text-3xl font-extrabold text-white font-mono tracking-tight">
              {traces.length}
            </div>
            <span className="text-xs text-indigo-400 font-mono font-medium">telemetry spans</span>
          </div>
          <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
            <span className="text-emerald-400 font-mono">{telemetryStatus?.written ?? 0}</span> persisted
            {telemetryStatus?.dropped ? (
              <span className="text-amber-400 font-mono">/ {telemetryStatus.dropped} dropped</span>
            ) : null}
          </p>
        </div>

        {/* Metric 2: Avg Latency */}
        <div className="glass-card glass-card-interactive p-6 rounded-2xl relative overflow-hidden group">
          <div className="flex items-center justify-between text-gray-400 mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider">Avg Stage Latency</span>
            <div className="h-8 w-8 rounded-lg bg-cyan-500/10 flex items-center justify-center text-cyan-400 border border-cyan-500/20">
              <Clock className="h-4 w-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <div className="text-3xl font-extrabold text-cyan-400 font-mono tracking-tight">
              {traces.length ? avgLatency : 'n/a'}
            </div>
            <span className="text-xs text-cyan-300 font-mono font-medium">ms / span</span>
          </div>
          <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
            <span className="text-cyan-400 font-mono">Measured</span> across {traces.length} stored spans
          </p>
        </div>

        {/* Metric 3: Cost Burn */}
        <div className="glass-card glass-card-interactive p-6 rounded-2xl relative overflow-hidden group">
          <div className="flex items-center justify-between text-gray-400 mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider">Total Pipeline Cost</span>
            <div className="h-8 w-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400 border border-emerald-500/20">
              <DollarSign className="h-4 w-4" />
            </div>
          </div>
          <div className="flex items-baseline gap-2">
            <div className="text-3xl font-extrabold text-emerald-400 font-mono tracking-tight">
              ${totalCost.toFixed(5)}
            </div>
            <span className="text-xs text-emerald-300 font-mono font-medium">USD</span>
          </div>
          <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
            <span className="text-emerald-400 font-mono">Estimated</span> from static pricing; not billing data
          </p>
        </div>

        {/* Metric 4: Quality Gate Health */}
        <div className="glass-card glass-card-interactive p-6 rounded-2xl relative overflow-hidden group">
          <div className="flex items-center justify-between text-gray-400 mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider">Quality Gate Status</span>
            <div className="h-8 w-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400 border border-emerald-500/20">
              <ShieldCheck className="h-4 w-4" />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${
              latestEval?.status === 'completed'
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
                : 'bg-amber-500/10 text-amber-300 border-amber-500/30'
            }`}>
              {latestEval?.status === 'completed' ? (
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
              ) : (
                <span className="h-2 w-2 rounded-full bg-amber-400" />
              )}
              {latestEval?.status ? latestEval.status.toUpperCase() : 'NO RUN'}
            </span>
            <span className="text-xs text-gray-400 font-mono">
              Recall: {formatMetric(latestEval?.recall_at_5)}
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
            Faithfulness: <span className="text-purple-400 font-mono font-semibold">{formatMetric(latestEval?.faithfulness)}</span>
          </p>
        </div>

      </div>

      {/* Interactive LangGraph Architecture Pipeline Flow */}
      <div className="glass-card p-6 sm:p-8 rounded-2xl border border-white/[0.08] shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Cpu className="h-5 w-5 text-cyan-400" />
              <span>LangGraph StateGraph Execution Pipeline</span>
            </h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Multi-stage deterministic pipeline orchestrated with LangGraph and OpenTelemetry instrumentation
            </p>
          </div>
          <span className="text-xs font-mono px-3 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 self-start sm:self-auto">
            Sequential State Machine
          </span>
        </div>

        {/* Pipeline Nodes Flow */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 relative">
          
          {/* Step 1 */}
          <div className="p-4 rounded-xl bg-white/[0.03] border border-cyan-500/20 hover:border-cyan-500/40 transition group">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="font-mono text-cyan-400 font-semibold">STAGE 01</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">{runtimeConfig.store_backend || 'vector store'}</span>
            </div>
            <h3 className="font-bold text-white text-sm">Hybrid Retrieval</h3>
            <p className="text-xs text-gray-400 mt-1">Dense embeddings + Sparse BM25 retrieval over chunk index</p>
            <div className="mt-3 flex items-center justify-between text-[11px] font-mono text-gray-400 pt-2 border-t border-white/[0.05]">
              <span>Latency</span>
              <span className="text-cyan-300 font-bold">
                {stageStats.find(s => s.name === 'RETRIEVE')?.count ? `${stageStats.find(s => s.name === 'RETRIEVE')?.avgLatency} ms` : 'n/a'}
              </span>
            </div>
          </div>

          {/* Step 2 */}
          <div className="p-4 rounded-xl bg-white/[0.03] border border-indigo-500/20 hover:border-indigo-500/40 transition group">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="font-mono text-indigo-400 font-semibold">STAGE 02</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">{runtimeConfig.rerank_policy || 'rerank policy'}</span>
            </div>
            <h3 className="font-bold text-white text-sm">Cross-Encoder Reranking</h3>
            <p className="text-xs text-gray-400 mt-1">Deep contextual score reordering across top-K candidates</p>
            <div className="mt-3 flex items-center justify-between text-[11px] font-mono text-gray-400 pt-2 border-t border-white/[0.05]">
              <span>Latency</span>
              <span className="text-indigo-300 font-bold">
                {stageStats.find(s => s.name === 'RERANK')?.count ? `${stageStats.find(s => s.name === 'RERANK')?.avgLatency} ms` : 'not recorded'}
              </span>
            </div>
          </div>

          {/* Step 3 */}
          <div className="p-4 rounded-xl bg-white/[0.03] border border-emerald-500/20 hover:border-emerald-500/40 transition group">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="font-mono text-emerald-400 font-semibold">STAGE 03</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">Guardrails</span>
            </div>
            <h3 className="font-bold text-white text-sm">Guardrails & Safety</h3>
            <p className="text-xs text-gray-400 mt-1">PII masking, prompt-injection defense and citation validation</p>
            <div className="mt-3 flex items-center justify-between text-[11px] font-mono text-gray-400 pt-2 border-t border-white/[0.05]">
              <span>Auth</span>
              <span className={`font-bold ${authStatus === 'enabled' ? 'text-emerald-300' : 'text-amber-300'}`}>{authStatus}</span>
            </div>
          </div>

          {/* Step 4 */}
          <div className="p-4 rounded-xl bg-white/[0.03] border border-purple-500/20 hover:border-purple-500/40 transition group">
            <div className="flex items-center justify-between text-xs mb-2">
              <span className="font-mono text-purple-400 font-semibold">STAGE 04</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">{runtimeConfig.llm_provider || 'generation'}</span>
            </div>
            <h3 className="font-bold text-white text-sm">Grounded Synthesis</h3>
            <p className="text-xs text-gray-400 mt-1">Generates a cited response conditioned on filtered context</p>
            <div className="mt-3 flex items-center justify-between text-[11px] font-mono text-gray-400 pt-2 border-t border-white/[0.05]">
              <span>Latency</span>
              <span className="text-purple-300 font-bold">
                {stageStats.find(s => s.name === 'GENERATE')?.count ? `${stageStats.find(s => s.name === 'GENERATE')?.avgLatency} ms` : 'not recorded'}
              </span>
            </div>
          </div>

        </div>
      </div>

      {/* Latency & Telemetry Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Stage Latency Distribution Chart */}
        <div className="lg:col-span-2 glass-card p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-cyan-400" />
                <span>Stage Latency Profile (ms)</span>
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">Average execution time per pipeline component</p>
            </div>
            <Link href="/traces" className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1 font-mono">
              <span>View Waterfall</span>
              <ChevronRight className="h-3.5 w-3.5" />
            </Link>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stageStats} margin={{ top: 20, right: 20, left: -10, bottom: 5 }}>
                <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 12, fill: '#94a3b8' }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 12, fill: '#94a3b8' }} />
                <Tooltip 
                  cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
                  contentStyle={{ 
                    backgroundColor: '#0b0f19', 
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderRadius: '12px',
                    color: '#fff',
                    fontFamily: 'JetBrains Mono',
                    boxShadow: '0 8px 30px rgba(0,0,0,0.8)'
                  }} 
                />
                <Bar dataKey="avgLatency" radius={[6, 6, 0, 0]}>
                  {stageStats.map((entry, index) => {
                    const colors = ['#06b6d4', '#6366f1', '#a855f7'];
                    return <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />;
                  })}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Quality Gates Reference Card */}
        <div className="glass-card p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-emerald-400" />
              <span>Regression Gates</span>
            </h3>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
              CI/CD ENFORCED
            </span>
          </div>
          
          <div className="space-y-3">
            <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05]">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-gray-300 font-medium">Recall@5 Gate</span>
                <span className="text-emerald-400 font-mono font-bold">&ge; 0.90</span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
                <div className="bg-emerald-500 h-1.5 rounded-full" style={{ width: `${(latestEval?.recall_at_5 ?? 0) * 100}%` }}></div>
              </div>
              <span className="text-[10px] text-gray-500 mt-1 block font-mono">
                Latest report: {latestEval?.recall_at_5 != null ? latestEval.recall_at_5.toFixed(3) : 'n/a'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05]">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-gray-300 font-medium">Faithfulness (Judge)</span>
                <span className="text-purple-400 font-mono font-bold">&ge; 0.80</span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
                <div className="bg-purple-500 h-1.5 rounded-full" style={{ width: `${(latestEval?.faithfulness ?? 0) * 100}%` }}></div>
              </div>
              <span className="text-[10px] text-gray-500 mt-1 block font-mono">
                Latest judge: {latestEval?.judge_status ?? 'not run'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05]">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="text-gray-300 font-medium">MRR Ranking Depth</span>
                <span className="text-cyan-400 font-mono font-bold">&ge; 0.85</span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-1.5 overflow-hidden">
                <div className="bg-cyan-500 h-1.5 rounded-full" style={{ width: `${(latestEval?.mrr ?? 0) * 100}%` }}></div>
              </div>
              <span className="text-[10px] text-gray-500 mt-1 block font-mono">
                Latest report: {latestEval?.mrr != null ? latestEval.mrr.toFixed(3) : 'n/a'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-300 font-medium">Retrieval Index</span>
                <span className={`font-mono font-bold ${indexStatus?.status === 'ready' ? 'text-emerald-400' : 'text-amber-300'}`}>
                  {indexStatus?.status ?? 'unknown'}
                </span>
              </div>
              <span className="text-[10px] text-gray-500 block font-mono">
                {indexStatus?.chunk_count != null ? `${indexStatus.chunk_count} chunks` : 'no index built'}
                {degradedModels.length ? ` · degraded: ${degradedModels.join(', ')}` : ''}
              </span>
            </div>
          </div>

          <Link
            href="/evals"
            className="w-full flex items-center justify-center gap-1.5 py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.09] text-xs font-semibold text-gray-300 border border-white/[0.08] transition"
          >
            <span>Open Evaluation Reports</span>
            <ChevronRight className="h-3.5 w-3.5" />
          </Link>
        </div>

      </div>

      {/* Recent Telemetry Stream */}
      <div className="glass-card rounded-2xl border border-white/[0.08] overflow-hidden shadow-xl">
        <div className="p-6 border-b border-white/[0.06] flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Zap className="h-4 w-4 text-cyan-400" />
              <span>Live OpenTelemetry Span Feed</span>
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">Most recent execution spans logged by the pipeline</p>
          </div>
          <Link
            href="/traces"
            className="text-xs text-cyan-400 hover:text-cyan-300 font-mono flex items-center gap-1"
          >
            <span>All Spans ({traces.length})</span>
            <ChevronRight className="h-3.5 w-3.5" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-gray-300">
            <thead className="bg-[#090d16] text-gray-400 uppercase text-[11px] font-mono border-b border-white/[0.05]">
              <tr>
                <th className="px-6 py-3.5">Query ID</th>
                <th className="px-6 py-3.5">Stage</th>
                <th className="px-6 py-3.5">Latency</th>
                <th className="px-6 py-3.5">Tokens (In / Out)</th>
                <th className="px-6 py-3.5">Cost (est.)</th>
                <th className="px-6 py-3.5">Model</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
              {traces.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-gray-500">
                    {loading ? (
                      <div className="flex items-center justify-center gap-2">
                        <span className="h-4 w-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                        <span>Loading telemetry...</span>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <p>No traces recorded yet. Head over to the Query Sandbox to run live queries!</p>
                        <Link 
                          href="/playground" 
                          className="inline-block px-3 py-1.5 rounded-lg bg-cyan-600/30 text-cyan-300 border border-cyan-500/30 text-xs font-semibold"
                        >
                          Run Sandbox Query →
                        </Link>
                      </div>
                    )}
                  </td>
                </tr>
              ) : (
                traces.slice(0, 5).map((t, idx) => (
                  <tr key={idx} className="hover:bg-white/[0.02] transition">
                    <td className="px-6 py-4 font-mono text-cyan-300 font-medium">
                      {t.query_id ? t.query_id.substring(0, 18) : 'q-unknown'}...
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-block px-2.5 py-0.5 rounded-full text-[11px] font-mono font-semibold uppercase ${
                          t.stage === 'generate'
                            ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                            : t.stage === 'rerank'
                            ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                            : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        }`}
                      >
                        {t.stage}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-mono text-white font-medium">
                      {t.latency_ms} ms
                    </td>
                    <td className="px-6 py-4 font-mono text-gray-400">
                      {t.tokens_in} / {t.tokens_out}
                    </td>
                    <td className="px-6 py-4 font-mono text-emerald-400">
                      ${(t.cost_usd || 0).toFixed(6)}
                    </td>
                    <td className="px-6 py-4 text-gray-400 font-mono text-[11px]">
                      {t.model_name || 'n/a'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
