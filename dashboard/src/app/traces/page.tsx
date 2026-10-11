'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { 
  Layers, 
  Search, 
  Filter, 
  Clock, 
  DollarSign, 
  Cpu, 
  Sparkles, 
  ArrowUpDown, 
  ChevronRight, 
  X, 
  Copy, 
  Check, 
  RefreshCw,
  Zap,
  BarChart2
} from 'lucide-react';

interface TraceItem {
  query_id?: string;
  stage?: string;
  latency_ms?: number;
  tokens_in?: number;
  tokens_out?: number;
  cost_usd?: number;
  model_name?: string;
  timestamp?: string;
  metadata?: any;
}

export default function TraceExplorerPage() {
  const [traces, setTraces] = useState<TraceItem[]>([]);
  const [filterStage, setFilterStage] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedTrace, setSelectedTrace] = useState<TraceItem | null>(null);
  const [copiedId, setCopiedId] = useState(false);

  const fetchTraces = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/traces?limit=100', { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        setTraces(Array.isArray(data) ? data : []);
      }
    } catch (err) {
      console.error('Failed to fetch traces', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTraces();
  }, []);

  const filtered = traces
    .filter((t) => filterStage === 'all' || t.stage === filterStage)
    .filter((t) => {
      if (!searchQuery) return true;
      const q = searchQuery.toLowerCase();
      return (
        (t.query_id && t.query_id.toLowerCase().includes(q)) ||
        (t.model_name && t.model_name.toLowerCase().includes(q)) ||
        (t.stage && t.stage.toLowerCase().includes(q))
      );
    });

  const maxLatency = Math.max(...traces.map((t) => t.latency_ms || 0), 100);
  const totalCost = traces.reduce((acc, t) => acc + (t.cost_usd || 0), 0);
  const totalTokens = traces.reduce((acc, t) => acc + (t.tokens_in || 0) + (t.tokens_out || 0), 0);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(true);
    setTimeout(() => setCopiedId(false), 2000);
  };

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <Layers className="h-3.5 w-3.5 text-indigo-400" />
              OpenTelemetry Telemetry Stream
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white mt-1">
            Trace Waterfall & <span className="gradient-text-cyan-indigo">Span Explorer</span>
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Inspect per-stage latency spans, tokens consumed, and exact execution overhead.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start md:self-auto">
          <button
            onClick={fetchTraces}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] text-gray-300 text-xs font-semibold border border-white/[0.08] transition"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
          <Link
            href="/playground"
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 text-white text-xs font-semibold shadow-md shadow-cyan-950/40 border border-cyan-400/30 transition"
          >
            <Sparkles className="h-3.5 w-3.5" />
            <span>Generate Traces</span>
          </Link>
        </div>
      </div>

      {/* Telemetry KPI Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="glass-card p-4 rounded-xl border border-white/[0.08] flex items-center justify-between">
          <div>
            <span className="text-xs text-gray-400 font-medium">Total Tracked Spans</span>
            <div className="text-2xl font-bold text-white font-mono mt-0.5">{traces.length}</div>
          </div>
          <div className="h-9 w-9 rounded-lg bg-indigo-500/10 flex items-center justify-center text-indigo-400 border border-indigo-500/20">
            <Layers className="h-4 w-4" />
          </div>
        </div>

        <div className="glass-card p-4 rounded-xl border border-white/[0.08] flex items-center justify-between">
          <div>
            <span className="text-xs text-gray-400 font-medium">Cumulative Tokens</span>
            <div className="text-2xl font-bold text-cyan-400 font-mono mt-0.5">{totalTokens.toLocaleString()}</div>
          </div>
          <div className="h-9 w-9 rounded-lg bg-cyan-500/10 flex items-center justify-center text-cyan-400 border border-cyan-500/20">
            <Cpu className="h-4 w-4" />
          </div>
        </div>

        <div className="glass-card p-4 rounded-xl border border-white/[0.08] flex items-center justify-between">
          <div>
            <span className="text-xs text-gray-400 font-medium">Cumulative Cost</span>
            <div className="text-2xl font-bold text-emerald-400 font-mono mt-0.5">${totalCost.toFixed(5)}</div>
          </div>
          <div className="h-9 w-9 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-400 border border-emerald-500/20">
            <DollarSign className="h-4 w-4" />
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="glass-card p-4 rounded-2xl border border-white/[0.08] flex flex-col md:flex-row items-center justify-between gap-4 shadow-lg">
        
        {/* Stage Filter Buttons */}
        <div className="flex items-center gap-1.5 w-full md:w-auto overflow-x-auto pb-1 md:pb-0">
          <span className="text-xs text-gray-400 mr-2 flex items-center gap-1 font-medium">
            <Filter className="h-3 w-3 text-cyan-400" />
            Stage:
          </span>
          {['all', 'retrieve', 'rerank', 'generate'].map((st) => (
            <button
              key={st}
              onClick={() => setFilterStage(st)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold capitalize font-mono transition ${
                filterStage === st
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm shadow-cyan-500/10'
                  : 'bg-white/[0.03] text-gray-400 hover:text-white border border-white/[0.05]'
              }`}
            >
              {st}
            </button>
          ))}
        </div>

        {/* Search Input */}
        <div className="relative w-full md:w-72">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search query ID, model..."
            className="w-full pl-9 pr-4 py-1.5 bg-[#0a0e1a]/80 border border-white/[0.1] rounded-xl text-white text-xs placeholder-gray-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/50 font-mono transition"
          />
        </div>

      </div>

      {/* Main Traces Table with Waterfall Bars */}
      <div className="glass-card rounded-2xl border border-white/[0.08] overflow-hidden shadow-2xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-gray-300">
            <thead className="bg-[#090d16] text-gray-400 uppercase text-[11px] font-mono border-b border-white/[0.05]">
              <tr>
                <th className="px-6 py-3.5">Query ID</th>
                <th className="px-6 py-3.5">Stage</th>
                <th className="px-6 py-3.5">Latency Waterfall</th>
                <th className="px-6 py-3.5">Tokens (In / Out)</th>
                <th className="px-6 py-3.5">Cost ($ USD)</th>
                <th className="px-6 py-3.5">Model</th>
                <th className="px-6 py-3.5 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-gray-500 space-y-3">
                    {loading ? (
                      <div className="flex items-center justify-center gap-2">
                        <span className="h-4 w-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                        <span>Loading telemetry traces...</span>
                      </div>
                    ) : (
                      <div className="space-y-3 max-w-sm mx-auto">
                        <p className="text-gray-400 text-sm">No telemetry spans match your criteria.</p>
                        <Link
                          href="/playground"
                          className="inline-block px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 text-white font-semibold text-xs shadow-md shadow-cyan-950/40"
                        >
                          Execute Test Query in Playground →
                        </Link>
                      </div>
                    )}
                  </td>
                </tr>
              ) : (
                filtered.map((t, idx) => {
                  const latency = t.latency_ms || 0;
                  const barWidthPercent = Math.max(Math.min((latency / maxLatency) * 100, 100), 4);
                  
                  return (
                    <tr 
                      key={idx} 
                      onClick={() => setSelectedTrace(t)}
                      className="hover:bg-white/[0.03] transition cursor-pointer group"
                    >
                      {/* Query ID */}
                      <td className="px-6 py-4 font-mono text-cyan-300 font-medium">
                        {t.query_id ? t.query_id.substring(0, 16) : 'q-unknown'}...
                      </td>

                      {/* Stage Pill */}
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

                      {/* Latency Waterfall Bar */}
                      <td className="px-6 py-4 w-64">
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-[11px] font-mono">
                            <span className="text-white font-semibold">{latency} ms</span>
                          </div>
                          <div className="w-full bg-white/[0.05] rounded-full h-1.5 overflow-hidden">
                            <div
                              className={`h-1.5 rounded-full transition-all duration-500 ${
                                t.stage === 'generate'
                                  ? 'bg-gradient-to-r from-purple-500 to-pink-500'
                                  : t.stage === 'rerank'
                                  ? 'bg-gradient-to-r from-indigo-500 to-blue-500'
                                  : 'bg-gradient-to-r from-cyan-400 to-teal-400'
                              }`}
                              style={{ width: `${barWidthPercent}%` }}
                            />
                          </div>
                        </div>
                      </td>

                      {/* Tokens */}
                      <td className="px-6 py-4 font-mono text-gray-400">
                        {t.tokens_in} / {t.tokens_out}
                      </td>

                      {/* Cost */}
                      <td className="px-6 py-4 font-mono text-emerald-400">
                        ${(t.cost_usd || 0).toFixed(6)}
                      </td>

                      {/* Model */}
                      <td className="px-6 py-4 text-gray-400 font-mono text-[11px]">
                        {t.model_name || 'gemini'}
                      </td>

                      {/* Action */}
                      <td className="px-6 py-4 text-right">
                        <span className="inline-flex items-center gap-1 text-[11px] text-cyan-400 opacity-0 group-hover:opacity-100 transition">
                          <span>Inspect</span>
                          <ChevronRight className="h-3 w-3" />
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Slide-out Span Inspector Drawer */}
      {selectedTrace && (
        <div className="fixed inset-0 z-50 flex items-center justify-end bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="w-full max-w-md h-full bg-[#0b0f19] border-l border-white/[0.1] shadow-2xl p-6 flex flex-col justify-between overflow-y-auto">
            
            <div className="space-y-6">
              <div className="flex items-center justify-between border-b border-white/[0.08] pb-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-cyan-400" />
                  <h3 className="text-base font-bold text-white">Span Telemetry Detail</h3>
                </div>
                <button
                  onClick={() => setSelectedTrace(null)}
                  className="p-1 rounded-lg hover:bg-white/[0.08] text-gray-400 hover:text-white transition"
                >
                  <X className="h-5 w-5" />
                </button>
              </div>

              {/* Span Attributes */}
              <div className="space-y-3">
                <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-1">
                  <span className="text-[10px] uppercase font-mono text-gray-500">Query ID</span>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs text-cyan-300 break-all">{selectedTrace.query_id || 'N/A'}</span>
                    <button
                      onClick={() => copyToClipboard(selectedTrace.query_id || '')}
                      className="p-1 hover:text-white text-gray-400 ml-2"
                    >
                      {copiedId ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-1">
                    <span className="text-[10px] uppercase font-mono text-gray-500">Stage</span>
                    <div className="font-mono text-xs text-white font-bold uppercase">{selectedTrace.stage}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-1">
                    <span className="text-[10px] uppercase font-mono text-gray-500">Latency</span>
                    <div className="font-mono text-xs text-cyan-400 font-bold">{selectedTrace.latency_ms} ms</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-1">
                    <span className="text-[10px] uppercase font-mono text-gray-500">Tokens In / Out</span>
                    <div className="font-mono text-xs text-white">{selectedTrace.tokens_in} / {selectedTrace.tokens_out}</div>
                  </div>
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-1">
                    <span className="text-[10px] uppercase font-mono text-gray-500">Cost</span>
                    <div className="font-mono text-xs text-emerald-400 font-bold">${(selectedTrace.cost_usd || 0).toFixed(6)}</div>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-1">
                  <span className="text-[10px] uppercase font-mono text-gray-500">Model Name</span>
                  <div className="font-mono text-xs text-gray-300">{selectedTrace.model_name || 'gemini'}</div>
                </div>

                {selectedTrace.timestamp && (
                  <div className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05] space-y-1">
                    <span className="text-[10px] uppercase font-mono text-gray-500">Timestamp</span>
                    <div className="font-mono text-xs text-gray-400">{selectedTrace.timestamp}</div>
                  </div>
                )}
              </div>

              {/* Raw JSON */}
              <div className="space-y-1">
                <span className="text-xs font-semibold text-gray-400">Raw Telemetry JSON</span>
                <pre className="p-3 rounded-xl bg-black/60 border border-white/[0.06] text-[11px] font-mono text-gray-300 overflow-x-auto max-h-48">
                  {JSON.stringify(selectedTrace, null, 2)}
                </pre>
              </div>
            </div>

            <button
              onClick={() => setSelectedTrace(null)}
              className="mt-6 w-full py-2.5 rounded-xl bg-white/[0.08] hover:bg-white/[0.12] text-xs font-semibold text-white transition"
            >
              Close Inspector
            </button>

          </div>
        </div>
      )}

    </div>
  );
}
