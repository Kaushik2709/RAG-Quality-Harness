'use client';

import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart
} from 'recharts';
import {
  Activity,
  ShieldCheck,
  Zap,
  TrendingUp,
  CheckCircle2,
  AlertCircle,
  Play,
  RefreshCw,
  Download,
  Layers,
  Award,
  ChevronRight,
  Sparkles
} from 'lucide-react';

export default function EvalTrendsPage() {
  const [evalData, setEvalData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [runningEval, setRunningEval] = useState<boolean>(false);
  const [activeMetrics, setActiveMetrics] = useState<Record<string, boolean>>({
    recall_at_5: true,
    faithfulness: true,
    mrr: true,
    relevancy: false,
    precision_at_5: false,
    context_precision: false,
  });

  const metricConfigs: Record<string, { label: string; color: string; threshold: number }> = {
    recall_at_5: { label: 'Recall@5', color: '#10b981', threshold: 0.75 },
    faithfulness: { label: 'Faithfulness (Judge)', color: '#8b5cf6', threshold: 0.80 },
    mrr: { label: 'MRR Depth', color: '#06b6d4', threshold: 0.70 },
    relevancy: { label: 'Answer Relevancy', color: '#f59e0b', threshold: 0.75 },
    precision_at_5: { label: 'Precision@5', color: '#3b82f6', threshold: 0.70 },
    context_precision: { label: 'Context Precision', color: '#ec4899', threshold: 0.70 },
  };

  const fetchReports = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/evals/reports', { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        setEvalData(Array.isArray(data) ? data : []);
      }
    } catch (err) {
      console.warn('Could not fetch evaluation reports:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchReports();
  }, []);

  const handleRunEvaluation = async () => {
    setRunningEval(true);
    try {
      const res = await fetch('http://localhost:8000/evals/run', {
        method: 'POST',
      });
      if (res.ok) {
        let attempts = 0;
        const initialCount = evalData.length;
        const interval = setInterval(async () => {
          attempts += 1;
          const reportsRes = await fetch('http://localhost:8000/evals/reports', { cache: 'no-store' });
          if (reportsRes.ok) {
            const data = await reportsRes.json();
            setEvalData(data);
            if (data.length > initialCount || attempts >= 15) {
              clearInterval(interval);
              setRunningEval(false);
            }
          }
        }, 4000);
      } else {
        alert('Evaluation run failed. Check server logs.');
        setRunningEval(false);
      }
    } catch (err) {
      alert('Error triggering evaluation run: ' + err);
      setRunningEval(false);
    }
  };

  const toggleMetric = (key: string) => {
    setActiveMetrics((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const latestRun = evalData.length > 0 ? evalData[evalData.length - 1] : null;

  return (
    <div className="space-y-8">

      <p className="text-sm text-slate-400">
        Run status: {latestRun?.status ?? 'No reports'} ? Judge: {latestRun?.judge_status ?? 'Unavailable'}.
        Offline scores use lexical test embeddings; judge scores appear only after an opt-in live run.
      </p>
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              Golden Dataset Regression Suite
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white mt-1">
            Evaluation Matrix & <span className="gradient-text-emerald">Quality Gates</span>
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Historical benchmarking of Recall@5, MRR, Context Precision, and LLM-as-a-Judge Faithfulness.
          </p>
        </div>

        <div className="flex items-center gap-3 self-start md:self-auto">
          <button
            onClick={fetchReports}
            disabled={loading}
            className="p-2.5 rounded-xl bg-white/[0.04] hover:bg-white/[0.08] text-gray-300 text-xs border border-white/[0.08] transition"
            title="Refresh reports"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>

          <button
            onClick={handleRunEvaluation}
            disabled={runningEval}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-emerald-950/50 border border-emerald-400/30 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            {runningEval ? (
              <>
                <RefreshCw className="h-4 w-4 animate-spin text-white" />
                <span>Running Golden Benchmark...</span>
              </>
            ) : (
              <>
                <Play className="h-4 w-4 fill-white" />
                <span>Run Benchmark Evaluation</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Quality Gate Scorecards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">

        {/* Recall@5 Gate */}
        <div className="glass-card p-5 rounded-2xl border border-white/[0.08] relative overflow-hidden group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase text-gray-400">Recall@5 Gate</span>
            <span className="h-2 w-2 rounded-full bg-emerald-400 badge-glow-emerald" />
          </div>
          <div className="flex items-baseline gap-2">
            <div className="text-3xl font-extrabold text-emerald-400 font-mono">
              {latestRun?.recall_at_5 != null ? `${(latestRun.recall_at_5 * 100).toFixed(1)}%` : 'Unavailable'}
            </div>
            <span className="text-[11px] font-mono text-emerald-300/80">Target: &ge; 75%</span>
          </div>
          <div className="w-full bg-white/[0.05] rounded-full h-1.5 mt-3 overflow-hidden">
            <div
              className="bg-emerald-500 h-1.5 rounded-full transition-all duration-500"
              style={{ width: `${(latestRun?.recall_at_5 ?? 0) * 100}%` }}
            />
          </div>
          <span className="text-[10px] text-gray-400 block mt-2 font-mono">Run: {latestRun?.status ?? 'Unavailable'}</span>
        </div>

        {/* Faithfulness Gate */}
        <div className="glass-card p-5 rounded-2xl border border-white/[0.08] relative overflow-hidden group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase text-gray-400">Faithfulness Gate</span>
            <span className="h-2 w-2 rounded-full bg-purple-400 badge-glow-violet" />
          </div>
          <div className="flex items-baseline gap-2">
            <div className="text-3xl font-extrabold text-purple-400 font-mono">
              {latestRun?.faithfulness != null ? `${(latestRun.faithfulness * 100).toFixed(1)}%` : 'Unavailable'}
            </div>
            <span className="text-[11px] font-mono text-purple-300/80">Target: &ge; 80%</span>
          </div>
          <div className="w-full bg-white/[0.05] rounded-full h-1.5 mt-3 overflow-hidden">
            <div
              className="bg-purple-500 h-1.5 rounded-full transition-all duration-500"
              style={{ width: `${(latestRun?.faithfulness ?? 0) * 100}%` }}
            />
          </div>
          <span className="text-[10px] text-gray-400 block mt-2 font-mono">Judge: {latestRun?.judge_status ?? 'Unavailable'}</span>
        </div>

        {/* MRR Gate */}
        <div className="glass-card p-5 rounded-2xl border border-white/[0.08] relative overflow-hidden group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase text-gray-400">MRR Ranking Depth</span>
            <span className="h-2 w-2 rounded-full bg-cyan-400 badge-glow-cyan" />
          </div>
          <div className="flex items-baseline gap-2">
            <div className="text-3xl font-extrabold text-cyan-400 font-mono">
              {latestRun?.mrr != null ? `${(latestRun.mrr * 100).toFixed(1)}%` : 'Unavailable'}
            </div>
            <span className="text-[11px] font-mono text-cyan-300/80">Target: &ge; 70%</span>
          </div>
          <div className="w-full bg-white/[0.05] rounded-full h-1.5 mt-3 overflow-hidden">
            <div
              className="bg-cyan-500 h-1.5 rounded-full transition-all duration-500"
              style={{ width: `${(latestRun?.mrr ?? 0) * 100}%` }}
            />
          </div>
          <span className="text-[10px] text-gray-400 block mt-2 font-mono">Mode: {latestRun?.config?.mode ?? 'Unavailable'}</span>
        </div>

        {/* Relevancy Gate */}
        <div className="glass-card p-5 rounded-2xl border border-white/[0.08] relative overflow-hidden group">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold uppercase text-gray-400">Answer Relevancy</span>
            <span className="h-2 w-2 rounded-full bg-amber-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <div className="text-3xl font-extrabold text-amber-400 font-mono">
              {latestRun?.relevancy != null ? `${(latestRun.relevancy * 100).toFixed(1)}%` : 'Unavailable'}
            </div>
            <span className="text-[11px] font-mono text-amber-300/80">Target: &ge; 75%</span>
          </div>
          <div className="w-full bg-white/[0.05] rounded-full h-1.5 mt-3 overflow-hidden">
            <div
              className="bg-amber-500 h-1.5 rounded-full transition-all duration-500"
              style={{ width: `${(latestRun?.relevancy ?? 0) * 100}%` }}
            />
          </div>
          <span className="text-[10px] text-gray-400 block mt-2 font-mono">Conditioned on query intent</span>
        </div>

      </div>

      {/* Interactive Trend Chart Card */}
      <div className="glass-card p-6 rounded-2xl border border-white/[0.08] shadow-2xl space-y-6">

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-emerald-400" />
              <span>Historical Metric Trajectory</span>
            </h2>
            <p className="text-xs text-gray-400 mt-0.5">Toggle metrics to isolate retrieval, ranking, and synthesis trends</p>
          </div>

          {/* Metric Selector Pills */}
          <div className="flex flex-wrap gap-2">
            {Object.entries(metricConfigs).map(([key, config]) => {
              const active = activeMetrics[key];
              return (
                <button
                  key={key}
                  onClick={() => toggleMetric(key)}
                  className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-mono font-medium transition border ${
                    active
                      ? 'bg-white/[0.08] text-white border-white/[0.2]'
                      : 'bg-white/[0.02] text-gray-500 border-white/[0.05] opacity-60'
                  }`}
                >
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: active ? config.color : '#4b5563' }}
                  />
                  <span>{config.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Recharts Multi-line Container */}
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={evalData} margin={{ top: 10, right: 30, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.05)" />
              <XAxis
                dataKey="run"
                stroke="#64748b"
                tick={{ fontSize: 11, fill: '#94a3b8' }}
              />
              <YAxis
                domain={[0.0, 1.0]}
                stroke="#64748b"
                tick={{ fontSize: 11, fill: '#94a3b8' }}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0b0f19',
                  borderColor: 'rgba(255, 255, 255, 0.1)',
                  borderRadius: '12px',
                  color: '#fff',
                  fontFamily: 'JetBrains Mono',
                  boxShadow: '0 8px 30px rgba(0,0,0,0.8)'
                }}
                formatter={(value: any) => [`${(Number(value) * 100).toFixed(1)}%`]}
              />
              <Legend
                wrapperStyle={{ paddingTop: '10px', fontSize: '12px', fontFamily: 'JetBrains Mono' }}
              />
              {activeMetrics.recall_at_5 && (
                <Line
                  type="monotone"
                  dataKey="recall_at_5"
                  stroke={metricConfigs.recall_at_5.color}
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                  name="Recall@5"
                />
              )}
              {activeMetrics.faithfulness && (
                <Line
                  type="monotone"
                  dataKey="faithfulness"
                  stroke={metricConfigs.faithfulness.color}
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                  name="Faithfulness"
                />
              )}
              {activeMetrics.mrr && (
                <Line
                  type="monotone"
                  dataKey="mrr"
                  stroke={metricConfigs.mrr.color}
                  strokeWidth={3}
                  dot={{ r: 4 }}
                  activeDot={{ r: 6 }}
                  name="MRR"
                />
              )}
              {activeMetrics.relevancy && (
                <Line
                  type="monotone"
                  dataKey="relevancy"
                  stroke={metricConfigs.relevancy.color}
                  strokeWidth={2}
                  name="Relevancy"
                />
              )}
              {activeMetrics.precision_at_5 && (
                <Line
                  type="monotone"
                  dataKey="precision_at_5"
                  stroke={metricConfigs.precision_at_5.color}
                  strokeWidth={2}
                  name="Precision@5"
                />
              )}
              {activeMetrics.context_precision && (
                <Line
                  type="monotone"
                  dataKey="context_precision"
                  stroke={metricConfigs.context_precision.color}
                  strokeWidth={2}
                  name="Context Precision"
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

      </div>

      {/* Historical Runs Table */}
      <div className="glass-card rounded-2xl border border-white/[0.08] overflow-hidden shadow-xl">
        <div className="p-6 border-b border-white/[0.06] flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Award className="h-4 w-4 text-emerald-400" />
              <span>Benchmark Runs Registry</span>
            </h3>
            <p className="text-xs text-gray-400 mt-0.5">Recorded regression test runs with complete scorecards</p>
          </div>
          <span className="text-xs font-mono text-gray-400">{evalData.length} Total Runs</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-gray-300">
            <thead className="bg-[#090d16] text-gray-400 uppercase text-[11px] font-mono border-b border-white/[0.05]">
              <tr>
                <th className="px-6 py-3.5">Run Label</th>
                <th className="px-6 py-3.5">Recall@5</th>
                <th className="px-6 py-3.5">MRR</th>
                <th className="px-6 py-3.5">Faithfulness</th>
                <th className="px-6 py-3.5">Relevancy</th>
                <th className="px-6 py-3.5">Context Prec.</th>
                <th className="px-6 py-3.5">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
              {evalData.map((row, idx) => (
                <tr key={idx} className="hover:bg-white/[0.02] transition">
                  <td className="px-6 py-4 font-medium text-white flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-cyan-400" />
                    <span>{row.run}</span>
                  </td>
                  <td className="px-6 py-4 font-mono text-emerald-400 font-semibold">
                    {row.recall_at_5 != null ? (row.recall_at_5 * 100).toFixed(1) : '-'}%
                  </td>
                  <td className="px-6 py-4 font-mono text-cyan-400 font-semibold">
                    {row.mrr != null ? (row.mrr * 100).toFixed(1) : '-'}%
                  </td>
                  <td className="px-6 py-4 font-mono text-purple-400 font-semibold">
                    {row.faithfulness != null ? (row.faithfulness * 100).toFixed(1) : '-'}%
                  </td>
                  <td className="px-6 py-4 font-mono text-amber-400">
                    {row.relevancy != null ? (row.relevancy * 100).toFixed(1) : '-'}%
                  </td>
                  <td className="px-6 py-4 font-mono text-pink-400">
                    {row.context_precision != null ? (row.context_precision * 100).toFixed(1) : '-'}%
                  </td>
                  <td className="px-6 py-4">
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      <CheckCircle2 className="h-3 w-3 text-emerald-400" />
                      {row.status ?? 'Unavailable'} / judge: {row.judge_status ?? 'Unavailable'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* CI/CD Command Snippet */}
      <div className="glass-card p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-3">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <Zap className="h-4 w-4 text-cyan-400" />
          <span>Automated CI/CD Quality Gate Runner</span>
        </h3>
        <p className="text-xs text-gray-400">
          Run this regression suite in GitHub Actions or locally to enforce threshold gates before deployment:
        </p>
        <pre className="p-3.5 rounded-xl bg-black/60 border border-white/[0.08] text-xs font-mono text-cyan-300 overflow-x-auto">
          python -m src.evaluation.regression_runner --dataset eval/fixtures/queries.jsonl --corpus eval/fixtures/corpus --mode hybrid --min-recall 0.9
        </pre>
      </div>

    </div>
  );
}
