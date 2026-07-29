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
} from 'recharts';

export default function EvalTrendsPage() {
  const [evalData, setEvalData] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [runningEval, setRunningEval] = useState<boolean>(false);

  const fetchReports = async () => {
    try {
      const res = await fetch('http://127.0.0.1:8000/evals/reports');
      if (res.ok) {
        const data = await res.json();
        setEvalData(data);
      }
    } catch (err) {
      console.warn('Could not fetch real evaluation reports:', err);
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
      const res = await fetch('http://127.0.0.1:8000/evals/run?accept_baseline=true', {
        method: 'POST',
      });
      if (res.ok) {
        // Poll for new report every 4 seconds up to 15 attempts (60 seconds)
        let attempts = 0;
        const initialCount = evalData.length;
        const interval = setInterval(async () => {
          attempts += 1;
          const reportsRes = await fetch('http://127.0.0.1:8000/evals/reports');
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

  return (
    <div className="space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white">Evaluation Metric Trends</h1>
          <p className="text-gray-400 text-sm mt-1">
            Historical tracking of Recall@5, MRR, and LLM-as-Judge Faithfulness across evaluation runs.
          </p>
        </div>
        <button
          onClick={handleRunEvaluation}
          disabled={runningEval}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white px-5 py-2.5 rounded-lg text-sm font-semibold shadow flex items-center gap-2 self-start md:self-auto transition"
        >
          {runningEval ? (
            <>
              <span className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
              Running Benchmark (Golden Dataset)...
            </>
          ) : (
            '⚡ Run Benchmark Evaluation'
          )}
        </button>
      </div>

      <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow space-y-6">
        <h2 className="text-xl font-semibold text-white">Retrieval Quality & Groundedness</h2>
        {evalData.length > 0 ? (
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={evalData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="run" stroke="#9CA3AF" />
                <YAxis domain={[0.0, 1.0]} stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1F2937', borderColor: '#4B5563', color: '#FFF' }}
                />
                <Legend />
                <Line type="monotone" dataKey="recall_at_5" stroke="#10B981" strokeWidth={3} name="Recall@5" />
                <Line type="monotone" dataKey="mrr" stroke="#3B82F6" strokeWidth={3} name="MRR" />
                <Line type="monotone" dataKey="faithfulness" stroke="#8B5CF6" strokeWidth={3} name="Faithfulness" />
                <Line type="monotone" dataKey="relevancy" stroke="#F59E0B" strokeWidth={2} name="Relevancy" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="p-8 text-center bg-gray-900 rounded-lg border border-gray-700 space-y-3">
            <p className="text-gray-300 font-medium">No live evaluation reports recorded yet.</p>
            <p className="text-gray-400 text-xs">
              To run a real benchmark against your Golden Dataset, execute:
            </p>
            <code className="block text-indigo-400 bg-gray-950 p-3 rounded text-sm font-mono max-w-md mx-auto">
              python -m src.evaluation.regression_runner --accept-baseline
            </code>
          </div>
        )}
      </div>

      <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow">
        <h3 className="text-lg font-bold text-white mb-4">Metric Quality Gates Reference</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div className="bg-gray-900 p-4 rounded-lg border border-gray-750">
            <span className="text-emerald-400 font-bold">Recall@5 Gate</span>
            <p className="text-gray-400 text-xs mt-1">Threshold: &ge; 0.75 | Max allowable drop: &le; 5% vs baseline</p>
          </div>
          <div className="bg-gray-900 p-4 rounded-lg border border-gray-750">
            <span className="text-purple-400 font-bold">Faithfulness Gate</span>
            <p className="text-gray-400 text-xs mt-1">Threshold: &ge; 0.80 | Score computed by LLM-as-Judge</p>
          </div>
          <div className="bg-gray-900 p-4 rounded-lg border border-gray-750">
            <span className="text-blue-400 font-bold">MRR Gate</span>
            <p className="text-gray-400 text-xs mt-1">Monitored for ranking position efficiency</p>
          </div>
        </div>
      </div>
    </div>
  );
}
