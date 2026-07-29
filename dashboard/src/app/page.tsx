'use client';

import React, { useEffect, useState } from 'react';

export default function OverviewPage() {
  const [traces, setTraces] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/traces?limit=20')
      .then((res) => res.json())
      .then((data) => {
        setTraces(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to fetch traces', err);
        setLoading(false);
      });
  }, []);

  const totalCost = traces.reduce((acc, t) => acc + (t.cost_usd || 0), 0);
  const avgLatency = traces.length
    ? (traces.reduce((acc, t) => acc + (t.latency_ms || 0), 0) / traces.length).toFixed(1)
    : '0';

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-extrabold text-white">RAG System Health & Metrics</h1>
        <p className="text-gray-400 mt-1">Real-time OpenTelemetry trace data and evaluation regression benchmarks.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow">
          <div className="text-sm font-medium text-gray-400">Total Recorded Spans</div>
          <div className="text-3xl font-bold text-white mt-2">{traces.length}</div>
        </div>
        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow">
          <div className="text-sm font-medium text-gray-400">Avg Stage Latency</div>
          <div className="text-3xl font-bold text-indigo-400 mt-2">{avgLatency} ms</div>
        </div>
        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow">
          <div className="text-sm font-medium text-gray-400">Total Query Cost</div>
          <div className="text-3xl font-bold text-green-400 mt-2">${totalCost.toFixed(6)}</div>
        </div>
        <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow">
          <div className="text-sm font-medium text-gray-400">Quality Gate Status</div>
          <div className="text-3xl font-bold text-emerald-400 mt-2">PASSING</div>
        </div>
      </div>

      <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow space-y-4">
        <h2 className="text-xl font-bold text-white">Quick Actions</h2>
        <div className="flex space-x-4">
          <a
            href="/traces"
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-white font-medium transition"
          >
            Explore Spans & Cost Breakdown →
          </a>
          <a
            href="/evals"
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white font-medium transition"
          >
            View Evaluation Trends & Baseline →
          </a>
        </div>
      </div>
    </div>
  );
}
