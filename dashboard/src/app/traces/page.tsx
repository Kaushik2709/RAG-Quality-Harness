'use client';

import React, { useEffect, useState } from 'react';

export default function TraceExplorerPage() {
  const [traces, setTraces] = useState<any[]>([]);
  const [filterStage, setFilterStage] = useState<string>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:8000/traces?limit=100')
      .then((res) => res.json())
      .then((data) => {
        setTraces(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  const filtered = filterStage === 'all'
    ? traces
    : traces.filter((t) => t.stage === filterStage);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">OpenTelemetry Trace Explorer</h1>
          <p className="text-gray-400 text-sm mt-1">Per-stage latency breakdown, token usage, and costs.</p>
        </div>
        <div className="flex items-center space-x-2">
          <label className="text-sm text-gray-400">Filter Stage:</label>
          <select
            value={filterStage}
            onChange={(e) => setFilterStage(e.target.value)}
            className="bg-gray-800 border border-gray-700 text-white rounded-lg px-3 py-1.5 focus:outline-none"
          >
            <option value="all">All Stages</option>
            <option value="retrieve">Retrieve</option>
            <option value="rerank">Rerank</option>
            <option value="generate">Generate</option>
          </select>
        </div>
      </div>

      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden shadow">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-gray-300">
            <thead className="bg-gray-900 text-gray-400 uppercase text-xs">
              <tr>
                <th className="px-6 py-3">Query ID</th>
                <th className="px-6 py-3">Stage</th>
                <th className="px-6 py-3">Latency (ms)</th>
                <th className="px-6 py-3">Tokens In / Out</th>
                <th className="px-6 py-3">Cost ($ USD)</th>
                <th className="px-6 py-3">Model</th>
                <th className="px-6 py-3">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-8 text-center text-gray-500">
                    {loading ? 'Loading traces...' : 'No telemetry traces recorded yet. Execute queries to see spans.'}
                  </td>
                </tr>
              ) : (
                filtered.map((t, idx) => (
                  <tr key={idx} className="hover:bg-gray-750 transition">
                    <td className="px-6 py-4 font-mono text-xs text-indigo-300">{t.query_id}</td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-semibold ${
                          t.stage === 'generate'
                            ? 'bg-purple-900 text-purple-200'
                            : t.stage === 'rerank'
                            ? 'bg-blue-900 text-blue-200'
                            : 'bg-emerald-900 text-emerald-200'
                        }`}
                      >
                        {t.stage}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-medium text-white">{t.latency_ms} ms</td>
                    <td className="px-6 py-4 text-xs font-mono">
                      {t.tokens_in} / {t.tokens_out}
                    </td>
                    <td className="px-6 py-4 text-xs font-mono text-green-400">
                      ${(t.cost_usd || 0).toFixed(6)}
                    </td>
                    <td className="px-6 py-4 text-xs text-gray-400">{t.model_name}</td>
                    <td className="px-6 py-4 text-xs text-gray-500">{t.timestamp}</td>
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
