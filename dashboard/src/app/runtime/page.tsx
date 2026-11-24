'use client';

import React, { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Database,
  FileText,
  RefreshCw,
  ServerCog,
  Trash2,
} from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

type RuntimeStatus = {
  readiness?: any;
  config?: Record<string, any>;
  telemetry?: Record<string, any>;
  metrics_enabled?: boolean;
};

type DocumentRow = {
  doc_id: string;
  tenant_id: string;
  source: string;
  active_version: number | null;
  status: string;
  chunk_count: number;
  content_sha256?: string | null;
};

export default function RuntimePage() {
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [ready, setReady] = useState<any>(null);
  const [documents, setDocuments] = useState<DocumentRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [runtimeRes, readyRes, docsRes] = await Promise.allSettled([
        fetch(`${API_BASE}/runtime/status`, { cache: 'no-store' }),
        fetch(`${API_BASE}/ready`, { cache: 'no-store' }),
        fetch(`${API_BASE}/documents`, { cache: 'no-store' }),
      ]);

      if (runtimeRes.status === 'fulfilled' && runtimeRes.value.ok) {
        setRuntime(await runtimeRes.value.json());
      }
      if (readyRes.status === 'fulfilled') {
        try {
          setReady(await readyRes.value.json());
        } catch {
          setReady(null);
        }
      }
      if (docsRes.status === 'fulfilled') {
        if (docsRes.value.ok) {
          const data = await docsRes.value.json();
          setDocuments(Array.isArray(data) ? data : []);
        } else {
          setError(`Documents endpoint returned ${docsRes.value.status}`);
        }
      }
    } catch (err: any) {
      setError(err?.message || 'Failed to load runtime status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const refreshIndex = async () => {
    setRefreshing(true);
    setMessage(null);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/index/refresh`, { method: 'POST' });
      const body = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(body?.error?.message || `Index refresh failed (${res.status})`);
      }
      setMessage(`Index refreshed to version ${body?.index?.version ?? 'n/a'} (${body?.index?.chunk_count ?? 0} chunks).`);
      await load();
    } catch (err: any) {
      setError(err?.message || 'Index refresh failed');
    } finally {
      setRefreshing(false);
    }
  };

  const deleteDocument = async (docId: string) => {
    setMessage(null);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}`, { method: 'DELETE' });
      const body = await res.json().catch(() => null);
      if (!res.ok) {
        throw new Error(body?.error?.message || `Delete failed (${res.status})`);
      }
      setMessage(`Deleted ${body?.chunks_deleted ?? 0} chunks for ${docId}. Index refreshed: ${String(body?.index_refreshed)}.`);
      await load();
    } catch (err: any) {
      setError(err?.message || 'Delete failed');
    }
  };

  const components = ready?.components || runtime?.readiness?.components || {};
  const config = runtime?.config || {};
  const index = components.index || {};
  const models = components.models?.models || {};
  const readinessStatus = ready?.status || runtime?.readiness?.status || 'unavailable';

  return (
    <div className="space-y-8">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <ServerCog className="h-3.5 w-3.5" />
            Live service state, no synthetic defaults
          </span>
          <h1 className="text-3xl font-extrabold text-white mt-1">
            Runtime & <span className="gradient-text-cyan-indigo">Index Operations</span>
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Readiness, model lifecycle, retrieval index, active documents and refresh controls backed by API responses.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={load}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.09] text-xs font-semibold text-gray-200 border border-white/[0.08] transition"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            Reload
          </button>
          <button
            onClick={refreshIndex}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 disabled:opacity-50 text-white text-xs font-semibold border border-cyan-400/30 transition"
          >
            <Activity className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            Rebuild Index
          </button>
          <Link
            href="/ingest"
            className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.09] text-xs font-semibold text-gray-200 border border-white/[0.08] transition"
          >
            <Database className="h-4 w-4 text-cyan-400" />
            Ingest
          </Link>
        </div>
      </div>

      {message && (
        <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/30 text-emerald-300 text-xs flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4" />
          <span>{message}</span>
        </div>
      )}
      {error && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="h-4 w-4" />
          <span>{error}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass-card p-6 rounded-2xl border border-white/[0.08]">
          <h2 className="text-sm font-bold text-white mb-3">Readiness</h2>
          <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold border ${
            readinessStatus === 'ready'
              ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30'
              : readinessStatus === 'degraded'
              ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
              : 'bg-rose-500/10 text-rose-300 border-rose-500/30'
          }`}>
            {readinessStatus.toUpperCase()}
          </div>
          <ul className="mt-3 space-y-1 text-xs text-gray-400">
            {(ready?.reasons || []).length === 0 ? (
              <li className="font-mono">No blocking readiness reasons.</li>
            ) : (
              (ready?.reasons || []).map((reason: string) => (
                <li key={reason} className="font-mono text-amber-300">• {reason}</li>
              ))
            )}
            {(ready?.optional_failures || []).map((name: string) => (
              <li key={name} className="font-mono text-gray-500">optional model failed: {name}</li>
            ))}
          </ul>
        </div>

        <div className="glass-card p-6 rounded-2xl border border-white/[0.08]">
          <h2 className="text-sm font-bold text-white mb-3">Retrieval index</h2>
          <dl className="space-y-2 text-xs">
            <div className="flex justify-between"><dt className="text-gray-400">Status</dt><dd className="font-mono text-white">{index.status ?? 'unknown'}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-400">Version</dt><dd className="font-mono text-cyan-300">{index.version ?? 'n/a'}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-400">Chunks</dt><dd className="font-mono text-cyan-300">{index.chunk_count ?? 0}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-400">Builds / failures</dt><dd className="font-mono text-gray-300">{index.builds ?? 0} / {index.failures ?? 0}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-400">Last error</dt><dd className="font-mono text-rose-300">{index.last_error ?? 'none'}</dd></div>
          </dl>
        </div>

        <div className="glass-card p-6 rounded-2xl border border-white/[0.08]">
          <h2 className="text-sm font-bold text-white mb-3">Telemetry</h2>
          <dl className="space-y-2 text-xs">
            <div className="flex justify-between"><dt className="text-gray-400">Sink</dt><dd className="font-mono text-white">{runtime?.telemetry?.sink ?? 'unknown'}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-400">Written</dt><dd className="font-mono text-emerald-300">{runtime?.telemetry?.written ?? 0}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-400">Dropped / failed</dt><dd className="font-mono text-amber-300">{runtime?.telemetry?.dropped ?? 0} / {runtime?.telemetry?.failed ?? 0}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-400">Queue</dt><dd className="font-mono text-gray-300">{runtime?.telemetry?.queue_depth ?? 0}/{runtime?.telemetry?.queue_capacity ?? 0}</dd></div>
            <div className="flex justify-between"><dt className="text-gray-400">Metrics</dt><dd className="font-mono text-gray-300">{runtime?.metrics_enabled ? 'enabled' : 'disabled'}</dd></div>
          </dl>
        </div>
      </div>

      <div className="glass-card p-6 rounded-2xl border border-white/[0.08]">
        <h2 className="text-sm font-bold text-white mb-4">Effective configuration</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {Object.entries(config).map(([key, value]) => (
            <div key={key} className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05]">
              <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono block">{key}</span>
              <span className="text-xs text-gray-200 font-mono break-all">{String(value)}</span>
            </div>
          ))}
          {Object.keys(config).length === 0 && <p className="text-xs text-gray-500">Configuration unavailable.</p>}
        </div>
      </div>

      <div className="glass-card p-6 rounded-2xl border border-white/[0.08]">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-bold text-white flex items-center gap-2">
            <FileText className="h-4 w-4 text-cyan-400" />
            Active documents
          </h2>
          <span className="text-[10px] font-mono text-gray-500">{documents.length} documents</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-gray-300">
            <thead className="text-gray-400 uppercase text-[10px] font-mono border-b border-white/[0.05]">
              <tr>
                <th className="px-3 py-2">Document</th>
                <th className="px-3 py-2">Tenant</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Version</th>
                <th className="px-3 py-2">Chunks</th>
                <th className="px-3 py-2">SHA-256</th>
                <th className="px-3 py-2"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04]">
              {documents.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-3 py-6 text-center text-gray-500 font-mono">
                    {loading ? 'Loading documents...' : 'No active documents. Ingest a document to build the index.'}
                  </td>
                </tr>
              ) : (
                documents.map((doc) => (
                  <tr key={`${doc.tenant_id}:${doc.doc_id}`} className="hover:bg-white/[0.02]">
                    <td className="px-3 py-2 font-mono text-cyan-300">{doc.doc_id}</td>
                    <td className="px-3 py-2 font-mono text-gray-400">{doc.tenant_id}</td>
                    <td className="px-3 py-2 font-mono">{doc.status}</td>
                    <td className="px-3 py-2 font-mono">{doc.active_version ?? 'n/a'}</td>
                    <td className="px-3 py-2 font-mono">{doc.chunk_count}</td>
                    <td className="px-3 py-2 font-mono text-gray-500">{doc.content_sha256 ? doc.content_sha256.slice(0, 12) : 'n/a'}</td>
                    <td className="px-3 py-2 text-right">
                      <button
                        onClick={() => deleteDocument(doc.doc_id)}
                        className="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/20 text-[10px] font-semibold transition"
                      >
                        <Trash2 className="h-3 w-3" />
                        Delete
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="glass-card p-6 rounded-2xl border border-white/[0.08]">
        <h2 className="text-sm font-bold text-white mb-3">Model lifecycle</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {Object.entries(models).map(([name, value]: [string, any]) => (
            <div key={name} className="p-3 rounded-xl bg-white/[0.02] border border-white/[0.05]">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-gray-200">{name}</span>
                <span className="text-[10px] font-mono text-gray-500">{value.required ? 'required' : 'optional'}</span>
              </div>
              <span className={`text-[11px] font-mono ${value.state === 'ready' ? 'text-emerald-300' : value.state === 'failed' ? 'text-rose-300' : 'text-amber-300'}`}>
                {value.state ?? 'unknown'}
              </span>
              {value.error && <p className="text-[10px] text-rose-400 font-mono mt-1 break-all">{value.error}</p>}
            </div>
          ))}
          {Object.keys(models).length === 0 && <p className="text-xs text-gray-500">Model lifecycle status unavailable.</p>}
        </div>
      </div>
    </div>
  );
}
