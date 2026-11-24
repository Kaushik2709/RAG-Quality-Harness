'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  Activity, 
  Terminal, 
  Layers, 
  Cpu, 
  CheckCircle2, 
  AlertCircle, 
  BarChart3, 
  Sparkles,
  Database,
  ServerCog
} from 'lucide-react';

export default function Navbar() {
  const pathname = usePathname();
  const [apiStatus, setApiStatus] = useState<{
    online: boolean;
    framework?: string;
    llm?: string;
    reranker?: string;
    embedding?: string;
    pingMs?: number;
  }>({ online: false });

  useEffect(() => {
    const checkHealth = async () => {
      const start = performance.now();
      try {
        const res = await fetch('http://localhost:8000/', { cache: 'no-store' });
        const ping = Math.round(performance.now() - start);
        if (res.ok) {
          const data = await res.json();
          setApiStatus({
            online: true,
            framework: data.framework || 'LangGraph',
            llm: data.llm_provider || 'Gemini',
            reranker: data.reranker_model || 'BGE Reranker',
            embedding: data.embedding_model || 'Gemini Embeddings',
            pingMs: ping,
          });
        } else {
          setApiStatus({ online: false });
        }
      } catch {
        setApiStatus({ online: false });
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const navLinks = [
    { href: '/', label: 'Overview', icon: BarChart3 },
    { href: '/playground', label: 'RAG Playground', icon: Sparkles, highlight: true },
    { href: '/traces', label: 'Trace Waterfall', icon: Layers },
    { href: '/evals', label: 'Eval Benchmarks', icon: Activity },
    { href: '/ingest', label: 'Ingest Studio', icon: Database },
    { href: '/runtime', label: 'Runtime', icon: ServerCog },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/[0.08] bg-[#07090e]/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between gap-4">
          
          {/* Logo & Brand Identity */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-600 to-violet-600 p-[1px] shadow-glow-cyan transition duration-300 group-hover:scale-105">
              <div className="flex h-full w-full items-center justify-center rounded-[11px] bg-[#07090e]/90">
                <Cpu className="h-5 w-5 text-cyan-400 group-hover:rotate-12 transition-transform duration-300" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold tracking-tight text-white text-base">
                  Neuro<span className="text-cyan-400">Trace</span>
                </span>
                <span className="text-[10px] uppercase font-mono tracking-wider px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                  LangGraph v2.0
                </span>
              </div>
              <p className="text-[11px] text-gray-400 font-mono tracking-tight -mt-0.5">
                RAG Observability & Eval Harness
              </p>
            </div>
          </Link>

          {/* Nav Items */}
          <nav className="hidden md:flex items-center gap-1 bg-white/[0.03] p-1 rounded-xl border border-white/[0.06]">
            {navLinks.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${
                    isActive
                      ? 'bg-gradient-to-r from-cyan-500/20 to-indigo-500/20 text-cyan-300 border border-cyan-500/30 shadow-sm shadow-cyan-500/10'
                      : 'text-gray-400 hover:text-white hover:bg-white/[0.04]'
                  } ${item.highlight && !isActive ? 'text-cyan-400' : ''}`}
                >
                  <Icon className={`h-3.5 w-3.5 ${isActive ? 'text-cyan-400' : ''}`} />
                  <span>{item.label}</span>
                  {item.highlight && (
                    <span className="flex h-1.5 w-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  )}
                </Link>
              );
            })}
          </nav>

          {/* Right Action & API Status Badge */}
          <div className="flex items-center gap-3">
            <div className={`hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-mono border backdrop-blur-md transition-all ${
              apiStatus.online 
                ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-300 shadow-[0_0_12px_rgba(16,185,129,0.2)]'
                : 'bg-rose-950/40 border-rose-500/30 text-rose-300'
            }`}>
              {apiStatus.online ? (
                <>
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  <span>API 8000 LIVE</span>
                  <span className="text-emerald-500/60 font-mono text-[10px]">({apiStatus.pingMs}ms)</span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-3 w-3 text-rose-400" />
                  <span>API OFFLINE</span>
                </>
              )}
            </div>

            <Link
              href="/playground"
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold text-white bg-gradient-to-r from-cyan-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 shadow-md shadow-cyan-900/40 border border-cyan-400/30 transition hover:scale-[1.02] active:scale-[0.98]"
            >
              <Terminal className="h-3.5 w-3.5" />
              <span>Query Sandbox</span>
            </Link>
          </div>

        </div>
      </div>
    </header>
  );
}
