import React from 'react';
import { Cpu, ShieldCheck, Sparkles, Terminal } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-white/[0.06] bg-[#05070a]/90 backdrop-blur-md py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-gray-500">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-gray-300">RAG Observability & Eval Platform</span>
            <span>•</span>
            <span className="font-mono text-cyan-400/80">LangGraph StateGraph Engine</span>
            <span>•</span>
            <span className="font-mono text-indigo-400/80">OpenTelemetry Traces</span>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-gray-400">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
              Guardrails Active
            </span>
            <span className="flex items-center gap-1.5 text-gray-400">
              <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
              Hybrid Retrieval
            </span>
            <span className="flex items-center gap-1.5 text-gray-400">
              <Cpu className="h-3.5 w-3.5 text-indigo-400" />
              BGE-Reranker Base
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
