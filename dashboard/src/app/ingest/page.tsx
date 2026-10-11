'use client';

import React, { useState } from 'react';
import { 
  Database, 
  UploadCloud, 
  FileText, 
  Layers, 
  CheckCircle2, 
  AlertCircle, 
  Sliders, 
  RefreshCw, 
  Sparkles, 
  HardDrive,
  Cpu,
  ArrowRight
} from 'lucide-react';

export default function IngestStudioPage() {
  const [activeTab, setActiveTab] = useState<'text' | 'file'>('text');
  const [text, setText] = useState('');
  const [docId, setDocId] = useState('');
  const [strategy, setStrategy] = useState('recursive');
  const [chunkSize, setChunkSize] = useState(500);
  const [chunkOverlap, setChunkOverlap] = useState(50);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  const [loading, setLoading] = useState(false);
  const [successResult, setSuccessResult] = useState<any>(null);
  const [errorResult, setErrorResult] = useState<string | null>(null);

  const handleIngestText = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setSuccessResult(null);
    setErrorResult(null);

    try {
      const res = await fetch('http://localhost:8000/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          doc_id: docId || undefined,
          strategy,
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
        }),
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }

      const data = await res.json();
      setSuccessResult(data);
      setText('');
      setDocId('');
    } catch (err: any) {
      setErrorResult(err.message || 'Ingestion failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleIngestFile = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setSuccessResult(null);
    setErrorResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('strategy', strategy);
      formData.append('chunk_size', String(chunkSize));
      formData.append('chunk_overlap', String(chunkOverlap));

      const res = await fetch('http://localhost:8000/ingest/file', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.text();
        throw new Error(err);
      }

      const data = await res.json();
      setSuccessResult(data);
      setSelectedFile(null);
    } catch (err: any) {
      setErrorResult(err.message || 'File ingestion failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              <Database className="h-3.5 w-3.5 text-cyan-400" />
              Vector Index & Chunking Studio
            </span>
          </div>
          <h1 className="text-3xl font-extrabold text-white mt-1">
            Knowledge Ingestion & <span className="gradient-text-cyan-indigo">Chunk Pipeline</span>
          </h1>
          <p className="text-gray-400 text-sm mt-1">
            Parse, segment, and index documents into Qdrant vector store with configurable strategies.
          </p>
        </div>
      </div>

      {/* Main 2-Column Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Form & Ingestion (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          
          <div className="glass-card p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-5">
            
            {/* Mode Switcher Tabs */}
            <div className="flex items-center gap-2 bg-[#090d16] p-1 rounded-xl border border-white/[0.06] w-fit">
              <button
                onClick={() => setActiveTab('text')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition ${
                  activeTab === 'text'
                    ? 'bg-gradient-to-r from-cyan-600 to-indigo-600 text-white shadow'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <FileText className="h-3.5 w-3.5" />
                <span>Raw Text / Markdown</span>
              </button>
              <button
                onClick={() => setActiveTab('file')}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition ${
                  activeTab === 'file'
                    ? 'bg-gradient-to-r from-cyan-600 to-indigo-600 text-white shadow'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <UploadCloud className="h-3.5 w-3.5" />
                <span>Upload Document File</span>
              </button>
            </div>

            {/* Tab 1: Text Mode */}
            {activeTab === 'text' && (
              <div className="space-y-4">
                <div>
                  <label className="text-xs font-medium text-gray-400 block mb-1">
                    Document Identifier (Optional)
                  </label>
                  <input
                    type="text"
                    value={docId}
                    onChange={(e) => setDocId(e.target.value)}
                    placeholder="e.g. docs/rag_architecture_v2"
                    className="w-full bg-[#0a0e1a]/80 border border-white/[0.1] rounded-xl px-4 py-2.5 text-white text-xs font-mono placeholder-gray-500 focus:outline-none focus:border-cyan-500 transition"
                  />
                </div>

                <div>
                  <label className="text-xs font-medium text-gray-400 block mb-1">
                    Document Text Content
                  </label>
                  <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Paste technical documentation, notes, or knowledge chunks to embed into Qdrant..."
                    rows={8}
                    className="w-full bg-[#0a0e1a]/80 border border-white/[0.1] rounded-xl p-4 text-white text-xs placeholder-gray-500 focus:outline-none focus:border-cyan-500 font-sans transition resize-none leading-relaxed"
                  />
                </div>

                <button
                  onClick={handleIngestText}
                  disabled={loading || !text.trim()}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-600 via-indigo-600 to-violet-600 hover:from-cyan-500 hover:to-indigo-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-cyan-950/50 border border-cyan-400/30 flex items-center justify-center gap-2 transition"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      <span>Embedding & Indexing Chunks...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      <span>Ingest Text Chunks to Vector DB</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Tab 2: File Upload Mode */}
            {activeTab === 'file' && (
              <div className="space-y-4">
                <div className="border-2 border-dashed border-white/[0.1] hover:border-cyan-500/50 rounded-2xl p-8 text-center transition bg-[#0a0e1a]/40">
                  <UploadCloud className="h-10 w-10 text-cyan-400 mx-auto mb-3" />
                  <p className="text-sm font-semibold text-white">Upload Knowledge Document</p>
                  <p className="text-xs text-gray-400 mt-1">Supports PDF, TXT, MD, and HTML files</p>
                  
                  <input
                    type="file"
                    id="docFileInput"
                    accept=".pdf,.txt,.md,.html"
                    onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                    className="hidden"
                  />
                  <label
                    htmlFor="docFileInput"
                    className="mt-4 inline-block px-4 py-2 rounded-xl bg-white/[0.06] hover:bg-white/[0.12] text-xs font-semibold text-cyan-300 border border-cyan-500/30 cursor-pointer transition"
                  >
                    Select File
                  </label>
                  
                  {selectedFile && (
                    <div className="mt-3 p-2 rounded-lg bg-cyan-950/40 text-cyan-300 border border-cyan-500/30 text-xs font-mono">
                      Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                    </div>
                  )}
                </div>

                <button
                  onClick={handleIngestFile}
                  disabled={loading || !selectedFile}
                  className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-600 via-indigo-600 to-violet-600 hover:from-cyan-500 hover:to-indigo-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-cyan-950/50 border border-cyan-400/30 flex items-center justify-center gap-2 transition"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="h-4 w-4 animate-spin" />
                      <span>Parsing & Ingesting File...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="h-4 w-4" />
                      <span>Process & Ingest File</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Results Feedback */}
            {successResult && (
              <div className="p-4 rounded-xl bg-emerald-950/40 border border-emerald-500/30 text-emerald-300 text-xs space-y-1 animate-in fade-in duration-200">
                <div className="flex items-center gap-2 font-bold text-sm">
                  <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  <span>Ingestion Successful!</span>
                </div>
                <p className="font-mono">
                  Created & embedded <strong>{successResult.chunks_created}</strong> chunks into Qdrant collection.
                </p>
                <p className="font-mono text-[10px] text-emerald-400/80">Doc ID: {successResult.doc_id}</p>
              </div>
            )}

            {errorResult && (
              <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs space-y-1">
                <div className="flex items-center gap-2 font-bold">
                  <AlertCircle className="h-4 w-4 text-rose-400" />
                  <span>Ingestion Failed</span>
                </div>
                <p className="font-mono">{errorResult}</p>
              </div>
            )}

          </div>

        </div>

        {/* Right Column: Chunking Config & Vector Store HUD (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          
          {/* Chunking Parameters Card */}
          <div className="glass-card p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-5">
            <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Sliders className="h-4 w-4 text-cyan-400" />
                <span>Chunking Configuration</span>
              </h3>
              <span className="text-[10px] font-mono text-cyan-400">Pre-processing</span>
            </div>

            {/* Strategy Select */}
            <div className="space-y-1.5">
              <label className="text-xs text-gray-300 font-medium">Splitter Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full bg-[#0a0e1a]/80 border border-white/[0.1] rounded-xl px-3 py-2 text-white text-xs font-mono focus:outline-none focus:border-cyan-500"
              >
                <option value="recursive">Recursive Character (General)</option>
                <option value="semantic">Semantic Similarity Splitter</option>
                <option value="sentence">Sentence Window Splitter</option>
              </select>
            </div>

            {/* Chunk Size */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-300 font-medium">Chunk Size (chars)</label>
                <span className="font-mono text-cyan-400 font-bold text-xs">{chunkSize}</span>
              </div>
              <input
                type="range"
                min="100"
                max="2000"
                step="50"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="w-full accent-cyan-500 cursor-pointer"
              />
              <span className="text-[10px] text-gray-500 block">Target tokens per vector segment</span>
            </div>

            {/* Chunk Overlap */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-300 font-medium">Chunk Overlap (chars)</label>
                <span className="font-mono text-indigo-400 font-bold text-xs">{chunkOverlap}</span>
              </div>
              <input
                type="range"
                min="0"
                max="300"
                step="10"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(Number(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer"
              />
              <span className="text-[10px] text-gray-500 block">Boundary continuity preservation</span>
            </div>

          </div>

          {/* Vector Store HUD */}
          <div className="glass-card p-6 rounded-2xl border border-white/[0.08] shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <HardDrive className="h-4 w-4 text-emerald-400" />
                <span>Vector Index Specs</span>
              </h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                CONNECTED
              </span>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                <span className="text-gray-400">Database Engine</span>
                <span className="font-mono text-white font-semibold">Qdrant In-Memory / Cloud</span>
              </div>
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                <span className="text-gray-400">Embedding Model</span>
                <span className="font-mono text-cyan-400">models/gemini-embedding-001</span>
              </div>
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                <span className="text-gray-400">Vector Dimensions</span>
                <span className="font-mono text-indigo-400">768 Dim (Dense)</span>
              </div>
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                <span className="text-gray-400">Distance Metric</span>
                <span className="font-mono text-white">Cosine Similarity</span>
              </div>
              <div className="flex items-center justify-between p-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]">
                <span className="text-gray-400">Reranker Model</span>
                <span className="font-mono text-purple-400">BAAI/bge-reranker-base</span>
              </div>
            </div>

          </div>

        </div>

      </div>

    </div>
  );
}
