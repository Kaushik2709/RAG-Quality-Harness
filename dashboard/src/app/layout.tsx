import React from 'react';
import './globals.css';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';

export const metadata = {
  title: 'NeuroTrace | RAG Observability & Eval Harness',
  description: 'Enterprise RAG Observability with LangGraph, OpenTelemetry traces, Latency/Cost Breakdown, and Regression Quality Gates.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#07090e] text-slate-100 font-sans antialiased min-h-screen flex flex-col selection:bg-cyan-500/30 selection:text-cyan-200">
        {/* Ambient background glows */}
        <div className="ambient-glow-top" />
        <div className="fixed inset-0 cyber-grid pointer-events-none opacity-40 z-0" />
        
        {/* App Shell */}
        <div className="relative z-10 flex flex-col min-h-screen">
          <Navbar />
          <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {children}
          </main>
          <Footer />
        </div>
      </body>
    </html>
  );
}
