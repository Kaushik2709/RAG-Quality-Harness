import React from 'react';

export const metadata = {
  title: 'RAG Observability & Evaluation Platform',
  description: 'Trace latency, cost, recall@k, and faithfulness evaluation metrics.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"
        />
      </head>
      <body className="bg-gray-900 text-gray-100 font-sans antialiased min-h-screen">
        <nav className="border-b border-gray-800 bg-gray-950 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white">
              ⚡
            </div>
            <span className="font-bold text-xl tracking-tight text-white">RAG Eval Harness</span>
          </div>
          <div className="flex space-x-6">
            <a href="/" className="hover:text-indigo-400 font-medium transition">Overview</a>
            <a href="/traces" className="hover:text-indigo-400 font-medium transition">Trace Explorer</a>
            <a href="/evals" className="hover:text-indigo-400 font-medium transition">Eval Trends</a>
          </div>
        </nav>
        <main className="p-8 max-w-7xl mx-auto">{children}</main>
      </body>
    </html>
  );
}
