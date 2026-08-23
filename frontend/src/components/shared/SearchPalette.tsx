'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { Search, Database, Layers, User, X } from 'lucide-react';
import { mockTransactions } from '../../data/mock/transactions';
import { mockRiskRings } from '../../data/mock/rings';

interface SearchResult {
  id: string;
  type: 'TRANSACTION' | 'RING' | 'CUSTOMER' | 'DEVICE' | 'IP';
  title: string;
  subtitle: string;
  riskScore: number;
  url: string;
}

export default function SearchPalette({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const modalRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
      setQuery('');
      setResults([]);
    }
  }, [isOpen]);

  // Click outside to close modal
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
    }
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, [isOpen, onClose]);

  // Handle escape key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  // Query matching filter
  useEffect(() => {
    if (!query) {
      setResults([]);
      return;
    }

    const term = query.toLowerCase();
    const matches: SearchResult[] = [];

    // Filter Transactions
    mockTransactions.forEach((txn) => {
      if (
        txn.id.toLowerCase().includes(term) ||
        txn.customerId.toLowerCase().includes(term) ||
        txn.deviceId.toLowerCase().includes(term) ||
        txn.ipAddress.toLowerCase().includes(term) ||
        txn.merchant.toLowerCase().includes(term)
      ) {
        matches.push({
          id: txn.id,
          type: 'TRANSACTION',
          title: `${txn.id} • ${txn.merchant}`,
          subtitle: `Customer: ${txn.customerId} | Amt: ₹${txn.amount.toLocaleString('en-US')} | IP: ${txn.ipAddress}`,
          riskScore: txn.riskDecision.score,
          url: `/transactions/${txn.id}`,
        });
      }
    });

    // Filter Rings
    mockRiskRings.forEach((ring) => {
      if (ring.id.toLowerCase().includes(term) || ring.name.toLowerCase().includes(term)) {
        matches.push({
          id: ring.id,
          type: 'RING',
          title: `${ring.id} • ${ring.name}`,
          subtitle: `Exposure: ₹${ring.historicalExposure.toLocaleString('en-US')} | Members: ${ring.entities.length}`,
          riskScore: ring.riskScore,
          url: `/rings/${ring.id}`,
        });
      }
    });

    setResults(matches.slice(0, 8));
  }, [query]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-start justify-center pt-24 px-4 select-none">
      <div
        ref={modalRef}
        className="w-full max-w-2xl bg-elevated border border-border rounded-xl shadow-2xl overflow-hidden flex flex-col"
      >
        {/* Search Input Area */}
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-3 flex-1">
            <Search className="w-5 h-5 text-text-secondary" />
            <input
              ref={inputRef}
              type="text"
              placeholder="Search by ID, customer code, device key, IP address, merchant..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="bg-transparent border-0 outline-none text-sm text-text-primary placeholder-text-secondary w-full"
            />
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-card text-text-secondary hover:text-text-primary transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results Panel */}
        <div className="max-h-96 overflow-y-auto p-2">
          {query === '' ? (
            <div className="py-12 text-center text-xs text-text-secondary">
              Type keywords to query risk logs...
            </div>
          ) : results.length === 0 ? (
            <div className="py-12 text-center text-xs text-text-secondary">
              No matching records found.
            </div>
          ) : (
            <div className="space-y-1">
              <span className="px-3 py-1.5 text-[10px] font-bold text-text-secondary block tracking-wider uppercase">
                Search Results ({results.length})
              </span>
              {results.map((result) => {
                const isCritical = result.riskScore >= 0.90;
                const isHigh = result.riskScore >= 0.70 && result.riskScore < 0.90;
                const isMedium = result.riskScore >= 0.40 && result.riskScore < 0.70;
                
                let scoreColor = 'text-risk-low border-risk-low/20 bg-risk-low/10';
                if (isCritical) scoreColor = 'text-risk-high border-risk-high/20 bg-risk-high/10';
                else if (isHigh) scoreColor = 'text-risk-medium border-risk-medium/20 bg-risk-medium/10';
                else if (isMedium) scoreColor = 'text-risk-medium border-risk-medium/20 bg-risk-medium/10';

                return (
                  <button
                    key={result.id}
                    onClick={() => {
                      router.push(result.url);
                      onClose();
                    }}
                    className="w-full text-left p-3 rounded-lg hover:bg-card border border-transparent hover:border-border transition-all flex items-center justify-between gap-4 cursor-pointer"
                  >
                    <div className="flex items-center gap-3">
                      <div className="bg-border p-2 rounded-lg border border-border">
                        {result.type === 'TRANSACTION' ? (
                          <Database className="w-4 h-4 text-risk-info" />
                        ) : (
                          <Layers className="w-4 h-4 text-risk-ai" />
                        )}
                      </div>
                      <div className="flex flex-col gap-0.5">
                        <span className="text-xs font-semibold text-text-primary">
                          {result.title}
                        </span>
                        <span className="text-[10px] text-text-secondary font-mono truncate max-w-md">
                          {result.subtitle}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${scoreColor}`}>
                        {(result.riskScore * 100).toFixed(0)}% Risk
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Keyboard instructions footer */}
        <div className="bg-background/80 border-t border-border px-4 py-2 flex items-center justify-between text-[10px] text-text-secondary select-none">
          <div className="flex items-center gap-4">
            <span>↑↓ to navigate</span>
            <span>↵ to select</span>
          </div>
          <span>esc to close</span>
        </div>
      </div>
    </div>
  );
}
