'use client';

import React, { useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import {
  Search,
  Filter,
  ArrowRight,
  Database,
  Calendar,
  AlertOctagon,
  ShieldCheck,
  ChevronRight
} from 'lucide-react';
import { useTransactionsQuery } from '../../hooks/useTransactions';

export default function TransactionsListPage() {
  const router = useRouter();
  const { data: transactions = [], isLoading } = useTransactionsQuery();

  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState<'ALL' | 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'>('ALL');
  const [actionFilter, setActionFilter] = useState<'ALL' | 'ALLOW' | 'STEP_UP' | 'MANUAL_REVIEW' | 'HOLD' | 'BLOCK'>('ALL');
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'PENDING_REVIEW' | 'REVIEW_COMPLETED' | 'CONFIRMED_ABUSE' | 'FALSE_POSITIVE' | 'DISMISSED'>('ALL');

  // Filter logic
  const filteredTxns = useMemo(() => {
    return transactions.filter((txn) => {
      // Risk filter
      const score = txn.riskDecision.score;
      if (riskFilter === 'CRITICAL' && score < 0.90) return false;
      if (riskFilter === 'HIGH' && (score < 0.70 || score >= 0.90)) return false;
      if (riskFilter === 'MEDIUM' && (score < 0.40 || score >= 0.70)) return false;
      if (riskFilter === 'LOW' && score >= 0.40) return false;

      // Recommended action filter
      if (actionFilter !== 'ALL' && txn.riskDecision.action !== actionFilter) return false;

      // Resolution status filter
      if (statusFilter !== 'ALL' && txn.status !== statusFilter) return false;

      // Search term
      if (searchTerm) {
        const term = searchTerm.toLowerCase();
        return (
          txn.id.toLowerCase().includes(term) ||
          txn.customerId.toLowerCase().includes(term) ||
          txn.deviceId.toLowerCase().includes(term) ||
          txn.ipAddress.toLowerCase().includes(term) ||
          txn.merchant.toLowerCase().includes(term)
        );
      }

      return true;
    });
  }, [transactions, searchTerm, riskFilter, actionFilter, statusFilter]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-extrabold tracking-tight text-text-primary uppercase flex items-center gap-2.5">
          <Database className="w-6 h-6 text-risk-info" />
          <span>Transaction Query Logs</span>
        </h2>
        <p className="text-xs text-text-secondary mt-1">
          Complete ledger of transaction risk scores, model decisions, and telemetry records.
        </p>
      </div>

      {/* Filter Toolbar Card */}
      <div className="bg-card border border-border rounded-xl p-5 select-none space-y-4">
        <div className="flex flex-col lg:flex-row gap-4 items-center justify-between">
          {/* Keyword Search */}
          <div className="relative w-full lg:max-w-md">
            <Search className="w-4 h-4 text-text-secondary absolute left-3.5 top-3" />
            <input
              type="text"
              placeholder="Search by Transaction ID, Customer, Device ID, IP..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-background border border-border rounded-lg pl-10 pr-4 py-2 text-xs text-text-primary placeholder-text-secondary outline-none focus:border-text-secondary/40 transition-colors"
            />
          </div>

          <div className="text-[11px] font-mono text-text-secondary flex items-center gap-1.5 self-end lg:self-center">
            <span>Query hits:</span>
            <span className="text-text-primary font-bold">{filteredTxns.length} records</span>
          </div>
        </div>

        <div className="h-px bg-border"></div>

        {/* Categories filters */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          {/* Risk Level Filter */}
          <div className="space-y-1.5">
            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Risk Severity</span>
            <select
              value={riskFilter}
              onChange={(e) => setRiskFilter(e.target.value as any)}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary outline-none focus:border-text-secondary/30 transition-colors cursor-pointer"
            >
              <option value="ALL">All Risk Levels</option>
              <option value="CRITICAL">Critical (90%+)</option>
              <option value="HIGH">High (70% - 90%)</option>
              <option value="MEDIUM">Medium (40% - 70%)</option>
              <option value="LOW">Low (Below 40%)</option>
            </select>
          </div>

          {/* Action Filter */}
          <div className="space-y-1.5">
            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Decision Action</span>
            <select
              value={actionFilter}
              onChange={(e) => setActionFilter(e.target.value as any)}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary outline-none focus:border-text-secondary/30 transition-colors cursor-pointer"
            >
              <option value="ALL">All Actions</option>
              <option value="ALLOW">ALLOW</option>
              <option value="STEP_UP">STEP_UP</option>
              <option value="MANUAL_REVIEW">MANUAL_REVIEW</option>
              <option value="HOLD">HOLD</option>
              <option value="BLOCK">BLOCK</option>
            </select>
          </div>

          {/* Review Status Filter */}
          <div className="space-y-1.5">
            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Resolution Status</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as any)}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary outline-none focus:border-text-secondary/30 transition-colors cursor-pointer"
            >
              <option value="ALL">All Statuses</option>
              <option value="PENDING_REVIEW">Pending Review</option>
              <option value="REVIEW_COMPLETED">Review Completed</option>
              <option value="CONFIRMED_ABUSE">Confirmed Abuse</option>
              <option value="FALSE_POSITIVE">False Positive</option>
              <option value="DISMISSED">Dismissed</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main ledger list table */}
      <div className="bg-card border border-border rounded-xl overflow-hidden select-none">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-sidebar/50 border-b border-border text-text-secondary font-mono font-bold">
              <th className="p-4">Transaction ID</th>
              <th className="p-4">Timestamp</th>
              <th className="p-4">Customer ID</th>
              <th className="p-4">Merchant</th>
              <th className="p-4">Risk Score</th>
              <th className="p-4">Decision</th>
              <th className="p-4">Device ID</th>
              <th className="p-4">Status</th>
              <th className="p-4 text-right">Action</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={9} className="p-12 text-center text-text-secondary font-medium">
                  Querying transaction data...
                </td>
              </tr>
            ) : filteredTxns.length === 0 ? (
              <tr>
                <td colSpan={9} className="p-12 text-center text-text-secondary font-medium">
                  No transaction matches found. Try adjusting filters.
                </td>
              </tr>
            ) : (
              filteredTxns.map((txn) => {
                const score = txn.riskDecision.score;
                const isCritical = score >= 0.90;
                const isHigh = score >= 0.70 && score < 0.90;
                const isMedium = score >= 0.40 && score < 0.70;

                let scoreColor = 'text-risk-low border-risk-low/20 bg-risk-low/10';
                if (isCritical) scoreColor = 'text-risk-high border-risk-high/30 bg-risk-high/15';
                else if (isHigh) scoreColor = 'text-risk-medium border-risk-medium/20 bg-risk-medium/10';
                else if (isMedium) scoreColor = 'text-risk-medium border-risk-medium/10 bg-risk-medium/5';

                let statusText = 'Pending Review';
                let statusColor = 'text-text-secondary bg-border border-border';
                if (txn.status === 'CONFIRMED_ABUSE') {
                  statusText = 'Confirmed Abuse';
                  statusColor = 'text-risk-high bg-risk-high/10 border-risk-high/20';
                } else if (txn.status === 'FALSE_POSITIVE') {
                  statusText = 'False Positive';
                  statusColor = 'text-risk-low bg-risk-low/10 border-risk-low/20';
                } else if (txn.status === 'REVIEW_COMPLETED') {
                  statusText = 'Completed';
                  statusColor = 'text-risk-low bg-risk-low/10 border-risk-low/20';
                }

                // Format timestamp
                const date = new Date(txn.timestamp);
                const formattedTime = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const formattedDate = date.toLocaleDateString([], { month: 'short', day: '2-digit' });

                return (
                  <tr
                    key={txn.id}
                    onClick={() => router.push(`/transactions/${txn.id}`)}
                    className="border-b border-border hover:bg-card/50 transition-all cursor-pointer"
                  >
                    <td className="p-4 font-bold text-text-primary">{txn.id}</td>
                    <td className="p-4 text-text-secondary font-medium">
                      <div className="flex items-center gap-1.5">
                        <Calendar className="w-3.5 h-3.5" />
                        <span>{formattedDate}, {formattedTime}</span>
                      </div>
                    </td>
                    <td className="p-4 font-mono text-text-secondary">{txn.customerId}</td>
                    <td className="p-4 text-text-primary font-medium">{txn.merchant}</td>
                    <td className="p-4">
                      <span className={`px-2 py-0.5 rounded border text-[10px] font-mono font-bold ${scoreColor}`}>
                        {Math.round(score * 100)}%
                      </span>
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        txn.riskDecision.action === 'BLOCK' ? 'bg-risk-high/20 text-risk-high' :
                        txn.riskDecision.action === 'HOLD' ? 'bg-risk-high/20 text-risk-high' :
                        txn.riskDecision.action === 'MANUAL_REVIEW' ? 'bg-risk-medium/20 text-risk-medium' :
                        'bg-risk-low/20 text-risk-low'
                      }`}>
                        {txn.riskDecision.action}
                      </span>
                    </td>
                    <td className="p-4 font-mono text-text-secondary truncate max-w-[100px]">{txn.deviceId}</td>
                    <td className="p-4">
                      <span className={`px-2 py-0.5 rounded border text-[9px] font-bold ${statusColor}`}>
                        {statusText}
                      </span>
                    </td>
                    <td className="p-4 text-right">
                      <button className="text-risk-ai hover:text-text-primary font-bold flex items-center gap-1 ml-auto cursor-pointer transition-colors">
                        <span>Inspect</span>
                        <ChevronRight className="w-3.5 h-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
