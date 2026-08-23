'use client';

import React, { useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { Users, Search, Clock, ShieldCheck, ChevronRight, AlertOctagon } from 'lucide-react';
import { useInvestigationsQuery } from '../../hooks/useInvestigation';

export default function InvestigationsListPage() {
  const router = useRouter();
  const { data: cases = [], isLoading } = useInvestigationsQuery();

  // Filters
  const [searchTerm, setSearchTerm] = useState('');
  const [priorityFilter, setPriorityFilter] = useState<'ALL' | 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'>('ALL');
  const [statusFilter, setStatusFilter] = useState<'ALL' | 'PENDING_REVIEW' | 'REVIEW_COMPLETED' | 'CONFIRMED_ABUSE' | 'FALSE_POSITIVE'>('ALL');
  const [assigneeFilter, setAssigneeFilter] = useState<'ALL' | 'MINE' | 'SYSTEM'>('ALL');

  const filteredCases = useMemo(() => {
    return cases.filter((item) => {
      // Priority filter
      if (priorityFilter !== 'ALL' && item.priority !== priorityFilter) return false;

      // Status filter
      if (statusFilter !== 'ALL' && item.status !== statusFilter) return false;

      // Assignee filter
      if (assigneeFilter === 'MINE' && item.assignedTo !== 'Arjun Mehta') return false;
      if (assigneeFilter === 'SYSTEM' && item.assignedTo !== 'Sentinel Auto-Block') return false;

      // Search keyword
      if (searchTerm) {
        const term = searchTerm.toLowerCase();
        return (
          item.id.toLowerCase().includes(term) ||
          item.caseId.toLowerCase().includes(term) ||
          item.riskType.toLowerCase().includes(term) ||
          item.assignedTo.toLowerCase().includes(term)
        );
      }

      return true;
    });
  }, [cases, searchTerm, priorityFilter, statusFilter, assigneeFilter]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-extrabold tracking-tight text-text-primary uppercase flex items-center gap-2.5">
            <Users className="w-6 h-6 text-risk-ai" />
            <span>Investigation Case Queue</span>
          </h2>
          <p className="text-xs text-text-secondary mt-1">
            Manage, audit, and resolve flagged alerts assigned for manual risk investigations.
          </p>
        </div>

        <div className="text-[10px] font-mono text-text-secondary bg-card border border-border px-3 py-1.5 rounded-lg flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-risk-low animate-pulse"></span>
          <span>Analyst Queue: {cases.filter(c => c.status === 'PENDING_REVIEW').length} Open</span>
        </div>
      </div>

      {/* Toolbar filters */}
      <div className="bg-card border border-border rounded-xl p-5 select-none space-y-4">
        <div className="flex flex-col lg:flex-row gap-4 items-center justify-between">
          {/* Keyword Search */}
          <div className="relative w-full lg:max-w-md">
            <Search className="w-4 h-4 text-text-secondary absolute left-3.5 top-3" />
            <input
              type="text"
              placeholder="Search case key, transaction, risk type..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full bg-background border border-border rounded-lg pl-10 pr-4 py-2 text-xs text-text-primary placeholder-text-secondary outline-none focus:border-text-secondary/40 transition-colors"
            />
          </div>

          <div className="flex bg-background border border-border rounded-lg p-0.5 text-[10px] font-mono font-bold self-end lg:self-center">
            {([
              { label: 'All Cases', value: 'ALL' },
              { label: 'Assigned to Me', value: 'MINE' },
              { label: 'Sentinel Auto-Blocks', value: 'SYSTEM' }
            ] as const).map((a) => (
              <button
                key={a.value}
                onClick={() => setAssigneeFilter(a.value)}
                className={`px-3 py-1.5 rounded cursor-pointer ${
                  assigneeFilter === a.value ? 'bg-card text-text-primary border border-border' : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                {a.label}
              </button>
            ))}
          </div>
        </div>

        <div className="h-px bg-border"></div>

        {/* Dynamic drop down filters */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          {/* Priority */}
          <div className="space-y-1.5">
            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Priority Classification</span>
            <select
              value={priorityFilter}
              onChange={(e) => setPriorityFilter(e.target.value as any)}
              className="w-full bg-background border border-border rounded-lg px-3 py-2 text-text-primary outline-none focus:border-text-secondary/30 transition-colors cursor-pointer"
            >
              <option value="ALL">All Priorities</option>
              <option value="CRITICAL">Critical Priority</option>
              <option value="HIGH">High Priority</option>
              <option value="MEDIUM">Medium Priority</option>
              <option value="LOW">Low Priority</option>
            </select>
          </div>

          {/* Status */}
          <div className="space-y-1.5">
            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Operational Status</span>
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
            </select>
          </div>
        </div>
      </div>

      {/* Case queue ledger table */}
      <div className="bg-card border border-border rounded-xl overflow-hidden select-none">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-sidebar/50 border-b border-border text-text-secondary font-mono font-bold">
              <th className="p-4">Case ID</th>
              <th className="p-4">Risk Type / Scenario</th>
              <th className="p-4">Priority</th>
              <th className="p-4">Risk Score</th>
              <th className="p-4">Amount</th>
              <th className="p-4">Assigned To</th>
              <th className="p-4">SLA Deadline</th>
              <th className="p-4">Status</th>
              <th className="p-4 text-right">Review</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={9} className="p-12 text-center text-text-secondary font-medium">
                  Loading investigation queue...
                </td>
              </tr>
            ) : filteredCases.length === 0 ? (
              <tr>
                <td colSpan={9} className="p-12 text-center text-text-secondary font-medium">
                  No active cases match current queue parameters.
                </td>
              </tr>
            ) : (
              filteredCases.map((item) => {
                const score = item.riskScore;
                const isCritical = item.priority === 'CRITICAL';
                const isHigh = item.priority === 'HIGH';

                let priorityBadge = 'bg-border text-text-secondary border-border';
                if (isCritical) priorityBadge = 'bg-risk-high/15 text-risk-high border-risk-high/20';
                else if (isHigh) priorityBadge = 'bg-risk-medium/10 text-risk-medium border-risk-medium/20';

                let statusColor = 'text-text-secondary bg-border border-border';
                if (item.status === 'CONFIRMED_ABUSE') {
                  statusColor = 'text-risk-high bg-risk-high/10 border-risk-high/20';
                } else if (item.status === 'FALSE_POSITIVE') {
                  statusColor = 'text-risk-low bg-risk-low/10 border-risk-low/20';
                } else if (item.status === 'REVIEW_COMPLETED') {
                  statusColor = 'text-risk-low bg-risk-low/10 border-risk-low/20';
                }

                return (
                  <tr
                    key={item.id}
                    onClick={() => router.push(`/investigations/${item.id}`)}
                    className="border-b border-border hover:bg-card/50 transition-all cursor-pointer"
                  >
                    <td className="p-4 font-bold text-text-primary">{item.id}</td>
                    <td className="p-4 text-text-primary font-medium">
                      <div className="flex flex-col gap-0.5">
                        <span>{item.riskType}</span>
                        <span className="text-[10px] text-text-secondary font-mono">TXN ref: {item.caseId}</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-0.5 rounded border text-[9px] font-bold ${priorityBadge}`}>
                        {item.priority}
                      </span>
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-0.5 rounded border text-[10px] font-mono font-bold ${
                        score >= 0.90 ? 'text-risk-high border-risk-high/20 bg-risk-high/10' : 'text-risk-medium border-risk-medium/20 bg-risk-medium/10'
                      }`}>
                        {Math.round(score * 100)}%
                      </span>
                    </td>
                    <td className="p-4 font-mono font-bold text-text-primary">
                      ₹{item.amount.toLocaleString('en-US')}
                    </td>
                    <td className="p-4 text-text-secondary font-medium">{item.assignedTo}</td>
                    <td className="p-4">
                      <div className="flex items-center gap-1.5 text-text-secondary font-medium">
                        <Clock className="w-3.5 h-3.5 text-text-secondary" />
                        <span>{item.sla}</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-0.5 rounded border text-[9px] font-bold ${statusColor}`}>
                        {item.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td className="p-4 text-right">
                      <button className="text-risk-ai hover:text-text-primary font-bold flex items-center gap-1 ml-auto cursor-pointer transition-colors">
                        <span>Open Workspace</span>
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
