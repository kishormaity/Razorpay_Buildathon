'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  ArrowLeft,
  Users,
  Clock,
  ShieldCheck,
  CheckCircle,
  XCircle,
  Database,
  FileText,
  AlertTriangle,
  MessageSquare,
  Zap,
  ChevronRight
} from 'lucide-react';
import { useInvestigationQuery, useUpdateInvestigationStatusMutation } from '@/hooks/useInvestigation';
import { useTransactionQuery } from '@/hooks/useTransactions';
import RiskScoreGauge from '@/components/shared/RiskScoreGauge';

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function InvestigationDetailPage({ params }: PageProps) {
  const { id } = React.use(params);
  const router = useRouter();
  const { data: item, isLoading: itemLoading } = useInvestigationQuery(id);
  const { data: txn, isLoading: txnLoading } = useTransactionQuery(item?.caseId || '');
  
  const updateStatusMutation = useUpdateInvestigationStatusMutation();

  const [outcome, setOutcome] = useState<
    'CONFIRMED_ABUSE' | 'FALSE_POSITIVE' | 'DISMISSED' | 'REVIEW_COMPLETED'
  >('CONFIRMED_ABUSE');
  const [notes, setNotes] = useState('');
  const [showSuccess, setShowSuccess] = useState(false);

  if (itemLoading || txnLoading) {
    return (
      <div className="py-12 text-center text-xs text-text-secondary select-none font-medium">
        Loading case workspace...
      </div>
    );
  }

  if (!item) {
    return (
      <div className="space-y-6 text-center select-none py-16">
        <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider">Case File Not Found</h3>
        <p className="text-xs text-text-secondary">The requested investigation ticket does not exist in our active queue.</p>
        <Link
          href="/investigations"
          className="text-xs font-bold text-risk-ai hover:underline inline-flex items-center gap-1 mt-4"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Investigation Queue</span>
        </Link>
      </div>
    );
  }

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updateStatusMutation.mutate(
      { id: item.id, status: outcome, notes },
      {
        onSuccess: () => {
          setShowSuccess(true);
          setNotes('');
          setTimeout(() => setShowSuccess(false), 3000);
        },
      }
    );
  };

  const isCritical = item.priority === 'CRITICAL';
  const isHigh = item.priority === 'HIGH';

  let borderClass = 'border-risk-low/20 bg-risk-low/5';
  let badgeColor = 'bg-risk-low/10 text-risk-low border-risk-low/20';

  if (isCritical) {
    borderClass = 'border-risk-high/30 bg-risk-high/5';
    badgeColor = 'bg-risk-high/15 text-risk-high border-risk-high/20';
  } else if (isHigh) {
    borderClass = 'border-risk-medium/30 bg-risk-medium/5';
    badgeColor = 'bg-risk-medium/10 text-risk-medium border-risk-medium/20';
  }

  // Resolved statuses
  const isResolved = item.status !== 'PENDING_REVIEW';
  let resolutionColor = 'text-text-secondary border-border bg-border/20';
  if (item.status === 'CONFIRMED_ABUSE') {
    resolutionColor = 'text-risk-high border-risk-high/20 bg-risk-high/10';
  } else if (item.status === 'FALSE_POSITIVE') {
    resolutionColor = 'text-risk-low border-risk-low/20 bg-risk-low/10';
  } else if (item.status === 'REVIEW_COMPLETED') {
    resolutionColor = 'text-risk-low border-risk-low/20 bg-risk-low/10';
  }

  return (
    <div className="space-y-6 select-none pb-12">
      {/* Back button */}
      <div className="flex justify-between items-center">
        <Link
          href="/investigations"
          className="text-xs font-bold text-text-secondary hover:text-text-primary flex items-center gap-1.5 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Investigation Queue</span>
        </Link>
      </div>

      {/* Case Header Card */}
      <div className={`border rounded-xl p-5 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 ${borderClass}`}>
        <div className="flex items-center gap-4">
          <div className="bg-card p-3 rounded-lg border border-border text-risk-ai">
            <Users className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-bold text-text-primary">{item.id}</h2>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${badgeColor}`}>
                {item.priority} PRIORITY
              </span>
              <span className="text-[10px] font-mono text-text-secondary bg-card border border-border px-2 py-0.5 rounded">
                Score: {Math.round(item.riskScore * 100)}%
              </span>
            </div>
            <p className="text-xs text-text-secondary mt-1">
              Case Category: <span className="font-semibold text-text-primary">{item.riskType}</span>
              {' '}• Assigned to: <span className="font-semibold text-text-primary">{item.assignedTo}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          <Clock className="w-4 h-4 text-text-secondary" />
          <span className="text-xs font-semibold text-text-primary">{item.sla}</span>
        </div>
      </div>

      {showSuccess && (
        <div className="bg-risk-low/10 border border-risk-low/20 text-risk-low text-xs font-bold rounded-lg p-3 text-center animate-bounce">
          Investigation verdict saved and propagated across active queues.
        </div>
      )}

      {/* Main Grid Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Transaction context & Evidence */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Linked Transaction Card */}
          {txn ? (
            <div className="bg-card border border-border rounded-xl p-5 space-y-4">
              <div className="flex justify-between items-center pb-2 border-b border-border">
                <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                  <Database className="w-4 h-4 text-risk-info" />
                  <span>Linked Transaction Evidence ({txn.id})</span>
                </h3>
                <Link
                  href={`/transactions/${txn.id}`}
                  className="text-xs font-bold text-risk-ai hover:underline flex items-center gap-1"
                >
                  <span>Open transaction workspace</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </Link>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                <div>
                  <span className="text-text-secondary block">Amount</span>
                  <span className="text-xs font-bold text-text-primary block mt-1">₹{txn.amount.toLocaleString('en-US')}</span>
                </div>
                <div>
                  <span className="text-text-secondary block">Merchant</span>
                  <span className="text-xs font-medium text-text-primary block mt-1">{txn.merchant}</span>
                </div>
                <div>
                  <span className="text-text-secondary block">Customer ID</span>
                  <span className="text-xs font-mono font-bold text-text-primary block mt-1">{txn.customerId}</span>
                </div>
                <div>
                  <span className="text-text-secondary block">Device ID</span>
                  <span className="text-xs font-mono font-bold text-text-primary block mt-1">{txn.deviceId}</span>
                </div>
              </div>

              <div className="bg-background border border-border p-3.5 rounded-lg">
                <span className="text-[10px] font-bold text-text-secondary block uppercase tracking-wider mb-1.5">Narrative context</span>
                <p className="text-xs text-text-secondary leading-relaxed">{txn.riskNarrative}</p>
              </div>
            </div>
          ) : (
            <div className="bg-card border border-border border-dashed rounded-xl p-5 text-center text-xs text-text-secondary">
              No underlying transaction context found.
            </div>
          )}

          {/* Evidence collected checklist */}
          {txn && (
            <div className="bg-card border border-border rounded-xl p-5 space-y-3.5">
              <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider pb-2 border-b border-border">
                Manual Audit Telemetry Checklist
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-3 text-xs">
                {txn.evidenceList.map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between py-1 bg-background/50 border border-border px-3 rounded-lg">
                    <span className="text-text-secondary">{item.name}</span>
                    {item.checked ? (
                      <CheckCircle className="w-4 h-4 text-risk-low shrink-0" />
                    ) : (
                      <XCircle className="w-4 h-4 text-text-secondary shrink-0" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>

        {/* Right Column: Risk Dial & Verdict Feedback Form */}
        <div className="space-y-6">
          
          {/* Radial score gauge */}
          <div className="bg-card border border-border rounded-xl p-5 flex flex-col items-center justify-center">
            <RiskScoreGauge score={item.riskScore} confidence={txn ? txn.riskDecision.confidence : 0.88} />
          </div>

          {/* Decision Verdict card */}
          <div className="bg-card border border-border rounded-xl p-5 space-y-4">
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider border-b border-border pb-3 flex items-center gap-1.5">
              <MessageSquare className="w-4 h-4 text-risk-ai" />
              <span>Analyst Verdict Audit</span>
            </h3>

            {isResolved ? (
              <div className="space-y-4">
                <div className={`p-4 border rounded-xl text-center flex flex-col gap-2 ${resolutionColor}`}>
                  <span className="text-[10px] font-bold tracking-widest uppercase">CASE RESOLVED</span>
                  <span className="text-sm font-extrabold">{item.status.replace('_', ' ')}</span>
                </div>

                {item.notes && (
                  <div className="bg-background border border-border rounded-lg p-3 text-xs">
                    <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase mb-1">Analyst Notes:</span>
                    <p className="text-text-secondary italic">{item.notes}</p>
                  </div>
                )}

                <button
                  onClick={() => {
                    // Re-open case simply updates back to PENDING_REVIEW
                    updateStatusMutation.mutate({ id: item.id, status: 'PENDING_REVIEW', notes: '' });
                  }}
                  className="w-full bg-border hover:bg-card border border-border text-text-primary font-bold text-xs py-2 rounded-lg cursor-pointer transition-colors"
                >
                  Re-open Audit Case
                </button>
              </div>
            ) : (
              <form onSubmit={handleFormSubmit} className="space-y-4">
                <div className="space-y-2 text-xs">
                  <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Select Case Verdict</span>
                  <div className="grid grid-cols-1 gap-2">
                    {[
                      { label: 'Confirm Abuse / Fraud Ring', value: 'CONFIRMED_ABUSE' },
                      { label: 'False Positive Alert', value: 'FALSE_POSITIVE' },
                      { label: 'Step-up Verification Completed', value: 'REVIEW_COMPLETED' }
                    ].map((opt) => (
                      <label
                        key={opt.value}
                        className={`flex items-center gap-2.5 p-2.5 rounded-lg border cursor-pointer select-none transition-all ${
                          outcome === opt.value
                            ? 'border-risk-ai bg-risk-ai/5 font-semibold text-text-primary'
                            : 'border-border bg-background hover:bg-card text-text-secondary'
                        }`}
                      >
                        <input
                          type="radio"
                          name="outcome"
                          value={opt.value}
                          checked={outcome === opt.value}
                          onChange={() => setOutcome(opt.value as any)}
                          className="sr-only"
                        />
                        <span>{opt.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* notes */}
                <div className="space-y-1.5 text-xs">
                  <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Analyst Explanatory Notes</span>
                  <textarea
                    rows={3}
                    value={notes}
                    onChange={(e) => setNotes(e.target.value)}
                    placeholder="Provide details backing your decision verdict..."
                    className="w-full bg-background border border-border rounded-lg p-2.5 text-xs text-text-primary outline-none focus:border-text-secondary/40 transition-colors resize-none placeholder-text-secondary"
                    required
                  />
                </div>

                <button
                  type="submit"
                  disabled={updateStatusMutation.isPending}
                  className="w-full bg-risk-ai hover:bg-risk-ai/90 text-white font-bold text-xs py-2.5 rounded-lg cursor-pointer transition-colors shadow-lg shadow-risk-ai/10"
                >
                  {updateStatusMutation.isPending ? 'Submitting Verdict...' : 'Submit Resolution Verdict'}
                </button>
              </form>
            )}
          </div>

        </div>

      </div>
    </div>
  );
}
