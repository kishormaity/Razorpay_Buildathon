'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  ArrowLeft,
  ChevronRight,
  Database,
  User,
  Smartphone,
  CreditCard,
  CheckCircle,
  XCircle,
  HelpCircle,
  Clock,
  Sparkles,
  ExternalLink,
  MessageSquare
} from 'lucide-react';
import { useTransactionQuery, useUpdateTransactionStatusMutation } from '@/hooks/useTransactions';
import RiskScoreGauge from '@/components/shared/RiskScoreGauge';
import CostComparisonBars from '@/components/shared/CostComparisonBars';

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function TransactionDetailPage({ params }: PageProps) {
  const { id } = React.use(params);
  const router = useRouter();
  const { data: txn, isLoading } = useTransactionQuery(id);
  const updateStatusMutation = useUpdateTransactionStatusMutation();

  const [feedbackOutcome, setFeedbackOutcome] = useState<
    'CONFIRMED_ABUSE' | 'FALSE_POSITIVE' | 'DISMISSED' | 'REVIEW_COMPLETED'
  >('CONFIRMED_ABUSE');
  const [feedbackNotes, setFeedbackNotes] = useState('');
  const [showStatusSuccess, setShowStatusSuccess] = useState(false);

  if (isLoading) {
    return (
      <div className="py-12 text-center text-xs text-text-secondary select-none font-medium">
        Loading transaction workspace...
      </div>
    );
  }

  if (!txn) {
    return (
      <div className="space-y-6 text-center select-none py-16">
        <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider">Transaction Not Found</h3>
        <p className="text-xs text-text-secondary">The requested transaction code does not exist in our threat database.</p>
        <Link
          href="/transactions"
          className="text-xs font-bold text-risk-ai hover:underline inline-flex items-center gap-1 mt-4"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Transaction Logs</span>
        </Link>
      </div>
    );
  }

  // Handle analyst decision buttons (Approve, Step-up, Hold, Block, etc.)
  const handleActionClick = (actionStatus: typeof feedbackOutcome) => {
    updateStatusMutation.mutate(
      { id: txn.id, status: actionStatus, notes: `Actioned directly from workspace toolbar: ${actionStatus}` },
      {
        onSuccess: () => {
          setShowStatusSuccess(true);
          setTimeout(() => setShowStatusSuccess(false), 3000);
        },
      }
    );
  };

  // Handle detailed analyst feedback form submission
  const handleFeedbackSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updateStatusMutation.mutate(
      { id: txn.id, status: feedbackOutcome, notes: feedbackNotes },
      {
        onSuccess: () => {
          setShowStatusSuccess(true);
          setFeedbackNotes('');
          setTimeout(() => setShowStatusSuccess(false), 3000);
        },
      }
    );
  };

  const score = txn.riskDecision.score;
  const isCritical = score >= 0.90;
  const isHigh = score >= 0.70 && score < 0.90;
  const isMedium = score >= 0.40 && score < 0.70;

  let headerColor = 'border-risk-low/20 bg-risk-low/5';
  let badgeColor = 'bg-risk-low/10 text-risk-low border-risk-low/20';
  let labelText = 'LOW RISK';

  if (isCritical) {
    headerColor = 'border-risk-high/30 bg-risk-high/5';
    badgeColor = 'bg-risk-high/15 text-risk-high border-risk-high/20';
    labelText = 'CRITICAL RISK';
  } else if (isHigh) {
    headerColor = 'border-risk-medium/30 bg-risk-medium/5';
    badgeColor = 'bg-risk-medium/10 text-risk-medium border-risk-medium/20';
    labelText = 'HIGH RISK';
  } else if (isMedium) {
    headerColor = 'border-risk-medium/15 bg-risk-medium/5';
    badgeColor = 'bg-risk-medium/10 text-risk-medium border-risk-medium/10';
    labelText = 'MEDIUM RISK';
  }

  // Status Badge configurations
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

  return (
    <div className="space-y-6 select-none pb-12">
      {/* Breadcrumb back navigation */}
      <div className="flex items-center justify-between">
        <Link
          href="/transactions"
          className="text-xs font-bold text-text-secondary hover:text-text-primary flex items-center gap-1.5 transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Transaction Logs</span>
        </Link>
        <span className="text-[10px] font-mono text-text-secondary bg-card border border-border px-2 py-0.5 rounded">
          Model Version: {txn.riskDecision.policyVersion}
        </span>
      </div>

      {/* Flagged Status Banner */}
      <div className={`border rounded-xl p-5 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 ${headerColor}`}>
        <div className="flex items-center gap-4">
          <div className="bg-card p-3 rounded-lg border border-border">
            <Database className="w-6 h-6 text-risk-info" />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-bold text-text-primary">{txn.id}</h2>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${badgeColor}`}>
                {labelText} ({Math.round(score * 100)}%)
              </span>
              <span className={`text-[9px] font-bold px-2 py-0.5 rounded border ${statusColor}`}>
                {statusText}
              </span>
            </div>
            <p className="text-xs text-text-secondary mt-1">
              Placed on <span className="font-semibold text-text-primary">{new Date(txn.timestamp).toLocaleString('en-US')}</span>
              {' '}via <span className="font-semibold text-text-primary">{txn.channel}</span>
            </p>
          </div>
        </div>

        {/* Rapid action toolbar buttons */}
        <div className="flex items-center gap-2 self-stretch md:self-auto">
          <button
            onClick={() => handleActionClick('REVIEW_COMPLETED')}
            disabled={updateStatusMutation.isPending}
            className="flex-1 md:flex-initial text-[11px] font-bold bg-risk-low/20 hover:bg-risk-low text-risk-low hover:text-white border border-risk-low/20 px-3.5 py-2 rounded-lg cursor-pointer transition-all disabled:opacity-50"
          >
            Approve
          </button>
          <button
            onClick={() => handleActionClick('DISMISSED')}
            disabled={updateStatusMutation.isPending}
            className="flex-1 md:flex-initial text-[11px] font-bold bg-border hover:bg-card text-text-primary border border-border px-3.5 py-2 rounded-lg cursor-pointer transition-all disabled:opacity-50"
          >
            Step-up
          </button>
          <button
            onClick={() => handleActionClick('CONFIRMED_ABUSE')}
            disabled={updateStatusMutation.isPending}
            className="flex-1 md:flex-initial text-[11px] font-bold bg-risk-high hover:bg-risk-high/90 text-white px-3.5 py-2 rounded-lg cursor-pointer transition-all disabled:opacity-50"
          >
            Block & Hold
          </button>
        </div>
      </div>

      {showStatusSuccess && (
        <div className="bg-risk-low/10 border border-risk-low/20 text-risk-low text-xs font-bold rounded-lg p-3 text-center animate-bounce">
          Database status updated successfully! Changes propagated back to risk queues.
        </div>
      )}

      {/* Narrative Card */}
      <div className="bg-card border border-border rounded-xl p-5 relative overflow-hidden">
        {/* Glow tint */}
        <div className="absolute top-0 right-0 w-24 h-24 bg-risk-ai/5 filter blur-xl rounded-full"></div>
        <div className="flex items-center gap-2 mb-2">
          <Sparkles className="w-4 h-4 text-risk-ai" />
          <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">AI Risk Explanation Narrative</h3>
        </div>
        <p className="text-xs text-text-secondary leading-relaxed font-medium">
          {txn.riskNarrative}
        </p>
      </div>

      {/* Main workspace layout grids */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left / Middle: Technical Details Panels */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Telemetry Details 4x4 Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* Panel 1: Transaction Info */}
            <div className="bg-card border border-border rounded-xl p-4.5 space-y-3">
              <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider pb-2 border-b border-border flex items-center gap-2">
                <Database className="w-4 h-4 text-risk-info" />
                <span>Transaction Metadata</span>
              </h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between"><span className="text-text-secondary">Amount</span><span className="text-text-primary font-bold">₹{txn.amount.toLocaleString('en-US')} {txn.currency}</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Merchant</span><span className="text-text-primary font-medium">{txn.merchant}</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Payment Method</span><span className="text-text-primary font-medium">{txn.paymentMethod}</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Location</span><span className="text-text-primary font-medium">{txn.location}</span></div>
              </div>
            </div>

            {/* Panel 2: Customer History */}
            <div className="bg-card border border-border rounded-xl p-4.5 space-y-3">
              <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider pb-2 border-b border-border flex items-center gap-2">
                <User className="w-4 h-4 text-risk-info" />
                <span>Customer Profile</span>
              </h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between"><span className="text-text-secondary">Customer ID</span><span className="text-text-primary font-bold font-mono">{txn.customerId}</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Account Age</span><span className="text-text-primary font-medium">12 Days</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Past Txns (30d)</span><span className="text-text-primary font-medium">4 approved</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Disputes Logged</span><span className="text-risk-high font-bold">3 refunds</span></div>
              </div>
            </div>

            {/* Panel 3: Device Telemetry */}
            <div className="bg-card border border-border rounded-xl p-4.5 space-y-3">
              <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider pb-2 border-b border-border flex items-center gap-2">
                <Smartphone className="w-4 h-4 text-risk-ai" />
                <span>Device Fingerprint</span>
              </h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between"><span className="text-text-secondary">Device Key</span><span className="text-text-primary font-bold font-mono">{txn.deviceId}</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Linked Accounts</span><span className="text-risk-high font-bold">8 distinct</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Linked IPs</span><span className="text-text-primary font-medium">5 addresses</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Emulator Score</span><span className="text-risk-medium font-bold">0.82 (High)</span></div>
              </div>
            </div>

            {/* Panel 4: Payment Instrument Network */}
            <div className="bg-card border border-border rounded-xl p-4.5 space-y-3">
              <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider pb-2 border-b border-border flex items-center gap-2">
                <CreditCard className="w-4 h-4 text-risk-info" />
                <span>Payment Binding</span>
              </h3>
              <div className="space-y-2 text-xs">
                <div className="flex justify-between"><span className="text-text-secondary">Issuer Bank</span><span className="text-text-primary font-medium">State Bank of India</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Country Origin</span><span className="text-text-primary font-medium">IN (India)</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Recent Overlap</span><span className="text-risk-high font-bold">3 accounts share card</span></div>
                <div className="flex justify-between"><span className="text-text-secondary">Card Binding Age</span><span className="text-text-primary font-medium">1.5 hours</span></div>
              </div>
            </div>

          </div>

          {/* Separated Explanation Panel */}
          <div className="bg-card border border-border rounded-xl p-5 space-y-4">
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider border-b border-border pb-3">
              Structured Risk Breakdown
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Risk Signals */}
              <div className="space-y-3">
                <span className="text-[10px] font-bold text-text-secondary uppercase tracking-widest block">Analyst-Readable Signals</span>
                <div className="space-y-2">
                  {txn.signals.length === 0 ? (
                    <span className="text-xs text-text-secondary">No risk signals flag triggered.</span>
                  ) : (
                    txn.signals.map((sig) => (
                      <div key={sig.id} className="p-3 bg-background border border-border rounded-lg flex items-start gap-2.5">
                        <span className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${
                          sig.severity === 'CRITICAL' ? 'bg-risk-high' :
                          sig.severity === 'HIGH' ? 'bg-risk-high' :
                          sig.severity === 'MEDIUM' ? 'bg-risk-medium' :
                          'bg-risk-low'
                        }`} />
                        <div>
                          <h4 className="text-xs font-bold text-text-primary">{sig.name}</h4>
                          <p className="text-[10px] text-text-secondary mt-0.5 leading-normal">{sig.description}</p>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Model Contributions */}
              <div className="space-y-3">
                <span className="text-[10px] font-bold text-text-secondary uppercase tracking-widest block">Model Weights Feature Vector</span>
                <div className="space-y-2 bg-background border border-border rounded-lg p-3">
                  {txn.contributions.map((item, idx) => (
                    <div key={idx} className="flex flex-col gap-1.5 py-1.5 first:pt-0 last:pb-0 border-b border-border last:border-b-0">
                      <div className="flex justify-between text-[11px] font-medium text-text-secondary">
                        <span>{item.featureName}</span>
                        <span className="text-risk-ai font-bold font-mono">+{item.weight.toFixed(2)}</span>
                      </div>
                      {/* Mini bar chart */}
                      <div className="h-1.5 w-full bg-border rounded-full overflow-hidden">
                        <div className="bg-risk-ai h-full rounded-full" style={{ width: `${item.weight * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

        </div>

        {/* Right Column: Gauges, Cost, Evidence, Feedback */}
        <div className="space-y-6">
          
          {/* Gauges & Cost Cards */}
          <div className="bg-card border border-border rounded-xl p-5 flex flex-col items-center justify-center">
            <RiskScoreGauge score={score} confidence={txn.riskDecision.confidence} />
          </div>

          <CostComparisonBars
            amount={txn.amount}
            expectedCost={txn.riskDecision.expectedCost}
            recommendedAction={txn.riskDecision.action}
          />

          {/* Evidence checklist */}
          <div className="bg-card border border-border rounded-xl p-5 space-y-3">
            <div className="flex justify-between items-center pb-2 border-b border-border">
              <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">Telemetry Evidence</h3>
              <span className="text-[10px] font-mono text-risk-low font-bold">
                {txn.evidenceCompleteness}% Checked
              </span>
            </div>
            <div className="space-y-2 text-xs">
              {txn.evidenceList.map((item, idx) => (
                <div key={idx} className="flex items-center justify-between py-0.5">
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

          {/* Analyst feedback component */}
          <div className="bg-card border border-border rounded-xl p-5 space-y-4">
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider border-b border-border pb-3 flex items-center gap-1.5">
              <MessageSquare className="w-4 h-4 text-risk-ai" />
              <span>Analyst Review Outcome</span>
            </h3>
            
            <form onSubmit={handleFeedbackSubmit} className="space-y-4">
              <div className="space-y-2 text-xs">
                <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Decision Label</span>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { label: 'Confirmed Abuse', value: 'CONFIRMED_ABUSE' },
                    { label: 'False Positive', value: 'FALSE_POSITIVE' },
                    { label: 'Dismiss Case', value: 'DISMISSED' },
                    { label: 'Verify Complete', value: 'REVIEW_COMPLETED' }
                  ].map((option) => (
                    <label
                      key={option.value}
                      className={`flex items-center gap-2 p-2 rounded-lg border cursor-pointer select-none transition-all ${
                        feedbackOutcome === option.value
                          ? 'border-risk-ai bg-risk-ai/5 font-semibold text-text-primary'
                          : 'border-border bg-background hover:bg-card text-text-secondary'
                      }`}
                    >
                      <input
                        type="radio"
                        name="outcome"
                        value={option.value}
                        checked={feedbackOutcome === option.value}
                        onChange={() => setFeedbackOutcome(option.value as any)}
                        className="sr-only"
                      />
                      <span>{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Notes Textarea */}
              <div className="space-y-1.5 text-xs">
                <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Investigation Notes</span>
                <textarea
                  rows={3}
                  value={feedbackNotes}
                  onChange={(e) => setFeedbackNotes(e.target.value)}
                  placeholder="Enter detailed reasons supporting your manual audit label..."
                  className="w-full bg-background border border-border rounded-lg p-2.5 text-xs text-text-primary outline-none focus:border-text-secondary/40 transition-colors resize-none placeholder-text-secondary"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={updateStatusMutation.isPending}
                className="w-full bg-risk-ai hover:bg-risk-ai/90 text-white font-bold text-xs py-2.5 rounded-lg cursor-pointer transition-colors shadow-lg shadow-risk-ai/10"
              >
                {updateStatusMutation.isPending ? 'Saving Label...' : 'Submit Risk Resolution'}
              </button>
            </form>
          </div>

        </div>

      </div>
    </div>
  );
}
