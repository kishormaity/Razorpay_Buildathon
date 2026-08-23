'use client';

import React, { useState, useMemo } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  TrendingUp,
  TrendingDown,
  AlertOctagon,
  Layers,
  Percent,
  Cpu,
  ChevronRight,
  Filter,
  Search,
  ExternalLink,
  ShieldCheck
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  Legend
} from 'recharts';
import { useTransactionsQuery } from '../../hooks/useTransactions';
import { useRiskRingsQuery } from '../../hooks/useRiskRing';

// Mock trend chart data
const mockTrendData = {
  '24H': [
    { time: '00:00', events: 12 },
    { time: '04:00', events: 8 },
    { time: '08:00', events: 15 },
    { time: '12:00', events: 35 }, // Fraud spike period
    { time: '16:00', events: 42 },
    { time: '20:00', events: 28 },
    { time: '24:00', events: 18 },
  ],
  '7D': [
    { time: 'Mon', events: 120 },
    { time: 'Tue', events: 135 },
    { time: 'Wed', events: 210 }, // Spike day
    { time: 'Thu', events: 140 },
    { time: 'Fri', events: 115 },
    { time: 'Sat', events: 95 },
    { time: 'Sun', events: 105 },
  ],
  '30D': [
    { time: 'W1', events: 450 },
    { time: 'W2', events: 680 },
    { time: 'W3', events: 920 }, // Elevated fraud cluster
    { time: 'W4', events: 510 },
  ],
};

const COLORS = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#f59e0b',
  low: '#10b981',
};

const donutData = [
  { name: 'Critical Risk (0.90+)', value: 18, color: COLORS.critical },
  { name: 'High Risk (0.70-0.90)', value: 45, color: COLORS.high },
  { name: 'Medium Risk (0.40-0.70)', value: 120, color: COLORS.medium },
  { name: 'Low Risk (0.00-0.40)', value: 1101, color: COLORS.low },
];

const lossComparisonData = [
  { name: 'Fraud Loss', value: 3.2, color: COLORS.critical, label: 'Actual Fraud Loss' },
  { name: 'Loss Prevented', value: 18.2, color: COLORS.low, label: 'Estimated Prevented' },
  { name: 'Potential Exposure', value: 24.8, color: '#3b82f6', label: 'Potential Exposure' },
];

const decisionDistributionData = [
  { name: 'Allow', count: 910 },
  { name: 'Step-up (OTP)', count: 182 },
  { name: 'Manual Review', count: 45 },
  { name: 'Hold', count: 18 },
  { name: 'Block', count: 29 },
];

export default function DashboardPage() {
  const router = useRouter();
  const { data: transactions = [], isLoading: txLoading } = useTransactionsQuery();
  const { data: rings = [], isLoading: ringsLoading } = useRiskRingsQuery();

  const [timeframe, setTimeframe] = useState<'24H' | '7D' | '30D'>('24H');
  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState<'ALL' | 'CRITICAL' | 'HIGH' | 'MEDIUM'>('ALL');

  // Filtered transactions for the events list
  const filteredEvents = useMemo(() => {
    return transactions.filter((txn) => {
      // Risk filter
      const score = txn.riskDecision.score;
      if (riskFilter === 'CRITICAL' && score < 0.90) return false;
      if (riskFilter === 'HIGH' && (score < 0.70 || score >= 0.90)) return false;
      if (riskFilter === 'MEDIUM' && (score < 0.40 || score >= 0.70)) return false;

      // Keyword filter
      if (searchTerm) {
        const term = searchTerm.toLowerCase();
        return (
          txn.id.toLowerCase().includes(term) ||
          txn.customerId.toLowerCase().includes(term) ||
          txn.merchant.toLowerCase().includes(term) ||
          txn.riskDecision.action.toLowerCase().includes(term)
        );
      }

      return true;
    });
  }, [transactions, searchTerm, riskFilter]);

  const activeSpike = true; // Simulating active elevation warning banner

  return (
    <div className="space-y-6">
      {/* Upper Alerts Banner */}
      {activeSpike && (
        <div className="bg-risk-medium/10 border border-risk-medium/20 rounded-xl p-4 flex items-center justify-between animate-pulse">
          <div className="flex items-center gap-3">
            <div className="bg-risk-medium/20 p-2 rounded-lg text-risk-medium border border-risk-medium/30">
              <AlertOctagon className="w-5 h-5" />
            </div>
            <div>
              <h4 className="text-xs font-bold text-text-primary uppercase tracking-wider">Active Threat Spike Warning</h4>
              <p className="text-xs text-text-secondary mt-0.5">
                Coordinated account refund velocities exceed baselines by <span className="text-risk-medium font-bold">2.4x</span> on retail merchants.
              </p>
            </div>
          </div>
          <Link
            href="/rings/AR-1042"
            className="text-xs font-bold text-risk-medium hover:text-text-primary flex items-center gap-1.5 transition-colors"
          >
            <span>Inspect Coordinated Ring AR-1042</span>
            <ChevronRight className="w-4 h-4" />
          </Link>
        </div>
      )}

      {/* Page Title & Subtitle */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-extrabold tracking-tight text-text-primary uppercase">
            AI Risk Command Center
          </h2>
          <p className="text-xs text-text-secondary mt-1">
            Real-time overview of transaction risk, coordinated abuse networks and automated policy savings.
          </p>
        </div>

        {/* Action controls */}
        <div className="flex items-center gap-3 select-none">
          <div className="bg-card border border-border rounded-lg p-0.5 flex">
            {(['24H', '7D', '30D'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTimeframe(t)}
                className={`px-3 py-1 text-[10px] font-bold uppercase rounded-md transition-all cursor-pointer ${
                  timeframe === t ? 'bg-sidebar text-text-primary shadow-sm' : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="text-[10px] font-mono text-text-secondary bg-card border border-border px-3 py-1.5 rounded-lg flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-risk-low animate-pulse"></span>
            <span>Live Monitor</span>
          </div>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
        {/* Card 1: Fraud Exposure */}
        <div className="bg-card border border-border rounded-xl p-4.5">
          <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Fraud Exposure</span>
          <h3 className="text-lg font-extrabold text-text-primary mt-1.5">₹24.8L</h3>
          <div className="flex items-center gap-1 mt-2.5 text-[10px] text-risk-low font-semibold">
            <TrendingDown className="w-3.5 h-3.5" />
            <span>12.4% vs prev week</span>
          </div>
        </div>

        {/* Card 2: Prevented Loss */}
        <div className="bg-card border border-border rounded-xl p-4.5">
          <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Prevented Loss</span>
          <h3 className="text-lg font-extrabold text-risk-low mt-1.5">₹18.2L</h3>
          <div className="flex items-center gap-1 mt-2.5 text-[10px] text-risk-low font-semibold">
            <TrendingUp className="w-3.5 h-3.5" />
            <span>8.7% caught rate</span>
          </div>
        </div>

        {/* Card 3: Risk Events */}
        <div className="bg-card border border-border rounded-xl p-4.5">
          <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">High-Risk Events</span>
          <h3 className="text-lg font-extrabold text-text-primary mt-1.5">1,284</h3>
          <div className="flex items-center gap-1 mt-2.5 text-[10px] text-risk-high font-semibold">
            <TrendingUp className="w-3.5 h-3.5" />
            <span>13.2% velocity change</span>
          </div>
        </div>

        {/* Card 4: Active Rings */}
        <div className="bg-card border border-border rounded-xl p-4.5">
          <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Abuse Rings</span>
          <h3 className="text-lg font-extrabold text-risk-high mt-1.5">37</h3>
          <div className="flex items-center gap-1.5 mt-2.5 text-[10px] text-text-secondary font-mono leading-none">
            <span className="h-1.5 w-1.5 rounded-full bg-risk-high animate-ping"></span>
            <span>9 Critical threat tags</span>
          </div>
        </div>

        {/* Card 5: False Positive Rate */}
        <div className="bg-card border border-border rounded-xl p-4.5">
          <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">False Positive Rate</span>
          <h3 className="text-lg font-extrabold text-text-primary mt-1.5">4.8%</h3>
          <div className="flex items-center gap-1 mt-2.5 text-[10px] text-risk-low font-semibold">
            <TrendingDown className="w-3.5 h-3.5" />
            <span>0.7% decrease</span>
          </div>
        </div>

        {/* Card 6: Decision Automation */}
        <div className="bg-card border border-border rounded-xl p-4.5">
          <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Auto-Decision Rate</span>
          <h3 className="text-lg font-extrabold text-risk-ai mt-1.5">82.4%</h3>
          <div className="flex items-center gap-1 mt-2.5 text-[10px] text-text-secondary font-mono">
            <Cpu className="w-3.5 h-3.5 text-risk-ai" />
            <span>Model v2.4 Active</span>
          </div>
        </div>
      </div>

      {/* Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Chart 1: Fraud Risk Trend (Area chart) */}
        <div className="bg-card border border-border rounded-xl p-5 lg:col-span-2 flex flex-col h-80">
          <span className="text-xs font-bold text-text-primary uppercase tracking-wider mb-4">
            Fraud Risk Events Volume ({timeframe})
          </span>
          <div className="flex-1 w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockTrendData[timeframe]}>
                <defs>
                  <linearGradient id="colorEvents" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" stroke="#9ca3af" fontSize={10} tickLine={false} />
                <YAxis stroke="#9ca3af" fontSize={10} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#151922', borderColor: '#252a34', borderRadius: '8px' }}
                  labelStyle={{ color: '#9ca3af', fontSize: '10px' }}
                  itemStyle={{ color: '#f5f7fa', fontSize: '11px' }}
                />
                <Area type="monotone" dataKey="events" stroke="#ef4444" strokeWidth={2} fillOpacity={1} fill="url(#colorEvents)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Risk Score Distribution (Donut chart) */}
        <div className="bg-card border border-border rounded-xl p-5 flex flex-col h-80">
          <span className="text-xs font-bold text-text-primary uppercase tracking-wider mb-4">
            Model Risk Distribution
          </span>
          <div className="flex-1 w-full min-h-0 flex items-center justify-center relative">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={donutData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {donutData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#151922', borderColor: '#252a34', borderRadius: '8px' }}
                  itemStyle={{ fontSize: '11px' }}
                />
              </PieChart>
            </ResponsiveContainer>

            {/* Centered label inside donut */}
            <div className="absolute flex flex-col items-center justify-center leading-none text-center pointer-events-none">
              <span className="text-xl font-black text-text-primary">1,284</span>
              <span className="text-[9px] text-text-secondary font-bold uppercase tracking-wider mt-1">Events Logged</span>
            </div>
          </div>
          {/* Custom legend */}
          <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[9px] font-mono text-text-secondary mt-2">
            {donutData.map((item) => (
              <div key={item.name} className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: item.color }}></span>
                <span className="truncate">{item.name}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Chart 3: Loss Prevention Performance (Bar chart) */}
        <div className="bg-card border border-border rounded-xl p-5 flex flex-col h-80">
          <span className="text-xs font-bold text-text-primary uppercase tracking-wider mb-4">
            Fraud Impact Assessment (₹ Lakhs)
          </span>
          <div className="flex-1 w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={lossComparisonData} layout="vertical">
                <XAxis type="number" stroke="#9ca3af" fontSize={9} tickLine={false} />
                <YAxis dataKey="name" type="category" stroke="#9ca3af" fontSize={10} tickLine={false} width={80} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#151922', borderColor: '#252a34', borderRadius: '8px' }}
                  itemStyle={{ fontSize: '11px', color: '#f5f7fa' }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {lossComparisonData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Middle Grid: Ring Previews & Policy Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Mini Abuse Ring Hero Preview Card */}
        <div className="bg-card border border-border rounded-xl p-5 lg:col-span-1 flex flex-col justify-between">
          <div>
            <div className="flex justify-between items-start">
              <span className="text-[10px] font-bold text-text-secondary tracking-widest uppercase">Hero Feature</span>
              <span className="text-[9px] font-mono text-risk-high border border-risk-high/20 bg-risk-high/10 px-2 py-0.5 rounded uppercase font-bold animate-pulse">
                CRITICAL THREAT
              </span>
            </div>
            <h4 className="text-sm font-bold text-text-primary mt-3">Coordinated Abuse Ring (AR-1042)</h4>
            <p className="text-xs text-text-secondary mt-1.5 leading-relaxed">
              17 accounts share device tokens, dynamic Jio IP networks, and payment codes. ₹3.8L exposure observed.
            </p>

            <div className="mt-4 grid grid-cols-2 gap-2 text-[10px] font-mono bg-background/50 border border-border rounded-lg p-2.5">
              <div>
                <span className="text-text-secondary">Exposure</span>
                <span className="text-text-primary font-bold block mt-0.5">₹3,80,000</span>
              </div>
              <div>
                <span className="text-text-secondary">Ring Members</span>
                <span className="text-text-primary font-bold block mt-0.5">17 Accounts</span>
              </div>
              <div className="mt-1">
                <span className="text-text-secondary">Risk Score</span>
                <span className="text-risk-high font-bold block mt-0.5">96%</span>
              </div>
              <div className="mt-1">
                <span className="text-text-secondary">Shared IPs</span>
                <span className="text-text-primary font-bold block mt-0.5">5 Addresses</span>
              </div>
            </div>
          </div>

          <button
            onClick={() => router.push('/rings/AR-1042')}
            className="w-full mt-4 bg-risk-ai hover:bg-risk-ai/90 border border-risk-ai/30 text-white font-bold text-xs py-2 rounded-lg flex items-center justify-center gap-2 cursor-pointer transition-colors shadow-lg shadow-risk-ai/10"
          >
            <span>Investigate Graph Network</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Decision Automation count chart */}
        <div className="bg-card border border-border rounded-xl p-5 lg:col-span-3 flex flex-col h-full">
          <span className="text-xs font-bold text-text-primary uppercase tracking-wider mb-4">
            Operational Decision Volume
          </span>
          <div className="flex-1 w-full min-h-[160px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={decisionDistributionData}>
                <XAxis dataKey="name" stroke="#9ca3af" fontSize={10} tickLine={false} />
                <YAxis stroke="#9ca3af" fontSize={10} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#151922', borderColor: '#252a34', borderRadius: '8px' }}
                  itemStyle={{ fontSize: '11px', color: '#f5f7fa' }}
                />
                <Bar dataKey="count" fill="#a855f7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Bottom Section: Top Risk Events Table */}
      <div className="bg-card border border-border rounded-xl p-5 select-none">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-5">
          <div>
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">Top Risk Events Queue</h3>
            <span className="text-[10px] text-text-secondary font-mono mt-0.5 block">
              Showing unresolved high-risk transactions requiring analyst review
            </span>
          </div>

          {/* Filtering and search controls */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Search inputs */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-text-secondary absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Filter events..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="bg-background border border-border text-xs text-text-primary placeholder-text-secondary pl-8 pr-3 py-1.5 rounded-lg outline-none w-48 focus:border-text-secondary/40 transition-colors"
              />
            </div>

            {/* Risk filter selector */}
            <div className="flex bg-background border border-border rounded-lg p-0.5 text-[9px] font-mono font-bold">
              {(['ALL', 'CRITICAL', 'HIGH', 'MEDIUM'] as const).map((level) => (
                <button
                  key={level}
                  onClick={() => setRiskFilter(level)}
                  className={`px-2 py-1.5 rounded cursor-pointer ${
                    riskFilter === level ? 'bg-card text-text-primary border border-border' : 'text-text-secondary hover:text-text-primary'
                  }`}
                >
                  {level}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Transactions Table */}
        <div className="overflow-x-auto border border-border rounded-lg">
          <table className="w-full text-left border-collapse text-xs select-none">
            <thead>
              <tr className="bg-sidebar/50 border-b border-border text-text-secondary font-mono font-bold">
                <th className="p-3.5">Risk ID</th>
                <th className="p-3.5">Transaction ID</th>
                <th className="p-3.5">Customer ID</th>
                <th className="p-3.5">Risk Score</th>
                <th className="p-3.5">Risk Category</th>
                <th className="p-3.5">Amount</th>
                <th className="p-3.5">Action</th>
                <th className="p-3.5">Status</th>
                <th className="p-3.5 text-right">Review</th>
              </tr>
            </thead>
            <tbody>
              {txLoading ? (
                <tr>
                  <td colSpan={9} className="p-8 text-center text-text-secondary">
                    Loading dashboard events...
                  </td>
                </tr>
              ) : filteredEvents.length === 0 ? (
                <tr>
                  <td colSpan={9} className="p-8 text-center text-text-secondary">
                    No risk events match current filters.
                  </td>
                </tr>
              ) : (
                filteredEvents.map((txn, index) => {
                  const score = txn.riskDecision.score;
                  const isCritical = score >= 0.90;
                  const isHigh = score >= 0.70 && score < 0.90;

                  let scoreColor = 'text-risk-low border-risk-low/20 bg-risk-low/10';
                  if (isCritical) scoreColor = 'text-risk-high border-risk-high/30 bg-risk-high/15';
                  else if (isHigh) scoreColor = 'text-risk-medium border-risk-medium/20 bg-risk-medium/10';

                  const signalName = txn.signals.length > 0 ? txn.signals[0].name : 'Anomaly Score Match';

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
                    <tr
                      key={txn.id}
                      onClick={() => router.push(`/transactions/${txn.id}`)}
                      className="border-b border-border hover:bg-card/50 transition-colors cursor-pointer select-none"
                    >
                      <td className="p-3.5 font-mono font-bold text-text-secondary">RSK-{10491 + index}</td>
                      <td className="p-3.5 font-semibold text-text-primary">{txn.id}</td>
                      <td className="p-3.5 font-mono text-text-secondary">{txn.customerId}</td>
                      <td className="p-3.5">
                        <span className={`px-2 py-0.5 rounded border text-[10px] font-mono font-bold ${scoreColor}`}>
                          {Math.round(score * 100)}%
                        </span>
                      </td>
                      <td className="p-3.5 text-text-primary max-w-[150px] truncate">{signalName}</td>
                      <td className="p-3.5 font-mono font-bold text-text-primary">
                        ₹{txn.amount.toLocaleString('en-US')}
                      </td>
                      <td className="p-3.5">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          txn.riskDecision.action === 'BLOCK' ? 'bg-risk-high/20 text-risk-high' :
                          txn.riskDecision.action === 'HOLD' ? 'bg-risk-high/20 text-risk-high' :
                          txn.riskDecision.action === 'MANUAL_REVIEW' ? 'bg-risk-medium/20 text-risk-medium' :
                          'bg-risk-low/20 text-risk-low'
                        }`}>
                          {txn.riskDecision.action}
                        </span>
                      </td>
                      <td className="p-3.5">
                        <span className={`px-2 py-0.5 rounded border text-[9px] font-bold ${statusColor}`}>
                          {statusText}
                        </span>
                      </td>
                      <td className="p-3.5 text-right">
                        <ChevronRight className="w-4 h-4 text-text-secondary inline" />
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
