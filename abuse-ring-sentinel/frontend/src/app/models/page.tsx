'use client';

import React from 'react';
import { BarChart2, ShieldAlert, Cpu } from 'lucide-react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useModelMetricsQuery, useBaselineComparisonQuery } from '@/hooks/useModelMetrics';

// Precision Recall Curve mock data points
const mockPrCurveData = [
  { threshold: 0.1, precision: 0.45, recall: 0.98 },
  { threshold: 0.2, precision: 0.55, recall: 0.95 },
  { threshold: 0.3, precision: 0.68, recall: 0.92 },
  { threshold: 0.4, precision: 0.78, recall: 0.90 },
  { threshold: 0.5, precision: 0.86, recall: 0.88 },
  { threshold: 0.55, precision: 0.912, recall: 0.876 }, // sweet spot
  { threshold: 0.6, precision: 0.93, recall: 0.82 },
  { threshold: 0.7, precision: 0.96, recall: 0.75 },
  { threshold: 0.8, precision: 0.98, recall: 0.62 },
  { threshold: 0.9, precision: 0.99, recall: 0.45 },
  { threshold: 1.0, precision: 1.00, recall: 0.00 },
];

export default function ModelsPage() {
  const { data: models = [], isLoading: modelsLoading } = useModelMetricsQuery();
  const { data: comparisons = [], isLoading: compsLoading } = useBaselineComparisonQuery();

  return (
    <div className="space-y-6 select-none pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-extrabold tracking-tight text-text-primary uppercase flex items-center gap-2.5">
            <Cpu className="w-6 h-6 text-risk-ai" />
            <span>Model Performance Monitoring</span>
          </h2>
          <p className="text-xs text-text-secondary mt-1">
            Precision, recall metrics and baseline lifts evaluated on historical test sets.
          </p>
        </div>

        {/* Prototype Metrics Badge */}
        <div className="flex items-center gap-2 bg-risk-ai/10 border border-risk-ai/20 px-3 py-1.5 rounded-lg text-risk-ai text-xs font-bold font-mono">
          <ShieldAlert className="w-4 h-4 text-risk-ai animate-pulse" />
          <span>PROTOTYPE EVALUATION DATA</span>
        </div>
      </div>

      {/* Model Cards Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {modelsLoading ? (
          <div className="col-span-3 text-center text-xs text-text-secondary font-medium py-10">
            Loading model telemetry...
          </div>
        ) : (
          models.map((model) => (
            <div
              key={model.id}
              className={`bg-card border rounded-xl p-5 relative overflow-hidden flex flex-col justify-between ${
                model.status === 'SHADOW' ? 'border-border opacity-75' : 'border-border'
              }`}
            >
              {model.status === 'SHADOW' && (
                <span className="absolute top-2 right-2 text-[8px] font-mono font-bold bg-border text-text-secondary px-1.5 py-0.5 rounded border border-border">
                  SHADOW MODE
                </span>
              )}
              
              <div className="space-y-3">
                <span className="text-[10px] text-text-secondary font-mono tracking-wider block">
                  Model: {model.version}
                </span>
                <h3 className="text-sm font-bold text-text-primary">
                  {model.name}
                </h3>

                {/* Metrics Details */}
                <div className="grid grid-cols-3 gap-2 mt-4 text-center text-[10px] font-mono bg-background/50 border border-border rounded-lg p-2.5">
                  <div>
                    <span className="text-text-secondary">Precision</span>
                    <span className="text-xs font-bold text-text-primary block mt-0.5">
                      {(model.precision * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-text-secondary">Recall</span>
                    <span className="text-xs font-bold text-text-primary block mt-0.5">
                      {(model.recall * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-text-secondary">PR-AUC</span>
                    <span className="text-xs font-bold text-risk-ai block mt-0.5">
                      {(model.prAuc * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Inference Latency & Drift metrics */}
              <div className="mt-5 pt-3 border-t border-border flex items-center justify-between text-[10px] font-mono text-text-secondary">
                <span>Latency: <span className="text-text-primary font-bold">{model.latency}ms</span></span>
                <span>Drift Index: <span className={`${model.drift > 0.04 ? 'text-risk-medium' : 'text-risk-low'} font-bold`}>{model.drift.toFixed(3)}</span></span>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Main Grid: Comparison Table & Precision-Recall Line Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Left Column: Baseline comparison */}
        <div className="bg-card border border-border rounded-xl p-5 space-y-4 flex flex-col justify-between">
          <div>
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">
              System Impact: Rule Engine vs AI Engine
            </h3>
            <span className="text-[10px] text-text-secondary font-mono mt-0.5 block">
              Evaluated on the synthetic dataset to show manual load and loss savings.
            </span>
          </div>

          <div className="overflow-x-auto border border-border rounded-lg mt-3">
            <table className="w-full text-left border-collapse text-xs select-none">
              <thead>
                <tr className="bg-sidebar/50 border-b border-border text-text-secondary font-mono font-bold">
                  <th className="p-3">Metric Type</th>
                  <th className="p-3">Rule-Based Setup</th>
                  <th className="p-3">AI Risk Engine</th>
                  <th className="p-3 text-right">Lift / Saved</th>
                </tr>
              </thead>
              <tbody>
                {compsLoading ? (
                  <tr>
                    <td colSpan={4} className="p-8 text-center text-text-secondary">
                      Calculating lift matrix...
                    </td>
                  </tr>
                ) : (
                  comparisons.map((row, idx) => {
                    const isFpr = row.metric.includes('False Positive') || row.metric.includes('Review Volume');
                    
                    return (
                      <tr key={idx} className="border-b border-border last:border-b-0 hover:bg-card/30 transition-colors">
                        <td className="p-3 font-semibold text-text-primary">{row.metric}</td>
                        <td className="p-3 text-text-secondary font-medium font-mono">{row.ruleEngine}</td>
                        <td className="p-3 text-text-primary font-bold font-mono">{row.aiEngine}</td>
                        <td className={`p-3 text-right font-bold font-mono ${
                          isFpr ? 'text-risk-low' : 'text-risk-low'
                        }`}>
                          {row.impact}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          <div className="mt-4 p-3 bg-risk-low/5 border border-risk-low/15 rounded-lg text-[11px] text-text-secondary leading-relaxed">
            <span className="font-bold text-risk-low block uppercase tracking-wide mb-0.5">Decision Lift Summary</span>
            Manual queues are reduced by <span className="text-text-primary font-bold">66.5%</span>, while fraud captures improve by <span className="text-text-primary font-bold">26.1%</span>, avoiding customer friction costs.
          </div>
        </div>

        {/* Right Column: Precision-Recall Curve Line Chart */}
        <div className="bg-card border border-border rounded-xl p-5 flex flex-col h-[380px]">
          <div className="mb-4">
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">
              Precision-Recall Trade-off curves
            </h3>
            <span className="text-[10px] text-text-secondary font-mono mt-0.5 block">
              Identify the optimal decision thresholds by plotting Precision vs. Recall curves.
            </span>
          </div>

          <div className="flex-1 w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockPrCurveData}>
                <XAxis dataKey="threshold" stroke="#9ca3af" fontSize={10} tickLine={false} label={{ value: 'Decision Threshold Score', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 10 }} />
                <YAxis stroke="#9ca3af" fontSize={10} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#151922', borderColor: '#252a34', borderRadius: '8px' }}
                  labelStyle={{ color: '#9ca3af', fontSize: '10px' }}
                  itemStyle={{ fontSize: '11px' }}
                />
                <Legend verticalAlign="top" height={36} iconSize={10} wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace' }} />
                <Line type="monotone" dataKey="precision" name="Precision (Friction reduction)" stroke="#a855f7" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="recall" name="Recall (Fraud caught)" stroke="#f97316" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>
    </div>
  );
}
