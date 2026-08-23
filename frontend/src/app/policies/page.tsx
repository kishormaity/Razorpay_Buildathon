'use client';

import React, { useState, useMemo } from 'react';
import { Sliders, ShieldAlert, CheckCircle, TrendingDown, HelpCircle } from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend
} from 'recharts';
import { mockPolicyRules, mockPolicySimulationData } from '../../data/mock/policies';

export default function PoliciesPage() {
  // Simulator bounds state
  const [allowThreshold, setAllowThreshold] = useState(0.40);
  const [stepUpThreshold, setStepUpThreshold] = useState(0.70);
  const [reviewThreshold, setReviewThreshold] = useState(0.90);

  const [simulatedScore, setSimulatedScore] = useState(0.55); // Vertical indicator line on chart
  const [successBanner, setSuccessBanner] = useState(false);

  // Dynamic cost calculations based on selected simulatedScore
  const simulatedCostMetrics = useMemo(() => {
    // Interpolate values from mockPolicySimulationData based on simulatedScore
    const data = mockPolicySimulationData;
    // find closest points
    let idx = 0;
    for (let i = 0; i < data.length; i++) {
      if (data[i].threshold >= simulatedScore) {
        idx = i;
        break;
      }
    }

    const point = data[idx] || data[data.length - 1];
    
    // Scale slightly based on slider adjustments
    const scalingFactor = (allowThreshold / 0.40) * 0.95;
    
    return {
      fraudLoss: Math.round(point.fraudLossCost * scalingFactor),
      friction: Math.round(point.frictionCost * (1.1 - scalingFactor)),
      review: Math.round(point.reviewCost),
      total: Math.round(point.fraudLossCost * scalingFactor + point.frictionCost * (1.1 - scalingFactor) + point.reviewCost),
    };
  }, [simulatedScore, allowThreshold, stepUpThreshold, reviewThreshold]);

  const handleSavePolicy = () => {
    setSuccessBanner(true);
    setTimeout(() => setSuccessBanner(false), 3000);
  };

  return (
    <div className="space-y-6 select-none pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-extrabold tracking-tight text-text-primary uppercase flex items-center gap-2.5">
            <Sliders className="w-6 h-6 text-risk-ai" />
            <span>Decision Policy Manager</span>
          </h2>
          <p className="text-xs text-text-secondary mt-1">
            Simulate and adjust automated action thresholds to balance customer friction against fraud loss.
          </p>
        </div>

        {/* Simulator mode banner */}
        <div className="flex items-center gap-2 bg-risk-medium/10 border border-risk-medium/20 px-3 py-1.5 rounded-lg text-risk-medium text-xs font-bold font-mono">
          <ShieldAlert className="w-4 h-4 text-risk-medium animate-pulse" />
          <span>SIMULATION MODE ONLY</span>
        </div>
      </div>

      {successBanner && (
        <div className="bg-risk-low/10 border border-risk-low/20 text-risk-low text-xs font-bold rounded-lg p-3 text-center animate-bounce">
          Policy simulation bounds updated locally. Changes recalculated below.
        </div>
      )}

      {/* Threshold Sliders and Recalculator Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Editable Sliders */}
        <div className="lg:col-span-1 bg-card border border-border rounded-xl p-5 space-y-6 flex flex-col justify-between">
          <div className="space-y-5">
            <div className="flex justify-between items-center pb-2 border-b border-border">
              <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">
                Risk Score Thresholds
              </h3>
              <span className="text-[9px] font-mono text-text-secondary">
                Auto-Enforcement Bounds
              </span>
            </div>

            {/* Slider 1: Allow */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-text-primary flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded bg-risk-low" />
                  ALLOW Threshold
                </span>
                <span className="text-risk-low font-mono">{allowThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.60"
                step="0.05"
                value={allowThreshold}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setAllowThreshold(val);
                }}
                className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-risk-low"
              />
              <p className="text-[10px] text-text-secondary leading-normal">
                Scores below this limit bypass secondary security checks and are immediately approved.
              </p>
            </div>

            {/* Slider 2: Step-up */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-text-primary flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded bg-risk-medium" />
                  STEP_UP Threshold
                </span>
                <span className="text-risk-medium font-mono">{stepUpThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.50"
                max="0.80"
                step="0.05"
                value={stepUpThreshold}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setStepUpThreshold(val);
                }}
                className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-risk-medium"
              />
              <p className="text-[10px] text-text-secondary leading-normal">
                Scores within this range trigger secondary verification checks (such as OTP or 3DS).
              </p>
            </div>

            {/* Slider 3: Review */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-text-primary flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded bg-risk-info" />
                  MANUAL REVIEW Threshold
                </span>
                <span className="text-risk-info font-mono">{reviewThreshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.75"
                max="0.95"
                step="0.05"
                value={reviewThreshold}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setReviewThreshold(val);
                }}
                className="w-full h-1 bg-border rounded-lg appearance-none cursor-pointer accent-risk-info"
              />
              <p className="text-[10px] text-text-secondary leading-normal">
                Scores within this range route to manual queues for analyst review. Above this trigger captures.
              </p>
            </div>
          </div>

          <button
            onClick={handleSavePolicy}
            className="w-full mt-6 bg-risk-ai hover:bg-risk-ai/90 border border-risk-ai/30 text-white font-bold text-xs py-2.5 rounded-lg cursor-pointer transition-colors"
          >
            Apply Simulated Settings
          </button>
        </div>

        {/* Right Column: Visualizer Chart & Simulated metrics outcomes */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Recharts Area cost-sweetspot chart */}
          <div className="bg-card border border-border rounded-xl p-5 flex flex-col h-[380px]">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-4">
              <div>
                <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">
                  Cost Optimization Sweet Spot
                </h3>
                <span className="text-[10px] text-text-secondary font-mono mt-0.5 block">
                  Find the optimal score boundary (X-Axis) by analyzing total expected cost curves (₹).
                </span>
              </div>

              {/* Slider for simulated score selection */}
              <div className="flex items-center gap-2 bg-background border border-border px-3 py-1.5 rounded-lg text-[10px] font-mono">
                <span className="text-text-secondary">Simulated Score:</span>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={simulatedScore}
                  onChange={(e) => setSimulatedScore(Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
                  className="bg-transparent border-0 outline-none w-10 text-risk-ai font-bold text-right"
                />
              </div>
            </div>

            <div className="flex-1 w-full min-h-0">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={mockPolicySimulationData}>
                  <XAxis dataKey="threshold" stroke="#9ca3af" fontSize={10} tickLine={false} />
                  <YAxis stroke="#9ca3af" fontSize={10} tickLine={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#151922', borderColor: '#252a34', borderRadius: '8px' }}
                    itemStyle={{ fontSize: '11px', color: '#f5f7fa' }}
                  />
                  <Legend verticalAlign="top" height={36} iconSize={10} wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace' }} />
                  {/* Areas for components */}
                  <Area type="monotone" dataKey="fraudLossCost" name="Fraud Loss Cost" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.15} />
                  <Area type="monotone" dataKey="frictionCost" name="Customer Friction Cost" stackId="1" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} />
                  <Area type="monotone" dataKey="reviewCost" name="Review Operations Cost" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} />
                  
                  {/* Reference line showing simulated threshold */}
                  <ReferenceLine x={simulatedScore} stroke="#a855f7" strokeWidth={2} strokeDasharray="3 3" label={{ value: `Cut-off: ${simulatedScore}`, fill: '#a855f7', fontSize: 10, position: 'top' }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Outcome metrics banner */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            
            <div className="bg-card border border-border rounded-xl p-4.5">
              <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Fraud Loss Cost</span>
              <h3 className="text-lg font-extrabold text-risk-high mt-1.5">₹{simulatedCostMetrics.fraudLoss.toLocaleString('en-US')}</h3>
              <span className="text-[9px] text-text-secondary mt-1 block">Uncaught chargebacks</span>
            </div>

            <div className="bg-card border border-border rounded-xl p-4.5">
              <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Friction Cost</span>
              <h3 className="text-lg font-extrabold text-risk-medium mt-1.5">₹{simulatedCostMetrics.friction.toLocaleString('en-US')}</h3>
              <span className="text-[9px] text-text-secondary mt-1 block">Abandoned checkouts</span>
            </div>

            <div className="bg-card border border-border rounded-xl p-4.5">
              <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Review Ops Cost</span>
              <h3 className="text-lg font-extrabold text-risk-info mt-1.5">₹{simulatedCostMetrics.review.toLocaleString('en-US')}</h3>
              <span className="text-[9px] text-text-secondary mt-1 block">Analyst audit times</span>
            </div>

            <div className="bg-card border border-risk-ai/20 rounded-xl p-4.5 bg-risk-ai/5">
              <span className="text-[10px] font-bold text-risk-ai tracking-wider block uppercase">Total Cost Impact</span>
              <h3 className="text-lg font-extrabold text-text-primary mt-1.5">₹{simulatedCostMetrics.total.toLocaleString('en-US')}</h3>
              <div className="flex items-center gap-1 mt-1 text-[9px] text-risk-low font-bold">
                <TrendingDown className="w-3.5 h-3.5" />
                <span>Optimal sweet spot</span>
              </div>
            </div>

          </div>

        </div>

      </div>
    </div>
  );
}
