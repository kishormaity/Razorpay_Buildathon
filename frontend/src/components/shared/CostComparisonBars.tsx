'use client';

import React from 'react';
import { CostEstimate } from '../../data/types/risk';

interface CostComparisonBarsProps {
  amount: number;
  expectedCost: CostEstimate;
  recommendedAction: string;
}

export default function CostComparisonBars({ amount, expectedCost, recommendedAction }: CostComparisonBarsProps) {
  // If allowed, fraud loss is full amount, friction and review costs are zero
  const costIfAllowed = {
    fraudLoss: amount,
    customerFriction: 0,
    manualReview: 0,
    total: amount,
  };

  // If held, fraud loss is usually zero (prevented), friction and review costs are active
  const costIfHeld = {
    fraudLoss: 0,
    customerFriction: expectedCost.customerFriction,
    manualReview: expectedCost.manualReview,
    total: expectedCost.customerFriction + expectedCost.manualReview,
  };

  const savings = costIfAllowed.total - costIfHeld.total;
  
  // Calculate relative widths based on max cost (which is costIfAllowed.total)
  const maxVal = costIfAllowed.total;
  const getPercent = (val: number) => `${Math.max((val / maxVal) * 100, 1.5)}%`;

  return (
    <div className="bg-card border border-border rounded-xl p-5 select-none">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">Decision Cost Analysis</h3>
        <span className="text-[10px] font-mono text-text-secondary bg-border px-2 py-0.5 rounded border border-border">
          Cost-Sensitive Decision Engine
        </span>
      </div>

      <div className="space-y-5">
        {/* Scenario 1: Allowed */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs font-medium">
            <span className="text-text-primary flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-risk-high"></span>
              If Transaction is Allowed (Expected Cost)
            </span>
            <span className="text-text-primary font-bold">₹{costIfAllowed.total.toLocaleString('en-US')}</span>
          </div>
          <div className="h-6 w-full bg-border/40 rounded-md overflow-hidden flex">
            {/* Red block for Fraud Loss */}
            <div
              className="bg-risk-high h-full transition-all duration-500 hover:opacity-90"
              style={{ width: getPercent(costIfAllowed.fraudLoss) }}
              title={`Fraud Loss: ₹${costIfAllowed.fraudLoss.toLocaleString('en-US')}`}
            />
          </div>
        </div>

        {/* Scenario 2: Held */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs font-medium">
            <span className="text-text-primary flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-risk-medium animate-pulse"></span>
              If Action is Hold/Review (Expected Cost)
            </span>
            <span className="text-text-primary font-bold">₹{costIfHeld.total.toLocaleString('en-US')}</span>
          </div>
          <div className="h-6 w-full bg-border/40 rounded-md overflow-hidden flex">
            {/* Amber block for Friction */}
            <div
              className="bg-risk-medium h-full border-r border-card transition-all duration-500 hover:opacity-90"
              style={{ width: getPercent(costIfHeld.customerFriction) }}
              title={`Friction: ₹${costIfHeld.customerFriction.toLocaleString('en-US')}`}
            />
            {/* Blue block for Review */}
            <div
              className="bg-risk-info h-full transition-all duration-500 hover:opacity-90"
              style={{ width: getPercent(costIfHeld.manualReview) }}
              title={`Review Cost: ₹${costIfHeld.manualReview.toLocaleString('en-US')}`}
            />
          </div>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex items-center gap-4 text-[10px] font-mono text-text-secondary border-t border-border pt-3.5">
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded bg-risk-high"></div>
          <span>Fraud Loss</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded bg-risk-medium"></div>
          <span>Customer Friction Cost</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded bg-risk-info"></div>
          <span>Manual Analyst Cost</span>
        </div>
      </div>

      {/* Saving Output Recommendation Banner */}
      <div className="mt-4 bg-risk-low/5 border border-risk-low/15 rounded-lg p-3 flex items-center justify-between">
        <div className="flex flex-col gap-0.5">
          <span className="text-[10px] font-bold text-risk-low tracking-wide">SENTINEL RECOMMENDATION</span>
          <span className="text-xs font-semibold text-text-primary leading-normal">
            Take <span className="text-risk-low font-bold">{recommendedAction}</span> Action
          </span>
        </div>
        <div className="text-right leading-none">
          <span className="text-[10px] text-text-secondary font-mono block">EST. LOSS PREVENTED</span>
          <span className="text-sm font-extrabold text-risk-low mt-1 block">
            ₹{savings.toLocaleString('en-US')}
          </span>
        </div>
      </div>
    </div>
  );
}
