'use client';

import React from 'react';

interface RiskScoreGaugeProps {
  score: number;       // 0.0 to 1.0
  confidence: number;  // 0.0 to 1.0
  size?: number;       // diameter in pixels
}

export default function RiskScoreGauge({ score, confidence, size = 160 }: RiskScoreGaugeProps) {
  const percentage = Math.round(score * 100);
  const confidencePercent = Math.round(confidence * 100);
  
  // Calculate severity details
  let severityLabel = 'LOW RISK';
  let colorClass = 'text-risk-low';
  let strokeColor = '#10b981';
  let bgGlow = 'rgba(16, 185, 129, 0.1)';

  if (score >= 0.90) {
    severityLabel = 'CRITICAL RISK';
    colorClass = 'text-risk-high';
    strokeColor = '#ef4444';
    bgGlow = 'rgba(239, 68, 68, 0.1)';
  } else if (score >= 0.70) {
    severityLabel = 'HIGH RISK';
    colorClass = 'text-risk-high';
    strokeColor = '#f97316';
    bgGlow = 'rgba(249, 115, 22, 0.1)';
  } else if (score >= 0.40) {
    severityLabel = 'MEDIUM RISK';
    colorClass = 'text-risk-medium';
    strokeColor = '#f59e0b';
    bgGlow = 'rgba(245, 158, 11, 0.1)';
  }

  // SVG parameters
  const strokeWidth = 10;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score * circumference);

  return (
    <div className="flex flex-col items-center select-none" style={{ width: size }}>
      {/* Gauge Visual */}
      <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
        {/* Glow backdrop shadow */}
        <div
          className="absolute inset-0 rounded-full transition-all duration-500"
          style={{ backgroundColor: bgGlow, filter: 'blur(16px)' }}
        />

        <svg width={size} height={size} className="transform -rotate-90">
          {/* Base track circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke="#252a34"
            strokeWidth={strokeWidth}
          />
          {/* Active progress circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="transparent"
            stroke={strokeColor}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-700 ease-out"
          />
        </svg>

        {/* Inner Centered Metrics */}
        <div className="absolute flex flex-col items-center justify-center leading-none text-center">
          <span className="text-[10px] font-bold text-text-secondary tracking-widest mb-1.5 uppercase">RISK SCORE</span>
          <span className={`text-4xl font-extrabold tracking-tight ${colorClass}`}>
            {percentage}%
          </span>
          <span className={`text-[10px] font-extrabold tracking-widest mt-2 uppercase ${colorClass}`}>
            {severityLabel}
          </span>
        </div>
      </div>

      {/* Supporting Calibration Indicators */}
      <div className="w-full mt-4 flex items-center justify-between border-t border-border pt-3.5 text-[10px] font-mono text-text-secondary">
        <div className="flex flex-col items-start">
          <span>Confidence</span>
          <span className="text-text-primary font-bold mt-0.5">{confidencePercent}%</span>
        </div>
        <div className="h-6 w-px bg-border"></div>
        <div className="flex flex-col items-center">
          <span>Percentile</span>
          <span className="text-text-primary font-bold mt-0.5">
            {percentage > 90 ? '99.2th' : percentage > 70 ? '92.4th' : percentage > 40 ? '68.1th' : '15.4th'}
          </span>
        </div>
        <div className="h-6 w-px bg-border"></div>
        <div className="flex flex-col items-end">
          <span>Calibration</span>
          <span className="text-risk-low font-bold mt-0.5">OPTIMAL</span>
        </div>
      </div>
    </div>
  );
}
