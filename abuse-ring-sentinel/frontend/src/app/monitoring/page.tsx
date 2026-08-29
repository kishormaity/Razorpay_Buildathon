'use client';

import React from 'react';
import {
  Settings,
  ShieldCheck,
  Activity,
  AlertTriangle,
  Cpu,
  Clock,
  RefreshCw,
  Info,
  AlertOctagon
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { useModelMetricsQuery } from '../../hooks/useModelMetrics';
import {
  mockDriftMetrics,
  mockSystemEvents,
  mockDataQualityTrends
} from '../../data/mock/monitoring';

export default function MonitoringPage() {
  const { data: models = [] } = useModelMetricsQuery();
  
  // Calculate aggregated stats
  const activeModels = models.filter((m) => m.status === 'ACTIVE');
  const avgLatency = activeModels.reduce((acc, m) => acc + m.latency, 0) / activeModels.length || 0;

  return (
    <div className="space-y-6 select-none pb-12">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-extrabold tracking-tight text-text-primary uppercase flex items-center gap-2.5">
            <Settings className="w-6 h-6 text-risk-ai" />
            <span>System & Model Health Monitoring</span>
          </h2>
          <p className="text-xs text-text-secondary mt-1">
            Telemetry metrics tracking model drift, API latencies, error states, and data quality indexes.
          </p>
        </div>

        <div className="text-[10px] font-mono text-text-secondary bg-card border border-border px-3 py-1.5 rounded-lg flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-risk-low animate-pulse"></span>
          <span>Online / Monitoring active</span>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Metric 1: Status */}
        <div className="bg-card border border-border rounded-xl p-4.5">
          <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Model Health</span>
          <h3 className="text-lg font-extrabold text-risk-low mt-1.5 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-risk-low" />
            <span>ONLINE</span>
          </h3>
          <span className="text-[9px] text-text-secondary mt-1 block">3 active model containers</span>
        </div>

        {/* Metric 2: Avg Latency */}
        <div className="bg-card border border-border rounded-xl p-4.5">
          <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Average Latency</span>
          <h3 className="text-lg font-extrabold text-text-primary mt-1.5 flex items-center gap-2">
            <Clock className="w-5 h-5 text-risk-info" />
            <span>{Math.round(avgLatency)}ms</span>
          </h3>
          <span className="text-[9px] text-text-secondary mt-1 block">Target limit &lt;100ms</span>
        </div>

        {/* Metric 3: Error Rate */}
        <div className="bg-card border border-border rounded-xl p-4.5">
          <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Error Rate</span>
          <h3 className="text-lg font-extrabold text-text-primary mt-1.5">0.12%</h3>
          <span className="text-[9px] text-text-secondary mt-1 block">Inference container restarts: 0</span>
        </div>

        {/* Metric 4: Max Drift */}
        <div className="bg-card border border-border rounded-xl p-4.5">
          <span className="text-[10px] font-bold text-text-secondary tracking-wider block uppercase">Peak Drift Warning</span>
          <h3 className="text-lg font-extrabold text-risk-medium mt-1.5 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-risk-medium animate-pulse" />
            <span>0.058 Index</span>
          </h3>
          <span className="text-[9px] text-text-secondary mt-1 block">Drift alert boundary: 0.05</span>
        </div>
      </div>

      {/* Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Chart 1: Model Prediction & Feature Drift (Line chart) */}
        <div className="bg-card border border-border rounded-xl p-5 flex flex-col h-[340px]">
          <div className="mb-4">
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">
              Concept & Feature Drift Trends (Drift index)
            </h3>
            <span className="text-[10px] text-text-secondary font-mono mt-0.5 block">
              Feature and prediction score distributions compared against model training baseline.
            </span>
          </div>

          <div className="flex-1 w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={mockDriftMetrics}>
                <XAxis dataKey="timestamp" stroke="#9ca3af" fontSize={10} tickLine={false} />
                <YAxis stroke="#9ca3af" fontSize={10} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#151922', borderColor: '#252a34', borderRadius: '8px' }}
                  itemStyle={{ fontSize: '11px', color: '#f5f7fa' }}
                />
                <Legend verticalAlign="top" height={36} iconSize={10} wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace' }} />
                <Line type="monotone" dataKey="predictionDrift" name="Prediction Score Drift" stroke="#a855f7" strokeWidth={2} />
                <Line type="monotone" dataKey="featureDrift" name="Feature Drift (Concept Shift)" stroke="#f97316" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Data Quality Drift - Device Sharing / overlap rates (Area chart) */}
        <div className="bg-card border border-border rounded-xl p-5 flex flex-col h-[340px]">
          <div className="mb-4">
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">
              Data Quality & Linking Velocities Index
            </h3>
            <span className="text-[10px] text-text-secondary font-mono mt-0.5 block">
              Device co-occurrence counts and payment cross-use overlaps recorded by Sentinel agent.
            </span>
          </div>

          <div className="flex-1 w-full min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={mockDataQualityTrends}>
                <XAxis dataKey="timestamp" stroke="#9ca3af" fontSize={10} tickLine={false} />
                <YAxis stroke="#9ca3af" fontSize={10} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#151922', borderColor: '#252a34', borderRadius: '8px' }}
                  itemStyle={{ fontSize: '11px', color: '#f5f7fa' }}
                />
                <Legend verticalAlign="top" height={36} iconSize={10} wrapperStyle={{ fontSize: '10px', fontFamily: 'monospace' }} />
                <Area type="monotone" dataKey="deviceSharingRate" name="Device Link Ratio" stroke="#10b981" fill="#10b981" fillOpacity={0.1} />
                <Area type="monotone" dataKey="paymentOverlapRatio" name="Card Binding Overlaps" stroke="#a855f7" fill="#a855f7" fillOpacity={0.1} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

      {/* Bottom Section: Recent System Events Timeline */}
      <div className="bg-card border border-border rounded-xl p-5 select-none">
        <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider mb-5 pb-2 border-b border-border flex items-center gap-2">
          <Activity className="w-4 h-4 text-risk-info" />
          <span>Risk Operations System Events log</span>
        </h3>

        <div className="space-y-4 relative pl-3.5 pr-2">
          {/* Vertical line */}
          <div className="absolute left-6 top-3 bottom-3 w-px bg-border"></div>

          {mockSystemEvents.map((evt) => {
            let Icon = Info;
            let iconColor = 'text-risk-info border-risk-info/20 bg-risk-info/10';

            if (evt.type === 'CRITICAL') {
              Icon = AlertOctagon;
              iconColor = 'text-risk-high border-risk-high/30 bg-risk-high/15';
            } else if (evt.type === 'WARNING') {
              Icon = AlertTriangle;
              iconColor = 'text-risk-medium border-risk-medium/20 bg-risk-medium/10';
            }

            return (
              <div key={evt.id} className="flex gap-4 items-start relative select-none">
                {/* Event Dot */}
                <div className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 relative z-10 border ${iconColor}`}>
                  <Icon className="w-3 h-3" />
                </div>

                <div className="flex-1 flex flex-col md:flex-row md:items-center justify-between gap-2 border border-border bg-background/50 rounded-lg p-3">
                  <div className="space-y-0.5">
                    <span className="text-[10px] text-text-secondary font-mono uppercase tracking-wider block">
                      {evt.source} • {evt.id}
                    </span>
                    <p className="text-xs font-semibold text-text-primary">
                      {evt.message}
                    </p>
                  </div>

                  <span className="text-[10px] font-mono text-text-secondary">
                    {new Date(evt.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
