'use client';

import React, { useState, useMemo } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  ArrowLeft,
  Layers,
  Calendar,
  Clock,
  ShieldCheck,
  CheckSquare,
  AlertTriangle,
  ZoomIn,
  Search,
  ExternalLink,
  ChevronRight
} from 'lucide-react';
import { useRiskRingQuery } from '@/hooks/useRiskRing';
import GraphViewer from '@/components/shared/GraphViewer';
import { Entity } from '@/data/types/risk';

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function RingDetailPage({ params }: PageProps) {
  const { id } = React.use(params);
  const router = useRouter();
  const { data: ring, isLoading } = useRiskRingQuery(id);

  // States
  const [timeFilter, setTimeFilter] = useState<'1h' | '24h' | '7d' | 'all'>('all');
  const [selectedNode, setSelectedNode] = useState<Entity | null>(null);

  if (isLoading) {
    return (
      <div className="py-12 text-center text-xs text-text-secondary select-none font-medium">
        Loading abuse ring graph workspace...
      </div>
    );
  }

  if (!ring) {
    return (
      <div className="space-y-6 text-center select-none py-16">
        <h3 className="text-sm font-bold text-text-primary uppercase tracking-wider">Abuse Ring Not Found</h3>
        <p className="text-xs text-text-secondary">The requested ring code does not exist in our threat database.</p>
        <Link
          href="/rings"
          className="text-xs font-bold text-risk-ai hover:underline inline-flex items-center gap-1 mt-4"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Back to Abuse Rings</span>
        </Link>
      </div>
    );
  }

  // Count types
  const userNodesCount = ring.entities.filter((e) => e.type === 'USER').length;
  const deviceNodesCount = ring.entities.filter((e) => e.type === 'DEVICE').length;
  const ipNodesCount = ring.entities.filter((e) => e.type === 'IP').length;
  const paymentNodesCount = ring.entities.filter((e) => e.type === 'PAYMENT').length;

  const isCritical = ring.severity === 'CRITICAL';

  return (
    <div className="space-y-6 select-none h-full flex flex-col pb-12">
      {/* Top Breadcrumbs & Temporal Selector */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link
            href="/rings"
            className="text-xs font-bold text-text-secondary hover:text-text-primary flex items-center gap-1.5 transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Abuse Rings</span>
          </Link>
        </div>

        {/* Temporal Filters */}
        <div className="flex items-center gap-2.5 text-xs font-medium self-end md:self-auto bg-card border border-border p-1 rounded-lg">
          <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider px-2">Temporal View:</span>
          <div className="flex bg-background border border-border/60 rounded p-0.5 text-[9px] font-mono font-bold">
            {([
              { label: 'Last 1H', value: '1h' },
              { label: 'Last 24H', value: '24h' },
              { label: 'Last 7D', value: '7d' },
              { label: 'All Time', value: 'all' }
            ] as const).map((t) => (
              <button
                key={t.value}
                onClick={() => setTimeFilter(t.value)}
                className={`px-2.5 py-1 rounded cursor-pointer transition-all ${
                  timeFilter === t.value ? 'bg-card text-text-primary border border-border/80' : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Ring Hero Summary Header */}
      <div className={`border rounded-xl p-5 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 ${
        isCritical ? 'border-risk-high/30 bg-risk-high/5' : 'border-risk-medium/20 bg-risk-medium/5'
      }`}>
        <div className="flex items-center gap-4">
          <div className="bg-card p-3 rounded-lg border border-border text-risk-ai">
            <Layers className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-bold text-text-primary">{ring.name} ({ring.id})</h2>
              <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${
                isCritical ? 'bg-risk-high/15 text-risk-high border-risk-high/25' : 'bg-risk-medium/10 text-risk-medium border-risk-medium/25'
              }`}>
                {ring.severity}
              </span>
            </div>
            <p className="text-xs text-text-secondary mt-1">
              Historical Exposure: <span className="font-bold text-text-primary">₹{ring.historicalExposure.toLocaleString('en-US')}</span>
              {' '}• Nodes: <span className="font-semibold text-text-primary">{ring.entities.length}</span>
              {' '}• Relations: <span className="font-semibold text-text-primary">{ring.relationships.length}</span>
            </p>
          </div>
        </div>

        <div className="text-[10px] font-mono text-text-secondary bg-card border border-border px-3 py-2 rounded-lg flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-risk-high animate-pulse"></span>
          <span>Risk Score: {Math.round(ring.riskScore * 100)}%</span>
        </div>
      </div>

      {/* Graph Visualizer Workspace Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 flex-1 min-h-[500px]">
        {/* Left Column: Interactive React Flow Graph Canvas */}
        <div className="lg:col-span-3 h-full min-h-[450px]">
          <GraphViewer
            entities={ring.entities}
            relationships={ring.relationships}
            selectedNodeId={selectedNode ? selectedNode.id : null}
            onSelectNode={setSelectedNode}
            timeFilter={timeFilter}
          />
        </div>

        {/* Right Column: Node Inspector & Ring Details */}
        <div className="space-y-6 flex flex-col h-full overflow-y-auto">
          
          {/* Selected Node Details Inspector */}
          {selectedNode ? (
            <div className="bg-card border border-risk-ai/20 rounded-xl p-5 space-y-4 shadow-lg shadow-risk-ai/5">
              <div className="flex justify-between items-center pb-2 border-b border-border">
                <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-risk-ai animate-pulse"></span>
                  Node Inspector
                </h3>
                <button
                  onClick={() => setSelectedNode(null)}
                  className="text-[10px] text-text-secondary hover:text-text-primary cursor-pointer transition-colors"
                >
                  Clear Selection
                </button>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="bg-background p-2 rounded-lg border border-border text-text-primary font-bold font-mono text-xs">
                    {selectedNode.id}
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-text-primary uppercase leading-none">{selectedNode.type}</h4>
                    <span className="text-[10px] text-text-secondary font-mono block mt-1">
                      Score: <span className="font-bold text-risk-high">{Math.round(selectedNode.riskScore * 100)}%</span>
                    </span>
                  </div>
                </div>

                <div className="h-px bg-border my-2"></div>

                <div className="space-y-2 text-xs">
                  <div className="flex flex-col gap-0.5">
                    <span className="text-[10px] text-text-secondary font-mono">First Seen</span>
                    <span className="text-text-primary font-medium">{new Date(selectedNode.firstSeen).toLocaleString('en-US')}</span>
                  </div>
                  <div className="flex flex-col gap-0.5">
                    <span className="text-[10px] text-text-secondary font-mono">Last Seen</span>
                    <span className="text-text-primary font-medium">{new Date(selectedNode.lastSeen).toLocaleString('en-US')}</span>
                  </div>

                  {/* Metadata key-value details */}
                  {selectedNode.details && (
                    <div className="mt-3 space-y-2 bg-background border border-border p-3.5 rounded-lg font-mono text-[10px]">
                      <span className="font-bold text-[9px] text-text-secondary block uppercase tracking-wider mb-1.5">Telemetry Meta</span>
                      {Object.entries(selectedNode.details).map(([k, v]) => (
                        <div key={k} className="flex justify-between">
                          <span className="text-text-secondary uppercase">{k}</span>
                          <span className="text-text-primary font-bold truncate max-w-[120px]">{v}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Inspect Entity redirects */}
                {selectedNode.type === 'TRANSACTION' && (
                  <button
                    onClick={() => router.push(`/transactions/${selectedNode.id}`)}
                    className="w-full mt-2 bg-risk-ai hover:bg-risk-ai/90 text-white font-bold text-xs py-2 rounded-lg cursor-pointer transition-all flex items-center justify-center gap-1.5"
                  >
                    <span>Investigate Transaction</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-card border border-border border-dashed rounded-xl p-5 text-center text-xs text-text-secondary py-12 flex flex-col items-center justify-center gap-2">
              <ZoomIn className="w-6 h-6 text-text-secondary opacity-60" />
              <span>Select any node on the graph canvas to inspect its risk profile and metadata relationships.</span>
            </div>
          )}

          {/* Ring Risk Summary Card */}
          <div className="bg-card border border-border rounded-xl p-5 space-y-3.5">
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider pb-2 border-b border-border">
              Ring Risk Diagnostics
            </h3>

            <div className="grid grid-cols-2 gap-2 text-center text-[10px] font-mono">
              <div className="bg-background border border-border rounded p-2">
                <span className="text-text-secondary">Devices</span>
                <span className="text-xs font-bold text-text-primary block mt-0.5">{deviceNodesCount}</span>
              </div>
              <div className="bg-background border border-border rounded p-2">
                <span className="text-text-secondary">Shared IPs</span>
                <span className="text-xs font-bold text-text-primary block mt-0.5">{ipNodesCount}</span>
              </div>
              <div className="bg-background border border-border rounded p-2 mt-1">
                <span className="text-text-secondary">Payments</span>
                <span className="text-xs font-bold text-text-primary block mt-0.5">{paymentNodesCount}</span>
              </div>
              <div className="bg-background border border-border rounded p-2 mt-1">
                <span className="text-text-secondary">Members</span>
                <span className="text-xs font-bold text-text-primary block mt-0.5">{userNodesCount}</span>
              </div>
            </div>

            {/* Checklist of Coordinated patterns */}
            <div className="space-y-2 pt-2">
              <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Coordinated Patterns</span>
              <div className="space-y-1.5 text-xs text-text-secondary">
                {[
                  'Multiple accounts share device fingerprint',
                  'Rapid back-to-back account creations',
                  'Coordinated refund chargeback exploit pattern',
                  'Abnormal card-holder payment ties',
                  'Similar timing on merchant checkout attempts'
                ].map((pattern, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <CheckSquare className="w-4 h-4 text-risk-high shrink-0" />
                    <span className="text-[11px] font-medium text-text-primary/95">{pattern}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Sequential Activity Timeline */}
          <div className="bg-card border border-border rounded-xl p-5 space-y-4 flex-1 min-h-[300px] flex flex-col">
            <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider pb-2 border-b border-border flex items-center gap-2">
              <Clock className="w-4 h-4 text-risk-info" />
              <span>Network Temporal Timeline</span>
            </h3>
            
            <div className="flex-1 overflow-y-auto space-y-4 pr-1 relative">
              {/* Timeline center line */}
              <div className="absolute left-3 top-2 bottom-2 w-px bg-border"></div>

              {ring.timeline.map((event, idx) => {
                let badgeColor = 'bg-border border-border text-text-secondary';
                if (event.type === 'REFUND_INITIATED') badgeColor = 'bg-risk-high/15 border-risk-high/30 text-risk-high';
                else if (event.type === 'TRANSACTION_MADE') badgeColor = 'bg-risk-info/10 border-risk-info/30 text-risk-info';

                // Format timestamp
                const date = new Date(event.timestamp);
                const timeString = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                const dateString = date.toLocaleDateString([], { month: 'short', day: '2-digit' });

                return (
                  <div key={idx} className="flex items-start gap-4 pl-1 relative select-none">
                    {/* Ring Dot */}
                    <div className="relative z-10 w-4 h-4 rounded-full bg-sidebar border-2 border-border flex items-center justify-center shrink-0 mt-0.5">
                      <div className={`w-1.5 h-1.5 rounded-full ${
                        event.type === 'REFUND_INITIATED' ? 'bg-risk-high' :
                        event.type === 'TRANSACTION_MADE' ? 'bg-risk-info' : 'bg-text-secondary'
                      }`}></div>
                    </div>

                    <div className="flex-1 flex flex-col gap-1">
                      <div className="flex justify-between items-center gap-2">
                        <span className={`text-[8px] font-mono font-bold px-1.5 py-0.5 rounded border uppercase tracking-wider ${badgeColor}`}>
                          {event.type.replace('_', ' ')}
                        </span>
                        <span className="text-[9px] font-mono text-text-secondary whitespace-nowrap">
                          {dateString}, {timeString}
                        </span>
                      </div>
                      <p className="text-[11px] font-medium text-text-primary leading-normal">
                        {event.description}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
