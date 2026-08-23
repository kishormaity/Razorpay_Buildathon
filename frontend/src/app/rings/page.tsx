'use client';

import React, { useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { Layers, Search, ShieldAlert, ArrowRight, ShieldCheck, HelpCircle } from 'lucide-react';
import { useRiskRingsQuery } from '../../hooks/useRiskRing';

export default function RingsListPage() {
  const router = useRouter();
  const { data: rings = [], isLoading } = useRiskRingsQuery();

  const [searchTerm, setSearchTerm] = useState('');
  const [severityFilter, setSeverityFilter] = useState<'ALL' | 'CRITICAL' | 'HIGH'>('ALL');

  // Filter rings
  const filteredRings = useMemo(() => {
    return rings.filter((ring) => {
      // Severity filter
      if (severityFilter !== 'ALL' && ring.severity !== severityFilter) return false;

      // Keyword filter
      if (searchTerm) {
        const term = searchTerm.toLowerCase();
        return ring.id.toLowerCase().includes(term) || ring.name.toLowerCase().includes(term);
      }

      return true;
    });
  }, [rings, searchTerm, severityFilter]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-extrabold tracking-tight text-text-primary uppercase flex items-center gap-2.5">
            <Layers className="w-6 h-6 text-risk-ai" />
            <span>Coordinated Abuse Rings</span>
          </h2>
          <p className="text-xs text-text-secondary mt-1">
            Explore linked clusters of accounts, devices, payment channels, and IP clusters identified by the Graph Risk Engine.
          </p>
        </div>

        <div className="text-[10px] font-mono text-text-secondary bg-card border border-border px-3 py-1.5 rounded-lg flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-risk-high animate-ping"></span>
          <span>Threat Scanner Active</span>
        </div>
      </div>

      {/* Query Search Card */}
      <div className="bg-card border border-border rounded-xl p-5 select-none flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Search */}
        <div className="relative w-full md:max-w-md">
          <Search className="w-4 h-4 text-text-secondary absolute left-3.5 top-3" />
          <input
            type="text"
            placeholder="Search by Ring ID, name, network keys..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-background border border-border rounded-lg pl-10 pr-4 py-2 text-xs text-text-primary placeholder-text-secondary outline-none focus:border-text-secondary/40 transition-colors"
          />
        </div>

        {/* Severity filter tabs */}
        <div className="flex items-center gap-3 self-end md:self-auto text-xs font-medium">
          <span className="text-text-secondary">Severity:</span>
          <div className="bg-background border border-border rounded-lg p-0.5 flex">
            {(['ALL', 'CRITICAL', 'HIGH'] as const).map((level) => (
              <button
                key={level}
                onClick={() => setSeverityFilter(level)}
                className={`px-3 py-1.5 text-[10px] font-bold rounded cursor-pointer ${
                  severityFilter === level ? 'bg-card text-text-primary border border-border' : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                {level}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Rings Queue Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 select-none">
        {isLoading ? (
          <div className="col-span-2 py-16 text-center text-xs text-text-secondary font-medium">
            Scanning graph networks...
          </div>
        ) : filteredRings.length === 0 ? (
          <div className="col-span-2 py-16 text-center text-xs text-text-secondary font-medium">
            No coordinated abuse rings found.
          </div>
        ) : (
          filteredRings.map((ring) => {
            const isCritical = ring.severity === 'CRITICAL';
            
            // Sub count nodes
            const userNodes = ring.entities.filter((e) => e.type === 'USER').length;
            const deviceNodes = ring.entities.filter((e) => e.type === 'DEVICE').length;
            const ipNodes = ring.entities.filter((e) => e.type === 'IP').length;
            const paymentNodes = ring.entities.filter((e) => e.type === 'PAYMENT').length;

            return (
              <div
                key={ring.id}
                className={`bg-card border rounded-xl p-5 flex flex-col justify-between transition-all hover:border-text-secondary/20 relative overflow-hidden ${
                  isCritical ? 'border-risk-high/30 shadow-[0_0_12px_rgba(239,68,68,0.02)]' : 'border-border'
                }`}
              >
                <div>
                  {/* Header info */}
                  <div className="flex justify-between items-start">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-bold text-text-secondary font-mono">
                        {ring.id}
                      </span>
                      <span className={`text-[9px] font-mono font-bold px-2 py-0.5 rounded border ${
                        isCritical
                          ? 'bg-risk-high/10 text-risk-high border-risk-high/20'
                          : 'bg-risk-medium/10 text-risk-medium border-risk-medium/20'
                      }`}>
                        {ring.severity}
                      </span>
                    </div>

                    <span className="text-[10px] font-mono text-text-secondary bg-border px-2 py-0.5 rounded border border-border">
                      {(ring.riskScore * 100).toFixed(0)}% Score
                    </span>
                  </div>

                  {/* Ring name */}
                  <h3 className="text-sm font-bold text-text-primary mt-3">
                    {ring.name}
                  </h3>

                  {/* Nodes Count Summary */}
                  <div className="grid grid-cols-4 gap-2 text-center mt-4 border-y border-border py-3 bg-background/30 rounded-lg">
                    <div>
                      <span className="text-[10px] text-text-secondary block">Users</span>
                      <span className="text-xs font-bold text-text-primary mt-0.5 block">{userNodes}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-text-secondary block">Devices</span>
                      <span className="text-xs font-bold text-text-primary mt-0.5 block">{deviceNodes}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-text-secondary block">IPs</span>
                      <span className="text-xs font-bold text-text-primary mt-0.5 block">{ipNodes}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-text-secondary block">Cards</span>
                      <span className="text-xs font-bold text-text-primary mt-0.5 block">{paymentNodes}</span>
                    </div>
                  </div>

                  {/* Narrative parameters */}
                  <div className="mt-4 flex items-center justify-between text-xs">
                    <span className="text-text-secondary">Ring Historical Exposure:</span>
                    <span className="font-bold text-text-primary font-mono">
                      ₹{ring.historicalExposure.toLocaleString('en-US')}
                    </span>
                  </div>
                </div>

                {/* Inspect Button */}
                <button
                  onClick={() => router.push(`/rings/${ring.id}`)}
                  className="w-full mt-5 bg-border hover:bg-card border border-border text-text-primary font-bold text-xs py-2 rounded-lg flex items-center justify-center gap-2 cursor-pointer transition-colors"
                >
                  <span>Investigate Connected Graph</span>
                  <ArrowRight className="w-3.5 h-3.5 text-risk-ai" />
                </button>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
