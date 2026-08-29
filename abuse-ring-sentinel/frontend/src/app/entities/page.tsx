'use client';

import React, { useState, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import {
  Shield,
  Search,
  User,
  Smartphone,
  Globe,
  CreditCard,
  ShoppingBag,
  MapPin,
  FileText,
  ChevronRight,
  ExternalLink
} from 'lucide-react';
import { mockRiskRings } from '../../data/mock/rings';
import { Entity } from '../../data/types/risk';

export default function EntitiesPage() {
  const router = useRouter();

  // Consolidate unique entities across all mock abuse rings
  const allEntities = useMemo(() => {
    const map = new Map<string, Entity>();
    mockRiskRings.forEach((ring) => {
      ring.entities.forEach((entity) => {
        map.set(entity.id, entity);
      });
    });
    return Array.from(map.values());
  }, []);

  // Filter states
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState<'ALL' | 'USER' | 'DEVICE' | 'IP' | 'PAYMENT' | 'MERCHANT' | 'ADDRESS'>('ALL');
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);

  const filteredEntities = useMemo(() => {
    return allEntities.filter((item) => {
      // Type filter
      if (typeFilter !== 'ALL' && item.type !== typeFilter) return false;

      // Keyword search
      if (searchTerm) {
        const term = searchTerm.toLowerCase();
        return (
          item.id.toLowerCase().includes(term) ||
          item.label.toLowerCase().includes(term) ||
          item.type.toLowerCase().includes(term)
        );
      }

      return true;
    });
  }, [allEntities, searchTerm, typeFilter]);

  return (
    <div className="space-y-6 select-none pb-12">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-extrabold tracking-tight text-text-primary uppercase flex items-center gap-2.5">
          <Shield className="w-6 h-6 text-risk-ai" />
          <span>Entity Explorer Network</span>
        </h2>
        <p className="text-xs text-text-secondary mt-1">
          Query and audit unique physical and logical identity nodes mapped across all threat rings.
        </p>
      </div>

      {/* Toolbar filters */}
      <div className="bg-card border border-border rounded-xl p-5 select-none flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Search */}
        <div className="relative w-full lg:max-w-md">
          <Search className="w-4 h-4 text-text-secondary absolute left-3.5 top-3" />
          <input
            type="text"
            placeholder="Search Entity ID or label name..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-background border border-border rounded-lg pl-10 pr-4 py-2 text-xs text-text-primary placeholder-text-secondary outline-none focus:border-text-secondary/40 transition-colors"
          />
        </div>

        {/* Type selector */}
        <div className="flex bg-background border border-border rounded-lg p-0.5 text-[9px] font-mono font-bold self-end lg:self-auto overflow-x-auto">
          {(['ALL', 'USER', 'DEVICE', 'IP', 'PAYMENT', 'MERCHANT'] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTypeFilter(t)}
              className={`px-3 py-1.5 rounded cursor-pointer whitespace-nowrap ${
                typeFilter === t ? 'bg-card text-text-primary border border-border' : 'text-text-secondary hover:text-text-primary'
              }`}
            >
              {t}
            </button>
          ))}
        </div>
      </div>

      {/* Grid: Main Table & Inspector Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
        
        {/* Table list */}
        <div className="lg:col-span-3 bg-card border border-border rounded-xl overflow-hidden">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-sidebar/50 border-b border-border text-text-secondary font-mono font-bold">
                <th className="p-4">Entity ID</th>
                <th className="p-4">Type</th>
                <th className="p-4">Label Reference</th>
                <th className="p-4">Risk Score</th>
                <th className="p-4">First Registered</th>
                <th className="p-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredEntities.length === 0 ? (
                <tr>
                  <td colSpan={6} className="p-12 text-center text-text-secondary font-medium">
                    No entities match current parameters.
                  </td>
                </tr>
              ) : (
                filteredEntities.map((item) => {
                  let Icon = User;
                  let colorClass = 'text-risk-info';

                  switch (item.type) {
                    case 'USER': Icon = User; colorClass = 'text-risk-info'; break;
                    case 'DEVICE': Icon = Smartphone; colorClass = 'text-risk-ai'; break;
                    case 'IP': Icon = Globe; colorClass = 'text-blue-400'; break;
                    case 'PAYMENT': Icon = CreditCard; colorClass = 'text-amber-400'; break;
                    case 'MERCHANT': Icon = ShoppingBag; colorClass = 'text-emerald-400'; break;
                    case 'TRANSACTION': Icon = FileText; colorClass = 'text-rose-400'; break;
                    case 'ADDRESS': Icon = MapPin; colorClass = 'text-indigo-400'; break;
                  }

                  const isCritical = item.riskScore >= 0.90;
                  const isHigh = item.riskScore >= 0.70 && item.riskScore < 0.90;

                  let scoreColor = 'text-risk-low border-risk-low/20 bg-risk-low/10';
                  if (isCritical) scoreColor = 'text-risk-high border-risk-high/30 bg-risk-high/15';
                  else if (isHigh) scoreColor = 'text-risk-medium border-risk-medium/20 bg-risk-medium/10';

                  return (
                    <tr
                      key={item.id}
                      onClick={() => setSelectedEntity(item)}
                      className={`border-b border-border hover:bg-card/50 transition-colors cursor-pointer ${
                        selectedEntity?.id === item.id ? 'bg-sidebar/40' : ''
                      }`}
                    >
                      <td className="p-4 font-bold text-text-primary">{item.id}</td>
                      <td className="p-4">
                        <div className="flex items-center gap-2">
                          <div className={`p-1 rounded bg-border border border-border ${colorClass}`}>
                            <Icon className="w-3.5 h-3.5" />
                          </div>
                          <span className="font-mono text-text-secondary">{item.type}</span>
                        </div>
                      </td>
                      <td className="p-4 text-text-primary font-medium">{item.label}</td>
                      <td className="p-4">
                        <span className={`px-2 py-0.5 rounded border text-[10px] font-mono font-bold ${scoreColor}`}>
                          {Math.round(item.riskScore * 100)}%
                        </span>
                      </td>
                      <td className="p-4 text-text-secondary font-medium">
                        {new Date(item.firstSeen).toLocaleDateString([], { month: 'short', day: '2-digit' })}
                      </td>
                      <td className="p-4 text-right">
                        <button className="text-risk-ai hover:text-text-primary font-bold flex items-center gap-1 ml-auto cursor-pointer transition-colors">
                          <span>Inspect</span>
                          <ChevronRight className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Inspector panel */}
        <div className="lg:col-span-1">
          {selectedEntity ? (
            <div className="bg-card border border-risk-ai/20 rounded-xl p-5 space-y-4 shadow-lg shadow-risk-ai/5 select-none">
              <div className="flex justify-between items-center pb-2 border-b border-border">
                <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-risk-ai animate-pulse"></span>
                  Entity Inspector
                </h3>
                <button
                  onClick={() => setSelectedEntity(null)}
                  className="text-[10px] text-text-secondary hover:text-text-primary cursor-pointer transition-colors"
                >
                  Clear
                </button>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <div className="bg-background p-2 rounded-lg border border-border text-text-primary font-bold font-mono text-xs">
                    {selectedEntity.id}
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-text-primary uppercase leading-none">{selectedEntity.type}</h4>
                    <span className="text-[10px] text-text-secondary font-mono block mt-1">
                      Score: <span className="font-bold text-risk-high">{Math.round(selectedEntity.riskScore * 100)}%</span>
                    </span>
                  </div>
                </div>

                <div className="h-px bg-border my-2"></div>

                <div className="space-y-2 text-xs">
                  <div className="flex flex-col gap-0.5">
                    <span className="text-[10px] text-text-secondary font-mono">First Seen</span>
                    <span className="text-text-primary font-medium">{new Date(selectedEntity.firstSeen).toLocaleString('en-US')}</span>
                  </div>
                  <div className="flex flex-col gap-0.5">
                    <span className="text-[10px] text-text-secondary font-mono">Last Seen</span>
                    <span className="text-text-primary font-medium">{new Date(selectedEntity.lastSeen).toLocaleString('en-US')}</span>
                  </div>

                  {selectedEntity.details && (
                    <div className="mt-3 space-y-2 bg-background border border-border p-3.5 rounded-lg font-mono text-[10px]">
                      <span className="font-bold text-[9px] text-text-secondary block uppercase tracking-wider mb-1.5">Metadata</span>
                      {Object.entries(selectedEntity.details).map(([k, v]) => (
                        <div key={k} className="flex justify-between">
                          <span className="text-text-secondary uppercase">{k}</span>
                          <span className="text-text-primary font-bold truncate max-w-[100px]">{v}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Inspect Entity redirect actions */}
                {selectedEntity.type === 'TRANSACTION' && (
                  <button
                    onClick={() => router.push(`/transactions/${selectedEntity.id}`)}
                    className="w-full mt-2 bg-risk-ai hover:bg-risk-ai/90 text-white font-bold text-xs py-2 rounded-lg cursor-pointer transition-all flex items-center justify-center gap-1.5"
                  >
                    <span>Investigate Transaction</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </button>
                )}
                {selectedEntity.type === 'USER' && selectedEntity.id === 'CUS-2031' && (
                  <button
                    onClick={() => router.push('/rings/AR-1042')}
                    className="w-full mt-2 bg-risk-ai hover:bg-risk-ai/90 text-white font-bold text-xs py-2 rounded-lg cursor-pointer transition-all flex items-center justify-center gap-1.5"
                  >
                    <span>Investigate Abuse Ring</span>
                    <ExternalLink className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-card border border-border border-dashed rounded-xl p-5 text-center text-xs text-text-secondary py-12">
              Select any entity row to inspect telemetry detail records.
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
