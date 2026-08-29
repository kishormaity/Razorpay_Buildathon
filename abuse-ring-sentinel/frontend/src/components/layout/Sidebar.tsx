'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Shield,
  Activity,
  Layers,
  Settings,
  Users,
  BarChart2,
  Sliders,
  Database,
  AlertTriangle,
  HelpCircle
} from 'lucide-react';

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<any>;
}

interface NavGroup {
  title: string;
  items: NavItem[];
}

export default function Sidebar() {
  const pathname = usePathname();

  const navigationGroups: NavGroup[] = [
    {
      title: 'OVERVIEW',
      items: [
        { name: 'Command Center', href: '/dashboard', icon: Activity },
      ],
    },
    {
      title: 'OPERATIONS',
      items: [
        { name: 'Transaction Logs', href: '/transactions', icon: Database },
        { name: 'Investigation Queue', href: '/investigations', icon: Users },
        { name: 'Coordinated Rings', href: '/rings', icon: Layers },
      ],
    },
    {
      title: 'ANALYTICS',
      items: [
        { name: 'Entity Explorer', href: '/entities', icon: Shield },
        { name: 'Model Performance', href: '/models', icon: BarChart2 },
        { name: 'Health Monitoring', href: '/monitoring', icon: Settings },
      ],
    },
    {
      title: 'CONFIGURATION',
      items: [
        { name: 'Policy Settings', href: '/policies', icon: Sliders },
      ],
    },
  ];

  // System State simulation
  const systemRiskLevel = 'ELEVATED'; // 'NORMAL' | 'ELEVATED'
  const systemAlertText = 'Fraud spike detected (2.4x baseline)';

  const isActive = (href: string) => {
    if (href === '/dashboard' && pathname === '/') return true;
    return pathname.startsWith(href);
  };

  return (
    <aside className="w-64 bg-sidebar border-r border-border flex flex-col h-full select-none">
      {/* Brand Header */}
      <div className="p-6 border-b border-border flex items-center gap-3">
        <div className="bg-risk-ai/20 p-2 rounded-lg border border-risk-ai/30">
          <Shield className="w-5 h-5 text-risk-ai" />
        </div>
        <div>
          <h1 className="text-sm font-bold text-text-primary tracking-wider leading-none">SENTINEL</h1>
          <span className="text-[10px] text-text-secondary font-mono tracking-widest uppercase">AI Risk Manager</span>
        </div>
      </div>

      {/* System-Wide Risk Status */}
      <div className="px-4 py-3 border-b border-border">
        <div className={`p-2.5 rounded-lg border flex flex-col gap-1.5 ${
          systemRiskLevel === 'ELEVATED' 
            ? 'bg-risk-medium/10 border-risk-medium/20' 
            : 'bg-risk-low/10 border-risk-low/20'
        }`}>
          <div className="flex items-center gap-1.5 justify-between">
            <span className="text-[10px] font-bold tracking-wider text-text-secondary">SYSTEM STATUS</span>
            <span className="relative flex h-2 w-2">
              <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                systemRiskLevel === 'ELEVATED' ? 'bg-risk-medium' : 'bg-risk-low'
              }`}></span>
              <span className={`relative inline-flex rounded-full h-2 w-2 ${
                systemRiskLevel === 'ELEVATED' ? 'bg-risk-medium' : 'bg-risk-low'
              }`}></span>
            </span>
          </div>
          <div className="flex items-center gap-1">
            <AlertTriangle className={`w-3.5 h-3.5 ${
              systemRiskLevel === 'ELEVATED' ? 'text-risk-medium' : 'text-risk-low'
            }`} />
            <span className="text-[11px] font-medium text-text-primary truncate">
              {systemRiskLevel === 'ELEVATED' ? 'ELEVATED RISK LEVEL' : 'NORMAL ACTIVITY'}
            </span>
          </div>
          {systemRiskLevel === 'ELEVATED' && (
            <p className="text-[10px] text-risk-medium/95 leading-snug">
              {systemAlertText}
            </p>
          )}
        </div>
      </div>

      {/* Navigation Grouping */}
      <div className="flex-1 py-4 overflow-y-auto px-3 space-y-6">
        {navigationGroups.map((group) => (
          <div key={group.title} className="space-y-1.5">
            <span className="px-3 text-[10px] font-bold text-text-secondary tracking-widest">
              {group.title}
            </span>
            <ul className="space-y-0.5">
              {group.items.map((item) => {
                const active = isActive(item.href);
                const Icon = item.icon;
                return (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      className={`flex items-center gap-3 px-3 py-2 rounded-md text-xs font-medium transition-all ${
                        active
                          ? 'bg-card text-text-primary border-l-2 border-risk-ai shadow-sm font-semibold'
                          : 'text-text-secondary hover:text-text-primary hover:bg-card/50'
                      }`}
                    >
                      <Icon className={`w-4 h-4 transition-colors ${
                        active ? 'text-risk-ai' : 'text-text-secondary group-hover:text-text-primary'
                      }`} />
                      <span>{item.name}</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </div>

      {/* Data Source Info Footnote */}
      <div className="p-4 border-t border-border bg-background/40 flex flex-col gap-2">
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-1.5 rounded-full bg-risk-info animate-pulse"></div>
          <span className="text-[10px] text-text-secondary font-mono uppercase tracking-wider">
            Source: Demo / Synthetic
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] text-text-secondary hover:text-text-primary transition-colors cursor-pointer">
          <HelpCircle className="w-3.5 h-3.5" />
          <span>Documentation Reference</span>
        </div>
      </div>
    </aside>
  );
}
