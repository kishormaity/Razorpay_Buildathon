'use client';

import React, { useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { Bell, X, AlertOctagon, AlertTriangle, Info, ArrowRight } from 'lucide-react';

interface NotificationItem {
  id: string;
  timestamp: string;
  message: string;
  timeAgo: string;
  type: 'CRITICAL' | 'WARNING' | 'INFO';
  url?: string;
}

export default function NotificationDrawer({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const router = useRouter();
  const drawerRef = useRef<HTMLDivElement>(null);

  const notifications: NotificationItem[] = [
    {
      id: 'N-1',
      timestamp: '2026-08-23T00:51:00Z',
      timeAgo: '15 min ago',
      message: 'Critical abuse ring AR-1042 detected linking 8 devices & 17 accounts',
      type: 'CRITICAL',
      url: '/rings/AR-1042',
    },
    {
      id: 'N-2',
      timestamp: '2026-08-23T00:34:00Z',
      timeAgo: '32 min ago',
      message: 'Coordinated velocity spike detected on Electronics Direct merchant checkout',
      type: 'WARNING',
      url: '/transactions/TXN-8291',
    },
    {
      id: 'N-3',
      timestamp: '2026-08-23T00:05:00Z',
      timeAgo: '1 hr ago',
      message: 'Model False-Positive rate crossed threshold: 4.8% vs target 4.0%',
      type: 'WARNING',
      url: '/models',
    },
    {
      id: 'N-4',
      timestamp: '2026-08-22T22:30:00Z',
      timeAgo: '2 hr ago',
      message: 'Risk Fusion Ensemble model calibration run succeeded. Accuracy: 94.1%',
      type: 'INFO',
      url: '/models',
    },
  ];

  // Click outside drawer to close
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (drawerRef.current && !drawerRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleOutsideClick);
    }
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, [isOpen, onClose]);

  // Handle escape key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-background/50 backdrop-blur-sm z-50 flex justify-end select-none">
      <div
        ref={drawerRef}
        className="w-full max-w-md bg-sidebar border-l border-border h-full flex flex-col shadow-2xl relative"
      >
        {/* Drawer Header */}
        <div className="p-6 border-b border-border flex items-center justify-between bg-background/25">
          <div className="flex items-center gap-2">
            <Bell className="w-4 h-4 text-risk-ai" />
            <h2 className="text-sm font-bold text-text-primary uppercase tracking-wider">Risk System Alerts</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-card text-text-secondary hover:text-text-primary transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Drawer Notifications List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {notifications.map((item) => {
            let Icon = Info;
            let iconColor = 'text-risk-info';
            let bgClass = 'bg-risk-info/5 border-risk-info/10';

            if (item.type === 'CRITICAL') {
              Icon = AlertOctagon;
              iconColor = 'text-risk-high';
              bgClass = 'bg-risk-high/5 border-risk-high/15';
            } else if (item.type === 'WARNING') {
              Icon = AlertTriangle;
              iconColor = 'text-risk-medium';
              bgClass = 'bg-risk-medium/5 border-risk-medium/15';
            }

            return (
              <div
                key={item.id}
                onClick={() => {
                  if (item.url) {
                    router.push(item.url);
                    onClose();
                  }
                }}
                className={`p-4 rounded-xl border ${bgClass} hover:bg-card hover:border-border transition-all flex flex-col gap-3 group cursor-pointer`}
              >
                <div className="flex gap-3">
                  <div className="mt-0.5">
                    <Icon className={`w-4 h-4 ${iconColor}`} />
                  </div>
                  <div className="flex-1 flex flex-col gap-1">
                    <div className="flex justify-between items-center gap-2">
                      <span className="text-[10px] font-bold tracking-widest text-text-secondary uppercase">
                        {item.type} ALERT
                      </span>
                      <span className="text-[9px] font-mono text-text-secondary">
                        {item.timeAgo}
                      </span>
                    </div>
                    <p className="text-xs font-medium text-text-primary leading-normal">
                      {item.message}
                    </p>
                  </div>
                </div>

                {item.url && (
                  <div className="flex items-center justify-end text-[10px] text-risk-ai font-semibold gap-1 opacity-80 group-hover:opacity-100 transition-opacity">
                    <span>Investigate</span>
                    <ArrowRight className="w-3 h-3 transition-transform group-hover:translate-x-0.5" />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Clear/Mute Settings footer */}
        <div className="p-4 border-t border-border bg-background/25 flex items-center justify-between text-[10px] text-text-secondary font-mono">
          <span>Active monitors: 4</span>
          <button className="hover:text-text-primary transition-colors cursor-pointer">
            Mark all read
          </button>
        </div>
      </div>
    </div>
  );
}
