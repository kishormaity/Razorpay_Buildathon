'use client';

import React, { useState, useEffect } from 'react';
import { Search, Bell, User, Command } from 'lucide-react';
import SearchPalette from '../shared/SearchPalette';
import NotificationDrawer from '../shared/NotificationDrawer';

export default function Header() {
  const [searchOpen, setSearchOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [unreadCount, setUnreadCount] = useState(3);

  // Bind Ctrl+K / Cmd+K search shortcut keys
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setSearchOpen((open) => !open);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <header className="h-16 border-b border-border bg-sidebar/55 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-40 select-none">
      {/* Search Input Trigger */}
      <div className="flex-1 max-w-md">
        <button
          onClick={() => setSearchOpen(true)}
          className="w-full flex items-center justify-between bg-card/65 hover:bg-card border border-border hover:border-text-secondary/30 px-3.5 py-1.5 rounded-lg text-xs text-text-secondary transition-all cursor-pointer text-left"
        >
          <div className="flex items-center gap-2">
            <Search className="w-3.5 h-3.5 text-text-secondary" />
            <span>Search transaction, device, IP, user, ring...</span>
          </div>
          <div className="flex items-center gap-1 font-mono text-[10px] bg-border px-1.5 py-0.5 rounded text-text-secondary border border-border">
            <Command className="w-2.5 h-2.5" />
            <span>K</span>
          </div>
        </button>
      </div>

      {/* Control Triggers & User Card */}
      <div className="flex items-center gap-4">
        {/* Notifications Icon Button */}
        <button
          onClick={() => {
            setNotificationsOpen(true);
            setUnreadCount(0);
          }}
          className="relative p-2 rounded-lg hover:bg-card text-text-secondary hover:text-text-primary transition-all cursor-pointer border border-transparent hover:border-border"
        >
          <Bell className="w-4 h-4" />
          {unreadCount > 0 && (
            <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-risk-high animate-pulse"></span>
          )}
        </button>

        <div className="h-4 w-px bg-border"></div>

        {/* Analyst Profile */}
        <div className="flex items-center gap-3">
          <div className="flex flex-col items-end leading-none">
            <span className="text-xs font-semibold text-text-primary">Arjun Mehta</span>
            <span className="text-[10px] text-text-secondary font-mono tracking-wider mt-0.5 uppercase">Risk Lead</span>
          </div>
          <div className="bg-risk-ai/10 p-2 rounded-lg border border-risk-ai/20 text-risk-ai">
            <User className="w-4 h-4" />
          </div>
        </div>
      </div>

      {/* Overlaid UI Modals */}
      <SearchPalette isOpen={searchOpen} onClose={() => setSearchOpen(false)} />
      <NotificationDrawer isOpen={notificationsOpen} onClose={() => setNotificationsOpen(false)} />
    </header>
  );
}
