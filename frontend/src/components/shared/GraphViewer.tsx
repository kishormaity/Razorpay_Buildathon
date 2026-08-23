'use client';

import React, { useMemo } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  Handle,
  Position,
  NodeProps,
  Edge
} from '@xyflow/react';
import {
  User,
  Smartphone,
  Globe,
  CreditCard,
  ShoppingBag,
  FileText,
  MapPin,
  ShieldAlert
} from 'lucide-react';
import { Entity, EntityRelationship } from '../../data/types/risk';
import '@xyflow/react/dist/style.css';

// Register custom node component
function CustomNode({ data }: { data: any }) {
  const entity = data.entity as Entity;
  
  // Decide icon based on entity type
  let IconComponent = User;
  let labelColor = 'text-risk-info';

  switch (entity.type) {
    case 'USER':
      IconComponent = User;
      labelColor = 'text-risk-info';
      break;
    case 'DEVICE':
      IconComponent = Smartphone;
      labelColor = 'text-risk-ai';
      break;
    case 'IP':
      IconComponent = Globe;
      labelColor = 'text-blue-400';
      break;
    case 'PAYMENT':
      IconComponent = CreditCard;
      labelColor = 'text-amber-400';
      break;
    case 'MERCHANT':
      IconComponent = ShoppingBag;
      labelColor = 'text-emerald-400';
      break;
    case 'TRANSACTION':
      IconComponent = FileText;
      labelColor = 'text-rose-400';
      break;
    case 'ADDRESS':
      IconComponent = MapPin;
      labelColor = 'text-indigo-400';
      break;
  }

  // Border and glow colors based on risk
  const score = entity.riskScore;
  let ringClass = 'border-risk-low/20 shadow-[0_0_8px_rgba(16,185,129,0.05)]';
  let badgeColor = 'bg-risk-low/10 text-risk-low border-risk-low/20';

  if (score >= 0.90) {
    ringClass = 'border-risk-high shadow-[0_0_12px_rgba(239,68,68,0.15)] animate-pulse';
    badgeColor = 'bg-risk-high/10 text-risk-high border-risk-high/20';
  } else if (score >= 0.70) {
    ringClass = 'border-risk-medium/70 shadow-[0_0_10px_rgba(249,115,22,0.1)]';
    badgeColor = 'bg-risk-medium/10 text-risk-medium border-risk-medium/20';
  } else if (score >= 0.40) {
    ringClass = 'border-risk-medium/40 shadow-[0_0_8px_rgba(245,158,11,0.05)]';
    badgeColor = 'bg-risk-medium/10 text-risk-medium border-risk-medium/20';
  }

  return (
    <div className={`px-3.5 py-2.5 bg-card border rounded-lg flex items-center gap-2.5 min-w-[150px] transition-all hover:scale-102 ${ringClass}`}>
      <Handle type="target" position={Position.Top} className="opacity-0" />
      
      <div className={`p-1.5 rounded bg-border border border-border ${labelColor}`}>
        <IconComponent className="w-4 h-4" />
      </div>

      <div className="flex flex-col gap-0.5 leading-none">
        <span className="text-[10px] font-bold text-text-primary whitespace-nowrap overflow-hidden text-ellipsis max-w-[120px]">
          {entity.id}
        </span>
        <span className="text-[9px] text-text-secondary whitespace-nowrap overflow-hidden text-ellipsis max-w-[120px]">
          {entity.type}
        </span>
      </div>

      <span className={`text-[8px] font-mono font-extrabold px-1.5 py-0.5 rounded border ml-auto ${badgeColor}`}>
        {Math.round(score * 100)}%
      </span>

      <Handle type="source" position={Position.Bottom} className="opacity-0" />
    </div>
  );
}

// Bind custom node registration
const nodeTypes = {
  custom: CustomNode,
};

interface GraphViewerProps {
  entities: Entity[];
  relationships: EntityRelationship[];
  selectedNodeId: string | null;
  onSelectNode: (entity: Entity) => void;
  timeFilter: string; // '1h' | '24h' | '7d' | 'all'
}

export default function GraphViewer({
  entities,
  relationships,
  selectedNodeId,
  onSelectNode,
  timeFilter
}: GraphViewerProps) {
  
  // Custom layout algorithm to position nodes radially based on type
  const processedNodes = useMemo(() => {
    // Separate by entity type
    const devices = entities.filter((e) => e.type === 'DEVICE');
    const ips = entities.filter((e) => e.type === 'IP');
    const users = entities.filter((e) => e.type === 'USER');
    const rest = entities.filter(
      (e) => e.type !== 'DEVICE' && e.type !== 'IP' && e.type !== 'USER'
    );

    const layoutNodes: any[] = [];

    // Place devices at the very center (radial spread if multiple)
    devices.forEach((entity, idx) => {
      const angle = (idx / devices.length) * 2 * Math.PI;
      const r = devices.length > 1 ? 50 : 0;
      layoutNodes.push({
        id: entity.id,
        type: 'custom',
        position: { x: Math.cos(angle) * r + 200, y: Math.sin(angle) * r + 200 },
        data: { entity },
      });
    });

    // Place IPs in inner-middle ring
    ips.forEach((entity, idx) => {
      const angle = (idx / ips.length) * 2 * Math.PI + Math.PI / 4;
      layoutNodes.push({
        id: entity.id,
        type: 'custom',
        position: { x: Math.cos(angle) * 160 + 200, y: Math.sin(angle) * 160 + 200 },
        data: { entity },
      });
    });

    // Place Users in outer-middle ring
    users.forEach((entity, idx) => {
      const angle = (idx / users.length) * 2 * Math.PI;
      layoutNodes.push({
        id: entity.id,
        type: 'custom',
        position: { x: Math.cos(angle) * 300 + 200, y: Math.sin(angle) * 300 + 200 },
        data: { entity },
      });
    });

    // Place other entities (transactions, merchants, payments, etc.) in the outermost ring
    rest.forEach((entity, idx) => {
      const angle = (idx / rest.length) * 2 * Math.PI + Math.PI / 6;
      layoutNodes.push({
        id: entity.id,
        type: 'custom',
        position: { x: Math.cos(angle) * 440 + 200, y: Math.sin(angle) * 440 + 200 },
        data: { entity },
      });
    });

    return layoutNodes;
  }, [entities]);

  // Process relationships to create edges, filtering by time context if needed
  const processedEdges = useMemo(() => {
    // In our prototype, we simulate temporal filtering by selecting fewer links for '1h' and '24h'
    let filteredLinks = [...relationships];
    if (timeFilter === '1h') {
      // Show only current transactions and dev ties
      filteredLinks = relationships.filter(
        (r) => r.type === 'MADE_TRANSACTION' || r.sourceId === 'CUS-2031'
      );
    } else if (timeFilter === '24h') {
      // Exclude older user links
      filteredLinks = relationships.filter((r) => r.sourceId !== 'CUS-2034');
    }

    return filteredLinks.map((rel) => {
      const isSelected =
        selectedNodeId === rel.sourceId || selectedNodeId === rel.targetId;
      
      const isCriticalStrength = rel.strength >= 0.90;

      return {
        id: rel.id,
        source: rel.sourceId,
        target: rel.targetId,
        animated: isSelected || isCriticalStrength,
        style: {
          stroke: isSelected ? '#a855f7' : isCriticalStrength ? '#f97316' : '#252a34',
          strokeWidth: isSelected ? 3 : 2,
        },
      } as Edge;
    });
  }, [relationships, selectedNodeId, timeFilter]);

  const onNodeClick = (_: any, node: any) => {
    onSelectNode(node.data.entity);
  };

  return (
    <div className="w-full h-full bg-background border border-border rounded-xl overflow-hidden relative select-none">
      <ReactFlow
        nodes={processedNodes}
        edges={processedEdges}
        nodeTypes={nodeTypes}
        onNodeClick={onNodeClick}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.2}
        maxZoom={2}
        className="w-full h-full"
      >
        <Background color="#252a34" gap={16} size={1} />
        <Controls showInteractive={false} className="react-flow__controls" />
      </ReactFlow>

      {/* Mini Legend overlay */}
      <div className="absolute bottom-4 right-4 bg-card/90 backdrop-blur border border-border p-3.5 rounded-lg flex flex-col gap-2 shadow-lg text-[9px] font-mono text-text-secondary z-10 pointer-events-none">
        <span className="font-bold text-[10px] text-text-primary mb-1 uppercase tracking-wider">Node Guide</span>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-border border border-border flex items-center justify-center text-risk-info">
              <User className="w-2.5 h-2.5" />
            </div>
            <span>User</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-border border border-border flex items-center justify-center text-risk-ai">
              <Smartphone className="w-2.5 h-2.5" />
            </div>
            <span>Device</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-border border border-border flex items-center justify-center text-blue-400">
              <Globe className="w-2.5 h-2.5" />
            </div>
            <span>IP Address</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-border border border-border flex items-center justify-center text-amber-400">
              <CreditCard className="w-2.5 h-2.5" />
            </div>
            <span>Payment</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-border border border-border flex items-center justify-center text-rose-400">
              <FileText className="w-2.5 h-2.5" />
            </div>
            <span>Transaction</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded bg-border border border-border flex items-center justify-center text-emerald-400">
              <ShoppingBag className="w-2.5 h-2.5" />
            </div>
            <span>Merchant</span>
          </div>
        </div>
        <div className="h-px bg-border my-1"></div>
        <div className="flex items-center gap-1.5 text-risk-high font-bold">
          <ShieldAlert className="w-3.5 h-3.5" />
          <span>Glow indicates Critical Risk</span>
        </div>
      </div>
    </div>
  );
}
