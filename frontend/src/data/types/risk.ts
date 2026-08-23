export interface CostEstimate {
  fraudLoss: number;
  customerFriction: number;
  manualReview: number;
  totalCost: number;
}

export interface RiskDecision {
  score: number;             // 0.0 to 1.0
  confidence: number;        // 0.0 to 1.0
  action: 'ALLOW' | 'STEP_UP' | 'MANUAL_REVIEW' | 'HOLD' | 'BLOCK';
  policyVersion: string;
  expectedCost: CostEstimate;
}

export interface RiskSignal {
  id: string;
  name: string;
  category: 'DEVICE' | 'VELOCITY' | 'BEHAVIOR' | 'NETWORK' | 'HISTORY';
  description: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

export interface ModelContribution {
  featureName: string;
  weight: number;            // e.g. +0.21 or -0.05
}

export interface Entity {
  id: string;
  type: 'USER' | 'DEVICE' | 'IP' | 'PAYMENT' | 'MERCHANT' | 'ADDRESS' | 'TRANSACTION';
  label: string;
  riskScore: number;
  firstSeen: string;
  lastSeen: string;
  details?: Record<string, string | number>;
}

export interface EntityRelationship {
  id: string;
  sourceId: string;
  targetId: string;
  type: 'USED_BY' | 'ASSOCIATED_WITH' | 'SHARED_IP' | 'LINKED_PAYMENT' | 'MADE_TRANSACTION';
  strength: number;          // 0 to 1
  firstSeen: string;
  lastSeen: string;
}

export interface RiskEvent {
  id: string;
  title: string;
  timestamp: string;
  riskScore: number;
  status: 'UNRESOLVED' | 'RESOLVED';
}

export interface ModelMetrics {
  id: string;
  name: string;
  version: string;
  status: 'ACTIVE' | 'SHADOW' | 'INACTIVE';
  precision: number;
  recall: number;
  prAuc: number;
  f1Score: number;
  fpr: number;
  latency: number; // in ms
  drift: number; // 0 to 1
  evaluationDate: string;
}

export interface BaselineComparison {
  metric: string;
  ruleEngine: string;
  aiEngine: string;
  impact: string;
  highlightRed?: boolean;
}

export interface PolicyRule {
  id: string;
  name: string;
  riskRange: [number, number]; // [min, max]
  action: 'ALLOW' | 'STEP_UP' | 'MANUAL_REVIEW' | 'HOLD' | 'BLOCK';
  isActive: boolean;
}

export interface DriftMetric {
  timestamp: string;
  predictionDrift: number;
  featureDrift: number;
}

