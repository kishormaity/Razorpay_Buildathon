import { RiskDecision, RiskSignal, ModelContribution } from './risk';

export interface Evidence {
  name: string;
  checked: boolean;
}

export interface Transaction {
  id: string;
  timestamp: string;
  amount: number;
  currency: string;
  merchant: string;
  paymentMethod: string;
  channel: string;
  location: string;
  riskDecision: RiskDecision;
  signals: RiskSignal[];
  contributions: ModelContribution[];
  evidenceCompleteness: number; // percentage, e.g., 94
  evidenceList: Evidence[];
  customerId: string;
  deviceId: string;
  ipAddress: string;
  riskNarrative: string; // Analyst-friendly natural language explanation
  status: 'PENDING_REVIEW' | 'REVIEW_COMPLETED' | 'CONFIRMED_ABUSE' | 'FALSE_POSITIVE' | 'DISMISSED';
  notes?: string; // Analyst notes entered during workspace investigation
}
