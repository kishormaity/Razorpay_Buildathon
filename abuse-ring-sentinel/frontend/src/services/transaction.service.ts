import { Transaction } from '../data/types/transaction';
import { RiskSignal } from '../data/types/risk';

const API_BASE_URL = 'http://127.0.0.1:8001/api';

export const transactionService = {
  async getTransactions(): Promise<Transaction[]> {
    const res = await fetch(`${API_BASE_URL}/alerts`, { cache: 'no-store' });
    if (!res.ok) {
      throw new Error(`Failed to fetch transactions queue: ${res.statusText}`);
    }
    const data = await res.json();
    
    // Map list of alerts to Transaction format
    return data.map((alert: any) => ({
      id: alert.alert_id,
      timestamp: alert.created_at,
      amount: alert.amount,
      currency: 'INR',
      merchant: 'E-Commerce Store',
      paymentMethod: 'Credit Card',
      channel: 'Web API',
      location: 'India',
      riskDecision: {
        score: alert.ring_risk_score,
        confidence: 0.92,
        action: alert.auto_decision as any,
        policyVersion: 'Sentinel-Hybrid-v1',
        expectedCost: {
          fraudLoss: alert.is_abuse === 1 ? alert.amount : 0,
          customerFriction: alert.auto_decision === 'HOLD' ? 1500 : 0,
          manualReview: alert.auto_decision === 'MANUAL_REVIEW' ? 500 : 0,
          totalCost: 0
        }
      },
      signals: [],
      contributions: [],
      evidenceCompleteness: 90,
      evidenceList: [],
      customerId: alert.user_id,
      deviceId: 'DEV-UNKNOWN',
      ipAddress: 'IP-UNKNOWN',
      riskNarrative: `Flagged coordinated account ring member user ID: ${alert.user_id}`,
      status: alert.status,
      notes: ''
    }));
  },

  async getTransaction(id: string): Promise<Transaction | undefined> {
    const res = await fetch(`${API_BASE_URL}/transaction/${id}`, { cache: 'no-store' });
    if (!res.ok) {
      if (res.status === 404) return undefined;
      throw new Error(`Failed to fetch transaction ${id}: ${res.statusText}`);
    }
    const data = await res.json();
    
    // Map full details
    const rf = data.risk_factors;
    const dec = data.decision;
    const exp = data.explanations;
    
    // Extract device/IP from graph links if present
    const devLink = (exp.graph || []).find((g: any) => g.entity_type === 'DEVICE');
    const ipLink = (exp.graph || []).find((g: any) => g.entity_type === 'IP');
    
    const signals: RiskSignal[] = (exp.tabular || []).map((t: any, idx: number) => ({
      id: `sig-${idx}`,
      name: t.feature,
      category: t.feature.includes('device') ? 'DEVICE' : t.feature.includes('count') ? 'VELOCITY' : 'BEHAVIOR',
      description: t.description,
      severity: t.shap_value > 0.4 ? 'HIGH' : 'MEDIUM'
    }));
    
    return {
      id: data.transaction_id,
      timestamp: data.timestamp,
      amount: data.amount,
      currency: 'INR',
      merchant: 'E-Commerce Merchant',
      paymentMethod: 'Credit Card (PMT)',
      channel: 'Web Checkout',
      location: data.user_country,
      status: data.investigation.status,
      notes: data.investigation.notes || '',
      customerId: data.user_id,
      deviceId: devLink ? devLink.entity_id : 'DEV-UNKNOWN',
      ipAddress: ipLink ? ipLink.entity_id : 'IP-UNKNOWN',
      riskNarrative: `This transaction was flagged by the Risk Fusion Engine due to a coordinated ring risk score of ${(rf.r_ring * 100).toFixed(0)}% and a transactional GBDT probability of ${(rf.r_gbm * 100).toFixed(0)}%.`,
      evidenceCompleteness: 95,
      evidenceList: [
        { name: 'Card Fingerprint Registered', checked: true },
        { name: 'Node2Vec Network Distance Scan', checked: true },
        { name: 'Modularity Cluster Partitioned', checked: true }
      ],
      riskDecision: {
        score: rf.r_final,
        confidence: 0.90,
        action: dec.recommended_action,
        policyVersion: 'Sentinel-Hybrid-v1',
        expectedCost: {
          fraudLoss: dec.expected_costs.ALLOW,
          customerFriction: dec.expected_costs.HOLD,
          manualReview: dec.expected_costs.MANUAL_REVIEW,
          totalCost: dec.expected_costs[dec.recommended_action]
        }
      },
      signals,
      contributions: (exp.tabular || []).map((t: any) => ({
        featureName: t.feature,
        weight: t.shap_value
      }))
    };
  },

  async updateTransactionStatus(
    id: string,
    status: Transaction['status'],
    notes?: string
  ): Promise<Transaction> {
    // Send update request to uvicorn endpoint
    const res = await fetch(`${API_BASE_URL}/investigate/${id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        status,
        decision: status === 'CONFIRMED_ABUSE' ? 'CONFIRMED_ABUSE' : status === 'FALSE_POSITIVE' ? 'FALSE_POSITIVE' : 'PENDING_REVIEW',
        notes
      })
    });
    
    if (!res.ok) {
      throw new Error(`Failed to update transaction ${id}: ${res.statusText}`);
    }
    
    // Return updated representation
    const tx = await this.getTransaction(id);
    if (!tx) {
      throw new Error(`Transaction ${id} not found after update.`);
    }
    return tx;
  },

  async getMerchantImpact(): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/merchant/impact`, { cache: 'no-store' });
    if (!res.ok) {
      throw new Error(`Failed to fetch merchant impact: ${res.statusText}`);
    }
    return res.json();
  },

  async getPolicyScorecard(): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/policy/scorecard`, { cache: 'no-store' });
    if (!res.ok) {
      throw new Error(`Failed to fetch policy scorecard: ${res.statusText}`);
    }
    return res.json();
  },

  async getDemoCases(): Promise<any> {
    const res = await fetch(`${API_BASE_URL}/demo/cases`, { cache: 'no-store' });
    if (!res.ok) {
      throw new Error(`Failed to fetch demo cases: ${res.statusText}`);
    }
    return res.json();
  }
};


