import { ModelMetrics, BaselineComparison } from '../types/risk';

export const mockModelMetrics: ModelMetrics[] = [
  {
    id: 'MOD-TXN-RISK',
    name: 'Transaction Risk Model',
    version: 'v2.4-hybrid',
    status: 'ACTIVE',
    precision: 0.912,
    recall: 0.876,
    prAuc: 0.898,
    f1Score: 0.893,
    fpr: 0.048,
    latency: 82,
    drift: 0.02,
    evaluationDate: '2026-08-20',
  },
  {
    id: 'MOD-GRAPH-RING',
    name: 'Graph Abuse Ring Model',
    version: 'v1.8-graph-sage',
    status: 'ACTIVE',
    precision: 0.934,
    recall: 0.842,
    prAuc: 0.912,
    f1Score: 0.885,
    fpr: 0.035,
    latency: 145,
    drift: 0.05,
    evaluationDate: '2026-08-20',
  },
  {
    id: 'MOD-RISK-FUSION',
    name: 'Risk Fusion Ensemble',
    version: 'v3.0-ensemble',
    status: 'SHADOW',
    precision: 0.941,
    recall: 0.882,
    prAuc: 0.925,
    f1Score: 0.910,
    fpr: 0.039,
    latency: 210,
    drift: 0.01,
    evaluationDate: '2026-08-20',
  }
];

export const mockBaselineComparison: BaselineComparison[] = [
  {
    metric: 'Model Precision',
    ruleEngine: '74.2%',
    aiEngine: '91.2%',
    impact: '+17.0% efficiency',
  },
  {
    metric: 'Model Recall (Detection Rate)',
    ruleEngine: '61.5%',
    aiEngine: '87.6%',
    impact: '+26.1% caught fraud',
  },
  {
    metric: 'PR-AUC (Model Robustness)',
    ruleEngine: '62.1%',
    aiEngine: '89.8%',
    impact: '+27.7% score separation',
  },
  {
    metric: 'False Positive Rate (Customer Friction)',
    ruleEngine: '11.4%',
    aiEngine: '4.8%',
    impact: '-6.6% fewer false alarms',
    highlightRed: false, // lower is better
  },
  {
    metric: 'Estimated Monthly Fraud Loss',
    ruleEngine: '₹14.8L',
    aiEngine: '₹3.2L',
    impact: '-78.3% loss reduction',
  },
  {
    metric: 'Manual Review Volume (Analyst Load)',
    ruleEngine: '24.5%',
    aiEngine: '8.2%',
    impact: '-66.5% lower queue size',
  }
];
