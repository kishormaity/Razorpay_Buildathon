import { PolicyRule } from '../types/risk';

export const mockPolicyRules: PolicyRule[] = [
  {
    id: 'POL-LOW',
    name: 'Standard Low Risk Pass',
    riskRange: [0.00, 0.40],
    action: 'ALLOW',
    isActive: true,
  },
  {
    id: 'POL-MED',
    name: 'Suspicious Activity Step-Up',
    riskRange: [0.40, 0.70],
    action: 'STEP_UP',
    isActive: true,
  },
  {
    id: 'POL-HIGH',
    name: 'High Risk Escalation Queue',
    riskRange: [0.70, 0.90],
    action: 'MANUAL_REVIEW',
    isActive: true,
  },
  {
    id: 'POL-CRIT',
    name: 'Coordinated Abuse Temporary Hold',
    riskRange: [0.90, 1.00],
    action: 'HOLD',
    isActive: true,
  }
];

export interface PolicySimulationPoint {
  threshold: number;
  fraudLossCost: number;
  frictionCost: number;
  reviewCost: number;
  totalCost: number;
}

export const mockPolicySimulationData: PolicySimulationPoint[] = [
  { threshold: 0.0, fraudLossCost: 0, frictionCost: 290000, reviewCost: 65000, totalCost: 355000 },
  { threshold: 0.1, fraudLossCost: 15000, frictionCost: 240000, reviewCost: 55000, totalCost: 310000 },
  { threshold: 0.2, fraudLossCost: 35000, frictionCost: 190000, reviewCost: 45000, totalCost: 270000 },
  { threshold: 0.3, fraudLossCost: 60000, frictionCost: 140000, reviewCost: 35000, totalCost: 235000 },
  { threshold: 0.4, fraudLossCost: 95000, frictionCost: 95000, reviewCost: 28000, totalCost: 218000 },
  { threshold: 0.5, fraudLossCost: 120000, frictionCost: 60000, reviewCost: 20000, totalCost: 200000 },
  { threshold: 0.55, fraudLossCost: 135000, frictionCost: 48000, reviewCost: 17000, totalCost: 200000 }, // Sweet spot threshold 0.55
  { threshold: 0.6, fraudLossCost: 160000, frictionCost: 40000, reviewCost: 15000, totalCost: 215000 },
  { threshold: 0.7, fraudLossCost: 220000, frictionCost: 22000, reviewCost: 10000, totalCost: 252000 },
  { threshold: 0.8, fraudLossCost: 310000, frictionCost: 10000, reviewCost: 6000, totalCost: 326000 },
  { threshold: 0.9, fraudLossCost: 440000, frictionCost: 2000, reviewCost: 2000, totalCost: 444000 },
  { threshold: 1.0, fraudLossCost: 620000, frictionCost: 0, reviewCost: 0, totalCost: 620000 },
];
