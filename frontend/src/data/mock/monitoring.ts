import { DriftMetric, RiskEvent } from '../types/risk';

export const mockDriftMetrics: DriftMetric[] = [
  { timestamp: '08-17', predictionDrift: 0.012, featureDrift: 0.015 },
  { timestamp: '08-18', predictionDrift: 0.014, featureDrift: 0.018 },
  { timestamp: '08-19', predictionDrift: 0.015, featureDrift: 0.022 },
  { timestamp: '08-20', predictionDrift: 0.019, featureDrift: 0.028 },
  { timestamp: '08-21', predictionDrift: 0.024, featureDrift: 0.035 },
  { timestamp: '08-22', predictionDrift: 0.038, featureDrift: 0.052 }, // drift rising slightly
  { timestamp: '08-23', predictionDrift: 0.045, featureDrift: 0.058 }, // current day
];

export interface SystemEvent {
  id: string;
  timestamp: string;
  message: string;
  type: 'CRITICAL' | 'WARNING' | 'INFO';
  source: string;
}

export const mockSystemEvents: SystemEvent[] = [
  {
    id: 'EVT-301',
    timestamp: '2026-08-23T00:51:00Z',
    message: 'Coordinated abuse ring AR-1042 detected on 4 fresh account profiles',
    type: 'CRITICAL',
    source: 'Graph Ring Agent',
  },
  {
    id: 'EVT-302',
    timestamp: '2026-08-23T00:34:00Z',
    message: 'Transaction risk volume spike observed on checkout channel Electronics Direct',
    type: 'WARNING',
    source: 'Anomaly Sentinel',
  },
  {
    id: 'EVT-303',
    timestamp: '2026-08-23T00:05:00Z',
    message: 'Feature Drift warning: device-linking density index crossed 0.05 threshold',
    type: 'WARNING',
    source: 'Drift Monitor',
  },
  {
    id: 'EVT-304',
    timestamp: '2026-08-22T22:30:00Z',
    message: 'Model calibration evaluation: PR-AUC remains stable at 89.8% (Target: 88%)',
    type: 'INFO',
    source: 'MLOps Evaluator',
  },
  {
    id: 'EVT-305',
    timestamp: '2026-08-22T20:10:00Z',
    message: 'Policy threshold v2.4 simulation executed. Sweet spot calibrated at 0.55',
    type: 'INFO',
    source: 'Policy Configurator',
  },
];

export interface DataQualityTrend {
  timestamp: string;
  deviceSharingRate: number;
  velocityZScore: number;
  accountAgeMedianDays: number;
  paymentOverlapRatio: number;
}

export const mockDataQualityTrends: DataQualityTrend[] = [
  { timestamp: '00:00', deviceSharingRate: 1.2, velocityZScore: 0.4, accountAgeMedianDays: 45, paymentOverlapRatio: 0.02 },
  { timestamp: '04:00', deviceSharingRate: 1.3, velocityZScore: 0.5, accountAgeMedianDays: 43, paymentOverlapRatio: 0.02 },
  { timestamp: '08:00', deviceSharingRate: 1.4, velocityZScore: 0.8, accountAgeMedianDays: 38, paymentOverlapRatio: 0.03 },
  { timestamp: '12:00', deviceSharingRate: 2.1, velocityZScore: 1.4, accountAgeMedianDays: 22, paymentOverlapRatio: 0.06 }, // elevation start
  { timestamp: '16:00', deviceSharingRate: 2.8, velocityZScore: 2.1, accountAgeMedianDays: 12, paymentOverlapRatio: 0.12 }, // spike!
  { timestamp: '20:00', deviceSharingRate: 3.4, velocityZScore: 2.8, accountAgeMedianDays: 4, paymentOverlapRatio: 0.19 },  // ring active
  { timestamp: '24:00', deviceSharingRate: 3.2, velocityZScore: 2.5, accountAgeMedianDays: 5, paymentOverlapRatio: 0.17 },
];
