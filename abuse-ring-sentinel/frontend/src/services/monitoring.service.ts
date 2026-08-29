import { mockDriftMetrics, mockSystemEvents, mockDataQualityTrends, SystemEvent, DataQualityTrend } from '../data/mock/monitoring';
import { DriftMetric } from '../data/types/risk';

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const monitoringService = {
  async getDriftMetrics(): Promise<DriftMetric[]> {
    await delay(200); // Simulate network latency
    return [...mockDriftMetrics];
  },

  async getSystemEvents(): Promise<SystemEvent[]> {
    await delay(150);
    return [...mockSystemEvents];
  },

  async getDataQualityTrends(): Promise<DataQualityTrend[]> {
    await delay(200);
    return [...mockDataQualityTrends];
  },
};
