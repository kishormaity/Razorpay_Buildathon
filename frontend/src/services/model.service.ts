import { mockModelMetrics, mockBaselineComparison } from '../data/mock/models';
import { ModelMetrics, BaselineComparison } from '../data/types/risk';

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const modelService = {
  async getModelMetrics(): Promise<ModelMetrics[]> {
    await delay(200);
    return [...mockModelMetrics];
  },

  async getBaselineComparison(): Promise<BaselineComparison[]> {
    await delay(150);
    return [...mockBaselineComparison];
  },
};
