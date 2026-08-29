import { useQuery } from '@tanstack/react-query';
import { modelService } from '../services/model.service';
import { ModelMetrics, BaselineComparison } from '../data/types/risk';

export function useModelMetricsQuery() {
  return useQuery<ModelMetrics[]>({
    queryKey: ['modelMetrics'],
    queryFn: () => modelService.getModelMetrics(),
  });
}

export function useBaselineComparisonQuery() {
  return useQuery<BaselineComparison[]>({
    queryKey: ['baselineComparison'],
    queryFn: () => modelService.getBaselineComparison(),
  });
}
