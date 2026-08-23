import { useQuery } from '@tanstack/react-query';
import { ringService } from '../services/ring.service';
import { RiskRing } from '../data/types/ring';

export function useRiskRingsQuery() {
  return useQuery<RiskRing[]>({
    queryKey: ['rings'],
    queryFn: () => ringService.getRiskRings(),
  });
}

export function useRiskRingQuery(id: string) {
  return useQuery<RiskRing | undefined>({
    queryKey: ['ring', id],
    queryFn: () => ringService.getRiskRing(id),
    enabled: !!id,
  });
}
