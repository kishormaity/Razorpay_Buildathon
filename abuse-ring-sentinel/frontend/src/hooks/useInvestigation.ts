import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { investigationService } from '../services/investigation.service';
import { Investigation } from '../data/types/investigation';

export function useInvestigationsQuery() {
  return useQuery<Investigation[]>({
    queryKey: ['investigations'],
    queryFn: () => investigationService.getInvestigations(),
  });
}

export function useInvestigationQuery(id: string) {
  return useQuery<Investigation | undefined>({
    queryKey: ['investigation', id],
    queryFn: () => investigationService.getInvestigation(id),
    enabled: !!id,
  });
}

export function useUpdateInvestigationStatusMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, status, notes }: { id: string; status: Investigation['status']; notes?: string }) =>
      investigationService.updateInvestigationStatus(id, status, notes),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['investigations'] });
      queryClient.invalidateQueries({ queryKey: ['investigation', variables.id] });
      queryClient.invalidateQueries({ queryKey: ['transactions'] });
      queryClient.invalidateQueries({ queryKey: ['transaction'] });
    },
  });
}
