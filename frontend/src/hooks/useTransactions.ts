import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { transactionService } from '../services/transaction.service';
import { Transaction } from '../data/types/transaction';

export function useTransactionsQuery() {
  return useQuery<Transaction[]>({
    queryKey: ['transactions'],
    queryFn: () => transactionService.getTransactions(),
  });
}

export function useTransactionQuery(id: string) {
  return useQuery<Transaction | undefined>({
    queryKey: ['transaction', id],
    queryFn: () => transactionService.getTransaction(id),
    enabled: !!id,
  });
}

export function useUpdateTransactionStatusMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, status, notes }: { id: string; status: Transaction['status']; notes?: string }) =>
      transactionService.updateTransactionStatus(id, status, notes),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['transactions'] });
      queryClient.invalidateQueries({ queryKey: ['transaction', variables.id] });
      queryClient.invalidateQueries({ queryKey: ['investigations'] });
      queryClient.invalidateQueries({ queryKey: ['investigation'] });
    },
  });
}
