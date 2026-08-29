import { Investigation } from '../data/types/investigation';
import { transactionService } from './transaction.service';

export const investigationService = {
  async getInvestigations(): Promise<Investigation[]> {
    const txs = await transactionService.getTransactions();
    
    // Map transactions back to Investigation interface
    return txs.map((t) => {
      const score = t.riskDecision.score;
      const priority = score >= 0.70 ? 'CRITICAL' : score >= 0.50 ? 'HIGH' : 'MEDIUM' as any;
      const assignedTo = score >= 0.70 ? 'Sentinel Auto-Block' : 'Arjun Mehta';
      
      return {
        id: `INV-${t.id.replace('TXN-', '')}`,
        caseId: t.id,
        riskType: 'Coordinated Ring Abuse',
        priority: priority,
        riskScore: score,
        amount: t.amount,
        assignedTo: assignedTo,
        created: t.timestamp,
        sla: '2h remaining',
        status: t.status as any,
        notes: t.notes
      };
    });
  },

  async getInvestigation(id: string): Promise<Investigation | undefined> {
    const txnId = `TXN-${id.replace('INV-', '')}`;
    const t = await transactionService.getTransaction(txnId);
    if (!t) return undefined;
    
    const score = t.riskDecision.score;
    const priority = score >= 0.70 ? 'CRITICAL' : score >= 0.50 ? 'HIGH' : 'MEDIUM' as any;
    const assignedTo = score >= 0.70 ? 'Sentinel Auto-Block' : 'Arjun Mehta';
    
    return {
      id,
      caseId: t.id,
      riskType: 'Coordinated Ring Abuse',
      priority: priority,
      riskScore: score,
      amount: t.amount,
      assignedTo: assignedTo,
      created: t.timestamp,
      sla: '2h remaining',
      status: t.status as any,
      notes: t.notes
    };
  },

  async updateInvestigationStatus(
    id: string,
    status: Investigation['status'],
    notes?: string
  ): Promise<Investigation> {
    const txnId = `TXN-${id.replace('INV-', '')}`;
    const t = await transactionService.updateTransactionStatus(txnId, status, notes);
    
    const score = t.riskDecision.score;
    const priority = score >= 0.70 ? 'CRITICAL' : score >= 0.50 ? 'HIGH' : 'MEDIUM' as any;
    const assignedTo = score >= 0.70 ? 'Sentinel Auto-Block' : 'Arjun Mehta';
    
    return {
      id,
      caseId: t.id,
      riskType: 'Coordinated Ring Abuse',
      priority: priority,
      riskScore: score,
      amount: t.amount,
      assignedTo: assignedTo,
      created: t.timestamp,
      sla: '2h remaining',
      status: t.status as any,
      notes: t.notes
    };
  }
};
