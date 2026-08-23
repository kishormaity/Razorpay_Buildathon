import { mockInvestigations } from '../data/mock/investigations';
import { Investigation } from '../data/types/investigation';
import { transactionService } from './transaction.service';

let investigationsDb = [...mockInvestigations];

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const investigationService = {
  async getInvestigations(): Promise<Investigation[]> {
    await delay(300); // Simulate network latency
    return [...investigationsDb];
  },

  async getInvestigation(id: string): Promise<Investigation | undefined> {
    await delay(200);
    return investigationsDb.find((i) => i.id === id);
  },

  async updateInvestigationStatus(
    id: string,
    status: Investigation['status'],
    notes?: string
  ): Promise<Investigation> {
    await delay(400);
    investigationsDb = investigationsDb.map((i) => {
      if (i.id === id) {
        // Automatically sync back to the underlying transaction status
        if (i.caseId) {
          transactionService.updateTransactionStatus(i.caseId, status, notes).catch(() => {});
        }
        return {
          ...i,
          status,
          notes: notes !== undefined ? notes : i.notes,
        };
      }
      return i;
    });
    const updated = investigationsDb.find((i) => i.id === id);
    if (!updated) {
      throw new Error(`Investigation with ID ${id} not found.`);
    }
    return updated;
  },
};
