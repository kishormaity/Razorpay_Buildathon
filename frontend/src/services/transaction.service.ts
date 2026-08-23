import { mockTransactions } from '../data/mock/transactions';
import { Transaction } from '../data/types/transaction';

// In-memory simulated database state
let transactionsDb = [...mockTransactions];

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const transactionService = {
  async getTransactions(): Promise<Transaction[]> {
    await delay(300); // Simulate network latency
    return [...transactionsDb];
  },

  async getTransaction(id: string): Promise<Transaction | undefined> {
    await delay(200);
    return transactionsDb.find((t) => t.id === id);
  },

  async updateTransactionStatus(
    id: string,
    status: Transaction['status'],
    notes?: string
  ): Promise<Transaction> {
    await delay(400);
    transactionsDb = transactionsDb.map((t) => {
      if (t.id === id) {
        return {
          ...t,
          status,
          notes: notes !== undefined ? notes : t.notes,
        };
      }
      return t;
    });
    const updated = transactionsDb.find((t) => t.id === id);
    if (!updated) {
      throw new Error(`Transaction with ID ${id} not found.`);
    }
    return updated;
  },
};
