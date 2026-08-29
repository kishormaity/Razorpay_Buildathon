import { mockRiskRings } from '../data/mock/rings';
import { RiskRing } from '../data/types/ring';

let ringsDb = [...mockRiskRings];

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const ringService = {
  async getRiskRings(): Promise<RiskRing[]> {
    await delay(300);
    return [...ringsDb];
  },

  async getRiskRing(id: string): Promise<RiskRing | undefined> {
    await delay(250);
    return ringsDb.find((r) => r.id === id);
  },
};
