import { mockPolicyRules, mockPolicySimulationData, PolicySimulationPoint } from '../data/mock/policies';
import { PolicyRule } from '../data/types/risk';

let policiesDb = [...mockPolicyRules];

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export const policyService = {
  async getPolicyRules(): Promise<PolicyRule[]> {
    await delay(200); // Simulate network latency
    return [...policiesDb];
  },

  async updatePolicyRuleRange(id: string, riskRange: [number, number]): Promise<PolicyRule> {
    await delay(300);
    policiesDb = policiesDb.map((p) => {
      if (p.id === id) {
        return {
          ...p,
          riskRange,
        };
      }
      return p;
    });
    const updated = policiesDb.find((p) => p.id === id);
    if (!updated) {
      throw new Error(`Policy rule with ID ${id} not found.`);
    }
    return updated;
  },

  async getPolicySimulationData(): Promise<PolicySimulationPoint[]> {
    await delay(250);
    return [...mockPolicySimulationData];
  },
};
