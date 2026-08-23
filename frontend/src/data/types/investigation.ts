export interface Investigation {
  id: string;
  caseId: string;
  riskType: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  riskScore: number;
  amount: number;
  assignedTo: string;
  created: string;
  sla: string;
  status: 'PENDING_REVIEW' | 'REVIEW_COMPLETED' | 'CONFIRMED_ABUSE' | 'FALSE_POSITIVE' | 'DISMISSED';
  notes?: string;
}
