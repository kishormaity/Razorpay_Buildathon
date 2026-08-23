import { Entity, EntityRelationship } from './risk';

export interface RingTimelineEvent {
  timestamp: string;
  description: string;
  type: 'ACCOUNT_CREATED' | 'DEVICE_LINKED' | 'TRANSACTION_MADE' | 'REFUND_INITIATED';
  entityId: string;
}

export interface RiskRing {
  id: string;
  name: string;
  riskScore: number;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  historicalExposure: number;
  entities: Entity[];
  relationships: EntityRelationship[];
  timeline: RingTimelineEvent[];
}
