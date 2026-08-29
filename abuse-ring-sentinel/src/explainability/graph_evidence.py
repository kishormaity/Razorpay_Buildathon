import os
import sys
import networkx as nx

class GraphEvidenceExtractor:
    def __init__(self, G):
        self.G = G
        
    def extract_evidence(self, user_id):
        evidence = []
        if not self.G or not self.G.has_node(user_id):
            return evidence
            
        neighbors = list(self.G.neighbors(user_id))
        for n in neighbors:
            ntype = self.G.nodes[n].get('type')
            if ntype in ('DEVICE', 'PAYMENT', 'IP'):
                # Find users sharing this entity
                linked_users = [u for u in self.G.neighbors(n) if self.G.nodes[u].get('type') == 'USER' and u != user_id]
                
                # Check for abuse labels
                for ou in linked_users:
                    ou_txns = [t for t in self.G.neighbors(ou) if self.G.nodes[t].get('type') == 'TRANSACTION']
                    is_ou_abuse = any(self.G.nodes[t].get('is_abuse', 0) == 1 for t in ou_txns)
                    
                    if is_ou_abuse:
                        evidence.append({
                            'entity_id': n,
                            'entity_type': ntype,
                            'connected_user': ou,
                            'relationship': f"Shares {ntype.lower()} ({n}) with known flagged account {ou}"
                        })
                        
        return evidence[:3] # Limit to top 3 paths
