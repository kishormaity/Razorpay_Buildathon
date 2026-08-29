class ReasonCodesCompiler:
    def __init__(self):
        # Maps raw features to explanation strings
        self.reason_map = {
            'card_tx_count_10m': "Excessive short-term card transaction frequency (10m velocity)",
            'card_tx_count_1h': "High card transaction velocity in past hour",
            'card_spend_sum_1h': "Unusual high volume of card spending within 1 hour",
            'spend_ratio_24h': "Transaction amount deviates significantly from 24h card average",
            'is_new_device': "Unseen device signature associated with payment credentials",
            'is_new_location': "New location address detected for user account profile",
            'device_connected_fraud_rate': "Device is linked to historical high-risk accounts",
            'addr_connected_fraud_rate': "IP address neighborhood has high card abuse history",
            'pagerank_centrality': "Highly central node position in coordinate card network",
            'clustering_coefficient': "Belongs to a tightly-knit entity reuse network cluster",
            'TransactionAmt': "Atypical transaction amount value size"
        }
        
    def compile(self, shap_dict, graph_evidence, temporal_gap=None):
        reasons = []
        
        # 1. Compile Tabular SHAP Reasons
        sorted_shaps = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
        for feat, val in sorted_shaps[:3]:
            desc = self.reason_map.get(feat, f"Feature '{feat}' value increased risk score")
            reasons.append({
                'source': 'TABULAR_SHAP',
                'feature': feat,
                'weight': float(val),
                'description': desc
            })
            
        # 2. Compile Graph Reasons
        for ge in graph_evidence:
            reasons.append({
                'source': 'GRAPH_RELATION',
                'feature': ge['entity_type'],
                'weight': 0.40,  # Standard relationship weight
                'description': ge['relationship']
            })
            
        # 3. Add Temporal Reasons if bursty
        if temporal_gap is not None and temporal_gap < 1.0: # under 1 second time gap
            reasons.append({
                'source': 'TEMPORAL_BURST',
                'feature': 'time_gap',
                'weight': 0.30,
                'description': "Sub-second transaction sequence execution interval (potential automated script)"
            })
            
        # 4. Generate Narrative Summary
        descriptions = [r['description'] for r in reasons[:3]]
        if descriptions:
            narrative = "Flagged due to: " + "; ".join(descriptions) + "."
        else:
            narrative = "No anomalous risk markers exceeded default baseline thresholds."
            
        return reasons, narrative
