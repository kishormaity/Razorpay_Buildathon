import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))

from src.data.connection import get_connection
from src.graph.builder import build_heterogeneous_graph
from src.evaluation.metrics import evaluate_predictions

class SAGEConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_neigh = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, adj_matrix):
        # x: [num_nodes, in_features]
        # adj_matrix: PyTorch sparse tensor [num_nodes, num_nodes] (normalized)
        neigh_agg = torch.sparse.mm(adj_matrix, x)
        out = self.linear_self(x) + self.linear_neigh(neigh_agg)
        return self.dropout(self.act(out))

class PyTorchGraphSAGE(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = SAGEConv(in_features, hidden_features)
        self.conv2 = SAGEConv(hidden_features, out_features)
        self.classifier = nn.Sequential(
            nn.Linear(out_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, adj):
        h = self.conv1(x, adj)
        h = self.conv2(h, adj)
        logits = self.classifier(h)
        return logits, h

def build_node_features_and_adj(G, train_txn_ids, user_to_label):
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    num_nodes = len(nodes)
    
    # 1. Build Node Feature Tensor (8-dimensional feature vectors)
    # Type representation: USER=0, DEVICE=1, IP=2, PAYMENT=3, MERCHANT=4, TRANSACTION=5
    type_map = {'USER': 0, 'DEVICE': 1, 'IP': 2, 'PAYMENT': 3, 'MERCHANT': 4, 'TRANSACTION': 5}
    features = np.zeros((num_nodes, 8), dtype=np.float32)
    
    for idx, node in enumerate(nodes):
        attr = G.nodes[node]
        ntype = attr.get('type', 'USER')
        
        # One-hot type encoding
        type_idx = type_map.get(ntype, 0)
        features[idx, type_idx] = 1.0
        
        # Normalized log degree
        degree = G.degree(node)
        features[idx, 6] = np.log1p(degree)
        
        # Leak-free target indicator: only show labels for training transactions
        if ntype == 'TRANSACTION' and node in train_txn_ids:
            features[idx, 7] = float(attr.get('is_abuse', 0))
        elif ntype == 'USER' and node in user_to_label:
            # Only use user labels if associated with training split
            pass
            
    x_tensor = torch.tensor(features, dtype=torch.float32)
    
    # 2. Build Sparse Adjacency Matrix using SciPy
    edges = list(G.edges())
    row = [node_to_idx[e[0]] for e in edges]
    col = [node_to_idx[e[1]] for e in edges]
    
    # Symmetric adjacency setup
    adj = sp.coo_matrix((np.ones(len(edges)), (row, col)), shape=(num_nodes, num_nodes))
    # Make symmetric (undirected link traversal)
    adj = adj + adj.T
    
    # Standard normalization D^-1 * A
    row_sum = np.array(adj.sum(axis=1)).flatten()
    d_inv = np.power(row_sum, -1.0, where=(row_sum > 0))
    d_inv[row_sum == 0] = 0.0
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv.dot(adj)
    
    # Convert to PyTorch Sparse Tensor
    coo = norm_adj.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i_tensor = torch.LongTensor(indices)
    v_tensor = torch.FloatTensor(coo.data)
    adj_tensor = torch.sparse_coo_tensor(i_tensor, v_tensor, torch.Size(coo.shape))
    
    return x_tensor, adj_tensor, node_to_idx

def train_graphsage_model(G, epochs=30):
    print("Preparing node features and sparse adjacency configurations...")
    
    # Query database and load chronological user labels
    conn = get_connection()
    tx_df = pd.read_sql_query("SELECT transaction_id, user_id, timestamp, is_abuse FROM transactions ORDER BY timestamp ASC;", conn)
    conn.close()
    
    # chronological partition: 70% of transactions are for training
    total_txns = len(tx_df)
    train_end = int(total_txns * 0.70)
    train_tx_ids = set(tx_df['transaction_id'].iloc[:train_end].values)
    
    # Retrieve user labels derived from training transactions
    train_tx_df = tx_df.iloc[:train_end]
    user_to_label = train_tx_df.groupby('user_id')['is_abuse'].max().to_dict()
    
    x, adj, node_to_idx = build_node_features_and_adj(G, train_tx_ids, user_to_label)
    
    # Define USER training targets
    user_nodes = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'USER']
    
    # Retrieve true labels for all users
    full_user_to_label = tx_df.groupby('user_id')['is_abuse'].max().to_dict()
    
    # Map users to indices and target tensors
    user_indices = [node_to_idx[u] for u in user_nodes]
    user_labels = [full_user_to_label.get(u, 0) for u in user_nodes]
    
    user_idx_tensor = torch.tensor(user_indices, dtype=torch.long)
    user_label_tensor = torch.tensor(user_labels, dtype=torch.float32).unsqueeze(1)
    
    # Model parameters
    model = PyTorchGraphSAGE(in_features=8, hidden_features=64, out_features=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.BCELoss()
    
    print(f"Training GraphSAGE classifier on {len(user_nodes)} users...")
    model.train()
    for epoch in range(epochs):
        logits, _ = model(x, adj)
        user_preds = logits[user_idx_tensor]
        
        loss = criterion(user_preds, user_label_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  * GraphSAGE Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f}")
            
    # Extract prediction risk values
    model.eval()
    with torch.no_grad():
        final_logits, _ = model(x, adj)
        all_preds = final_logits[user_idx_tensor].squeeze().numpy()
        
    user_risks = {user: float(all_preds[idx]) for idx, user in enumerate(user_nodes)}
    
    # Save predictions
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    models_dir = os.path.join(proj_root, "models", "graphsage")
    os.makedirs(models_dir, exist_ok=True)
    
    out_path = os.path.join(models_dir, "graphsage_user_risks.json")
    with open(out_path, "w") as f:
        json.dump(user_risks, f, indent=2)
        
    print(f"GraphSAGE user risk scores saved to: {out_path}")
    
    # Save the PyTorch model weights for real-time inference pipeline
    model_weights_path = os.path.join(models_dir, "graphsage_weights.pt")
    torch.save(model.state_dict(), model_weights_path)
    print(f"GraphSAGE model weights saved to: {model_weights_path}")
    
    return user_risks

def main():
    print("Building global graph for GraphSAGE...")
    G = build_heterogeneous_graph()
    
    # Fetch parameters from config
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    config_path = os.path.join(proj_root, "configs", "models.yaml")
    with open(config_path, "r") as f:
        m_config = yaml.safe_load(f)["graphsage"]
        
    train_graphsage_model(G, epochs=m_config["epochs"])

if __name__ == "__main__":
    main()
