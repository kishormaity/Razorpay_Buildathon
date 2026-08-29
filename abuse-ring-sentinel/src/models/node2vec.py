import os
import sys
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))

from src.data.connection import get_connection
from src.graph.builder import build_heterogeneous_graph
from src.evaluation.metrics import evaluate_predictions

class PurePyTorchNode2Vec(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        
    def forward(self, u, v):
        emb_u = self.embeddings(u)
        emb_v = self.embeddings(v)
        scores = torch.sum(emb_u * emb_v, dim=1)
        return scores

def generate_random_walks(G, walk_length=5, walks_per_node=2):
    nodes = list(G.nodes())
    walks = []
    
    print("Generating Node2Vec random walks on graph...")
    for _ in range(walks_per_node):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            curr = node
            for _ in range(walk_length - 1):
                neighbors = list(G.neighbors(curr))
                if len(neighbors) == 0:
                    break
                curr = random.choice(neighbors)
                walk.append(curr)
            if len(walk) > 1:
                walks.append(walk)
    return walks

def build_skip_gram_pairs(walks, context_size=3):
    pairs = []
    for walk in walks:
        for i, u in enumerate(walk):
            start = max(0, i - context_size)
            end = min(len(walk), i + context_size + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((u, walk[j]))
    return pairs

def train_node2vec_embeddings(G, embedding_dim=16, epochs=5):
    nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for idx, node in enumerate(nodes)}
    
    walks = generate_random_walks(G, walk_length=5, walks_per_node=2)
    pairs = build_skip_gram_pairs(walks, context_size=3)
    print(f"Extracted {len(pairs)} co-occurrence pairs for skip-gram training.")
    
    model = PurePyTorchNode2Vec(len(nodes), embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    batch_size = 1024
    num_batches = max(1, len(pairs) // batch_size)
    
    model.train()
    for epoch in range(epochs):
        random.shuffle(pairs)
        total_loss = 0.0
        
        for b in range(num_batches):
            batch = pairs[b * batch_size : (b + 1) * batch_size]
            if not batch:
                continue
                
            u_indices = torch.tensor([node_to_idx[p[0]] for p in batch], dtype=torch.long)
            v_indices = torch.tensor([node_to_idx[p[1]] for p in batch], dtype=torch.long)
            
            pos_scores = model(u_indices, v_indices)
            pos_loss = criterion(pos_scores, torch.ones_like(pos_scores))
            
            neg_indices = torch.randint(0, len(nodes), (len(batch),), dtype=torch.long)
            neg_scores = model(u_indices, neg_indices)
            neg_loss = criterion(neg_scores, torch.zeros_like(neg_scores))
            
            loss = pos_loss + neg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"  * Node2Vec Epoch {epoch+1}/{epochs} - Loss: {total_loss / num_batches:.4f}")
        
    model.eval()
    embeddings_matrix = model.embeddings.weight.detach().numpy()
    
    node_embeddings = {}
    for idx, node in idx_to_node.items():
        node_embeddings[node] = embeddings_matrix[idx]
        
    return node_embeddings

def train_node2vec_classifier(G, embeddings, epochs=10):
    user_nodes = [node for node, attrs in G.nodes(data=True) if attrs.get('type') == 'USER']
    
    conn = get_connection()
    user_labels_df = pd.read_sql_query(
        "SELECT user_id, max(is_abuse) as is_fraud FROM transactions GROUP BY user_id;", conn
    )
    conn.close()
    
    user_to_label = dict(zip(user_labels_df['user_id'], user_labels_df['is_fraud']))
    
    X = []
    y = []
    for user in user_nodes:
        if user in embeddings:
            X.append(embeddings[user])
            y.append(user_to_label.get(user, 0))
            
    X = np.array(X)
    y = np.array(y)
    
    # 70/15/15 Chronological split index mapping (from user transaction dates)
    # We can match user risks predictions
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    classifier = nn.Sequential(
        nn.Linear(X.shape[1], 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )
    
    opt = optim.Adam(classifier.parameters(), lr=0.01)
    crit = nn.BCELoss()
    
    classifier.train()
    for ep in range(epochs):
        preds = classifier(X_tensor)
        loss = crit(preds, y_tensor)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    classifier.eval()
    with torch.no_grad():
        all_preds = classifier(X_tensor).squeeze().numpy()
        
    user_risks = {user: float(all_preds[idx]) for idx, user in enumerate(user_nodes)}
    
    # Save the Node2Vec model artifact
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    models_dir = os.path.join(proj_root, "models", "node2vec")
    os.makedirs(models_dir, exist_ok=True)
    
    out_path = os.path.join(models_dir, "node2vec_user_risks.json")
    with open(out_path, "w") as f:
        json.dump(user_risks, f, indent=2)
        
    print(f"Node2Vec user risk scores saved to: {out_path}")
    return user_risks

def main():
    print("Building global graph for Node2Vec...")
    G = build_heterogeneous_graph()
    
    # Fetch parameters from config
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    config_path = os.path.join(proj_root, "configs", "models.yaml")
    with open(config_path, "r") as f:
        m_config = yaml.safe_load(f)["node2vec"]
        
    embeddings = train_node2vec_embeddings(G, embedding_dim=m_config["embedding_dim"], epochs=3)
    train_node2vec_classifier(G, embeddings, epochs=10)

if __name__ == "__main__":
    main()
