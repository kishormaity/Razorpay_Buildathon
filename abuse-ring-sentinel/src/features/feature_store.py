import os
import sys
import time
import pandas as pd
import yaml

# Adjust path to import database connection
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CURRENT_DIR, "..")))

from data.connection import get_connection
from features.transaction import extract_transaction_features
from features.behavioral import compute_behavioral_features
from features.temporal import compute_temporal_features
from features.graph import compute_graph_features

def load_config():
    config_path = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "configs", "data.yaml"))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_features_matrix():
    print("=" * 70)
    print("           ABUSE-RING SENTINEL FEATURE STORE BUILDER")
    print("=" * 70)
    
    start_time = time.time()
    config = load_config()
    
    # 1. Connect to SQL
    conn = get_connection()
    
    # 2. Extract transaction features
    print("Extracting transaction features from SQLite...")
    df = extract_transaction_features(conn)
    conn.close()
    print(f"Extracted transaction base: {df.shape}")
    
    # 3. Compute behavioral features
    print("Computing rolling behavioral velocities...")
    df = compute_behavioral_features(df)
    
    # 4. Compute temporal features
    print("Computing temporal time gaps...")
    df = compute_temporal_features(df)
    
    # 5. Compute graph features
    print("Computing graph structural & centrality features...")
    df = compute_graph_features(df)
    
    # 5. Save output Parquet
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    processed_dir = config["paths"]["processed_dir"]
    output_dir = os.path.join(proj_root, processed_dir, "features")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "features.parquet")
    print(f"Saving compiled features matrix ({df.shape[0]} rows, {df.shape[1]} columns) to:")
    print(f"  * {output_path}")
    
    t0 = time.time()
    df.to_parquet(output_path, engine="pyarrow", index=False)
    print(f"Parquet file written in {time.time() - t0:.2f} seconds.")
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds.")
    print("=" * 70)
    return df

if __name__ == "__main__":
    build_features_matrix()
