import os
import sqlite3
import yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "configs", "data.yaml"))

def get_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_db_path():
    config = get_config()
    db_rel_path = config["database"]["db_path"]
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    return os.path.join(proj_root, db_rel_path)

def get_connection():
    db_path = get_db_path()
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    config = get_config()
    schema_rel_path = config["database"]["schema_path"]
    proj_root = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
    schema_path = os.path.join(proj_root, schema_rel_path)
    
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema SQL file not found at: {schema_path}")
        
    conn = get_connection()
    cursor = conn.cursor()
    
    print(f"Loading SQLite schema migrations from: {schema_path}")
    with open(schema_path, "r") as f:
        schema_sql = f.read()
        
    print("Executing schema scripts...")
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()
    print("Relational schema successfully applied.")

if __name__ == "__main__":
    init_db()
