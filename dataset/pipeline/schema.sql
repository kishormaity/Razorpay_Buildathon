-- SQLite Schema for AI Risk Sentinel

-- Entities: Users, Devices, IPs, Payments, Merchants, Addresses
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    type TEXT CHECK(type IN ('USER', 'DEVICE', 'IP', 'PAYMENT', 'MERCHANT', 'ADDRESS', 'TRANSACTION')),
    label TEXT,
    risk_score REAL DEFAULT 0.0,
    first_seen TEXT,
    last_seen TEXT,
    details TEXT -- Store JSON formatted metadata fields
);

-- Relationships: Links between entities representing logins, checkouts, shared details
CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT,
    target_id TEXT,
    type TEXT CHECK(type IN ('USED_BY', 'ASSOCIATED_WITH', 'SHARED_IP', 'LINKED_PAYMENT', 'MADE_TRANSACTION')),
    strength REAL DEFAULT 1.0,
    first_seen TEXT,
    last_seen TEXT,
    FOREIGN KEY (source_id) REFERENCES entities(id),
    FOREIGN KEY (target_id) REFERENCES entities(id)
);

-- Transactions: Single purchase events with model decisions and analyst resolution outcomes
CREATE TABLE IF NOT EXISTS transactions (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    amount REAL,
    currency TEXT DEFAULT 'INR',
    merchant TEXT,
    payment_method TEXT,
    channel TEXT,
    location TEXT,
    risk_score REAL,
    recommended_action TEXT CHECK(recommended_action IN ('ALLOW', 'STEP_UP', 'MANUAL_REVIEW', 'HOLD', 'BLOCK')),
    customer_id TEXT,
    device_id TEXT,
    ip_address TEXT,
    risk_narrative TEXT,
    status TEXT CHECK(status IN ('PENDING_REVIEW', 'REVIEW_COMPLETED', 'CONFIRMED_ABUSE', 'FALSE_POSITIVE', 'DISMISSED')) DEFAULT 'PENDING_REVIEW',
    notes TEXT,
    FOREIGN KEY (customer_id) REFERENCES entities(id),
    FOREIGN KEY (device_id) REFERENCES entities(id)
);

-- Model Telemetry: Track F1 scores, latency, error boundaries, evaluation parameters
CREATE TABLE IF NOT EXISTS model_metrics (
    id TEXT PRIMARY KEY,
    name TEXT,
    version TEXT,
    status TEXT CHECK(status IN ('ACTIVE', 'SHADOW', 'INACTIVE')),
    precision REAL,
    recall REAL,
    pr_auc REAL,
    f1_score REAL,
    fpr REAL,
    latency INTEGER, -- in milliseconds
    drift REAL,
    evaluation_date TEXT
);

-- Drift Telemetry: Concept and prediction drift logs
CREATE TABLE IF NOT EXISTS drift_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    prediction_drift REAL,
    feature_drift REAL
);

-- System Events Logging: Live telemetry alarms and warnings
CREATE TABLE IF NOT EXISTS system_events (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    message TEXT,
    type TEXT CHECK(type IN ('CRITICAL', 'WARNING', 'INFO')),
    source TEXT
);

-- Indexing for fast search and traversal queries
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status);
CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(type);
