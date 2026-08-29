-- SQLite Schema for Abuse-Ring Sentinel V2 Relational Store

-- 1. Core Entities
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    country TEXT,
    email_hash TEXT,
    phone_hash TEXT
);

CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    device_type TEXT,
    os TEXT
);

CREATE TABLE IF NOT EXISTS ips (
    ip_id TEXT PRIMARY KEY,
    ip_hash TEXT NOT NULL,
    country TEXT,
    first_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS payment_methods (
    payment_id TEXT PRIMARY KEY,
    payment_type TEXT,
    fingerprint_hash TEXT NOT NULL,
    first_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id TEXT PRIMARY KEY,
    category TEXT,
    country TEXT
);

-- 2. Transaction Records
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    merchant_id TEXT,
    payment_id TEXT,
    amount REAL NOT NULL,
    timestamp TEXT NOT NULL,
    status TEXT CHECK(status IN ('ALLOW', 'MONITOR', 'MANUAL_REVIEW', 'HOLD', 'BLOCK')) DEFAULT 'MONITOR',
    is_abuse INTEGER CHECK(is_abuse IN (0, 1)) DEFAULT 0,
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(merchant_id) REFERENCES merchants(merchant_id),
    FOREIGN KEY(payment_id) REFERENCES payment_methods(payment_id)
);

-- 3. Login Telemetry
CREATE TABLE IF NOT EXISTS logins (
    login_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    device_id TEXT,
    ip_id TEXT,
    timestamp TEXT NOT NULL,
    success INTEGER CHECK(success IN (0, 1)),
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(device_id) REFERENCES devices(device_id),
    FOREIGN KEY(ip_id) REFERENCES ips(ip_id)
);

-- 4. Bipartite Edge Tables
CREATE TABLE IF NOT EXISTS user_devices (
    user_id TEXT,
    device_id TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    usage_count INTEGER DEFAULT 1,
    PRIMARY KEY (user_id, device_id),
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(device_id) REFERENCES devices(device_id)
);

CREATE TABLE IF NOT EXISTS user_ips (
    user_id TEXT,
    ip_id TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    usage_count INTEGER DEFAULT 1,
    PRIMARY KEY (user_id, ip_id),
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(ip_id) REFERENCES ips(ip_id)
);

CREATE TABLE IF NOT EXISTS user_payments (
    user_id TEXT,
    payment_id TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    usage_count INTEGER DEFAULT 1,
    PRIMARY KEY (user_id, payment_id),
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(payment_id) REFERENCES payment_methods(payment_id)
);

-- 5. Investigator Feedback Loop
CREATE TABLE IF NOT EXISTS investigations (
    investigation_id TEXT PRIMARY KEY,
    alert_id TEXT UNIQUE NOT NULL,
    status TEXT CHECK(status IN ('PENDING_REVIEW', 'REVIEW_COMPLETED', 'CONFIRMED_ABUSE', 'FALSE_POSITIVE')) DEFAULT 'PENDING_REVIEW',
    analyst_decision TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY(alert_id) REFERENCES transactions(transaction_id)
);

-- 6. Model Performance Telemetry
CREATE TABLE IF NOT EXISTS model_metrics (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT CHECK(status IN ('ACTIVE', 'SHADOW', 'INACTIVE')) DEFAULT 'INACTIVE',
    precision REAL,
    recall REAL,
    pr_auc REAL,
    f1_score REAL,
    fpr REAL,
    latency INTEGER, -- millisecond latency
    drift REAL,
    evaluation_date TEXT
);

-- 7. Database Query Indexing
CREATE INDEX IF NOT EXISTS idx_transactions_user ON transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_logins_user ON logins(user_id);
CREATE INDEX IF NOT EXISTS idx_investigations_status ON investigations(status);
