PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wallets (
  user_id INTEGER PRIMARY KEY,
  bean_balance INTEGER NOT NULL DEFAULT 0 CHECK (bean_balance >= 0),
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS jars (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT NOT NULL UNIQUE,
  display_name TEXT NOT NULL,
  note TEXT NOT NULL,
  comment TEXT NOT NULL,
  total_beans INTEGER NOT NULL DEFAULT 0 CHECK (total_beans >= 0),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS contributions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  jar_id INTEGER NOT NULL,
  amount INTEGER NOT NULL CHECK (amount > 0),
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (jar_id) REFERENCES jars(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_contributions_user_id
ON contributions(user_id);

CREATE INDEX IF NOT EXISTS idx_contributions_jar_id
ON contributions(jar_id);

CREATE TABLE IF NOT EXISTS user_emails (
  user_id INTEGER PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  verified_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS email_login_codes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT NOT NULL,
  code TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  used_at TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  token_hash TEXT NOT NULL UNIQUE,
  expires_at TEXT NOT NULL,
  last_used_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  revoked_at TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_emails_email
ON user_emails(email);

CREATE INDEX IF NOT EXISTS idx_email_login_codes_email
ON email_login_codes(email);

CREATE INDEX IF NOT EXISTS idx_email_login_codes_expires_at
ON email_login_codes(expires_at);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id
ON sessions(user_id);

CREATE INDEX IF NOT EXISTS idx_sessions_token_hash
ON sessions(token_hash);

CREATE TABLE IF NOT EXISTS wallet_topups (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  amount INTEGER NOT NULL CHECK (amount > 0),
  balance_after INTEGER NOT NULL CHECK (balance_after >= 0),
  source TEXT NOT NULL DEFAULT 'manual_purchase',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_wallet_topups_user_id
ON wallet_topups(user_id);

CREATE INDEX IF NOT EXISTS idx_wallet_topups_created_at
ON wallet_topups(created_at);
