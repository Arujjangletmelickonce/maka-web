PRAGMA foreign_keys = ON;

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
