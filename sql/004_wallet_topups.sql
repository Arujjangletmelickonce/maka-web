PRAGMA foreign_keys = ON;

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
