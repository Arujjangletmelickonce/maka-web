CREATE TABLE IF NOT EXISTS jar_requests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT NOT NULL,
  requested_by_user_id INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'merged')),
  approved_jar_id INTEGER,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  reviewed_at TEXT,
  FOREIGN KEY (requested_by_user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (approved_jar_id) REFERENCES jars(id) ON DELETE SET NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_jar_requests_pending_ticker
ON jar_requests(ticker)
WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_jar_requests_requested_by_user_id
ON jar_requests(requested_by_user_id);

CREATE INDEX IF NOT EXISTS idx_jar_requests_status_created_at
ON jar_requests(status, created_at);
