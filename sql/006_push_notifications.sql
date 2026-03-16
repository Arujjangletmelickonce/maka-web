CREATE TABLE IF NOT EXISTS device_tokens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  token TEXT NOT NULL UNIQUE,
  platform TEXT NOT NULL DEFAULT 'ios',
  environment TEXT NOT NULL DEFAULT 'sandbox',
  user_id INTEGER,
  is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
  last_registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  last_notified_at TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_device_tokens_active_platform
ON device_tokens(is_active, platform);

CREATE INDEX IF NOT EXISTS idx_device_tokens_user_id
ON device_tokens(user_id);

CREATE TABLE IF NOT EXISTS post_push_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_key TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  sent_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_post_push_events_created_at
ON post_push_events(created_at);
