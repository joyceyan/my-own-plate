-- Users are created on first app launch via /v1/bootstrap with a device_id.
-- apple_user_id stays NULL until the user optionally signs in with Apple.
CREATE TABLE users (
    id              TEXT PRIMARY KEY,
    apple_user_id   TEXT UNIQUE,
    device_id       TEXT,
    referral_code   TEXT NOT NULL UNIQUE,
    referred_by     TEXT REFERENCES users(id),
    created_at      INTEGER NOT NULL  -- unix seconds; server-side trial anchor
);
CREATE INDEX idx_users_device_id ON users(device_id);

-- Bearer session tokens returned by /v1/bootstrap.
CREATE TABLE sessions (
    token       TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id),
    created_at  INTEGER NOT NULL,
    expires_at  INTEGER NOT NULL
);
CREATE INDEX idx_sessions_user_id ON sessions(user_id);

-- One referral row per referee; referee_id is the PK so a user can only
-- ever be attributed to one referrer.
CREATE TABLE referrals (
    referee_id  TEXT PRIMARY KEY REFERENCES users(id),
    referrer_id TEXT NOT NULL REFERENCES users(id),
    created_at  INTEGER NOT NULL
);
CREATE INDEX idx_referrals_referrer ON referrals(referrer_id);

-- Append-only mirror of Superwall subscription webhooks.
-- id is the provider event id (or a content hash) for INSERT OR IGNORE idempotency.
CREATE TABLE subscription_events (
    id          TEXT PRIMARY KEY,
    user_id     TEXT,
    product_id  TEXT,
    event_type  TEXT,
    expires_at  INTEGER,
    raw_payload TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE INDEX idx_subscription_events_user ON subscription_events(user_id);
