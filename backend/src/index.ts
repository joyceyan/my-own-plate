import { verifyAppleIdentityToken } from "./apple";
import {
    HttpError,
    UserRow,
    generateReferralCode,
    json,
    nowSeconds,
    sha256Hex,
} from "./util";

export interface Env {
    DB: D1Database;
    APPLE_BUNDLE_ID: string;
    APPLE_TEAM_ID: string;
    APP_STORE_ID: string;
    WEBHOOK_SECRET: string;
}

const SESSION_TTL_S = 90 * 24 * 3600;
const REF_CODE_PATTERN = /^[A-Z2-9]{6,12}$/;

export default {
    async fetch(request: Request, env: Env): Promise<Response> {
        const url = new URL(request.url);
        const path = url.pathname;
        try {
            if (request.method === "POST" && path === "/v1/bootstrap") {
                return await bootstrap(request, env);
            }
            if (request.method === "POST" && path === "/v1/auth/apple") {
                return await authApple(request, env);
            }
            if (request.method === "POST" && path === "/v1/referrals/claim") {
                return await claimReferral(request, env);
            }
            if (request.method === "POST" && path.startsWith("/v1/webhooks/superwall/")) {
                return await superwallWebhook(request, env, path);
            }
            if (request.method === "GET" && path === "/v1/me") {
                return await getMe(request, env);
            }
            if (request.method === "GET" && path.startsWith("/r/")) {
                return referralLanding(path.slice("/r/".length), env);
            }
            if (request.method === "GET" && path === "/.well-known/apple-app-site-association") {
                return appleAppSiteAssociation(env);
            }
            return json({ error: "not_found" }, 404);
        } catch (e) {
            if (e instanceof HttpError) return json({ error: e.code }, e.status);
            console.error(e);
            return json({ error: "internal" }, 500);
        }
    },
};

// ---------------------------------------------------------------------------
// Auth helper

async function requireUser(request: Request, env: Env): Promise<UserRow> {
    const auth = request.headers.get("Authorization") ?? "";
    if (!auth.startsWith("Bearer ")) throw new HttpError(401, "unauthorized");
    const row = await env.DB.prepare(
        `SELECT u.* FROM sessions s
         JOIN users u ON u.id = s.user_id
         WHERE s.token = ? AND s.expires_at > ?`,
    )
        .bind(auth.slice("Bearer ".length), nowSeconds())
        .first<UserRow>();
    if (!row) throw new HttpError(401, "unauthorized");
    return row;
}

async function createSession(env: Env, userId: string): Promise<string> {
    const token = crypto.randomUUID();
    const now = nowSeconds();
    await env.DB.prepare(
        "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
    )
        .bind(token, userId, now, now + SESSION_TTL_S)
        .run();
    return token;
}

function publicUser(user: UserRow, sessionToken: string) {
    return {
        user_id: user.id,
        session_token: sessionToken,
        referral_code: user.referral_code,
        is_referred: user.referred_by !== null,
        is_apple_linked: user.apple_user_id !== null,
        created_at: user.created_at,
    };
}

// ---------------------------------------------------------------------------
// POST /v1/bootstrap  { device_id }
// Creates the user on first launch (trial anchor = created_at), or returns the
// existing user for this device.

async function bootstrap(request: Request, env: Env): Promise<Response> {
    const body = (await request.json().catch(() => null)) as { device_id?: string } | null;
    const deviceId = body?.device_id?.trim();
    if (!deviceId || deviceId.length > 128) throw new HttpError(400, "invalid_device_id");

    const existing = await env.DB.prepare("SELECT * FROM users WHERE device_id = ?")
        .bind(deviceId)
        .first<UserRow>();
    if (existing) {
        return json(publicUser(existing, await createSession(env, existing.id)));
    }

    // Retry on the (astronomically unlikely) referral-code collision.
    for (let attempt = 0; attempt < 5; attempt++) {
        const user: UserRow = {
            id: crypto.randomUUID(),
            apple_user_id: null,
            device_id: deviceId,
            referral_code: generateReferralCode(),
            referred_by: null,
            created_at: nowSeconds(),
        };
        const result = await env.DB.prepare(
            `INSERT OR IGNORE INTO users (id, apple_user_id, device_id, referral_code, referred_by, created_at)
             VALUES (?, NULL, ?, ?, NULL, ?)`,
        )
            .bind(user.id, user.device_id, user.referral_code, user.created_at)
            .run();
        if (result.meta.changes > 0) {
            return json(publicUser(user, await createSession(env, user.id)));
        }
    }
    throw new HttpError(500, "internal");
}

// ---------------------------------------------------------------------------
// POST /v1/auth/apple  (Bearer)  { identity_token }
// Verifies the Sign in with Apple credential and links it to this user.

async function authApple(request: Request, env: Env): Promise<Response> {
    const user = await requireUser(request, env);
    const body = (await request.json().catch(() => null)) as { identity_token?: string } | null;
    if (!body?.identity_token) throw new HttpError(400, "missing_identity_token");

    let payload;
    try {
        payload = await verifyAppleIdentityToken(body.identity_token, env.APPLE_BUNDLE_ID);
    } catch (e) {
        console.error("Apple token verification failed:", e);
        throw new HttpError(401, "invalid_identity_token");
    }

    const conflict = await env.DB.prepare(
        "SELECT id FROM users WHERE apple_user_id = ? AND id != ?",
    )
        .bind(payload.sub, user.id)
        .first();
    if (conflict) throw new HttpError(409, "apple_id_already_linked");

    await env.DB.prepare("UPDATE users SET apple_user_id = ? WHERE id = ?")
        .bind(payload.sub, user.id)
        .run();
    return json({ ok: true, apple_user_id: payload.sub });
}

// ---------------------------------------------------------------------------
// POST /v1/referrals/claim  (Bearer)  { code }
// Attributes this user to a referrer. Reward (3-month trial) is delivered via
// the referral IAP product in Superwall, keyed off is_referred.

async function claimReferral(request: Request, env: Env): Promise<Response> {
    const user = await requireUser(request, env);
    const body = (await request.json().catch(() => null)) as { code?: string } | null;
    const code = body?.code?.trim().toUpperCase() ?? "";
    if (!REF_CODE_PATTERN.test(code)) throw new HttpError(400, "invalid_code");

    if (user.referred_by) throw new HttpError(409, "already_referred");

    const referrer = await env.DB.prepare("SELECT * FROM users WHERE referral_code = ?")
        .bind(code)
        .first<UserRow>();
    if (!referrer) throw new HttpError(404, "invalid_code");
    if (referrer.id === user.id) throw new HttpError(400, "self_referral");
    if (referrer.device_id && referrer.device_id === user.device_id) {
        throw new HttpError(400, "same_device");
    }

    const now = nowSeconds();
    const results = await env.DB.batch([
        env.DB.prepare("UPDATE users SET referred_by = ? WHERE id = ? AND referred_by IS NULL")
            .bind(referrer.id, user.id),
        env.DB.prepare(
            "INSERT OR IGNORE INTO referrals (referee_id, referrer_id, created_at) VALUES (?, ?, ?)",
        ).bind(user.id, referrer.id, now),
    ]);
    if (results[0].meta.changes === 0) throw new HttpError(409, "already_referred");
    return json({ ok: true, is_referred: true });
}

// ---------------------------------------------------------------------------
// POST /v1/webhooks/superwall/<WEBHOOK_SECRET>
// Append-only mirror of Superwall subscription lifecycle events.
// Set the webhook URL in the Superwall dashboard including the secret.
//
// Superwall v2 webhook payload is nested under `data`: the event id is
// `data.id`, the event name is `data.name`, the product is `data.productId`,
// the SDK user id is `data.originalAppUserId`, and the expiration is
// `data.expirationAt`. We also accept the older flat payload for robustness.

async function superwallWebhook(request: Request, env: Env, path: string): Promise<Response> {
    const secret = path.slice("/v1/webhooks/superwall/".length);
    if (!secret || secret !== env.WEBHOOK_SECRET) throw new HttpError(401, "unauthorized");

    const raw = await request.text();
    let body: Record<string, unknown>;
    try {
        body = JSON.parse(raw);
    } catch {
        throw new HttpError(400, "invalid_json");
    }

    const data = body.data && typeof body.data === "object" ? (body.data as Record<string, unknown>) : {};

    const eventId = nestedString(data, "id") ?? str(body.id) ?? str(body.eventId) ?? str(body.event_id) ?? await sha256Hex(raw);
    const eventType = nestedString(data, "name") ?? str(body.eventName) ?? str(body.event_name) ?? str(body.type);
    const productId = nestedString(data, "productId") ?? str(body.productId) ?? str(body.product_id);
    const userId = nestedString(data, "originalAppUserId") ?? str(body.userId) ?? str(body.appUserId) ?? str(body.user_id);

    const expiresMs = nestedNumber(data, "expirationAt") ?? num(body.expiresAt) ?? num(body.expires_at);
    const expiresAt = typeof expiresMs === "number" ? Math.floor(expiresMs / 1000) : null;

    await env.DB.prepare(
        `INSERT OR IGNORE INTO subscription_events
         (id, user_id, product_id, event_type, expires_at, raw_payload, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`,
    )
        .bind(eventId, userId, productId, eventType, expiresAt, raw, nowSeconds())
        .run();
    return json({ ok: true });
}

function nestedString(obj: unknown, ...path: string[]): string | null {
    let cur = obj;
    for (const key of path) {
        if (!cur || typeof cur !== "object") return null;
        cur = (cur as Record<string, unknown>)[key];
    }
    return typeof cur === "string" ? cur : null;
}

function nestedNumber(obj: unknown, ...path: string[]): number | null {
    let cur = obj;
    for (const key of path) {
        if (!cur || typeof cur !== "object") return null;
        cur = (cur as Record<string, unknown>)[key];
    }
    return typeof cur === "number" ? cur : null;
}

function num(v: unknown): number | null {
    return typeof v === "number" ? v : null;
}

function str(v: unknown): string | null {
    return typeof v === "string" ? v : null;
}

const ACTIVE_EVENT_TYPES = new Set([
    "initial_purchase",
    "renewal",
    "uncancellation",
    "product_change",
    "non_renewing_purchase",
]);
const INACTIVE_EVENT_TYPES = new Set(["cancellation", "expiration", "billing_issue"]);

// ---------------------------------------------------------------------------
// GET /v1/me  (Bearer)
// Returns the current user and a server-side entitlement summary built from
// Superwall subscription events. The iOS app primarily reads subscription state
// from the Superwall SDK, but this endpoint is useful for diagnostics, web, and
// any future server-side gating.

async function getMe(request: Request, env: Env): Promise<Response> {
    const user = await requireUser(request, env);
    const { results } = await env.DB.prepare(
        `SELECT event_type, product_id, expires_at, created_at
         FROM subscription_events
         WHERE user_id = ?
         ORDER BY created_at DESC, id DESC`,
    ).bind(user.id).all<{ event_type: string; product_id: string | null; expires_at: number | null; created_at: number }>();

    const now = nowSeconds();
    let subscription: { active: boolean; expires_at: number | null; product_id: string | null } = {
        active: false,
        expires_at: null,
        product_id: null,
    };
    for (const row of results ?? []) {
        if (row.expires_at && ACTIVE_EVENT_TYPES.has(row.event_type) && row.expires_at > now) {
            subscription = { active: true, expires_at: row.expires_at, product_id: row.product_id };
            break;
        }
        if (INACTIVE_EVENT_TYPES.has(row.event_type)) {
            break;
        }
    }

    return json({
        user_id: user.id,
        referral_code: user.referral_code,
        is_referred: user.referred_by !== null,
        is_apple_linked: user.apple_user_id !== null,
        created_at: user.created_at,
        subscription,
    });
}

// ---------------------------------------------------------------------------
// GET /r/<CODE>
// Referral landing page. On tap, stashes the code in the pasteboard (read by
// the app on first launch) and redirects to the App Store. Also the universal-
// link target for users who already have the app installed.

function referralLanding(rawCode: string, env: Env): Response {
    const code = rawCode.trim().toUpperCase();
    if (!REF_CODE_PATTERN.test(code)) return new Response("Not found", { status: 404 });
    const marker = `mop-ref:${code}`;
    const appStoreUrl = `https://apps.apple.com/app/id${env.APP_STORE_ID}`;
    const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MyOwnPlate invite</title>
<style>
  body { margin: 0; min-height: 100vh; display: flex; flex-direction: column; align-items: center;
         justify-content: center; gap: 16px; background: #0d0f12; color: #f2f3f5;
         font-family: -apple-system, system-ui, sans-serif; text-align: center; padding: 24px; }
  h1 { font-size: 24px; margin: 0; }
  p { color: #9aa0a8; margin: 0 0 8px; }
  button { font-size: 17px; font-weight: 600; padding: 14px 32px; border: none; border-radius: 12px;
           background: #ffffff; color: #0d0f12; cursor: pointer; }
</style>
</head>
<body>
<h1>You've been invited</h1>
<p>Get 3 months of MyOwnPlate Premium free.</p>
<button id="go">Get the app</button>
<script>
document.getElementById("go").addEventListener("click", async function () {
  try { await navigator.clipboard.writeText(${JSON.stringify(marker)}); } catch (e) {}
  location.href = ${JSON.stringify(appStoreUrl)};
});
</script>
</body>
</html>`;
    return new Response(html, { headers: { "content-type": "text/html; charset=utf-8" } });
}

// ---------------------------------------------------------------------------
// GET /.well-known/apple-app-site-association
// Serves the AASA file so /r/* links open directly in the installed app.

function appleAppSiteAssociation(env: Env): Response {
    return json({
        applinks: {
            details: [
                {
                    appIDs: [`${env.APPLE_TEAM_ID}.${env.APPLE_BUNDLE_ID}`],
                    paths: ["/r/*"],
                },
            ],
        },
    });
}
