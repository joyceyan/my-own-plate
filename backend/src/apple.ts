// Verifies Sign in with Apple identity tokens (RS256 JWTs) against Apple's JWKS.

export interface AppleTokenPayload {
    iss: string;
    aud: string;
    exp: number;
    sub: string; // stable Apple user id — store as users.apple_user_id
    email?: string;
}

const JWKS_URL = "https://appleid.apple.com/auth/keys";
const EXPECTED_ISS = "https://appleid.apple.com";
const JWKS_CACHE_MS = 60 * 60 * 1000;

let cachedKeys: { keys: JsonWebKey[]; fetchedAt: number } | null = null;

async function applePublicKeys(): Promise<JsonWebKey[]> {
    if (cachedKeys && Date.now() - cachedKeys.fetchedAt < JWKS_CACHE_MS) {
        return cachedKeys.keys;
    }
    const res = await fetch(JWKS_URL);
    if (!res.ok) throw new Error(`Apple JWKS fetch failed: ${res.status}`);
    const body = (await res.json()) as { keys: JsonWebKey[] };
    cachedKeys = { keys: body.keys, fetchedAt: Date.now() };
    return body.keys;
}

function base64UrlDecode(input: string): Uint8Array {
    const b64 = input.replace(/-/g, "+").replace(/_/g, "/");
    const padded = b64 + "=".repeat((4 - (b64.length % 4)) % 4);
    const bin = atob(padded);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
}

function decodeJsonPart<T>(part: string): T {
    return JSON.parse(new TextDecoder().decode(base64UrlDecode(part))) as T;
}

export async function verifyAppleIdentityToken(
    token: string,
    bundleId: string,
): Promise<AppleTokenPayload> {
    const parts = token.split(".");
    if (parts.length !== 3) throw new Error("malformed JWT");
    const [headerB64, payloadB64, signatureB64] = parts;

    const header = decodeJsonPart<{ alg: string; kid: string }>(headerB64);
    if (header.alg !== "RS256") throw new Error("unexpected alg");

    const keys = await applePublicKeys();
    const jwk = keys.find((k) => (k as { kid?: string }).kid === header.kid);
    if (!jwk) throw new Error("unknown kid");

    const key = await crypto.subtle.importKey(
        "jwk",
        jwk,
        { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
        false,
        ["verify"],
    );
    const signed = new TextEncoder().encode(`${headerB64}.${payloadB64}`);
    const valid = await crypto.subtle.verify(
        "RSASSA-PKCS1-v1_5",
        key,
        base64UrlDecode(signatureB64) as BufferSource,
        signed,
    );
    if (!valid) throw new Error("invalid signature");

    const payload = decodeJsonPart<AppleTokenPayload>(payloadB64);
    if (payload.iss !== EXPECTED_ISS) throw new Error("bad issuer");
    if (payload.aud !== bundleId) throw new Error("bad audience");
    if (payload.exp * 1000 < Date.now()) throw new Error("expired token");
    return payload;
}
