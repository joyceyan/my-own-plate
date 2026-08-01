export class HttpError extends Error {
    constructor(
        public status: number,
        public code: string,
    ) {
        super(code);
    }
}

export function json(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" },
    });
}

const CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"; // no I/L/O/0/1

export function generateReferralCode(length = 8): string {
    const bytes = crypto.getRandomValues(new Uint8Array(length));
    let code = "";
    for (const b of bytes) code += CODE_ALPHABET[b % CODE_ALPHABET.length];
    return code;
}

export function nowSeconds(): number {
    return Math.floor(Date.now() / 1000);
}

export interface UserRow {
    id: string;
    apple_user_id: string | null;
    device_id: string | null;
    referral_code: string;
    referred_by: string | null;
    created_at: number;
}

export async function sha256Hex(text: string): Promise<string> {
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
