import Foundation

struct APIError: Error, LocalizedError {
    let code: String

    var errorDescription: String? {
        switch code {
        case "invalid_code": return "That referral code isn't valid."
        case "already_referred": return "This account was already referred."
        case "self_referral": return "You can't use your own referral code."
        case "same_device": return "Referrals can't be claimed on the same device."
        case "apple_id_already_linked": return "This Apple ID is linked to another account."
        case "unauthorized": return "Session expired."
        default: return "Something went wrong (\(code))."
        }
    }
}

struct BootstrapResponse: Decodable {
    let userId: String
    let sessionToken: String
    let referralCode: String
    let isReferred: Bool
    let isAppleLinked: Bool

    enum CodingKeys: String, CodingKey {
        case userId = "user_id"
        case sessionToken = "session_token"
        case referralCode = "referral_code"
        case isReferred = "is_referred"
        case isAppleLinked = "is_apple_linked"
    }
}

final class BackendClient: Sendable {
    static let shared = BackendClient()

    static let baseURL = URL(string: "https://myownplate-api.myownplate-app-aj-labs.workers.dev")!

    private init() {}

    func bootstrap(deviceId: String) async throws -> BootstrapResponse {
        try await post("/v1/bootstrap", body: ["device_id": deviceId], token: nil)
    }

    func claimReferral(code: String, sessionToken: String) async throws {
        struct Ok: Decodable { let ok: Bool }
        let _: Ok = try await post("/v1/referrals/claim", body: ["code": code], token: sessionToken)
    }

    func linkAppleAccount(identityToken: String, sessionToken: String) async throws {
        struct Ok: Decodable { let ok: Bool }
        let _: Ok = try await post("/v1/auth/apple", body: ["identity_token": identityToken], token: sessionToken)
    }

    private func post<T: Decodable>(_ path: String, body: [String: String], token: String?) async throws -> T {
        var request = URLRequest(url: Self.baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw APIError(code: "network")
        }
        guard (200..<300).contains(http.statusCode) else {
            let serverCode = (try? JSONDecoder().decode([String: String].self, from: data))?["error"]
            throw APIError(code: serverCode ?? "http_\(http.statusCode)")
        }
        return try JSONDecoder().decode(T.self, from: data)
    }
}
