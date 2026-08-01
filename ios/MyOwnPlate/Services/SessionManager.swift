import UIKit

@MainActor
@Observable
final class SessionManager {
    /// Pasteboard marker written by the referral landing page before the App Store redirect.
    static let pasteboardMarker = "mop-ref:"

    private(set) var userId: String?
    private(set) var referralCode: String?
    private(set) var isReferred = false
    private(set) var isAppleLinked = false

    private let client = BackendClient.shared

    var referralURL: URL? {
        referralCode.map { BackendClient.baseURL.appendingPathComponent("r/\($0)") }
    }

    /// Stable per-install device identity, stored in the Keychain so it survives reinstall.
    var deviceId: String {
        if let existing = Keychain.string(forKey: "deviceId") { return existing }
        let id = UUID().uuidString
        Keychain.set(id, forKey: "deviceId")
        return id
    }

    private var sessionToken: String? { Keychain.string(forKey: "sessionToken") }

    func bootstrapIfNeeded() async {
        guard userId == nil else { return }
        do {
            let response = try await client.bootstrap(deviceId: deviceId)
            userId = response.userId
            referralCode = response.referralCode
            isReferred = response.isReferred
            isAppleLinked = response.isAppleLinked
            Keychain.set(response.sessionToken, forKey: "sessionToken")
        } catch {
            // Offline-tolerant: the app works fully without an account session.
            print("Backend bootstrap failed: \(error.localizedDescription)")
        }
    }

    /// Reads the pasteboard once for a referral code stashed by the landing page.
    /// Triggers the system paste prompt on first launch — call only when userId exists.
    func claimPasteboardReferralIfNeeded() async {
        let attemptedKey = "referralClaimAttempted"
        guard !isReferred, !UserDefaults.standard.bool(forKey: attemptedKey) else { return }
        UserDefaults.standard.set(true, forKey: attemptedKey)
        guard let contents = UIPasteboard.general.string,
              contents.hasPrefix(Self.pasteboardMarker) else { return }
        let code = String(contents.dropFirst(Self.pasteboardMarker.count))
        UIPasteboard.general.string = ""
        _ = await claimReferral(code: code)
    }

    /// Handles universal links (https://<domain>/r/CODE) when the app is already installed.
    func handleUniversalLink(_ url: URL) {
        let components = url.pathComponents
        guard components.count >= 2, components[components.count - 2] == "r" else { return }
        Task { _ = await claimReferral(code: components[components.count - 1]) }
    }

    @discardableResult
    func claimReferral(code: String) async -> Bool {
        guard let sessionToken, !isReferred else { return false }
        do {
            try await client.claimReferral(code: code, sessionToken: sessionToken)
            isReferred = true
            return true
        } catch {
            print("Referral claim failed: \(error.localizedDescription)")
            return false
        }
    }

    @discardableResult
    func linkAppleAccount(identityToken: Data) async -> Bool {
        guard let sessionToken,
              let tokenString = String(data: identityToken, encoding: .utf8) else { return false }
        do {
            try await client.linkAppleAccount(identityToken: tokenString, sessionToken: sessionToken)
            isAppleLinked = true
            return true
        } catch {
            print("Apple account link failed: \(error.localizedDescription)")
            return false
        }
    }
}
