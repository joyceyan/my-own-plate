import Foundation
import SuperwallKit

@MainActor
@Observable
final class EntitlementService: SuperwallDelegate {
    private static let superwallAPIKey = "pk_hzf4rt9Wz6WZ07OHpqj9D"

    private(set) var isPremium = false

    init() {
        Superwall.configure(apiKey: Self.superwallAPIKey)
        Superwall.shared.delegate = self
        isPremium = Superwall.shared.subscriptionStatus.isActive
    }

    /// Identify the user after backend bootstrap so Superwall events are attributed
    /// to the server-side user id.
    func identify(userId: String, referred: Bool) {
        Superwall.shared.identify(userId: userId)
        Superwall.shared.setUserAttributes(["referred": referred])
    }

    func setReferred(_ referred: Bool) {
        Superwall.shared.setUserAttributes(["referred": referred])
    }

    /// Registers the History tab placement. If the user is not premium, Superwall
    /// presents the configured paywall; the completion runs once access is granted.
    func registerHistory(completion: @escaping @MainActor () -> Void) {
        Superwall.shared.register(placement: "history") { [weak self] in
            self?.isPremium = Superwall.shared.subscriptionStatus.isActive
            completion()
        }
    }

    /// Registers an optional upgrade placement for explicit "Upgrade" buttons.
    func registerUpgrade(placement: String = "Upgrade") {
        Superwall.shared.register(placement: placement) { [weak self] in
            self?.isPremium = Superwall.shared.subscriptionStatus.isActive
        }
    }

    nonisolated func subscriptionStatusDidChange(
        from oldValue: SubscriptionStatus,
        to newValue: SubscriptionStatus
    ) {
        let active: Bool
        if case .active = newValue {
            active = true
        } else {
            active = false
        }
        Task { @MainActor [weak self] in
            self?.isPremium = active
        }
    }
}
