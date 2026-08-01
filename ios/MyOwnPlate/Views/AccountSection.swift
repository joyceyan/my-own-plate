import SwiftUI

struct AccountSection: View {
    @Environment(EntitlementService.self) var entitlements

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Account")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                Spacer()
                if entitlements.isPremium {
                    Text("Premium")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(Theme.accent)
                }
            }

            if !entitlements.isPremium {
                Button {
                    entitlements.registerUpgrade()
                } label: {
                    Text("Upgrade to Premium")
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .background(Theme.accent)
                        .foregroundStyle(Theme.background)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

