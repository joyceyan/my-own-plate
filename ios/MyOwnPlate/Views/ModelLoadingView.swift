import SwiftUI

struct ModelLoadingView: View {
    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            ProgressView()
                .scaleEffect(1.5)
                .tint(.white)

            Text("Loading model...")
                .font(.headline)
                .foregroundStyle(Theme.textPrimary)

            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Theme.background)
    }
}
