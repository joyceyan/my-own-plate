import SwiftUI

struct AnalyzingView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel

    var body: some View {
        VStack(spacing: 0) {
            // Nav bar
            HStack {
                Spacer()
                Text("Analyzing")
                    .font(.headline)
                    .foregroundStyle(Theme.textPrimary)
                Spacer()
            }
            .padding(.vertical, 12)

            // Photo
            if let image = viewModel.selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 350)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .padding(.horizontal, 20)
            }

            Spacer()

            VStack(spacing: 16) {
                if let error = viewModel.analysisError {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.title)
                        .foregroundStyle(Theme.protein)
                    Text(error)
                        .font(.subheadline)
                        .foregroundStyle(Theme.textSecondary)
                        .multilineTextAlignment(.center)
                    Button("Dismiss") {
                        viewModel.resetAnalysis()
                    }
                    .font(.headline)
                    .foregroundStyle(Theme.background)
                    .padding(.horizontal, 32)
                    .padding(.vertical, 12)
                    .background(Color.white)
                    .clipShape(Capsule())
                } else {
                    ProgressView()
                        .tint(.white)

                    Text("Identifying your meal...")
                        .font(.headline)
                        .foregroundStyle(Theme.textPrimary)

                    Text("Estimating calories and macros")
                        .font(.subheadline)
                        .foregroundStyle(Theme.textSecondary)

                    Label("Running on this device", systemImage: "lock.shield.fill")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 8)
                        .background(Theme.cardBackground)
                        .clipShape(Capsule())
                }
            }
            .padding()

            Spacer()
        }
        .background(Theme.background)
    }
}
