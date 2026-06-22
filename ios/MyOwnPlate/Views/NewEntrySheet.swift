import SwiftUI

struct NewEntrySheet: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Log a meal")
                .font(.title3.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
                .padding(.horizontal, 20)
                .padding(.top, 24)
                .padding(.bottom, 20)

            entryOption(icon: "camera.fill", title: "Take photo", subtitle: "Use camera to capture meal") {
                dismiss()
                viewModel.startCamera()
            }

            Divider().opacity(0.15).padding(.leading, 60)

            entryOption(icon: "photo.fill", title: "Choose from library", subtitle: "Pick an existing photo") {
                dismiss()
                viewModel.startPhotoPicker()
            }

            Divider().opacity(0.15).padding(.leading, 60)

            entryOption(icon: "plus", title: "Manual entry", subtitle: "Skip photo, enter directly") {
                dismiss()
                viewModel.startManualEntry()
            }

            Spacer()
        }
        .background(Theme.cardBackground)
    }

    private func entryOption(icon: String, title: String, subtitle: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 16) {
                Image(systemName: icon)
                    .font(.body)
                    .frame(width: 24, height: 24)
                    .foregroundStyle(Theme.textPrimary)

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.body.weight(.medium))
                        .foregroundStyle(Theme.textPrimary)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }

                Spacer()
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
        }
    }
}
