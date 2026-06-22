import SwiftUI

struct SplashView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel

    var body: some View {
        VStack(spacing: 0) {
            Spacer()

            // Title
            Text("MyOwnPlate")
                .font(.system(size: 34, weight: .bold, design: .rounded))
                .foregroundStyle(Theme.textPrimary)
                .padding(.bottom, 32)

            // Phone illustration
            phoneIllustration
                .padding(.bottom, 32)

            // Subtitle
            Text("Easy, private calorie tracking")
                .font(.title3.weight(.medium))
                .foregroundStyle(Theme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            Spacer()

            // Let's Go button
            Button {
                viewModel.showOnboarding = true
            } label: {
                Text("Let's Go")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(Color.white)
                    .foregroundStyle(Theme.background)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 32)
        }
        .background(Theme.background)
    }

    // MARK: - Phone illustration

    private var phoneIllustration: some View {
        ZStack {
            // Phone body
            RoundedRectangle(cornerRadius: 20)
                .fill(Theme.cardBackground)
                .frame(width: 160, height: 260)
                .overlay(
                    RoundedRectangle(cornerRadius: 20)
                        .stroke(Color.white.opacity(0.15), lineWidth: 1)
                )

            VStack(spacing: 16) {
                // Camera icon
                ZStack {
                    Circle()
                        .fill(Theme.cardBackgroundLight)
                        .frame(width: 72, height: 72)
                    Image(systemName: "camera.fill")
                        .font(.system(size: 28))
                        .foregroundStyle(Theme.textPrimary)
                }

                // Nutrition readout
                VStack(spacing: 6) {
                    HStack(spacing: 12) {
                        miniStat(value: "420", label: "cal", color: Theme.calories)
                        miniStat(value: "32g", label: "protein", color: Theme.protein)
                    }
                    HStack(spacing: 12) {
                        miniStat(value: "20g", label: "carbs", color: Theme.carbs)
                        miniStat(value: "5g", label: "fat", color: Theme.fat)
                    }
                }
            }
        }
    }

    private func miniStat(value: String, label: String, color: Color) -> some View {
        HStack(spacing: 3) {
            Text(value)
                .font(.system(size: 13, weight: .bold, design: .rounded))
                .foregroundStyle(color)
            Text(label)
                .font(.system(size: 11))
                .foregroundStyle(Theme.textSecondary)
        }
    }
}
