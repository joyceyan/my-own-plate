import SwiftUI

struct ReviewView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel
    @State private var showEdit = false

    var body: some View {
        VStack(spacing: 0) {
            // Nav bar
            HStack {
                Button("Cancel") { viewModel.resetAnalysis() }
                    .font(.subheadline)
                    .foregroundStyle(Theme.textSecondary)
                Spacer()
                Text("Review")
                    .font(.headline)
                    .foregroundStyle(Theme.textPrimary)
                Spacer()
                // Balance the cancel button
                Text("Cancel").font(.subheadline).opacity(0)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Photo
                    if let image = viewModel.selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                    }

                    // Food description
                    if let result = viewModel.analysisResult {
                        Text(viewModel.foodDescription.isEmpty ? "Meal" : viewModel.foodDescription)
                            .font(.body.weight(.medium))
                            .foregroundStyle(Theme.textPrimary)
                            .padding(14)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Theme.cardBackground)
                            .clipShape(RoundedRectangle(cornerRadius: 12))

                        nutritionSummary(result)
                    }
                }
                .padding(.horizontal, 20)
            }

            // Bottom buttons
            HStack(spacing: 12) {
                Button {
                    showEdit = true
                } label: {
                    Text("Edit")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(Theme.cardBackground)
                        .foregroundStyle(Theme.textPrimary)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                }

                Button {
                    viewModel.saveMeal()
                } label: {
                    Text("Save")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(Color.white)
                        .foregroundStyle(Theme.background)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
        }
        .background(Theme.background)
        .fullScreenCover(isPresented: $showEdit) {
            EditMealView()
        }
    }

    // MARK: - Read-only nutrition summary

    private func nutritionSummary(_ result: NutritionResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            // Calories
            VStack(alignment: .leading, spacing: 6) {
                Text("Calories")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(result.calories.map { "\(Int($0))" } ?? "-")
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundStyle(Theme.textPrimary)
                    Text("kcal")
                        .font(.subheadline)
                        .foregroundStyle(Theme.textSecondary)
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.cardBackground)
            .clipShape(RoundedRectangle(cornerRadius: 12))

            // Macros
            HStack(spacing: 10) {
                macroSummary(label: "Protein", grams: result.protein, color: Theme.protein, icon: .system("figure.arms.open"))
                macroSummary(label: "Carbs", grams: result.carbs, color: Theme.carbs, icon: .system("leaf.fill"))
                macroSummary(label: "Fat", grams: result.fat, color: Theme.fat, icon: .system("drop.halffull"))
            }
        }
    }

    private func macroSummary(label: String, grams: Double?, color: Color, icon: IconRef) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                icon.image
                    .font(.caption2)
                    .foregroundStyle(color)
                Text(label)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(grams.map { "\(Int($0))" } ?? "-")
                    .font(.system(.title3, design: .rounded, weight: .bold))
                    .foregroundStyle(Theme.textPrimary)
                Text("g")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
