import SwiftUI

struct EditMealView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        @Bindable var vm = viewModel

        VStack(spacing: 0) {
            // Nav bar
            HStack {
                Button("Cancel") { dismiss() }
                    .font(.subheadline)
                    .foregroundStyle(Theme.textSecondary)
                Spacer()
                Text("Edit Meal")
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
                    // Photo (read-only preview)
                    if let image = viewModel.selectedImage {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 200)
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                    }

                    // Food description
                    TextField("Describe your meal", text: $vm.foodDescription)
                        .font(.body)
                        .foregroundStyle(Theme.textPrimary)
                        .padding(14)
                        .background(Theme.cardBackground)
                        .clipShape(RoundedRectangle(cornerRadius: 12))

                    nutritionEditor
                }
                .padding(.horizontal, 20)
            }

            // Save button
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
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
        }
        .background(Theme.background)
    }

    // MARK: - Editable nutrition

    @ViewBuilder
    private var nutritionEditor: some View {
        @Bindable var vm = viewModel

        // Calories
        optionalField(
            label: "Calories",
            unit: "kcal",
            color: Theme.calories,
            icon: .system("flame"),
            value: Binding(
                get: { vm.analysisResult?.calories },
                set: { vm.analysisResult?.calories = $0 }
            ),
            large: true
        )

        // Macros
        HStack(spacing: 10) {
            optionalField(
                label: "Protein", unit: "g", color: Theme.protein, icon: .system("figure.arms.open"),
                value: Binding(
                    get: { vm.analysisResult?.protein },
                    set: { vm.analysisResult?.protein = $0 }
                )
            )
            optionalField(
                label: "Carbs", unit: "g", color: Theme.carbs, icon: .system("leaf.fill"),
                value: Binding(
                    get: { vm.analysisResult?.carbs },
                    set: { vm.analysisResult?.carbs = $0 }
                )
            )
            optionalField(
                label: "Fat", unit: "g", color: Theme.fat, icon: .system("drop.halffull"),
                value: Binding(
                    get: { vm.analysisResult?.fat },
                    set: { vm.analysisResult?.fat = $0 }
                )
            )
        }
    }

    private func optionalField(
        label: String, unit: String, color: Color, icon: IconRef,
        value: Binding<Double?>, large: Bool = false
    ) -> some View {
        let isUnknown = value.wrappedValue == nil
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                icon.image.font(.caption2).foregroundStyle(color)
                Text(label).font(.caption).foregroundStyle(Theme.textSecondary)
            }
            if isUnknown {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text("-")
                        .font(large ? .system(size: 36, weight: .bold, design: .rounded) : .system(.title3, design: .rounded, weight: .bold))
                        .foregroundStyle(Theme.textTertiary)
                    Text(unit).font(large ? .subheadline : .caption).foregroundStyle(Theme.textSecondary)
                }
            } else {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    TextField("0", value: Binding(
                        get: { value.wrappedValue ?? 0 },
                        set: { value.wrappedValue = $0 }
                    ), format: .number.precision(.fractionLength(0)))
                        .font(large ? .system(size: 36, weight: .bold, design: .rounded) : .system(.title3, design: .rounded, weight: .bold))
                        .foregroundStyle(Theme.textPrimary)
                        .keyboardType(.numberPad)
                        .fixedSize()
                    Text(unit).font(large ? .subheadline : .caption).foregroundStyle(Theme.textSecondary)
                }
            }
            Button {
                value.wrappedValue = isUnknown ? 0 : nil
            } label: {
                Text(isUnknown ? "Enter value" : "I don't know")
                    .font(.caption2)
                    .foregroundStyle(Theme.accent)
            }
        }
        .padding(large ? 14 : 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
