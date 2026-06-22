import SwiftUI

struct ManualEntryView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel

    @State private var mealName: String = ""
    @State private var mealTime: Date = Date()

    @State private var caloriesText: String = ""
    @State private var proteinText: String = ""
    @State private var carbsText: String = ""
    @State private var fatText: String = ""

    @State private var caloriesUnknown = false
    @State private var proteinUnknown = false
    @State private var carbsUnknown = false
    @State private var fatUnknown = false

    private var canSave: Bool {
        !mealName.trimmingCharacters(in: .whitespaces).isEmpty
            && (caloriesUnknown || Double(caloriesText) != nil)
            && (proteinUnknown || Double(proteinText) != nil)
            && (carbsUnknown || Double(carbsText) != nil)
            && (fatUnknown || Double(fatText) != nil)
    }

    var body: some View {
        VStack(spacing: 0) {
            // Nav bar
            HStack {
                Button("Cancel") { viewModel.resetAnalysis() }
                    .font(.subheadline)
                    .foregroundStyle(Theme.textSecondary)
                Spacer()
                Text("Manual Entry")
                    .font(.headline)
                    .foregroundStyle(Theme.textPrimary)
                Spacer()
                Text("Cancel").font(.subheadline).opacity(0)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Meal name
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Meal name")
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                        TextField("e.g. Chicken salad", text: $mealName)
                            .font(.body)
                            .foregroundStyle(Theme.textPrimary)
                            .padding(14)
                            .background(Theme.cardBackground)
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                    }

                    // Time
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Time")
                            .font(.caption)
                            .foregroundStyle(Theme.textSecondary)
                        DatePicker("", selection: $mealTime, displayedComponents: [.date, .hourAndMinute])
                            .labelsHidden()
                            .colorScheme(.dark)
                    }

                    // Calories
                    nutritionField(
                        label: "Calories",
                        unit: "kcal",
                        color: Theme.calories,
                        icon: .system("flame"),
                        text: $caloriesText,
                        unknown: $caloriesUnknown,
                        large: true
                    )

                    // Macros
                    HStack(spacing: 10) {
                        nutritionField(
                            label: "Protein",
                            unit: "g",
                            color: Theme.protein,
                            icon: .system("figure.arms.open"),
                            text: $proteinText,
                            unknown: $proteinUnknown
                        )
                        nutritionField(
                            label: "Carbs",
                            unit: "g",
                            color: Theme.carbs,
                            icon: .system("leaf.fill"),
                            text: $carbsText,
                            unknown: $carbsUnknown
                        )
                        nutritionField(
                            label: "Fat",
                            unit: "g",
                            color: Theme.fat,
                            icon: .system("drop.halffull"),
                            text: $fatText,
                            unknown: $fatUnknown
                        )
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
            }

            // Save button
            Button {
                save()
            } label: {
                Text("Save")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(canSave ? Color.white : Color.white.opacity(0.3))
                    .foregroundStyle(Theme.background)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
            }
            .disabled(!canSave)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
        }
        .background(Theme.background)
    }

    // MARK: - Nutrition field

    private func nutritionField(
        label: String,
        unit: String,
        color: Color,
        icon: IconRef,
        text: Binding<String>,
        unknown: Binding<Bool>,
        large: Bool = false
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                icon.image
                    .font(.caption2)
                    .foregroundStyle(color)
                Text(label)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }

            if unknown.wrappedValue {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text("-")
                        .font(large
                            ? .system(size: 36, weight: .bold, design: .rounded)
                            : .system(.title3, design: .rounded, weight: .bold))
                        .foregroundStyle(Theme.textTertiary)
                    Text(unit)
                        .font(large ? .subheadline : .caption)
                        .foregroundStyle(Theme.textSecondary)
                }
            } else {
                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    TextField("0", text: text)
                        .font(large
                            ? .system(size: 36, weight: .bold, design: .rounded)
                            : .system(.title3, design: .rounded, weight: .bold))
                        .foregroundStyle(Theme.textPrimary)
                        .keyboardType(.numberPad)
                        .fixedSize()
                    Text(unit)
                        .font(large ? .subheadline : .caption)
                        .foregroundStyle(Theme.textSecondary)
                }
            }

            Button {
                unknown.wrappedValue.toggle()
                if unknown.wrappedValue {
                    text.wrappedValue = ""
                }
            } label: {
                Text(unknown.wrappedValue ? "Enter value" : "I don't know")
                    .font(.caption2)
                    .foregroundStyle(Theme.accent)
            }
        }
        .padding(large ? 14 : 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Save

    private func save() {
        let nutrition = NutritionResult(
            calories: caloriesUnknown ? nil : Double(caloriesText),
            protein: proteinUnknown ? nil : Double(proteinText),
            fat: fatUnknown ? nil : Double(fatText),
            carbs: carbsUnknown ? nil : Double(carbsText)
        )
        viewModel.saveManualMeal(
            description: mealName.trimmingCharacters(in: .whitespaces),
            timestamp: mealTime,
            nutrition: nutrition
        )
    }
}
