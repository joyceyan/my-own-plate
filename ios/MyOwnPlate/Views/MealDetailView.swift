import SwiftUI

struct MealDetailView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel
    @Environment(\.dismiss) private var dismiss

    let meal: Meal

    @State private var isEditing = false
    @State private var editDescription: String = ""
    @State private var editTimestamp: Date = Date()
    @State private var editCalories: Double?
    @State private var editProtein: Double?
    @State private var editFat: Double?
    @State private var editCarbs: Double?
    @State private var showDeleteConfirm = false

    private var store: MealStore { viewModel.mealStore }

    var body: some View {
        VStack(spacing: 0) {
            navBar
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    // Photo
                    if let uiImage = meal.loadImage() {
                        Image(uiImage: uiImage)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 16))
                    }

                    if isEditing {
                        editingContent
                    } else {
                        viewingContent
                    }
                }
                .padding(.horizontal, 20)
                .padding(.bottom, 20)
            }
            bottomBar
        }
        .background(Theme.background)
        .confirmationDialog("Are you sure you want to delete this meal?", isPresented: $showDeleteConfirm, titleVisibility: .visible) {
            Button("Delete", role: .destructive) {
                store.delete(meal)
                dismiss()
            }
            Button("Cancel", role: .cancel) {}
        }
        .onAppear { loadFields() }
    }

    // MARK: - Nav bar

    private var navBar: some View {
        HStack {
            if !isEditing && !hasImage {
                Text(meal.description.isEmpty ? "Meal" : meal.description)
                    .font(.headline)
                    .foregroundStyle(Theme.textPrimary)
                    .lineLimit(1)
            } else if isEditing {
                Text("Edit Meal")
                    .font(.headline)
                    .foregroundStyle(Theme.textPrimary)
            }
            Spacer()
            Button("Cancel") {
                if isEditing {
                    isEditing = false
                    loadFields()
                } else {
                    dismiss()
                }
            }
            .font(.subheadline)
            .foregroundStyle(Theme.textSecondary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private var hasImage: Bool { meal.imageFileName != nil }

    // MARK: - Viewing content

    private var viewingContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Description (only in body for meals with images; no-image meals show it in nav bar)
            if hasImage {
                Text(meal.description.isEmpty ? "Meal" : meal.description)
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
            }

            // Time
            HStack(spacing: 6) {
                Image(systemName: "clock")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                Text(meal.timestamp.formatted(date: .abbreviated, time: .shortened))
                    .font(.subheadline)
                    .foregroundStyle(Theme.textSecondary)
            }

            // Calories
            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Image(systemName: "flame")
                    .font(.subheadline)
                    .foregroundStyle(Theme.calories)
                Text(meal.nutrition.calories.map { "\(Int($0))" } ?? "-")
                    .font(.system(size: 36, weight: .bold, design: .rounded))
                    .foregroundStyle(Theme.textPrimary)
                Text("kcal")
                    .font(.subheadline)
                    .foregroundStyle(Theme.textSecondary)
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.cardBackground)
            .clipShape(RoundedRectangle(cornerRadius: 12))

            // Macros
            HStack(spacing: 10) {
                readonlyMacro(label: "Protein", value: meal.nutrition.protein, color: Theme.protein, icon: .system("figure.arms.open"))
                readonlyMacro(label: "Carbs", value: meal.nutrition.carbs, color: Theme.carbs, icon: .system("leaf.fill"))
                readonlyMacro(label: "Fat", value: meal.nutrition.fat, color: Theme.fat, icon: .system("drop.halffull"))
            }
        }
    }

    private func readonlyMacro(label: String, value: Double?, color: Color, icon: IconRef) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                icon.image.font(.caption2).foregroundStyle(color)
                Text(label).font(.caption).foregroundStyle(Theme.textSecondary)
            }
            HStack(alignment: .firstTextBaseline, spacing: 2) {
                Text(value.map { "\(Int($0))" } ?? "-")
                    .font(.system(.title3, design: .rounded, weight: .bold))
                    .foregroundStyle(Theme.textPrimary)
                Text("g").font(.caption).foregroundStyle(Theme.textSecondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Editing content

    private var editingContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Description
            VStack(alignment: .leading, spacing: 6) {
                Text("Meal name")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                TextField("Describe your meal", text: $editDescription)
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
                DatePicker("", selection: $editTimestamp, displayedComponents: [.date, .hourAndMinute])
                    .labelsHidden()
                    .colorScheme(.dark)
            }

            // Calories
            optionalField(label: "Calories", unit: "kcal", color: Theme.calories, icon: .system("flame"),
                          value: $editCalories, large: true)

            // Macros
            HStack(spacing: 10) {
                optionalField(label: "Protein", unit: "g", color: Theme.protein, icon: .system("figure.arms.open"),
                              value: $editProtein)
                optionalField(label: "Carbs", unit: "g", color: Theme.carbs, icon: .system("leaf.fill"),
                              value: $editCarbs)
                optionalField(label: "Fat", unit: "g", color: Theme.fat, icon: .system("drop.halffull"),
                              value: $editFat)
            }
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

    // MARK: - Bottom bar

    private var bottomBar: some View {
        HStack(spacing: 12) {
            if isEditing {
                Button {
                    saveEdits()
                } label: {
                    Text("Save")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(Color.white)
                        .foregroundStyle(Theme.background)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                }
            } else {
                Button {
                    isEditing = true
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
                    showDeleteConfirm = true
                } label: {
                    Text("Delete")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(Theme.calories.opacity(0.2))
                        .foregroundStyle(Theme.calories)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                }
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    // MARK: - Helpers

    private func loadFields() {
        editDescription = meal.description
        editTimestamp = meal.timestamp
        editCalories = meal.nutrition.calories
        editProtein = meal.nutrition.protein
        editFat = meal.nutrition.fat
        editCarbs = meal.nutrition.carbs
    }

    private func saveEdits() {
        var updated = meal
        updated.description = editDescription
        updated.timestamp = editTimestamp
        updated.nutrition = NutritionResult(
            calories: editCalories,
            protein: editProtein,
            fat: editFat,
            carbs: editCarbs
        )
        store.update(updated)
        dismiss()
    }
}
