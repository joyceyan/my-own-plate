import SwiftUI

/// Wraps either an SF Symbol name or a custom asset image name.
enum IconRef {
    case system(String)
    case custom(String)

    var image: Image {
        switch self {
        case .system(let name): Image(systemName: name)
        case .custom(let name): Image(name).renderingMode(.template)
        }
    }
}

struct TodayView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel
    @State private var selectedMeal: Meal?

    private var store: MealStore { viewModel.mealStore }
    private var totals: NutritionResult { store.todayTotals }
    private var goal: DailyGoal { store.dailyGoal }
    private var remaining: NutritionResult { store.todayRemaining }
    private var calorieProgress: Double {
        guard let cal = totals.calories, goal.calories > 0 else { return 0 }
        return min(cal / goal.calories, 1.0)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                calorieCard
                macroCards
                mealsSection
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 80)
        }
        .background(Theme.background)
        .fullScreenCover(item: $selectedMeal) { meal in
            MealDetailView(meal: meal)
        }
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("MyOwnPlate")
                .font(.title2.weight(.bold))
                .foregroundStyle(Theme.textPrimary)

            Text("Today")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
        }
    }

    // MARK: - Calorie Card

    private var calorieCard: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(remaining.calories.map { "\(Int($0))" } ?? "-")
                    .font(.system(size: 44, weight: .bold, design: .rounded))
                    .foregroundStyle(Theme.textPrimary)
                Text("Calories left")
                    .font(.subheadline)
                    .foregroundStyle(Theme.textSecondary)
            }

            Spacer()

            // Calorie ring
            ZStack {
                Circle()
                    .stroke(Color.white.opacity(0.1), lineWidth: 10)
                    .frame(width: 80, height: 80)
                Circle()
                    .trim(from: 0, to: calorieProgress)
                    .stroke(Theme.calories, style: StrokeStyle(lineWidth: 10, lineCap: .round))
                    .frame(width: 80, height: 80)
                    .rotationEffect(.degrees(-90))
                Image(systemName: "flame")
                    .font(.title3)
                    .foregroundStyle(Theme.calories)
            }
        }
        .padding(20)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 20))
    }

    // MARK: - Macro Cards

    private var macroCards: some View {
        HStack(spacing: 10) {
            macroCard(
                label: "Protein",
                grams: totals.protein,
                target: goal.protein,
                color: Theme.protein,
                icon: .system("figure.arms.open")
            )
            macroCard(
                label: "Carbs",
                grams: totals.carbs,
                target: goal.carbs,
                color: Theme.carbs,
                icon: .system("leaf.fill")
            )
            macroCard(
                label: "Fats",
                grams: totals.fat,
                target: goal.fat,
                color: Theme.fat,
                icon: .system("drop.halffull")
            )
        }
    }

    private func macroCard(label: String, grams: Double?, target: Double, color: Color, icon: IconRef) -> some View {
        let left = grams.map { max(0, target - $0) }
        let progress = grams.map { target > 0 ? min($0 / target, 1.0) : 0 } ?? 0
        let over = grams.map { $0 > target } ?? false

        return VStack(alignment: .leading, spacing: 8) {
            Text(grams.map { "\(Int(over ? $0 - target : (left ?? 0)))g" } ?? "-")
                .font(.system(.title3, design: .rounded, weight: .bold))
                .foregroundStyle(Theme.textPrimary)
            Text(grams != nil ? "\(label) \(over ? "over" : "left")" : label)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)

            ZStack {
                Circle()
                    .stroke(color.opacity(0.2), lineWidth: 6)
                    .frame(width: 44, height: 44)
                Circle()
                    .trim(from: 0, to: progress)
                    .stroke(color, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                    .frame(width: 44, height: 44)
                    .rotationEffect(.degrees(-90))
                icon.image
                    .font(.caption)
                    .foregroundStyle(color)
            }
            .frame(maxWidth: .infinity)
        }
        .padding(12)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Meals Section

    private var mealsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Recently uploaded")
                .font(.headline)
                .foregroundStyle(Theme.textPrimary)

            if store.todayMeals.isEmpty {
                Text("No meals logged yet")
                    .font(.subheadline)
                    .foregroundStyle(Theme.textTertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                    .padding(.vertical, 24)
            } else {
                ForEach(store.todayMeals) { meal in
                    Button { selectedMeal = meal } label: {
                        MealCard(meal: meal)
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }
}

// MARK: - Meal Card

struct MealCard: View {
    let meal: Meal

    private var hasImage: Bool { meal.imageFileName != nil }

    var body: some View {
        Group {
            if hasImage {
                photoLayout
            } else {
                textLayout
            }
        }
        .padding(12)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Photo layout (thumbnail + info)

    private var photoLayout: some View {
        HStack(spacing: 14) {
            if let uiImage = meal.loadImage() {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 72, height: 72)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
            }

            VStack(alignment: .leading, spacing: 6) {
                headerRow
                calorieRow
                macroRow
            }
        }
    }

    // MARK: - Text-only layout (no photo)

    private var textLayout: some View {
        VStack(alignment: .leading, spacing: 10) {
            headerRow

            HStack(spacing: 0) {
                // Calories on the left
                HStack(spacing: 4) {
                    Image(systemName: "flame")
                        .font(.caption)
                        .foregroundStyle(Theme.calories)
                    Text(meal.nutrition.calories.map { "\(Int($0))" } ?? "-")
                        .font(.system(.body, design: .rounded, weight: .semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text("kcal")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }

                Spacer()

                // Macros on the right
                HStack(spacing: 12) {
                    macroChip(icon: .system("figure.arms.open"), value: meal.nutrition.protein, color: Theme.protein)
                    macroChip(icon: .system("leaf.fill"), value: meal.nutrition.carbs, color: Theme.carbs)
                    macroChip(icon: .system("drop.halffull"), value: meal.nutrition.fat, color: Theme.fat)
                }
            }
        }
    }

    // MARK: - Shared components

    private var headerRow: some View {
        HStack {
            Text(meal.description)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
                .lineLimit(1)
            Spacer()
            Text(meal.formattedTime)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
    }

    private var calorieRow: some View {
        HStack(spacing: 4) {
            Image(systemName: "flame")
                .font(.caption2)
                .foregroundStyle(Theme.calories)
            Text(meal.nutrition.calories.map { "\(Int($0)) kcal" } ?? "- kcal")
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
        }
    }

    private var macroRow: some View {
        HStack(spacing: 12) {
            macroTag(icon: .system("figure.arms.open"), value: meal.nutrition.protein, color: Theme.protein)
            macroTag(icon: .system("leaf.fill"), value: meal.nutrition.carbs, color: Theme.carbs)
            macroTag(icon: .system("drop.halffull"), value: meal.nutrition.fat, color: Theme.fat)
        }
    }

    private func macroTag(icon: IconRef, value: Double?, color: Color) -> some View {
        HStack(spacing: 3) {
            icon.image
                .font(.system(size: 9))
                .foregroundStyle(color)
            Text(value.map { "\(Int($0))g" } ?? "-")
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)
        }
    }

    private func macroChip(icon: IconRef, value: Double?, color: Color) -> some View {
        HStack(spacing: 3) {
            icon.image
                .font(.system(size: 9))
                .foregroundStyle(color)
            Text(value.map { "\(Int($0))g" } ?? "-")
                .font(.caption2)
                .foregroundStyle(Theme.textSecondary)
        }
    }
}

