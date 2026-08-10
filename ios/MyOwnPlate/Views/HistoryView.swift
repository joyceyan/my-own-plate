import SwiftUI

struct HistoryView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel
    @State private var selectedRange = 0
    @State private var selectedMeal: Meal?

    private var store: MealStore { viewModel.mealStore }
    private let rangeLabels = ["Week", "Month", "All"]

    private var filteredDays: [Date] {
        let calendar = Calendar.current
        let now = Date()
        return store.daysWithMeals.filter { date in
            switch selectedRange {
            case 0: return calendar.dateComponents([.day], from: date, to: now).day ?? 0 < 7
            case 1: return calendar.dateComponents([.day], from: date, to: now).day ?? 0 < 30
            default: return true
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("History")
                .font(.title2.weight(.bold))
                .foregroundStyle(Theme.textPrimary)
                .padding(.horizontal, 20)
                .padding(.bottom, 12)

            Picker("Range", selection: $selectedRange) {
                ForEach(0..<rangeLabels.count, id: \.self) { i in
                    Text(rangeLabels[i]).tag(i)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 20)
            .padding(.bottom, 16)

            if filteredDays.isEmpty {
                Spacer()
                Text("No meals logged yet.")
                    .font(.subheadline)
                    .foregroundStyle(Theme.textTertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
                Spacer()
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(filteredDays, id: \.self) { day in
                            daySection(day)
                        }
                    }
                    .padding(.horizontal, 20)
                }
            }
        }
        .background(Theme.background)
        .fullScreenCover(item: $selectedMeal) { meal in
            MealDetailView(meal: meal)
        }
    }

    private func daySection(_ date: Date) -> some View {
        let totals = store.totals(for: date)
        let dayMeals = store.meals(for: date)
        let dateStr = Calendar.current.isDateInToday(date)
            ? "Today"
            : Calendar.current.isDateInYesterday(date)
                ? "Yesterday"
                : date.formatted(.dateTime.weekday(.wide).month().day())

        return VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text("\(dateStr) - \(totals.calories.map { "\(Int($0))" } ?? "-") cal")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Theme.textPrimary)
                Text("Protein \(totals.protein.map { "\(Int($0))g" } ?? "-")  ·  Carbs \(totals.carbs.map { "\(Int($0))g" } ?? "-")  ·  Fat \(totals.fat.map { "\(Int($0))g" } ?? "-")")
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
            .padding(.top, 16)

            ForEach(dayMeals) { meal in
                Button { selectedMeal = meal } label: {
                    MealCard(meal: meal)
                }
                .buttonStyle(.plain)
            }
        }
    }
}
