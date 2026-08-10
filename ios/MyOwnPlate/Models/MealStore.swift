import Foundation

@MainActor
@Observable
final class MealStore {
    private(set) var meals: [Meal] = []
    var dailyGoal: DailyGoal {
        didSet { dailyGoal.save() }
    }

    private static var mealsFileURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("meals.json")
    }

    init() {
        self.dailyGoal = DailyGoal.load()
        self.meals = Self.loadMeals()
    }

    // MARK: - CRUD

    func add(_ meal: Meal) {
        meals.append(meal)
        saveMeals()
    }

    func update(_ meal: Meal) {
        guard let index = meals.firstIndex(where: { $0.id == meal.id }) else { return }
        meals[index] = meal
        saveMeals()
    }

    func delete(_ meal: Meal) {
        meal.deleteImage()
        meals.removeAll { $0.id == meal.id }
        saveMeals()
    }

    // MARK: - Queries

    func meals(for date: Date) -> [Meal] {
        meals.filter { Calendar.current.isDate($0.timestamp, inSameDayAs: date) }
            .sorted { $0.timestamp < $1.timestamp }
    }

    func totals(for date: Date) -> NutritionResult {
        meals(for: date).reduce(.zero) { $0 + $1.nutrition }
    }

    func remaining(for date: Date) -> NutritionResult {
        let dayMeals = meals(for: date)
        if dayMeals.isEmpty {
            // No meals logged — full goal remaining.
            return NutritionResult(
                calories: dailyGoal.calories,
                protein: dailyGoal.protein,
                fat: dailyGoal.fat,
                carbs: dailyGoal.carbs
            )
        }
        let consumed = totals(for: date)
        return NutritionResult(
            calories: consumed.calories.map { max(0, dailyGoal.calories - $0) },
            protein: consumed.protein.map { max(0, dailyGoal.protein - $0) },
            fat: consumed.fat.map { max(0, dailyGoal.fat - $0) },
            carbs: consumed.carbs.map { max(0, dailyGoal.carbs - $0) }
        )
    }

    var todayMeals: [Meal] { meals(for: Date()) }
    var todayTotals: NutritionResult { totals(for: Date()) }
    var todayRemaining: NutritionResult { remaining(for: Date()) }

    /// All unique days that have meals, most recent first.
    var daysWithMeals: [Date] {
        let calendar = Calendar.current
        let grouped = Dictionary(grouping: meals) { calendar.startOfDay(for: $0.timestamp) }
        return grouped.keys.sorted(by: >)
    }

    // MARK: - Sample Data

    #if DEBUG
    func loadSampleDataIfEmpty() {
        guard meals.isEmpty else { return }

        let calendar = Calendar.current
        let now = Date()

        // Meal templates: (description, calories, protein, fat, carbs)
        let breakfasts: [(String, Double, Double, Double, Double)] = [
            ("Oatmeal with banana", 350, 12, 8, 58),
            ("Greek yogurt & granola", 420, 24, 14, 48),
            ("Scrambled eggs & toast", 480, 28, 22, 38),
            ("Avocado toast", 390, 14, 18, 42),
            ("Protein smoothie", 340, 30, 8, 40),
            ("Pancakes with syrup", 520, 12, 16, 78),
            ("Overnight oats", 380, 16, 10, 54),
        ]

        let lunches: [(String, Double, Double, Double, Double)] = [
            ("Grilled chicken salad", 520, 42, 22, 28),
            ("Turkey sandwich", 580, 34, 18, 62),
            ("Burrito bowl", 650, 38, 24, 68),
            ("Poke bowl", 560, 36, 16, 64),
            ("Chicken Caesar wrap", 510, 32, 20, 48),
            ("Lentil soup & bread", 480, 22, 12, 68),
            ("Sushi (8 pc)", 440, 20, 10, 66),
        ]

        let dinners: [(String, Double, Double, Double, Double)] = [
            ("Salmon & roasted veggies", 620, 44, 28, 32),
            ("Pasta with meat sauce", 720, 36, 22, 82),
            ("Stir-fry with rice", 580, 30, 18, 72),
            ("Grilled steak & potato", 750, 52, 32, 48),
            ("Chicken tikka masala", 680, 38, 26, 64),
            ("Fish tacos", 540, 28, 20, 56),
            ("Bean & cheese quesadilla", 610, 26, 28, 62),
        ]

        let snacks: [(String, Double, Double, Double, Double)] = [
            ("Apple & peanut butter", 280, 8, 16, 28),
            ("Protein bar", 220, 20, 8, 24),
            ("Trail mix", 260, 8, 16, 22),
            ("Cheese & crackers", 240, 10, 14, 20),
            ("Banana", 105, 1, 0, 27),
        ]

        var sampleMeals: [Meal] = []

        for dayOffset in 0..<30 {
            guard let date = calendar.date(byAdding: .day, value: -dayOffset, to: now) else { continue }

            // Skip ~20% of days randomly to look realistic
            if dayOffset > 0 && dayOffset.hashValue % 5 == 0 { continue }

            let dayIndex = dayOffset

            // Breakfast ~8am
            let b = breakfasts[dayIndex % breakfasts.count]
            if let ts = calendar.date(bySettingHour: 8, minute: 15 + (dayIndex * 7) % 30, second: 0, of: date) {
                sampleMeals.append(Meal(
                    description: b.0,
                    nutrition: NutritionResult(calories: b.1, protein: b.2, fat: b.3, carbs: b.4),
                    timestamp: ts
                ))
            }

            // Lunch ~12:30pm
            let l = lunches[dayIndex % lunches.count]
            if let ts = calendar.date(bySettingHour: 12, minute: 20 + (dayIndex * 11) % 40, second: 0, of: date) {
                sampleMeals.append(Meal(
                    description: l.0,
                    nutrition: NutritionResult(calories: l.1, protein: l.2, fat: l.3, carbs: l.4),
                    timestamp: ts
                ))
            }

            // Dinner ~7pm
            let d = dinners[dayIndex % dinners.count]
            if let ts = calendar.date(bySettingHour: 19, minute: 5 + (dayIndex * 13) % 45, second: 0, of: date) {
                sampleMeals.append(Meal(
                    description: d.0,
                    nutrition: NutritionResult(calories: d.1, protein: d.2, fat: d.3, carbs: d.4),
                    timestamp: ts
                ))
            }

            // Snack on ~60% of days, ~3:30pm
            if dayIndex % 3 != 2 {
                let s = snacks[dayIndex % snacks.count]
                if let ts = calendar.date(bySettingHour: 15, minute: 20 + (dayIndex * 9) % 30, second: 0, of: date) {
                    sampleMeals.append(Meal(
                        description: s.0,
                        nutrition: NutritionResult(calories: s.1, protein: s.2, fat: s.3, carbs: s.4),
                        timestamp: ts
                    ))
                }
            }
        }

        meals = sampleMeals
        saveMeals()
    }
    #endif

    // MARK: - Persistence

    private func saveMeals() {
        do {
            let data = try JSONEncoder().encode(meals)
            try data.write(to: Self.mealsFileURL, options: .atomic)
        } catch {
            #if DEBUG
            print("Failed to save meals: \(error)")
            #endif
        }
    }

    private static func loadMeals() -> [Meal] {
        guard let data = try? Data(contentsOf: mealsFileURL),
              let meals = try? JSONDecoder().decode([Meal].self, from: data) else {
            return []
        }
        return meals
    }
}
