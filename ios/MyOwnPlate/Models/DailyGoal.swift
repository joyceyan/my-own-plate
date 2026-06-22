import Foundation

struct DailyGoal: Codable, Equatable {
    var calories: Double
    var protein: Double
    var fat: Double
    var carbs: Double

    static let `default` = DailyGoal(calories: 2100, protein: 150, fat: 70, carbs: 250)

    private static let key = "dailyGoal"

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.key)
        }
    }

    static func load() -> DailyGoal {
        guard let data = UserDefaults.standard.data(forKey: key),
              let goal = try? JSONDecoder().decode(DailyGoal.self, from: data) else {
            return .default
        }
        return goal
    }
}
