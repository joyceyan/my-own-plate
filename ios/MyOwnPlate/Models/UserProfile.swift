import Foundation

struct UserProfile: Codable, Equatable {
    var name: String
    var phone: String
    var birthday: Date?
    var sex: Sex
    var heightCm: Double?
    var weightKg: Double?
    var targetWeightKg: Double?
    var usesMetric: Bool
    var goal: Goal?
    var activityLevel: ActivityLevel?
    var exerciseTimesPerWeek: Int?

    enum Sex: String, Codable, CaseIterable {
        case male = "Male"
        case female = "Female"
        case other = "Other"
    }

    enum Goal: String, Codable, CaseIterable {
        case lose = "Lose weight"
        case maintain = "Maintain"
        case gain = "Gain weight"
    }

    enum ActivityLevel: String, Codable, CaseIterable {
        case notActive = "Not active"
        case exerciseRegularly = "I exercise regularly"
        case physicalJob = "I have a physically demanding job"
    }

    static let `default` = UserProfile(
        name: "",
        phone: "",
        birthday: nil,
        sex: .other,
        heightCm: nil,
        weightKg: nil,
        targetWeightKg: nil,
        usesMetric: false,
        goal: nil,
        activityLevel: nil,
        exerciseTimesPerWeek: nil
    )

    var age: Int? {
        guard let birthday else { return nil }
        return Calendar.current.dateComponents([.year], from: birthday, to: Date()).year
    }

    var formattedHeight: String? {
        guard let cm = heightCm else { return nil }
        if usesMetric {
            return "\(Int(cm)) cm"
        } else {
            let totalInches = cm / 2.54
            let feet = Int(totalInches) / 12
            let inches = Int(totalInches) % 12
            return "\(feet)'\(inches)\""
        }
    }

    var formattedWeight: String? {
        guard let kg = weightKg else { return nil }
        if usesMetric {
            return "\(Int(kg)) kg"
        } else {
            return "\(Int(kg * 2.20462)) lbs"
        }
    }

    // MARK: - Nutrition Calculation

    /// Physical Activity Level multiplier based on activity and exercise frequency.
    var pal: Double {
        switch activityLevel {
        case .notActive, nil:
            return 1.2
        case .physicalJob:
            return 2.0
        case .exerciseRegularly:
            switch exerciseTimesPerWeek ?? 0 {
            case 1...2: return 1.4
            case 3: return 1.6
            case 4...5: return 1.75
            default: return 2.0 // 6+
            }
        }
    }

    /// Mifflin-St Jeor BMR. Uses male formula for .other sex.
    var bmr: Double? {
        guard let kg = weightKg, let cm = heightCm, let age else { return nil }
        let base = 10.0 * kg + 6.25 * cm - 5.0 * Double(age)
        return sex == .female ? base - 161 : base + 5
    }

    /// Recommended daily goals based on profile, activity, and goal.
    func recommendedGoals() -> DailyGoal? {
        guard let bmr, let kg = weightKg else { return nil }

        // TDEE = BMR × PAL
        let tdee = bmr * pal

        // Adjust for goal — 1 lb/week = 500 cal/day deficit or surplus
        let targetCalories: Double
        switch goal {
        case .lose: targetCalories = tdee - 500
        case .gain: targetCalories = tdee + 500
        case .maintain, nil: targetCalories = tdee
        }
        let calories = max(1200, targetCalories)

        // Protein: 1g per lb of goal body weight (capped at 35% of calories)
        let goalKg = targetWeightKg ?? kg
        let goalLbs = goalKg * 2.20462
        let proteinFromWeight = goalLbs // grams
        let proteinCaloriesCap = calories * 0.35 / 4.0
        let protein = min(proteinFromWeight, proteinCaloriesCap)

        // Fat: 25% of calories (9 cal/g)
        let fat = calories * 0.25 / 9.0

        // Carbs: remainder (4 cal/g)
        let usedCalories = protein * 4.0 + fat * 9.0
        let carbs = max(0, (calories - usedCalories) / 4.0)

        return DailyGoal(
            calories: round(calories),
            protein: round(protein),
            fat: round(fat),
            carbs: round(carbs)
        )
    }

    // MARK: - Persistence

    private static let key = "userProfile"

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.key)
        }
    }

    static func load() -> UserProfile {
        guard let data = UserDefaults.standard.data(forKey: key),
              let profile = try? JSONDecoder().decode(UserProfile.self, from: data) else {
            return .default
        }
        return profile
    }
}
