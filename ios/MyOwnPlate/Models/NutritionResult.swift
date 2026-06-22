import Foundation

struct NutritionResult: Codable, Equatable {
    var calories: Double?
    var protein: Double?
    var fat: Double?
    var carbs: Double?

    /// Identity for reduce — all nil (no data).
    static let zero = NutritionResult(calories: nil, protein: nil, fat: nil, carbs: nil)

    static func + (lhs: NutritionResult, rhs: NutritionResult) -> NutritionResult {
        NutritionResult(
            calories: addOptional(lhs.calories, rhs.calories),
            protein: addOptional(lhs.protein, rhs.protein),
            fat: addOptional(lhs.fat, rhs.fat),
            carbs: addOptional(lhs.carbs, rhs.carbs)
        )
    }

    private static func addOptional(_ a: Double?, _ b: Double?) -> Double? {
        switch (a, b) {
        case (.some(let x), .some(let y)): x + y
        case (.some(let x), nil): x
        case (nil, .some(let y)): y
        case (nil, nil): nil
        }
    }
}
