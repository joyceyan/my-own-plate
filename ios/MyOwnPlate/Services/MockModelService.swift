import Foundation
import UIKit

/// Returns realistic fake data for UI development and testing.
final class MockModelService: ModelService, Sendable {
    func loadModel() async throws {}

    func analyze(image: UIImage) async throws -> (NutritionResult, String) {
        try await Task.sleep(for: .seconds(1.5))

        let result = NutritionResult(
            calories: Double.random(in: 150...650),
            protein: Double.random(in: 5...45),
            fat: Double.random(in: 3...35),
            carbs: Double.random(in: 10...80)
        )

        let raw = """
        {"calories": \(result.calories ?? 0), "protein": \(result.protein ?? 0), "fat": \(result.fat ?? 0), "carbs": \(result.carbs ?? 0)}
        """
        return (result, raw)
    }
}
