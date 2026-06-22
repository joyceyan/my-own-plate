import Foundation
import UIKit

// The exact prompt used in training (from data/prepare_nutrition5k.py).
// Keeping this constant ensures consistency between training and inference.
let kNutritionPrompt = "Estimate the nutritional content of this food image. Respond as JSON with keys: calories (kcal), protein (g), fat (g), carbs (g)."

protocol ModelService: Sendable {
    func analyze(image: UIImage) async throws -> (NutritionResult, String)
}

enum ModelServiceError: LocalizedError {
    case modelNotFound
    case modelNotLoaded
    case parseFailed(raw: String)
    case imageEncodingFailed

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "Model files not found in the app bundle."
        case .modelNotLoaded:
            return "The model is not loaded yet."
        case .parseFailed(let raw):
            return "Failed to parse nutrition data from model output: \(raw)"
        case .imageEncodingFailed:
            return "Failed to encode the image."
        }
    }
}
