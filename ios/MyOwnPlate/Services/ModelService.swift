import Foundation
import UIKit

// The exact prompt used in training (from data/prepare_nutrition5k.py).
// Keeping this constant ensures consistency between training and inference.
let kNutritionPrompt = "Estimate the nutritional content of this food image. Respond as JSON with keys: calories (kcal), protein (g), fat (g), carbs (g)."

protocol ModelService: Sendable {
    func loadModel() async throws
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

// MARK: - Response Parsing

/// Parses JSON output from the model, with regex fallback.
/// Mirrors the logic in training/evaluate.py parse_completion().
func parseNutritionResult(from raw: String) throws -> NutritionResult {
    let nutrients = ["calories", "protein", "fat", "carbs"]

    // Strip markdown code fences
    let cleaned = raw.replacingOccurrences(
        of: "```(?:json)?\\s*", with: "", options: .regularExpression
    ).trimmingCharacters(in: .whitespacesAndNewlines)

    // Try JSON parse: find first { ... } block
    if let jsonRange = cleaned.range(of: "\\{[\\s\\S]*\\}", options: .regularExpression) {
        var jsonStr = String(cleaned[jsonRange])
        let openCount = jsonStr.filter { $0 == "{" }.count
        let closeCount = jsonStr.filter { $0 == "}" }.count
        if openCount > closeCount {
            jsonStr += "}"
        }
        if let data = jsonStr.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            var values = [String: Double]()
            for key in nutrients {
                if let val = parsed[key] as? Double {
                    values[key] = val
                } else if let val = parsed[key] as? Int {
                    values[key] = Double(val)
                } else if let str = parsed[key] as? String, let val = Double(str) {
                    values[key] = val
                }
            }
            if values.count == nutrients.count {
                return NutritionResult(
                    calories: values["calories"]!,
                    protein: values["protein"]!,
                    fat: values["fat"]!,
                    carbs: values["carbs"]!
                )
            }
        }
    }

    // Regex fallback
    var values = [String: Double]()
    for nutrient in nutrients {
        let pattern = "[\"']?\(nutrient)[\"']?\\s*[:=]\\s*([0-9]+(?:\\.[0-9]+)?)"
        if let range = raw.range(of: pattern, options: [.regularExpression, .caseInsensitive]) {
            let match = String(raw[range])
            if let numRange = match.range(of: "([0-9]+(?:\\.[0-9]+)?)", options: .regularExpression, range: match.range(of: "[:=]", options: .regularExpression)!.upperBound..<match.endIndex) {
                if let val = Double(String(match[numRange])) {
                    values[nutrient] = val
                }
            }
        }
    }

    if values.count == nutrients.count {
        return NutritionResult(
            calories: values["calories"]!,
            protein: values["protein"]!,
            fat: values["fat"]!,
            carbs: values["carbs"]!
        )
    }

    throw ModelServiceError.parseFailed(raw: raw)
}
