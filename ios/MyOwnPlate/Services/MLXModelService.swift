import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXRandom
import MLXVLM
import Tokenizers
import UIKit

// MARK: - Tokenizer Loader

/// Loads tokenizers from a local model directory using swift-transformers.
private struct HFTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return TokenizerWrapper(upstream)
    }
}

/// Bridges swift-transformers Tokenizer -> MLXLMCommon.Tokenizer.
private struct TokenizerWrapper: MLXLMCommon.Tokenizer {
    private let wrapped: any Tokenizers.Tokenizer

    init(_ tokenizer: any Tokenizers.Tokenizer) {
        self.wrapped = tokenizer
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        wrapped.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        wrapped.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? {
        wrapped.convertTokenToId(token)
    }

    func convertIdToToken(_ id: Int) -> String? {
        wrapped.convertIdToToken(id)
    }

    var bosToken: String? { wrapped.bosToken }
    var eosToken: String? { wrapped.eosToken }
    var unknownToken: String? { wrapped.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try wrapped.applyChatTemplate(
            messages: messages,
            tools: tools,
            additionalContext: additionalContext
        )
    }
}

// MARK: - MLX Model Service

/// On-device inference using mlx-swift.
/// Loads the model from the app bundle (no network download needed).
final class MLXModelService: ModelService, @unchecked Sendable {
    enum State: Sendable {
        case idle
        case loading
        case ready
        case error(String)
    }

    private var container: ModelContainer?

    @MainActor var state: State = .idle

    /// Load the model from the app bundle. Safe to call multiple times.
    @MainActor
    func loadModel() async throws {
        guard container == nil else { return }

        state = .loading

        do {
            guard let modelDir = Bundle.main.url(forResource: "Model", withExtension: nil) else {
                throw ModelServiceError.modelNotFound
            }

            let modelContainer = try await VLMModelFactory.shared.loadContainer(
                from: modelDir,
                using: HFTokenizerLoader()
            )

            container = modelContainer
            state = .ready
        } catch {
            state = .error(error.localizedDescription)
            throw error
        }
    }

    func analyze(image: UIImage) async throws -> (NutritionResult, String) {
        guard let container else {
            throw ModelServiceError.modelNotLoaded
        }

        guard let cgImage = image.cgImage else {
            throw ModelServiceError.imageEncodingFailed
        }

        let ciImage = CIImage(cgImage: cgImage)

        let userInput = UserInput(
            prompt: kNutritionPrompt,
            images: [.ciImage(ciImage)]
        )

        let input = try await container.prepare(input: userInput)

        let parameters = GenerateParameters(
            maxTokens: 512,
            temperature: 0.0
        )

        MLXRandom.seed(0)

        var rawOutput = ""
        let stream = try await container.generate(input: input, parameters: parameters)
        for await generation in stream {
            if let chunk = generation.chunk {
                rawOutput += chunk
            }
        }

        let nutrition = try parseNutritionResult(from: rawOutput)
        return (nutrition, rawOutput)
    }
}

// MARK: - Response Parsing

/// Parses JSON output from the model, with regex fallback.
/// Mirrors the logic in training/evaluate.py parse_completion().
private func parseNutritionResult(from raw: String) throws -> NutritionResult {
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
