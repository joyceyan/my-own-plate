import Foundation
import LlamaSwift
import UIKit

/// Prompt for food description (second inference pass).
private let kDescriptionPrompt = "Briefly describe the food in this image in one sentence."

/// On-device VLM inference using llama.cpp via the LlamaSwift package.
/// Loads a two-file GGUF model (language model + vision projector) from the app bundle.
final class LlamaCppModelService: ModelService, @unchecked Sendable {

    private var model: OpaquePointer?                          // llama_model *
    private var context: OpaquePointer?                        // llama_context *
    private var mtmdCtx: OpaquePointer?                        // mtmd_context *
    private var sampler: UnsafeMutablePointer<llama_sampler>?

    deinit {
        if let sampler { llama_sampler_free(sampler) }
        if let mtmdCtx { mtmd_free(mtmdCtx) }
        if let context { llama_free(context) }
        if let model { llama_model_free(model) }
        llama_backend_free()
    }

    @MainActor
    func loadModel() async throws {
        guard model == nil else { return }

        llama_backend_init()

        // Locate GGUF files in app bundle
        guard let modelsDir = Bundle.main.url(forResource: "GGUFModels", withExtension: nil) else {
            throw ModelServiceError.modelNotFound
        }
        let modelPath = modelsDir.appendingPathComponent("myownplate-q4km.gguf").path
        let mmprojPath = modelsDir.appendingPathComponent("mmproj-myownplate-f16.gguf").path

        guard FileManager.default.fileExists(atPath: modelPath),
              FileManager.default.fileExists(atPath: mmprojPath) else {
            throw ModelServiceError.modelNotFound
        }

        // Load language model
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = 99  // offload all layers to Metal GPU
        guard let loadedModel = llama_model_load_from_file(modelPath, modelParams) else {
            throw ModelServiceError.modelNotFound
        }
        self.model = loadedModel

        // Create context
        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = 2048
        ctxParams.n_batch = 512
        ctxParams.n_threads = 4
        guard let ctx = llama_init_from_model(loadedModel, ctxParams) else {
            throw ModelServiceError.modelNotLoaded
        }
        self.context = ctx

        // Load multimodal projector
        var mtmdParams = mtmd_context_params_default()
        mtmdParams.use_gpu = true
        mtmdParams.n_threads = 4
        guard let mctx = mtmd_init_from_file(mmprojPath, loadedModel, mtmdParams) else {
            throw ModelServiceError.modelNotFound
        }
        self.mtmdCtx = mctx

        // Set up sampler (greedy / temperature 0)
        let samplerParams = llama_sampler_chain_default_params()
        guard let chain = llama_sampler_chain_init(samplerParams) else {
            throw ModelServiceError.modelNotLoaded
        }
        llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        self.sampler = chain
    }

    func analyze(image: UIImage) async throws -> (NutritionResult, String) {
        guard let model, let context, let mtmdCtx, let sampler else {
            throw ModelServiceError.modelNotLoaded
        }

        // Convert UIImage to RGB pixel data
        let rgbData = try imageToRGB(image, size: 384)

        // Create mtmd bitmap — rgbData must stay alive until we're done with both calls
        guard let bitmap = mtmd_bitmap_init(384, 384, rgbData) else {
            throw ModelServiceError.imageEncodingFailed
        }
        defer { mtmd_bitmap_free(bitmap) }

        // First call: nutrition estimation
        let nutritionRaw = try runInference(prompt: kNutritionPrompt, bitmap: bitmap, maxTokens: 512)
        let nutrition = try parseNutritionResult(from: nutritionRaw)

        // Second call: food description
        let description = try runInference(prompt: kDescriptionPrompt, bitmap: bitmap, maxTokens: 128)

        return (nutrition, Self.cleanFoodDescription(description))
    }

    /// Strip markdown, quotes, and trailing punctuation from the description.
    private static func cleanFoodDescription(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // Remove surrounding quotes
        if s.hasPrefix("\"") && s.hasSuffix("\"") { s = String(s.dropFirst().dropLast()) }
        // Remove markdown bold
        s = s.replacingOccurrences(of: "**", with: "")
        // Trim again
        s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return s.isEmpty ? "Meal" : s
    }

    // MARK: - Single Inference Pass

    private func runInference(prompt: String, bitmap: OpaquePointer, maxTokens: Int32) throws -> String {
        guard let model, let context, let mtmdCtx, let sampler else {
            throw ModelServiceError.modelNotLoaded
        }

        let marker = String(cString: mtmd_get_marker(mtmdCtx))
        let promptStr = "<|im_start|>user\n\(marker)\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        let cPrompt = strdup(promptStr)!
        defer { free(cPrompt) }

        var textInput = mtmd_input_text()
        textInput.text = UnsafePointer(cPrompt)
        textInput.text_len = strlen(cPrompt)
        textInput.add_special = true
        textInput.parse_special = true

        guard let chunks = mtmd_input_chunks_init() else {
            throw ModelServiceError.imageEncodingFailed
        }
        defer { mtmd_input_chunks_free(chunks) }

        var bitmapPtr: OpaquePointer? = bitmap
        let tokenizeResult = mtmd_tokenize(mtmdCtx, chunks, &textInput, &bitmapPtr, 1)
        guard tokenizeResult == 0 else {
            throw ModelServiceError.parseFailed(raw: "mtmd_tokenize failed with code \(tokenizeResult)")
        }

        let memory = llama_get_memory(context)
        llama_memory_clear(memory, true)

        var nPast: Int32 = 0
        let evalResult = mtmd_helper_eval_chunks(
            mtmdCtx, context, chunks,
            0, 0,
            Int32(llama_n_batch(context)),
            true,
            &nPast
        )
        guard evalResult == 0 else {
            throw ModelServiceError.parseFailed(raw: "mtmd_helper_eval_chunks failed with code \(evalResult)")
        }

        let vocab = llama_model_get_vocab(model)
        var output = ""

        for _ in 0..<maxTokens {
            let token = llama_sampler_sample(sampler, context, -1)

            if llama_vocab_is_eog(vocab, token) {
                break
            }

            var buf = [CChar](repeating: 0, count: 256)
            let len = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, true)
            if len > 0 {
                buf[Int(len)] = 0
                output += String(cString: buf)
            }

            var tokenCopy = token
            var batch = llama_batch_get_one(&tokenCopy, 1)
            let decodeResult = llama_decode(context, batch)
            if decodeResult != 0 {
                break
            }
        }

        return output
    }

    // MARK: - Image Conversion

    /// Convert UIImage to a flat RGB byte array at the given size.
    private func imageToRGB(_ image: UIImage, size: Int) throws -> [UInt8] {
        // Resize
        let targetSize = CGSize(width: size, height: size)
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        image.draw(in: CGRect(origin: .zero, size: targetSize))
        guard let resized = UIGraphicsGetImageFromCurrentImageContext() else {
            UIGraphicsEndImageContext()
            throw ModelServiceError.imageEncodingFailed
        }
        UIGraphicsEndImageContext()

        guard let cgImage = resized.cgImage else {
            throw ModelServiceError.imageEncodingFailed
        }

        // Render to RGBA buffer
        let width = size
        let height = size
        let bytesPerRow = width * 4
        var rgba = [UInt8](repeating: 0, count: height * bytesPerRow)

        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(
                data: &rgba,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
              ) else {
            throw ModelServiceError.imageEncodingFailed
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Convert RGBA → RGB
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            rgb[i * 3 + 0] = rgba[i * 4 + 0]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return rgb
    }
}
