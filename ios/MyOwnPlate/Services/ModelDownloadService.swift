import Foundation

/// Downloads the on‑device VLM weights from a remote CDN on first launch.
/// The downloaded files live in the app’s Documents directory and persist across updates.
enum ModelDownloadError: LocalizedError {
    case notConfigured
    case insufficientSpace(requiredBytes: Int64, availableBytes: Int64)
    case downloadFailed(underlying: Error)
    case moveFailed(underlying: Error)
    case unexpectedResponse
    case cancelled

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "Model download URLs are not configured."
        case .insufficientSpace(let required, let available):
            let reqGB = Double(required) / 1_000_000_000
            let availGB = Double(available) / 1_000_000_000
            return String(
                format: "Not enough storage space. The model needs %.1f GB available, but only %.1f GB is free.",
                reqGB, availGB
            )
        case .downloadFailed(let error):
            return "Download failed: \(error.localizedDescription)"
        case .moveFailed(let error):
            return "Could not save the downloaded model: \(error.localizedDescription)"
        case .unexpectedResponse:
            return "The model server returned an unexpected response."
        case .cancelled:
            return "Download was cancelled."
        }
    }
}

// MARK: - Configuration

enum ModelDownloadConfig {
    // TODO: Replace these placeholder URLs with the real CDN URLs before submitting to the App Store.
    // Both files are currently expected at:
    //   - myownplate-q4km.gguf          (language model, ~1.1 GB)
    //   - mmproj-myownplate-f16.gguf    (vision projector, ~780 MB)
    static let modelURL = URL(string: "https://your-cdn.example.com/myownplate-q4km.gguf")
    static let mmprojURL = URL(string: "https://your-cdn.example.com/mmproj-myownplate-f16.gguf")

    /// Headroom included in the space check to leave room for temp files and extraction.
    static let requiredFreeBytes: Int64 = 2_500_000_000
}

// MARK: - Service

actor ModelDownloadService {

    private(set) var isDownloading = false

    /// A long‑timeout session so large model files don’t fail on slow connections.
    private static var downloadSession: URLSession {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 0
        config.timeoutIntervalForResource = 24 * 60 * 60
        config.waitsForConnectivity = true
        return URLSession(configuration: config)
    }

    // MARK: - Paths

    static var modelsDirectory: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static var modelFileURL: URL {
        modelsDirectory.appendingPathComponent("myownplate-q4km.gguf")
    }

    static var mmprojFileURL: URL {
        modelsDirectory.appendingPathComponent("mmproj-myownplate-f16.gguf")
    }

    static var modelFilesExist: Bool {
        FileManager.default.fileExists(atPath: modelFileURL.path) &&
        FileManager.default.fileExists(atPath: mmprojFileURL.path)
    }

    // MARK: - Public API

    /// Downloads both model files if they are not already present in Documents/Models.
    /// `onProgress` reports a fraction in [0, 1].
    func downloadIfNeeded(onProgress: @escaping @Sendable (Double) -> Void) async throws {
        guard !Self.modelFilesExist else {
            await MainActor.run { onProgress(1.0) }
            return
        }

        guard let modelURL = ModelDownloadConfig.modelURL,
              let mmprojURL = ModelDownloadConfig.mmprojURL else {
            throw ModelDownloadError.notConfigured
        }

        isDownloading = true
        defer { isDownloading = false }

        // Storage sanity check.
        let freeSpace = try availableFreeSpace()
        guard freeSpace >= ModelDownloadConfig.requiredFreeBytes else {
            throw ModelDownloadError.insufficientSpace(
                requiredBytes: ModelDownloadConfig.requiredFreeBytes,
                availableBytes: freeSpace
            )
        }

        let filePairs: [(source: URL, destination: URL)] = [
            (modelURL, Self.modelFileURL),
            (mmprojURL, Self.mmprojFileURL),
        ]

        let fm = FileManager.default

        for (index, pair) in filePairs.enumerated() {
            if Task.isCancelled {
                throw ModelDownloadError.cancelled
            }

            let tempURL: URL
            do {
                let (url, response) = try await Self.downloadSession.download(from: pair.source)
                guard let httpResponse = response as? HTTPURLResponse,
                      (200...299).contains(httpResponse.statusCode) else {
                    throw ModelDownloadError.unexpectedResponse
                }
                tempURL = url
            } catch {
                throw ModelDownloadError.downloadFailed(underlying: error)
            }

            do {
                if fm.fileExists(atPath: pair.destination.path) {
                    try fm.removeItem(at: pair.destination)
                }
                try fm.moveItem(at: tempURL, to: pair.destination)
            } catch {
                throw ModelDownloadError.moveFailed(underlying: error)
            }

            let progress = Double(index + 1) / Double(filePairs.count)
            await MainActor.run { onProgress(progress) }
        }

        await MainActor.run { onProgress(1.0) }
    }

    // MARK: - Helpers

    private func availableFreeSpace() throws -> Int64 {
        let attrs = try FileManager.default.attributesOfFileSystem(forPath: NSTemporaryDirectory())
        guard let freeSize = attrs[.systemFreeSize] as? Int64 else {
            return 0
        }
        return freeSize
    }
}
