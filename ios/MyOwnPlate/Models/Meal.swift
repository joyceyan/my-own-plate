import Foundation
import UIKit

struct Meal: Identifiable, Codable {
    let id: UUID
    var imageFileName: String?
    var description: String
    var nutrition: NutritionResult
    var timestamp: Date

    init(
        id: UUID = UUID(),
        imageFileName: String? = nil,
        description: String = "",
        nutrition: NutritionResult,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.imageFileName = imageFileName
        self.description = description
        self.nutrition = nutrition
        self.timestamp = timestamp
    }

    var formattedTime: String {
        timestamp.formatted(date: .omitted, time: .shortened)
    }

    // MARK: - Image I/O

    private static var imagesDirectory: URL {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("meal_images", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Save a UIImage to disk and return the filename.
    static func saveImage(_ image: UIImage) -> String? {
        guard let data = image.jpegData(compressionQuality: 0.8) else { return nil }
        let fileName = UUID().uuidString + ".jpg"
        let url = imagesDirectory.appendingPathComponent(fileName)
        do {
            try data.write(to: url)
            return fileName
        } catch {
            return nil
        }
    }

    /// Load the image from disk, if it exists.
    func loadImage() -> UIImage? {
        guard let fileName = imageFileName else { return nil }
        let url = Self.imagesDirectory.appendingPathComponent(fileName)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return UIImage(data: data)
    }

    /// Delete the image file from disk.
    func deleteImage() {
        guard let fileName = imageFileName else { return }
        let url = Self.imagesDirectory.appendingPathComponent(fileName)
        try? FileManager.default.removeItem(at: url)
    }
}
