import SwiftUI

enum AnalysisPhase {
    case photoPicker
    case camera
    case analyzing
    case review
    case manualEntry
}

@MainActor
@Observable
final class FoodAnalysisViewModel {
    let modelService: any ModelService
    let mealStore = MealStore()
    var userProfile: UserProfile = UserProfile.load() {
        didSet { userProfile.save() }
    }
    /// Set to true to always show onboarding on launch, even if previously completed.
    private static let DEBUG_RESET_ONBOARDING = true

    var hasCompletedOnboarding: Bool = {
        #if DEBUG
        if DEBUG_RESET_ONBOARDING { return false }
        #endif
        return UserDefaults.standard.bool(forKey: "hasCompletedOnboarding")
    }() {
        didSet { UserDefaults.standard.set(hasCompletedOnboarding, forKey: "hasCompletedOnboarding") }
    }
    var showOnboarding = false

    var modelReady = false
    var modelError: String?

    // Analysis flow state
    var showNewEntry = false
    var selectedImage: UIImage?
    var analysisPhase: AnalysisPhase?
    var analysisResult: NutritionResult?
    var foodDescription: String = ""
    var analysisError: String?

    init() {
        #if targetEnvironment(simulator)
        self.modelService = MockModelService()
        #else
        self.modelService = LlamaCppModelService()
        #endif
    }

    func loadModel() async {
        #if targetEnvironment(simulator)
        modelReady = true
        #else
        do {
            try await modelService.loadModel()
            modelReady = true
        } catch {
            modelError = error.localizedDescription
        }
        #endif
    }

    func startPhotoPicker() {
        showNewEntry = false
        analysisPhase = .photoPicker
    }

    func startCamera() {
        showNewEntry = false
        analysisPhase = .camera
    }

    func startManualEntry() {
        showNewEntry = false
        analysisPhase = .manualEntry
    }

    func saveManualMeal(description: String, timestamp: Date, nutrition: NutritionResult) {
        let meal = Meal(
            description: description,
            nutrition: nutrition,
            timestamp: timestamp
        )
        mealStore.add(meal)
        resetAnalysis()
    }

    func analyze(image: UIImage) async {
        selectedImage = image
        analysisPhase = .analyzing
        analysisError = nil

        do {
            let (nutrition, raw) = try await modelService.analyze(image: image)
            analysisResult = nutrition
            foodDescription = "Meal"
            analysisPhase = .review
        } catch {
            analysisError = error.localizedDescription
        }
    }

    func saveMeal() {
        guard let nutrition = analysisResult else { return }
        let imageFileName = selectedImage.flatMap { Meal.saveImage($0) }
        let meal = Meal(
            imageFileName: imageFileName,
            description: foodDescription,
            nutrition: nutrition
        )
        mealStore.add(meal)
        resetAnalysis()
    }

    func resetAnalysis() {
        selectedImage = nil
        analysisPhase = nil
        analysisResult = nil
        foodDescription = ""
        analysisError = nil
        showNewEntry = false
    }
}
