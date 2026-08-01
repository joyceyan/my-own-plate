import SwiftUI

@main
struct MyOwnPlateApp: App {
    @State private var viewModel = FoodAnalysisViewModel()
    @State private var entitlementService = EntitlementService()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(viewModel)
                .environment(entitlementService)
                .task {
                    #if DEBUG
                    viewModel.mealStore.loadSampleDataIfEmpty()
                    #endif
                    await viewModel.loadModel()
                    identifyUserForSuperwall()
                }
        }
    }

    private func identifyUserForSuperwall() {
        let key = "superwallUserId"
        let userId: String
        if let existing = UserDefaults.standard.string(forKey: key) {
            userId = existing
        } else {
            userId = UUID().uuidString
            UserDefaults.standard.set(userId, forKey: key)
        }
        entitlementService.identify(userId: userId, referred: false)
    }
}
