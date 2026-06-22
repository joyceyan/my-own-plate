import SwiftUI

@main
struct MyOwnPlateApp: App {
    @State private var viewModel = FoodAnalysisViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(viewModel)
                .task {
                    #if DEBUG
                    viewModel.mealStore.loadSampleDataIfEmpty()
                    #endif
                    await viewModel.loadModel()
                }
        }
    }
}
