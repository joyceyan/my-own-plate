import SwiftUI

struct AnalysisFlowView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel

    var body: some View {
        Group {
            switch viewModel.analysisPhase {
            case .photoPicker:
                PhotoPickerView(
                    onSelect: { image in Task { await viewModel.analyze(image: image) } },
                    onCancel: { viewModel.resetAnalysis() }
                )
            case .camera:
                CameraView(
                    onCapture: { image in Task { await viewModel.analyze(image: image) } },
                    onCancel: { viewModel.resetAnalysis() }
                )
            case .analyzing:
                AnalyzingView()
            case .review:
                ReviewView()
            case .manualEntry:
                ManualEntryView()
            case nil:
                EmptyView()
            }
        }
        .animation(.default, value: viewModel.analysisPhase == .analyzing)
        .animation(.default, value: viewModel.analysisPhase == .review)
    }
}
