import SwiftUI

struct ContentView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel
    @State private var selectedTab = 0

    var body: some View {
        Group {
            if !viewModel.hasCompletedOnboarding {
                SplashView()
                    .fullScreenCover(isPresented: Binding(
                        get: { viewModel.showOnboarding },
                        set: { viewModel.showOnboarding = $0 }
                    )) {
                        OnboardingView()
                    }
            } else if let error = viewModel.modelError, !viewModel.modelReady {
                modelErrorView(error)
            } else if !viewModel.modelReady {
                ModelLoadingView()
            } else {
                mainView
            }
        }
        .preferredColorScheme(.dark)
    }

    private var mainView: some View {
        VStack(spacing: 0) {
            Group {
                switch selectedTab {
                case 0: TodayView()
                case 1: HistoryView()
                case 2: SettingsView()
                default: EmptyView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            // Custom tab bar
            customTabBar
        }
        .background(Theme.background)
        .sheet(isPresented: Binding(
            get: { viewModel.showNewEntry },
            set: { viewModel.showNewEntry = $0 }
        )) {
            NewEntrySheet()
                .presentationDetents([.medium])
        }
        .fullScreenCover(isPresented: Binding(
            get: { viewModel.analysisPhase != nil },
            set: { if !$0 { viewModel.resetAnalysis() } }
        )) {
            AnalysisFlowView()
        }
    }

    private var customTabBar: some View {
        HStack {
            tabBarItem(icon: "house.fill", label: "Home", index: 0)
            tabBarItem(icon: "clock.fill", label: "History", index: 1)
            tabBarItem(icon: "gearshape.fill", label: "Settings", index: 2)

            // + button
            Button {
                viewModel.showNewEntry = true
            } label: {
                Image(systemName: "plus")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(Theme.background)
                    .frame(width: 48, height: 48)
                    .background(Color.white)
                    .clipShape(Circle())
            }
            .padding(.leading, 4)
        }
        .padding(.horizontal, 20)
        .padding(.top, 10)
        .padding(.bottom, 6)
        .background(Theme.tabBarBackground)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(Color.white.opacity(0.08))
                .frame(height: 0.5)
        }
    }

    private func tabBarItem(icon: String, label: String, index: Int) -> some View {
        Button {
            selectedTab = index
        } label: {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 18))
                Text(label)
                    .font(.system(size: 10))
            }
            .foregroundStyle(selectedTab == index ? Theme.tabActive : Theme.tabInactive)
            .frame(maxWidth: .infinity)
        }
    }

    private func modelErrorView(_ message: String) -> some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 60))
                .foregroundStyle(Theme.protein)
            Text("Failed to load model")
                .font(.headline)
                .foregroundStyle(Theme.textPrimary)
            Text(message)
                .font(.callout)
                .foregroundStyle(Theme.textSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 32)
            Button("Retry") {
                Task { await viewModel.loadModel() }
            }
            .buttonStyle(.borderedProminent)
            Spacer()
        }
        .background(Theme.background)
    }
}
