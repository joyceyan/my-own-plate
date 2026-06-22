import SwiftUI

struct RecalculatePlanView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel
    @Environment(\.dismiss) private var dismiss

    @State private var stepIndex = 0

    // Step 0: Height & Weight
    @State private var usesMetric = false
    @State private var editFeet = ""
    @State private var editInches = ""
    @State private var editCm = ""
    @State private var editWeightLbs = ""
    @State private var editWeightKg = ""

    // Step 1: Goal
    @State private var selectedGoal: UserProfile.Goal?

    // Step 2: Target Weight (conditional)
    @State private var editTargetLbs = ""
    @State private var editTargetKg = ""

    // Step 3: Activity
    @State private var selectedActivity: UserProfile.ActivityLevel?
    @State private var exerciseFrequency: Int = 3

    // Step 4: Plan review
    @State private var planCalories = ""
    @State private var planProtein = ""
    @State private var planFat = ""
    @State private var planCarbs = ""

    private enum Step {
        case body, goal, targetWeight, activity, plan
    }

    private var steps: [Step] {
        var s: [Step] = [.body, .goal]
        if selectedGoal == .lose || selectedGoal == .gain {
            s.append(.targetWeight)
        }
        s.append(contentsOf: [.activity, .plan])
        return s
    }

    private var currentStep: Step { steps[stepIndex] }

    private var canContinue: Bool {
        switch currentStep {
        case .body: hasValidBody
        case .goal: selectedGoal != nil
        case .targetWeight: hasValidTargetWeight
        case .activity: selectedActivity != nil
        case .plan: Double(planCalories) != nil && Double(planProtein) != nil && Double(planFat) != nil && Double(planCarbs) != nil
        }
    }

    private var hasValidBody: Bool {
        if usesMetric {
            return Double(editCm) != nil && Double(editWeightKg) != nil
        } else {
            return Int(editFeet) != nil && Double(editWeightLbs) != nil
        }
    }

    private var currentWeightKg: Double? {
        if usesMetric {
            return Double(editWeightKg)
        } else if let lbs = Double(editWeightLbs) {
            return lbs / 2.20462
        }
        return nil
    }

    private var targetWeightKg: Double? {
        if usesMetric {
            return Double(editTargetKg)
        } else if let lbs = Double(editTargetLbs) {
            return lbs / 2.20462
        }
        return nil
    }

    private var hasValidTargetWeight: Bool {
        guard let current = currentWeightKg, let target = targetWeightKg, target > 0 else {
            return false
        }
        switch selectedGoal {
        case .lose: return target < current
        case .gain: return target > current
        default: return true
        }
    }

    private var targetWeightError: String? {
        guard let current = currentWeightKg, let target = targetWeightKg, target > 0 else {
            return nil
        }
        if selectedGoal == .lose && target >= current {
            return "Target weight must be less than your current weight"
        }
        if selectedGoal == .gain && target <= current {
            return "Target weight must be more than your current weight"
        }
        return nil
    }

    var body: some View {
        VStack(spacing: 0) {
            // Nav + Progress
            HStack(spacing: 12) {
                Button {
                    if stepIndex > 0 {
                        withAnimation { stepIndex -= 1 }
                    } else {
                        dismiss()
                    }
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.body.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                }

                HStack(spacing: 6) {
                    ForEach(0..<steps.count, id: \.self) { i in
                        Capsule()
                            .fill(i <= stepIndex ? Color.white : Color.white.opacity(0.2))
                            .frame(height: 4)
                    }
                }
            }
            .padding(.top, 12)
            .padding(.horizontal, 20)

            Spacer()

            Group {
                switch currentStep {
                case .body: bodyStep
                case .goal: goalStep
                case .targetWeight: targetWeightStep
                case .activity: activityStep
                case .plan: planStep
                }
            }

            Spacer()

            Button {
                advance()
            } label: {
                Text(currentStep == .plan ? "Save Plan" : "Continue")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .background(canContinue ? Color.white : Color.white.opacity(0.3))
                    .foregroundStyle(Theme.background)
                    .clipShape(RoundedRectangle(cornerRadius: 14))
            }
            .disabled(!canContinue)
            .padding(.horizontal, 20)
            .padding(.bottom, 32)
        }
        .background(Theme.background)
        .onAppear { loadFromProfile() }
    }

    // MARK: - Body

    private var bodyStep: some View {
        VStack(spacing: 32) {
            header(title: "Height & Weight")

            VStack(spacing: 16) {
                Picker("Units", selection: $usesMetric) {
                    Text("Imperial").tag(false)
                    Text("Metric").tag(true)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 20)

                if usesMetric {
                    fieldRow(label: "Height") {
                        HStack(spacing: 6) {
                            TextField("170", text: $editCm)
                                .keyboardType(.numberPad).font(.body).foregroundStyle(Theme.textPrimary)
                            Text("cm").font(.caption).foregroundStyle(Theme.textSecondary)
                        }
                    }
                    fieldRow(label: "Weight") {
                        HStack(spacing: 6) {
                            TextField("70", text: $editWeightKg)
                                .keyboardType(.decimalPad).font(.body).foregroundStyle(Theme.textPrimary)
                            Text("kg").font(.caption).foregroundStyle(Theme.textSecondary)
                        }
                    }
                } else {
                    fieldRow(label: "Height") {
                        HStack(spacing: 12) {
                            HStack(spacing: 4) {
                                TextField("5", text: $editFeet)
                                    .keyboardType(.numberPad).font(.body).foregroundStyle(Theme.textPrimary).frame(width: 40)
                                Text("ft").font(.caption).foregroundStyle(Theme.textSecondary)
                            }
                            HStack(spacing: 4) {
                                TextField("10", text: $editInches)
                                    .keyboardType(.numberPad).font(.body).foregroundStyle(Theme.textPrimary).frame(width: 40)
                                Text("in").font(.caption).foregroundStyle(Theme.textSecondary)
                            }
                        }
                    }
                    fieldRow(label: "Weight") {
                        HStack(spacing: 6) {
                            TextField("155", text: $editWeightLbs)
                                .keyboardType(.decimalPad).font(.body).foregroundStyle(Theme.textPrimary)
                            Text("lbs").font(.caption).foregroundStyle(Theme.textSecondary)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Goal

    private var goalStep: some View {
        VStack(spacing: 32) {
            header(title: "What is your goal?")

            VStack(spacing: 12) {
                ForEach(UserProfile.Goal.allCases, id: \.self) { goal in
                    selectionRow(label: goal.rawValue, selected: selectedGoal == goal) {
                        selectedGoal = goal
                    }
                }
            }
            .padding(.horizontal, 20)
        }
    }

    // MARK: - Target Weight

    private var targetWeightStep: some View {
        VStack(spacing: 32) {
            header(title: "What's your target weight?")

            VStack(spacing: 12) {
                if usesMetric {
                    fieldRow(label: "Target weight") {
                        HStack(spacing: 6) {
                            TextField("65", text: $editTargetKg)
                                .keyboardType(.decimalPad).font(.body).foregroundStyle(Theme.textPrimary)
                            Text("kg").font(.caption).foregroundStyle(Theme.textSecondary)
                        }
                    }
                } else {
                    fieldRow(label: "Target weight") {
                        HStack(spacing: 6) {
                            TextField("140", text: $editTargetLbs)
                                .keyboardType(.decimalPad).font(.body).foregroundStyle(Theme.textPrimary)
                            Text("lbs").font(.caption).foregroundStyle(Theme.textSecondary)
                        }
                    }
                }

                if let error = targetWeightError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(Theme.calories)
                        .padding(.horizontal, 20)
                }
            }
        }
    }

    // MARK: - Activity

    private var activityStep: some View {
        VStack(spacing: 32) {
            header(title: "How active are you?")

            VStack(spacing: 12) {
                ForEach(UserProfile.ActivityLevel.allCases, id: \.self) { level in
                    selectionRow(label: level.rawValue, selected: selectedActivity == level) {
                        selectedActivity = level
                    }
                }
            }
            .padding(.horizontal, 20)

            if selectedActivity == .exerciseRegularly {
                VStack(spacing: 12) {
                    Text("How many times per week?")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(Theme.textPrimary)

                    HStack(spacing: 0) {
                        ForEach(1...7, id: \.self) { n in
                            Button {
                                exerciseFrequency = n
                            } label: {
                                Text("\(n)")
                                    .font(.body.weight(.semibold))
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 12)
                                    .background(exerciseFrequency == n ? Theme.accent : Theme.cardBackground)
                                    .foregroundStyle(exerciseFrequency == n ? Theme.background : Theme.textPrimary)
                            }
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal, 20)
                }
                .transition(.opacity)
            }
        }
        .animation(.default, value: selectedActivity)
    }

    // MARK: - Plan

    private var planStep: some View {
        VStack(spacing: 24) {
            header(title: "Your updated plan")

            VStack(spacing: 12) {
                planField(label: "Calories", value: $planCalories, unit: "kcal", color: Theme.calories, icon: "flame")
                planField(label: "Protein", value: $planProtein, unit: "g", color: Theme.protein, icon: "figure.arms.open")
                planField(label: "Carbs", value: $planCarbs, unit: "g", color: Theme.carbs, icon: "leaf.fill")
                planField(label: "Fat", value: $planFat, unit: "g", color: Theme.fat, icon: "drop.halffull")
            }
            .padding(.horizontal, 20)
        }
    }

    private func planField(label: String, value: Binding<String>, unit: String, color: Color, icon: String) -> some View {
        HStack(spacing: 14) {
            Image(systemName: icon)
                .font(.body)
                .foregroundStyle(color)
                .frame(width: 28)
            Text(label)
                .font(.body.weight(.medium))
                .foregroundStyle(Theme.textPrimary)
            Spacer()
            HStack(spacing: 4) {
                TextField("0", text: value)
                    .keyboardType(.numberPad)
                    .font(.system(.body, design: .rounded, weight: .bold))
                    .foregroundStyle(Theme.textPrimary)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 60)
                Text(unit)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .padding(16)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - Shared

    private func header(title: String) -> some View {
        Text(title)
            .font(.title2.weight(.bold))
            .foregroundStyle(Theme.textPrimary)
    }

    private func selectionRow(label: String, selected: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack {
                Text(label)
                    .font(.body.weight(.medium))
                    .foregroundStyle(Theme.textPrimary)
                Spacer()
                Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(selected ? Theme.accent : Theme.textTertiary)
            }
            .padding(16)
            .background(selected ? Theme.accent.opacity(0.15) : Theme.cardBackground)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(selected ? Theme.accent : Color.clear, lineWidth: 1.5)
            )
        }
    }

    private func fieldRow<Content: View>(label: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label).font(.caption).foregroundStyle(Theme.textSecondary)
            content()
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.cardBackground)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding(.horizontal, 20)
    }

    // MARK: - Lifecycle

    private func loadFromProfile() {
        let p = viewModel.userProfile
        usesMetric = p.usesMetric
        if let cm = p.heightCm {
            editCm = "\(Int(cm))"
            let totalInches = cm / 2.54
            editFeet = "\(Int(totalInches) / 12)"
            editInches = "\(Int(totalInches) % 12)"
        }
        if let kg = p.weightKg {
            editWeightKg = "\(Int(kg))"
            editWeightLbs = "\(Int(kg * 2.20462))"
        }
        if let tkg = p.targetWeightKg {
            editTargetKg = "\(Int(tkg))"
            editTargetLbs = "\(Int(tkg * 2.20462))"
        }
        selectedGoal = p.goal
        selectedActivity = p.activityLevel
        exerciseFrequency = p.exerciseTimesPerWeek ?? 3
    }

    private func advance() {
        switch currentStep {
        case .body:
            saveBody()
        case .goal:
            viewModel.userProfile.goal = selectedGoal
        case .targetWeight:
            viewModel.userProfile.targetWeightKg = targetWeightKg
        case .activity:
            viewModel.userProfile.activityLevel = selectedActivity
            viewModel.userProfile.exerciseTimesPerWeek =
                selectedActivity == .exerciseRegularly ? exerciseFrequency : nil

            if let goals = viewModel.userProfile.recommendedGoals() {
                planCalories = "\(Int(goals.calories))"
                planProtein = "\(Int(goals.protein))"
                planFat = "\(Int(goals.fat))"
                planCarbs = "\(Int(goals.carbs))"
            }
        case .plan:
            let goals = DailyGoal(
                calories: Double(planCalories) ?? 2100,
                protein: Double(planProtein) ?? 150,
                fat: Double(planFat) ?? 70,
                carbs: Double(planCarbs) ?? 250
            )
            viewModel.mealStore.dailyGoal = goals
            dismiss()
            return
        }
        withAnimation { stepIndex += 1 }
    }

    private func saveBody() {
        viewModel.userProfile.usesMetric = usesMetric
        if usesMetric {
            viewModel.userProfile.heightCm = Double(editCm)
            viewModel.userProfile.weightKg = Double(editWeightKg)
        } else {
            let feet = Double(editFeet) ?? 0
            let inches = Double(editInches) ?? 0
            let totalInches = feet * 12 + inches
            viewModel.userProfile.heightCm = totalInches > 0 ? totalInches * 2.54 : nil
            if let lbs = Double(editWeightLbs) {
                viewModel.userProfile.weightKg = lbs / 2.20462
            } else {
                viewModel.userProfile.weightKg = nil
            }
        }
    }
}
