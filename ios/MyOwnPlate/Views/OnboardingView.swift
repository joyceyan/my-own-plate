import SwiftUI

struct OnboardingView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel

    @State private var stepIndex = 0

    // Step: Name
    @State private var editName = ""

    // Step: Goal
    @State private var selectedGoal: UserProfile.Goal?

    // Step: Gender
    @State private var selectedSex: UserProfile.Sex?

    // Step: Height & Weight
    @State private var usesMetric = false
    @State private var editFeet = ""
    @State private var editInches = ""
    @State private var editCm = ""
    @State private var editWeightLbs = ""
    @State private var editWeightKg = ""

    // Step: Target Weight
    @State private var editTargetLbs = ""
    @State private var editTargetKg = ""

    // Step: Birthday
    @State private var birthday = Calendar.current.date(byAdding: .year, value: -25, to: Date()) ?? Date()

    // Step: Activity
    @State private var selectedActivity: UserProfile.ActivityLevel?
    @State private var exerciseFrequency: Int = 3

    // Step: Plan
    @State private var planCalories = ""
    @State private var planProtein = ""
    @State private var planFat = ""
    @State private var planCarbs = ""

    private enum Step {
        case name, goal, gender, body, targetWeight, birthday, activity, plan
    }

    private var steps: [Step] {
        var s: [Step] = [.name, .goal, .gender, .body]
        if selectedGoal == .lose || selectedGoal == .gain {
            s.append(.targetWeight)
        }
        s.append(contentsOf: [.birthday, .activity, .plan])
        return s
    }

    private var currentStep: Step { steps[stepIndex] }
    private var isLastStep: Bool { stepIndex == steps.count - 1 }

    private var canContinue: Bool {
        switch currentStep {
        case .name: !editName.trimmingCharacters(in: .whitespaces).isEmpty
        case .goal: selectedGoal != nil
        case .gender: selectedSex != nil
        case .body: hasValidBody
        case .targetWeight: hasValidTargetWeight
        case .birthday: birthday <= maxBirthday
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

    private var maxBirthday: Date {
        Calendar.current.date(byAdding: .year, value: -13, to: Date()) ?? Date()
    }

    var body: some View {
        VStack(spacing: 0) {
            // Nav + Progress bar
            HStack(spacing: 12) {
                if stepIndex > 0 {
                    Button {
                        withAnimation { stepIndex -= 1 }
                    } label: {
                        Image(systemName: "chevron.left")
                            .font(.body.weight(.semibold))
                            .foregroundStyle(Theme.textPrimary)
                    }
                }
                progressBar
            }
            .padding(.top, 12)
            .padding(.horizontal, 20)

            Spacer()

            // Current step content
            Group {
                switch currentStep {
                case .name: nameStep
                case .goal: goalStep
                case .gender: genderStep
                case .body: bodyStep
                case .targetWeight: targetWeightStep
                case .birthday: birthdayStep
                case .activity: activityStep
                case .plan: planStep
                }
            }

            Spacer()

            // Continue button
            Button {
                advance()
            } label: {
                Text(currentStep == .plan ? "Get Started" : "Continue")
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
    }

    // MARK: - Progress bar

    private var progressBar: some View {
        HStack(spacing: 6) {
            ForEach(0..<steps.count, id: \.self) { i in
                Capsule()
                    .fill(i <= stepIndex ? Color.white : Color.white.opacity(0.2))
                    .frame(height: 4)
            }
        }
    }

    // MARK: - Name

    private var nameStep: some View {
        VStack(spacing: 32) {
            questionHeader(title: "What's your name?", subtitle: nil)

            fieldRow(label: "Name") {
                TextField("Your name", text: $editName)
                    .font(.body)
                    .foregroundStyle(Theme.textPrimary)
                    .autocorrectionDisabled()
            }
        }
    }

    // MARK: - Goal

    private var goalStep: some View {
        VStack(spacing: 32) {
            questionHeader(title: "What is your goal?", subtitle: "This helps us generate a custom plan")

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

    // MARK: - Gender

    private var genderStep: some View {
        VStack(spacing: 32) {
            questionHeader(title: "Choose your gender", subtitle: nil)

            VStack(spacing: 12) {
                ForEach(UserProfile.Sex.allCases, id: \.self) { sex in
                    selectionRow(label: sex.rawValue, selected: selectedSex == sex) {
                        selectedSex = sex
                    }
                }
            }
            .padding(.horizontal, 20)
        }
    }

    // MARK: - Height & Weight

    private var bodyStep: some View {
        VStack(spacing: 32) {
            questionHeader(title: "Height & Weight", subtitle: nil)

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
                                .keyboardType(.numberPad)
                                .font(.body)
                                .foregroundStyle(Theme.textPrimary)
                            Text("cm").font(.caption).foregroundStyle(Theme.textSecondary)
                        }
                    }
                    fieldRow(label: "Weight") {
                        HStack(spacing: 6) {
                            TextField("70", text: $editWeightKg)
                                .keyboardType(.decimalPad)
                                .font(.body)
                                .foregroundStyle(Theme.textPrimary)
                            Text("kg").font(.caption).foregroundStyle(Theme.textSecondary)
                        }
                    }
                } else {
                    fieldRow(label: "Height") {
                        HStack(spacing: 12) {
                            HStack(spacing: 4) {
                                TextField("5", text: $editFeet)
                                    .keyboardType(.numberPad)
                                    .font(.body)
                                    .foregroundStyle(Theme.textPrimary)
                                    .frame(width: 40)
                                Text("ft").font(.caption).foregroundStyle(Theme.textSecondary)
                            }
                            HStack(spacing: 4) {
                                TextField("10", text: $editInches)
                                    .keyboardType(.numberPad)
                                    .font(.body)
                                    .foregroundStyle(Theme.textPrimary)
                                    .frame(width: 40)
                                Text("in").font(.caption).foregroundStyle(Theme.textSecondary)
                            }
                        }
                    }
                    fieldRow(label: "Weight") {
                        HStack(spacing: 6) {
                            TextField("155", text: $editWeightLbs)
                                .keyboardType(.decimalPad)
                                .font(.body)
                                .foregroundStyle(Theme.textPrimary)
                            Text("lbs").font(.caption).foregroundStyle(Theme.textSecondary)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Target Weight

    private var targetWeightStep: some View {
        VStack(spacing: 32) {
            questionHeader(title: "What's your target weight?", subtitle: nil)

            VStack(spacing: 12) {
                if usesMetric {
                    fieldRow(label: "Target weight") {
                        HStack(spacing: 6) {
                            TextField("65", text: $editTargetKg)
                                .keyboardType(.decimalPad)
                                .font(.body)
                                .foregroundStyle(Theme.textPrimary)
                            Text("kg").font(.caption).foregroundStyle(Theme.textSecondary)
                        }
                    }
                } else {
                    fieldRow(label: "Target weight") {
                        HStack(spacing: 6) {
                            TextField("140", text: $editTargetLbs)
                                .keyboardType(.decimalPad)
                                .font(.body)
                                .foregroundStyle(Theme.textPrimary)
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

    // MARK: - Birthday

    private var birthdayStep: some View {
        VStack(spacing: 32) {
            questionHeader(title: "When were you born?", subtitle: "You must be at least 13 years old")

            DatePicker("", selection: $birthday, in: ...maxBirthday, displayedComponents: .date)
                .datePickerStyle(.wheel)
                .labelsHidden()
                .colorScheme(.dark)
                .padding(.horizontal, 20)
        }
    }

    // MARK: - Activity

    private var activityStep: some View {
        VStack(spacing: 32) {
            questionHeader(title: "How active are you?", subtitle: nil)

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
            questionHeader(
                title: "Your custom plan is ready!",
                subtitle: "You can edit this at anytime"
            )

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

    // MARK: - Shared components

    private func questionHeader(title: String, subtitle: String?) -> some View {
        VStack(spacing: 8) {
            Text(title)
                .font(.title2.weight(.bold))
                .foregroundStyle(Theme.textPrimary)
            if let subtitle {
                Text(subtitle)
                    .font(.subheadline)
                    .foregroundStyle(Theme.textSecondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 20)
            }
        }
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
            Text(label)
                .font(.caption)
                .foregroundStyle(Theme.textSecondary)
            content()
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.cardBackground)
                .clipShape(RoundedRectangle(cornerRadius: 12))
        }
        .padding(.horizontal, 20)
    }

    // MARK: - Navigation

    private func advance() {
        switch currentStep {
        case .name:
            viewModel.userProfile.name = editName.trimmingCharacters(in: .whitespaces)
        case .goal:
            viewModel.userProfile.goal = selectedGoal
        case .gender:
            if let sex = selectedSex { viewModel.userProfile.sex = sex }
        case .body:
            saveBody()
        case .targetWeight:
            viewModel.userProfile.targetWeightKg = targetWeightKg
        case .birthday:
            viewModel.userProfile.birthday = birthday
        case .activity:
            viewModel.userProfile.activityLevel = selectedActivity
            viewModel.userProfile.exerciseTimesPerWeek =
                selectedActivity == .exerciseRegularly ? exerciseFrequency : nil

            // Compute recommended goals and populate plan fields
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

            viewModel.hasCompletedOnboarding = true
            viewModel.showOnboarding = false
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
