import SwiftUI

struct SettingsView: View {
    @Environment(FoodAnalysisViewModel.self) var viewModel

    @State private var editingSection: EditingSection?
    @State private var showRecalculate = false

    // Personal info edit state
    @State private var editName = ""
    @State private var editBirthday: Date = Date()
    @State private var hasBirthday = false
    @State private var editSex: UserProfile.Sex = .other

    // Body edit state
    @State private var editUsesMetric = false
    @State private var editFeet = ""
    @State private var editInches = ""
    @State private var editCm = ""
    @State private var editWeightLbs = ""
    @State private var editWeightKg = ""

    // Goals edit state
    @State private var editCalories = ""
    @State private var editProtein = ""
    @State private var editFat = ""
    @State private var editCarbs = ""

    private enum EditingSection {
        case personal, body, goals
    }

    private var profile: UserProfile { viewModel.userProfile }
    private var goal: DailyGoal { viewModel.mealStore.dailyGoal }

    private var maxBirthday: Date {
        Calendar.current.date(byAdding: .year, value: -13, to: Date()) ?? Date()
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Settings")
                    .font(.title2.weight(.bold))
                    .foregroundStyle(Theme.textPrimary)

                personalSection
                bodySection
                goalsSection
                recalculateButton
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 80)
        }
        .background(Theme.background)
        .fullScreenCover(isPresented: $showRecalculate) {
            RecalculatePlanView()
        }
    }

    // MARK: - Recalculate

    private var recalculateButton: some View {
        Button {
            showRecalculate = true
        } label: {
            Text("Recalculate my plan")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(Theme.accent)
                .frame(maxWidth: .infinity)
        }
    }

    // MARK: - Personal Information

    private var personalSection: some View {
        settingsCard {
            sectionHeader("Personal Information", section: .personal)

            if editingSection == .personal {
                personalEditFields
                editButtons(save: savePersonal, cancel: { editingSection = nil })
            } else {
                personalReadFields
            }
        }
    }

    private var personalReadFields: some View {
        VStack(alignment: .leading, spacing: 10) {
            readRow(label: "Name", value: profile.name.isEmpty ? nil : profile.name)
            readRow(label: "Birthday", value: profile.birthday.map {
                let formatted = $0.formatted(date: .abbreviated, time: .omitted)
                if let age = profile.age { return "\(formatted) (age \(age))" }
                return formatted
            })
            readRow(label: "Gender", value: profile.sex.rawValue)
        }
    }

    private var personalEditFields: some View {
        VStack(alignment: .leading, spacing: 12) {
            editTextField(label: "Name", text: $editName, placeholder: "Your name")

            VStack(alignment: .leading, spacing: 6) {
                Text("Birthday").font(.caption).foregroundStyle(Theme.textSecondary)
                HStack {
                    if hasBirthday {
                        DatePicker("", selection: $editBirthday, in: ...maxBirthday, displayedComponents: .date)
                            .labelsHidden()
                            .colorScheme(.dark)
                        Button {
                            hasBirthday = false
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(Theme.textTertiary)
                        }
                    } else {
                        Button("Set birthday") {
                            hasBirthday = true
                        }
                        .font(.subheadline)
                        .foregroundStyle(Theme.accent)
                    }
                }
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Gender").font(.caption).foregroundStyle(Theme.textSecondary)
                Picker("Gender", selection: $editSex) {
                    ForEach(UserProfile.Sex.allCases, id: \.self) { sex in
                        Text(sex.rawValue).tag(sex)
                    }
                }
                .pickerStyle(.segmented)
            }
        }
    }

    // MARK: - Body

    private var bodySection: some View {
        settingsCard {
            sectionHeader("Body", section: .body)

            if editingSection == .body {
                bodyEditFields
                editButtons(save: saveBody, cancel: { editingSection = nil })
            } else {
                bodyReadFields
            }
        }
    }

    private var bodyReadFields: some View {
        VStack(alignment: .leading, spacing: 10) {
            readRow(label: "Height", value: profile.formattedHeight)
            readRow(label: "Weight", value: profile.formattedWeight)
        }
    }

    private var bodyEditFields: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Unit toggle
            VStack(alignment: .leading, spacing: 6) {
                Text("Units").font(.caption).foregroundStyle(Theme.textSecondary)
                Picker("Units", selection: $editUsesMetric) {
                    Text("Imperial").tag(false)
                    Text("Metric").tag(true)
                }
                .pickerStyle(.segmented)
            }

            if editUsesMetric {
                editTextField(label: "Height", text: $editCm, placeholder: "170", keyboard: .numberPad, suffix: "cm")
                editTextField(label: "Weight", text: $editWeightKg, placeholder: "70", keyboard: .decimalPad, suffix: "kg")
            } else {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Height").font(.caption).foregroundStyle(Theme.textSecondary)
                    HStack(spacing: 8) {
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
                    .padding(12)
                    .background(Theme.cardBackgroundLight)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                editTextField(label: "Weight", text: $editWeightLbs, placeholder: "155", keyboard: .decimalPad, suffix: "lbs")
            }
        }
    }

    // MARK: - Daily Goals

    private var goalsSection: some View {
        settingsCard {
            sectionHeader("Daily Goals", section: .goals)

            if editingSection == .goals {
                goalsEditFields
                editButtons(save: saveGoals, cancel: { editingSection = nil })
            } else {
                goalsReadFields
            }
        }
    }

    private var goalsReadFields: some View {
        VStack(alignment: .leading, spacing: 10) {
            readRow(label: "Calories", value: "\(Int(goal.calories)) kcal")
            readRow(label: "Protein", value: "\(Int(goal.protein))g")
            readRow(label: "Fat", value: "\(Int(goal.fat))g")
            readRow(label: "Carbs", value: "\(Int(goal.carbs))g")
        }
    }

    private var goalsEditFields: some View {
        VStack(alignment: .leading, spacing: 12) {
            editTextField(label: "Calories", text: $editCalories, placeholder: "2100", keyboard: .numberPad, suffix: "kcal")
            editTextField(label: "Protein", text: $editProtein, placeholder: "150", keyboard: .numberPad, suffix: "g")
            editTextField(label: "Fat", text: $editFat, placeholder: "70", keyboard: .numberPad, suffix: "g")
            editTextField(label: "Carbs", text: $editCarbs, placeholder: "250", keyboard: .numberPad, suffix: "g")
        }
    }

    // MARK: - Reusable components

    private func settingsCard<Content: View>(@ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            content()
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.cardBackground)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private func sectionHeader(_ title: String, section: EditingSection) -> some View {
        HStack {
            Text(title)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(Theme.textPrimary)
            Spacer()
            if editingSection != section {
                Button("Edit") { startEditing(section) }
                    .font(.subheadline)
                    .foregroundStyle(Theme.accent)
            }
        }
    }

    private func readRow(label: String, value: String?) -> some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(Theme.textSecondary)
            Spacer()
            Text(value ?? "Not set")
                .font(.subheadline)
                .foregroundStyle(value != nil ? Theme.textPrimary : Theme.textTertiary)
        }
    }

    private func editTextField(label: String, text: Binding<String>, placeholder: String, keyboard: UIKeyboardType = .default, suffix: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label).font(.caption).foregroundStyle(Theme.textSecondary)
            HStack(spacing: 6) {
                TextField(placeholder, text: text)
                    .keyboardType(keyboard)
                    .font(.body)
                    .foregroundStyle(Theme.textPrimary)
                if let suffix {
                    Text(suffix).font(.caption).foregroundStyle(Theme.textSecondary)
                }
            }
            .padding(12)
            .background(Theme.cardBackgroundLight)
            .clipShape(RoundedRectangle(cornerRadius: 10))
        }
    }

    private func editButtons(save: @escaping () -> Void, cancel: @escaping () -> Void) -> some View {
        HStack(spacing: 12) {
            Button {
                cancel()
            } label: {
                Text("Cancel")
                    .font(.subheadline.weight(.medium))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(Theme.cardBackgroundLight)
                    .foregroundStyle(Theme.textPrimary)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
            Button {
                save()
            } label: {
                Text("Save")
                    .font(.subheadline.weight(.medium))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(Color.white)
                    .foregroundStyle(Theme.background)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        }
        .padding(.top, 4)
    }

    // MARK: - Edit lifecycle

    private func startEditing(_ section: EditingSection) {
        editingSection = section
        switch section {
        case .personal:
            editName = profile.name
            if let bday = profile.birthday {
                editBirthday = bday
                hasBirthday = true
            } else {
                editBirthday = Calendar.current.date(byAdding: .year, value: -25, to: Date()) ?? Date()
                hasBirthday = false
            }
            editSex = profile.sex
        case .body:
            editUsesMetric = profile.usesMetric
            if let cm = profile.heightCm {
                editCm = "\(Int(cm))"
                let totalInches = cm / 2.54
                editFeet = "\(Int(totalInches) / 12)"
                editInches = "\(Int(totalInches) % 12)"
            } else {
                editCm = ""
                editFeet = ""
                editInches = ""
            }
            if let kg = profile.weightKg {
                editWeightKg = "\(Int(kg))"
                editWeightLbs = "\(Int(kg * 2.20462))"
            } else {
                editWeightKg = ""
                editWeightLbs = ""
            }
        case .goals:
            editCalories = "\(Int(goal.calories))"
            editProtein = "\(Int(goal.protein))"
            editFat = "\(Int(goal.fat))"
            editCarbs = "\(Int(goal.carbs))"
        }
    }

    private func savePersonal() {
        viewModel.userProfile.name = editName
        viewModel.userProfile.birthday = hasBirthday ? min(editBirthday, maxBirthday) : nil
        viewModel.userProfile.sex = editSex
        editingSection = nil
    }

    private func saveBody() {
        viewModel.userProfile.usesMetric = editUsesMetric
        if editUsesMetric {
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
        editingSection = nil
    }

    private func saveGoals() {
        if let cal = Double(editCalories) { viewModel.mealStore.dailyGoal.calories = cal }
        if let pro = Double(editProtein) { viewModel.mealStore.dailyGoal.protein = pro }
        if let fat = Double(editFat) { viewModel.mealStore.dailyGoal.fat = fat }
        if let carbs = Double(editCarbs) { viewModel.mealStore.dailyGoal.carbs = carbs }
        editingSection = nil
    }
}
