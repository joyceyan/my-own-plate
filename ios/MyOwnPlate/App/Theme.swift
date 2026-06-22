import SwiftUI

enum Theme {
    // Backgrounds — Charcoal Blue base
    static let background = Color(hex: 0x1E3340)
    static let cardBackground = Color(hex: 0x264653)
    static let cardBackgroundLight = Color(hex: 0x2F5466)

    // Text
    static let textPrimary = Color.white
    static let textSecondary = Color(white: 0.72)
    static let textTertiary = Color(white: 0.50)

    // Macro colors
    static let protein = Color(hex: 0x2A9D8F)     // Verdigris — teal/green
    static let carbs = Color(hex: 0xE9C46A)        // Jasmine — golden
    static let fat = Color(hex: 0xF4A261)          // Sandy Brown — warm orange
    static let calories = Color(hex: 0xE76F51)     // Coral accent for fire

    // Accent
    static let accent = Color(hex: 0x2A9D8F)

    // Tab bar
    static let tabBarBackground = Color(hex: 0x1E3340)
    static let tabActive = Color.white
    static let tabInactive = Color(white: 0.50)
}

extension Color {
    init(hex: UInt, opacity: Double = 1.0) {
        self.init(
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: opacity
        )
    }
}
