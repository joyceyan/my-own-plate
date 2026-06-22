import SwiftUI
import PhotosUI

struct PhotoPickerView: UIViewControllerRepresentable {
    let onSelect: @MainActor (UIImage) -> Void
    let onCancel: @MainActor () -> Void

    func makeUIViewController(context: Context) -> PHPickerViewController {
        var config = PHPickerConfiguration()
        config.filter = .images
        config.selectionLimit = 1
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: PHPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onSelect: onSelect, onCancel: onCancel)
    }

    final class Coordinator: NSObject, PHPickerViewControllerDelegate {
        let onSelect: @MainActor (UIImage) -> Void
        let onCancel: @MainActor () -> Void

        init(onSelect: @MainActor @escaping (UIImage) -> Void, onCancel: @MainActor @escaping () -> Void) {
            self.onSelect = onSelect
            self.onCancel = onCancel
        }

        func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
            guard let provider = results.first?.itemProvider,
                  provider.canLoadObject(ofClass: UIImage.self) else {
                Task { @MainActor in onCancel() }
                return
            }

            let onSelect = self.onSelect
            provider.loadObject(ofClass: UIImage.self) { object, _ in
                if let image = object as? UIImage {
                    Task { @MainActor in onSelect(image) }
                }
            }
        }
    }
}
