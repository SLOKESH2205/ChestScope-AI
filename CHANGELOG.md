# Changelog

All notable changes to the ChestScope-AI workspace will be documented in this file.

## [1.2.0-stable] - 2026-07-18
### Added
- **Batch Processing Mode**: Upload multiple patient scans, run sequential inference, view summaries, export batch CSV logs, and download combined PDFs.
- **Monte Carlo Dropout Uncertainty**: Integrated Shannon Entropy quantification over 15 forward passes to identify diagnostic warnings.
- **Explainable AI Attributions**: Grad-CAM, Grad-CAM++, and Integrated Gradients overlays for localizing features.
- **PDF Report Generation**: Downloadable clinical reports compiling metadata, XAI charts, recommendations, and disclaimers.
- **Session History Logging**: Tab tracking session diagnostics with CSV download capabilities.
- **Unit Testing**: Pytest coverage with 7 test validations.

### Fixed
- **EfficientNetB0 double-rescaling bug**: Restored baseline accuracy to 78% by adding a `Rescaling(255.0)` wrapper.
- **AttributeError on Keras 3 flat Sequential models**: Fixed Grad-CAM layers tracking by executing sequential tape tracing.

## [1.0.0] - 2026-06-12
### Added
- Initial release with baseline models (Custom CNN, MobileNetV2, and original EfficientNetB0).
- Simple prediction tab and metrics comparison section.
