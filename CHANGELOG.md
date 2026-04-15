# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog and this project follows Semantic Versioning where possible.

## [Unreleased]

### Added
- Added explicit API precondition checks in DSModel methods that require predictions.
- Added convergence-based angle propagation loop stability check.
- Added robust prediction runtime checks for non-finite outputs.
- Added configurable inference batch size support through DSModel.predict.
- Added changelog tracking for project-level changes.

### Changed
- Hardened plane geometry calculations against zero-division in depth and axis computations.
- Replaced fragile secondary-weight string sentinel logic with explicit None handling.
- Updated TensorFlow dependency constraint to `tensorflow>=2.13,<3.0`.
- Filled previously empty neural network architecture module with compatibility exports.

### Fixed
- Fixed mutable shared default markers list construction in QUINT JSON writer.
- Improved inference error surfacing by wrapping model prediction failures with actionable RuntimeError messages.
