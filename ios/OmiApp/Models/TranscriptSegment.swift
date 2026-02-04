import Foundation

/// Represents a single segment of transcribed speech
///
/// Contains the original text, optional translation, speaker information,
/// and timestamp for when the segment was captured.
struct TranscriptSegment: Identifiable, Codable {
    /// Unique identifier for the segment
    let id: UUID

    /// When this segment was captured
    let timestamp: Date

    /// Speaker name (if identified via speaker recognition)
    let speaker: String?

    /// Original transcribed text
    let text: String

    /// Translated text (if translation enabled)
    let translation: String?

    /// Detected language code (e.g., "en", "es", "ja")
    let language: String

    /// Initialize a new transcript segment
    /// - Parameters:
    ///   - id: Unique identifier (generates new UUID if not provided)
    ///   - timestamp: Capture time (defaults to now)
    ///   - speaker: Speaker name
    ///   - text: Transcribed text
    ///   - translation: Translated text
    ///   - language: Language code
    init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        speaker: String? = nil,
        text: String,
        translation: String? = nil,
        language: String
    ) {
        self.id = id
        self.timestamp = timestamp
        self.speaker = speaker
        self.text = text
        self.translation = translation
        self.language = language
    }

    /// Cached time formatter for performance
    private static let timeFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter
    }()

    /// Formatted time string (HH:mm:ss)
    var timeString: String {
        Self.timeFormatter.string(from: timestamp)
    }
}
