import Foundation

/// Represents a recording session with transcript segments
///
/// Tracks the start/end time, device information, and all transcribed
/// segments captured during the session.
struct SessionModel: Identifiable, Codable {
    /// Unique identifier for the session
    let id: UUID

    /// When the session started
    let startTime: Date

    /// When the session ended (nil if still in progress)
    let endTime: Date?

    /// All transcript segments captured during this session
    let segments: [TranscriptSegment]

    /// Name of the connected BLE device
    let deviceName: String?

    /// Initialize a new session
    /// - Parameters:
    ///   - id: Unique identifier (generates new UUID if not provided)
    ///   - startTime: Session start time (defaults to now)
    ///   - endTime: Session end time (nil for in-progress)
    ///   - segments: Transcript segments (defaults to empty)
    ///   - deviceName: Connected device name
    init(
        id: UUID = UUID(),
        startTime: Date = Date(),
        endTime: Date? = nil,
        segments: [TranscriptSegment] = [],
        deviceName: String? = nil
    ) {
        self.id = id
        self.startTime = startTime
        self.endTime = endTime
        self.segments = segments
        self.deviceName = deviceName
    }

    /// Cached date formatter for performance
    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()

    /// Formatted duration string ("Xm Ys" or "In progress")
    var durationString: String {
        guard let endTime = endTime else {
            return "In progress"
        }

        let duration = endTime.timeIntervalSince(startTime)
        let minutes = Int(duration) / 60
        let seconds = Int(duration) % 60

        return "\(minutes)m \(seconds)s"
    }

    /// Formatted date string (medium date + short time)
    var dateString: String {
        Self.dateFormatter.string(from: startTime)
    }
}
