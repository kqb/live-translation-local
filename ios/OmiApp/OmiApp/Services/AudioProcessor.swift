//
//  AudioProcessor.swift
//  OmiApp
//
//  Orchestrates audio decoding, buffering, and server upload
//

import Foundation
import Combine

/// Notification sent when a transcript is received from the server
extension Notification.Name {
    static let transcriptReceived = Notification.Name("transcriptReceived")
}

/// Orchestrates the audio processing pipeline: BLE packets → decoding → buffering → upload
class AudioProcessor: ObservableObject {

    // MARK: - Published Properties

    /// Whether the processor is actively processing audio
    @Published var isProcessing = false

    /// Current buffer fill percentage (0.0 to 1.0)
    @Published var bufferFillPercentage: Double = 0.0

    /// Number of chunks processed in current session
    @Published var chunksProcessed: Int = 0

    /// Total samples decoded in current session
    @Published var totalSamplesDecoded: Int = 0

    // MARK: - Private Properties

    private let decoder: OpusDecoder
    private let buffer: AudioBuffer
    private let apiClient: APIClient
    private var processingTask: Task<Void, Never>?
    private var metricsUpdateTask: Task<Void, Never>?

    // MARK: - Configuration

    /// How often to check if buffer is ready (in nanoseconds)
    private let bufferCheckInterval: UInt64 = 100_000_000 // 100ms

    /// How often to update buffer metrics (in nanoseconds)
    private let metricsUpdateInterval: UInt64 = 250_000_000 // 250ms

    // MARK: - Initialization

    /// Initialize audio processor
    /// - Parameters:
    ///   - apiClient: API client for server communication
    ///   - chunkDuration: Target chunk duration in seconds (default: 3.0)
    /// - Throws: OpusError if decoder initialization fails
    init(apiClient: APIClient, chunkDuration: Double = 3.0) throws {
        self.decoder = try OpusDecoder()
        self.buffer = AudioBuffer(durationSeconds: chunkDuration, sampleRate: 16000)
        self.apiClient = apiClient
    }

    // MARK: - Public Methods

    /// Start processing audio
    func start() {
        guard !isProcessing else { return }

        isProcessing = true
        chunksProcessed = 0
        totalSamplesDecoded = 0

        // Start buffer monitoring task
        processingTask = Task {
            await monitorBuffer()
        }

        // Start metrics update task
        metricsUpdateTask = Task {
            await updateMetrics()
        }
    }

    /// Stop processing audio
    func stop() {
        isProcessing = false
        processingTask?.cancel()
        metricsUpdateTask?.cancel()
        buffer.clear()
        bufferFillPercentage = 0.0
    }

    /// Handle a BLE packet containing Opus audio data
    /// - Parameter packet: BLE packet with 3-byte header + Opus data
    /// - Note: This method is thread-safe and can be called from BLE callback queue
    func handleBLEPacket(_ packet: Data) {
        guard isProcessing else { return }

        do {
            // Decode Opus packet to PCM samples
            let pcmSamples = try decoder.decode(packet)

            // Append to buffer
            buffer.append(pcmSamples)

            // Update statistics
            totalSamplesDecoded += pcmSamples.count

        } catch {
            // Log error but continue processing
            // Don't crash the pipeline for a single bad packet
            print("⚠️ Audio decode error: \(error.localizedDescription)")
        }
    }

    /// Reset all statistics and clear buffer
    func reset() {
        stop()
        chunksProcessed = 0
        totalSamplesDecoded = 0
        bufferFillPercentage = 0.0
    }

    // MARK: - Private Methods

    /// Continuously monitor buffer and upload chunks when ready
    private func monitorBuffer() async {
        while isProcessing {
            if buffer.isChunkReady() {
                await uploadChunk()
            }

            // Sleep before next check
            try? await Task.sleep(nanoseconds: bufferCheckInterval)
        }
    }

    /// Periodically update buffer fill metrics
    private func updateMetrics() async {
        while isProcessing {
            await MainActor.run {
                bufferFillPercentage = buffer.fillPercentage
            }

            try? await Task.sleep(nanoseconds: metricsUpdateInterval)
        }
    }

    /// Upload a complete audio chunk to the server
    private func uploadChunk() async {
        guard let chunk = buffer.drainChunk() else { return }

        // Convert Float32 array to Data (raw bytes)
        let audioData = chunk.withUnsafeBufferPointer { ptr in
            Data(buffer: ptr)
        }

        do {
            // Upload to server API
            let response = try await apiClient.transcribe(audio: audioData)

            // Update statistics
            await MainActor.run {
                chunksProcessed += 1
            }

            // Notify UI on main thread
            await MainActor.run {
                NotificationCenter.default.post(
                    name: .transcriptReceived,
                    object: response
                )
            }

            print("✅ Chunk \(chunksProcessed) uploaded successfully")

        } catch {
            print("⚠️ Upload error: \(error.localizedDescription)")
            // Could implement retry logic here
        }
    }
}

// MARK: - Debug Extension

extension AudioProcessor {
    /// Get detailed debug information
    func debugInfo() -> [String: Any] {
        return [
            "isProcessing": isProcessing,
            "chunksProcessed": chunksProcessed,
            "totalSamplesDecoded": totalSamplesDecoded,
            "bufferInfo": buffer.debugInfo()
        ]
    }
}
