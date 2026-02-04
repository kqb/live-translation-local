//
//  AudioBuffer.swift
//  OmiApp
//
//  Thread-safe circular buffer for accumulating PCM audio samples
//

import Foundation

/// Thread-safe buffer for accumulating PCM audio samples into chunks
class AudioBuffer {

    // MARK: - Properties

    private var samples: [Float] = []
    private let lock = NSLock()
    private let targetSamples: Int
    private let sampleRate: Int

    /// Current buffer fill percentage (0.0 to 1.0)
    var fillPercentage: Double {
        lock.lock()
        defer { lock.unlock() }
        return Double(samples.count) / Double(targetSamples)
    }

    /// Current number of samples in buffer
    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return samples.count
    }

    // MARK: - Initialization

    /// Initialize audio buffer
    /// - Parameters:
    ///   - durationSeconds: Target chunk duration in seconds (default: 3.0)
    ///   - sampleRate: Audio sample rate in Hz (default: 16000)
    init(durationSeconds: Double = 3.0, sampleRate: Int = 16000) {
        self.sampleRate = sampleRate
        self.targetSamples = Int(durationSeconds * Double(sampleRate))
    }

    // MARK: - Public Methods

    /// Append new PCM samples to the buffer (thread-safe)
    /// - Parameter newSamples: Array of PCM samples to append
    func append(_ newSamples: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        samples.append(contentsOf: newSamples)
    }

    /// Check if a complete chunk is ready for extraction
    /// - Returns: True if buffer contains at least targetSamples
    func isChunkReady() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return samples.count >= targetSamples
    }

    /// Extract a complete chunk from the buffer
    /// - Returns: Array of exactly targetSamples, or nil if not enough samples
    /// - Note: Remaining samples are kept in buffer for next chunk
    func drainChunk() -> [Float]? {
        lock.lock()
        defer { lock.unlock() }

        guard samples.count >= targetSamples else {
            return nil
        }

        // Extract exactly targetSamples
        let chunk = Array(samples.prefix(targetSamples))

        // Keep remaining samples for next chunk
        samples.removeFirst(targetSamples)

        return chunk
    }

    /// Clear all buffered samples (thread-safe)
    func clear() {
        lock.lock()
        defer { lock.unlock() }
        samples.removeAll(keepingCapacity: true)
    }

    /// Get current buffer state for debugging
    /// - Returns: Dictionary with buffer statistics
    func debugInfo() -> [String: Any] {
        lock.lock()
        defer { lock.unlock() }

        return [
            "sampleCount": samples.count,
            "targetSamples": targetSamples,
            "fillPercentage": Double(samples.count) / Double(targetSamples) * 100,
            "durationSeconds": Double(samples.count) / Double(sampleRate),
            "isReady": samples.count >= targetSamples
        ]
    }
}
