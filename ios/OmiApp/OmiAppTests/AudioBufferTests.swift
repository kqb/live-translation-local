//
//  AudioBufferTests.swift
//  OmiAppTests
//
//  Unit tests for AudioBuffer
//

import XCTest
@testable import OmiApp

class AudioBufferTests: XCTestCase {

    var buffer: AudioBuffer!

    override func setUp() {
        super.setUp()
        buffer = AudioBuffer(durationSeconds: 3.0, sampleRate: 16000)
    }

    override func tearDown() {
        buffer = nil
        super.tearDown()
    }

    // MARK: - Initialization Tests

    func testBufferInitialization() {
        XCTAssertNotNil(buffer)
        XCTAssertEqual(buffer.count, 0)
        XCTAssertEqual(buffer.fillPercentage, 0.0)
        XCTAssertFalse(buffer.isChunkReady())
    }

    // MARK: - Append Tests

    func testAppendSamples() {
        let samples = [Float](repeating: 0.5, count: 1000)
        buffer.append(samples)

        XCTAssertEqual(buffer.count, 1000)
        XCTAssertGreaterThan(buffer.fillPercentage, 0.0)
    }

    func testAppendMultipleTimes() {
        // Add samples in chunks
        for _ in 0..<10 {
            let samples = [Float](repeating: 0.5, count: 100)
            buffer.append(samples)
        }

        XCTAssertEqual(buffer.count, 1000)
    }

    // MARK: - Chunk Ready Tests

    func testChunkReady() {
        // 3 seconds at 16kHz = 48,000 samples
        let targetSamples = 48000

        // Add just under target
        buffer.append([Float](repeating: 0.5, count: targetSamples - 1))
        XCTAssertFalse(buffer.isChunkReady())

        // Add one more sample
        buffer.append([0.5])
        XCTAssertTrue(buffer.isChunkReady())
    }

    // MARK: - Drain Tests

    func testDrainChunk() {
        let targetSamples = 48000

        // Add exactly target samples
        buffer.append([Float](repeating: 0.5, count: targetSamples))

        XCTAssertTrue(buffer.isChunkReady())

        let chunk = buffer.drainChunk()
        XCTAssertNotNil(chunk)
        XCTAssertEqual(chunk?.count, targetSamples)
        XCTAssertEqual(buffer.count, 0)
    }

    func testDrainChunkWithOverflow() {
        let targetSamples = 48000
        let overflow = 1000

        // Add more than target samples
        buffer.append([Float](repeating: 0.5, count: targetSamples + overflow))

        let chunk = buffer.drainChunk()
        XCTAssertNotNil(chunk)
        XCTAssertEqual(chunk?.count, targetSamples)

        // Overflow samples should remain in buffer
        XCTAssertEqual(buffer.count, overflow)
    }

    func testDrainWhenNotReady() {
        buffer.append([Float](repeating: 0.5, count: 1000))

        let chunk = buffer.drainChunk()
        XCTAssertNil(chunk)

        // Samples should still be in buffer
        XCTAssertEqual(buffer.count, 1000)
    }

    // MARK: - Clear Tests

    func testClear() {
        buffer.append([Float](repeating: 0.5, count: 10000))
        XCTAssertGreaterThan(buffer.count, 0)

        buffer.clear()
        XCTAssertEqual(buffer.count, 0)
        XCTAssertEqual(buffer.fillPercentage, 0.0)
    }

    // MARK: - Fill Percentage Tests

    func testFillPercentage() {
        let targetSamples = 48000

        // Add 25% of target
        buffer.append([Float](repeating: 0.5, count: targetSamples / 4))
        XCTAssertEqual(buffer.fillPercentage, 0.25, accuracy: 0.01)

        // Add another 25%
        buffer.append([Float](repeating: 0.5, count: targetSamples / 4))
        XCTAssertEqual(buffer.fillPercentage, 0.5, accuracy: 0.01)

        // Add 50% more to reach 100%
        buffer.append([Float](repeating: 0.5, count: targetSamples / 2))
        XCTAssertEqual(buffer.fillPercentage, 1.0, accuracy: 0.01)
    }

    // MARK: - Thread Safety Tests

    func testConcurrentAppends() {
        let expectation = XCTestExpectation(description: "Concurrent appends")
        let iterations = 100

        DispatchQueue.concurrentPerform(iterations: iterations) { i in
            let samples = [Float](repeating: Float(i), count: 100)
            buffer.append(samples)
        }

        expectation.fulfill()
        wait(for: [expectation], timeout: 5.0)

        // Should have all samples
        XCTAssertEqual(buffer.count, iterations * 100)
    }

    func testConcurrentDrains() {
        // Fill buffer with enough samples for multiple chunks
        buffer.append([Float](repeating: 0.5, count: 48000 * 3))

        let expectation = XCTestExpectation(description: "Concurrent drains")
        var drainedChunks = 0
        let queue = DispatchQueue(label: "test.drain", attributes: .concurrent)

        for _ in 0..<3 {
            queue.async {
                if let chunk = self.buffer.drainChunk() {
                    XCTAssertEqual(chunk.count, 48000)
                    drainedChunks += 1
                }
            }
        }

        queue.async {
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 5.0)
    }

    // MARK: - Debug Info Tests

    func testDebugInfo() {
        buffer.append([Float](repeating: 0.5, count: 24000)) // 50% full

        let info = buffer.debugInfo()

        XCTAssertEqual(info["sampleCount"] as? Int, 24000)
        XCTAssertEqual(info["targetSamples"] as? Int, 48000)
        XCTAssertEqual(info["fillPercentage"] as? Double, 50.0, accuracy: 0.1)
        XCTAssertEqual(info["durationSeconds"] as? Double, 1.5, accuracy: 0.01)
        XCTAssertEqual(info["isReady"] as? Bool, false)
    }

    // MARK: - Performance Tests

    func testAppendPerformance() {
        let samples = [Float](repeating: 0.5, count: 1000)

        measure {
            for _ in 0..<100 {
                buffer.append(samples)
            }
            buffer.clear()
        }
    }

    func testDrainPerformance() {
        // Pre-fill buffer
        buffer.append([Float](repeating: 0.5, count: 480000)) // 10 chunks

        measure {
            for _ in 0..<10 {
                _ = buffer.drainChunk()
            }
        }
    }
}
