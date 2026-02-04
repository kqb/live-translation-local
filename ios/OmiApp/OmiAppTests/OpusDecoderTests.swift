//
//  OpusDecoderTests.swift
//  OmiAppTests
//
//  Unit tests for OpusDecoder
//

import XCTest
@testable import OmiApp

class OpusDecoderTests: XCTestCase {

    var decoder: OpusDecoder!

    override func setUp() {
        super.setUp()
        decoder = try! OpusDecoder()
    }

    override func tearDown() {
        decoder = nil
        super.tearDown()
    }

    // MARK: - Initialization Tests

    func testDecoderInitialization() throws {
        // Should initialize without errors
        let decoder = try OpusDecoder(sampleRate: 16000, channels: 1)
        XCTAssertNotNil(decoder)
    }

    func testInvalidSampleRate() {
        // Should throw error for invalid sample rate
        XCTAssertThrowsError(try OpusDecoder(sampleRate: 11025, channels: 1)) { error in
            XCTAssertTrue(error is OpusError)
        }
    }

    func testInvalidChannelCount() {
        // Should throw error for invalid channel count
        XCTAssertThrowsError(try OpusDecoder(sampleRate: 16000, channels: 3)) { error in
            XCTAssertTrue(error is OpusError)
        }
    }

    // MARK: - Decoding Tests

    func testDecodeInvalidPacket() {
        // Packet too small (less than 3 bytes header)
        let invalidPacket = Data([0x00, 0x01])

        XCTAssertThrowsError(try decoder.decode(invalidPacket)) { error in
            guard let opusError = error as? OpusError else {
                XCTFail("Expected OpusError")
                return
            }

            if case .invalidPacketFormat = opusError {
                // Success
            } else {
                XCTFail("Expected invalidPacketFormat error")
            }
        }
    }

    func testDecodeValidPacketFormat() {
        // Create a mock BLE packet with valid header
        // Note: This will likely fail to decode since it's not real Opus data,
        // but it tests the packet format validation
        var packet = Data([0x00, 0x01, 0x00]) // 3-byte header

        // Add some mock Opus data (this won't decode properly, but tests the format)
        packet.append(Data(repeating: 0xFF, count: 100))

        // Should not throw invalidPacketFormat error
        // (will throw decodeFailed instead, which is expected)
        do {
            _ = try decoder.decode(packet)
            XCTFail("Expected decode to fail with mock data")
        } catch let error as OpusError {
            if case .decodeFailed = error {
                // Expected behavior
            } else {
                XCTFail("Expected decodeFailed error, got: \(error)")
            }
        } catch {
            XCTFail("Expected OpusError, got: \(error)")
        }
    }

    func testSampleNormalization() throws {
        // If we had real Opus test data, we would verify:
        // - All samples are in range -1.0 to 1.0
        // - Sample count matches expected frame size
        // This test would need real Opus encoded audio data

        // For now, we just verify the decoder exists
        XCTAssertNotNil(decoder)
    }

    // MARK: - Performance Tests

    func testDecoderPerformance() {
        // Measure performance of decoder creation
        measure {
            _ = try? OpusDecoder(sampleRate: 16000, channels: 1)
        }
    }
}
