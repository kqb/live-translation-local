//
//  OpusDecoder.swift
//  OmiApp
//
//  Decodes Opus audio packets from BLE to PCM Float32 arrays
//

import Foundation

/// Decodes Opus audio packets to PCM Float32 samples
class OpusDecoder {

    // MARK: - Properties

    private var decoder: OpaquePointer?
    private let sampleRate: Int32
    private let channels: Int32
    private let maxFrameSize: Int

    // MARK: - Constants

    /// Omi device uses 16kHz sample rate
    static let omiSampleRate: Int32 = 16000

    /// Omi device uses mono audio
    static let omiChannels: Int32 = 1

    /// Maximum frame size for 120ms at 48kHz (Opus max)
    private static let maxOpusFrameSize: Int = 5760

    /// BLE packet header size (2 bytes packet_number + 1 byte index)
    private static let bleHeaderSize: Int = 3

    // MARK: - Initialization

    /// Initialize Opus decoder with specified sample rate and channel count
    /// - Parameters:
    ///   - sampleRate: Sample rate in Hz (8000, 12000, 16000, 24000, or 48000)
    ///   - channels: Number of channels (1 for mono, 2 for stereo)
    /// - Throws: OpusError if initialization fails
    init(sampleRate: Int32 = omiSampleRate, channels: Int32 = omiChannels) throws {
        self.sampleRate = sampleRate
        self.channels = channels

        // Calculate max frame size based on sample rate
        // 120ms is the maximum Opus frame size
        self.maxFrameSize = Int(sampleRate * 120 / 1000)

        // Validate parameters
        let validSampleRates: [Int32] = [8000, 12000, 16000, 24000, 48000]
        guard validSampleRates.contains(sampleRate) else {
            throw OpusError.invalidSampleRate
        }

        guard channels == 1 || channels == 2 else {
            throw OpusError.invalidChannelCount
        }

        // Create Opus decoder
        var error: Int32 = 0
        decoder = opus_decoder_create(sampleRate, channels, &error)

        guard error == OPUS_OK else {
            throw OpusError.initializationFailed(error)
        }

        guard decoder != nil else {
            throw OpusError.initializationFailed(-1)
        }
    }

    deinit {
        if let decoder = decoder {
            opus_decoder_destroy(decoder)
        }
    }

    // MARK: - Public Methods

    /// Decode a BLE packet containing Opus audio data
    /// - Parameter packet: BLE packet with 3-byte header + Opus data
    /// - Returns: Array of PCM samples as Float32 (normalized to -1.0...1.0)
    /// - Throws: OpusError if decoding fails
    func decode(_ packet: Data) throws -> [Float] {
        // Validate packet size
        guard packet.count > Self.bleHeaderSize else {
            throw OpusError.invalidPacketFormat
        }

        // Extract Opus payload (skip BLE header)
        let opusData = packet.subdata(in: Self.bleHeaderSize..<packet.count)

        // Allocate buffer for decoded PCM samples (Int16)
        var pcmInt16 = [Int16](repeating: 0, count: maxFrameSize)

        // Decode Opus data to PCM
        let frameSize = opusData.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) -> Int32 in
            guard let baseAddress = bytes.baseAddress else { return -1 }

            return opus_decode(
                decoder,
                baseAddress.assumingMemoryBound(to: UInt8.self),
                Int32(opusData.count),
                &pcmInt16,
                Int32(maxFrameSize),
                0  // decode_fec = 0 (no forward error correction)
            )
        }

        // Check for decode errors
        guard frameSize > 0 else {
            throw OpusError.decodeFailed(frameSize)
        }

        // Convert Int16 PCM to Float32 (normalized to -1.0...1.0)
        let samples = pcmInt16.prefix(Int(frameSize)).map { sample in
            Float(sample) / Float(Int16.max)
        }

        return samples
    }

    /// Decode Opus data without BLE header (for testing or alternative sources)
    /// - Parameter opusData: Raw Opus encoded data
    /// - Returns: Array of PCM samples as Float32
    /// - Throws: OpusError if decoding fails
    func decodeRaw(_ opusData: Data) throws -> [Float] {
        var pcmInt16 = [Int16](repeating: 0, count: maxFrameSize)

        let frameSize = opusData.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) -> Int32 in
            guard let baseAddress = bytes.baseAddress else { return -1 }

            return opus_decode(
                decoder,
                baseAddress.assumingMemoryBound(to: UInt8.self),
                Int32(opusData.count),
                &pcmInt16,
                Int32(maxFrameSize),
                0
            )
        }

        guard frameSize > 0 else {
            throw OpusError.decodeFailed(frameSize)
        }

        let samples = pcmInt16.prefix(Int(frameSize)).map { sample in
            Float(sample) / Float(Int16.max)
        }

        return samples
    }
}

// MARK: - Opus Constants

/// Opus library return codes
private let OPUS_OK: Int32 = 0
