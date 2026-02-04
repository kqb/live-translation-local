//
//  OpusError.swift
//  OmiApp
//
//  Error types for Opus audio processing
//

import Foundation

enum OpusError: LocalizedError {
    case initializationFailed(Int32)
    case decodeFailed(Int32)
    case invalidPacketFormat
    case bufferOverflow
    case invalidSampleRate
    case invalidChannelCount

    var errorDescription: String? {
        switch self {
        case .initializationFailed(let code):
            return "Opus decoder initialization failed with code: \(code)"
        case .decodeFailed(let code):
            return "Opus decode failed with code: \(code)"
        case .invalidPacketFormat:
            return "Invalid BLE packet format (expected 3+ bytes header)"
        case .bufferOverflow:
            return "Audio buffer overflow - too many samples"
        case .invalidSampleRate:
            return "Invalid sample rate (must be 8000, 12000, 16000, 24000, or 48000)"
        case .invalidChannelCount:
            return "Invalid channel count (must be 1 or 2)"
        }
    }
}
