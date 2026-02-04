//
//  APIClient.swift
//  OmiApp
//
//  HTTP client for communicating with local transcription server
//

import Foundation

/// API client for transcription server communication
class APIClient {

    // MARK: - Properties

    let baseURL: URL
    private let session: URLSession

    // MARK: - Configuration

    private let timeoutInterval: TimeInterval = 30.0

    // MARK: - Initialization

    /// Initialize API client
    /// - Parameter baseURL: Base URL of the transcription server (e.g., "http://192.168.1.100:8000")
    init(baseURL: String) {
        guard let url = URL(string: baseURL) else {
            fatalError("Invalid base URL: \(baseURL)")
        }
        self.baseURL = url

        // Configure URLSession
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = timeoutInterval
        config.timeoutIntervalForResource = timeoutInterval
        self.session = URLSession(configuration: config)
    }

    // MARK: - Public Methods

    /// Upload audio data for transcription
    /// - Parameter audio: PCM Float32 audio data (16kHz, mono)
    /// - Returns: Transcription response with text, speakers, translation
    /// - Throws: APIError if request fails
    func transcribe(audio: Data) async throws -> TranscriptResponse {
        let url = baseURL.appendingPathComponent("/transcribe")

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
        request.httpBody = audio

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }

        guard httpResponse.statusCode == 200 else {
            throw APIError.serverError(statusCode: httpResponse.statusCode)
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        do {
            return try decoder.decode(TranscriptResponse.self, from: data)
        } catch {
            throw APIError.decodingFailed(error)
        }
    }

    /// Check if server is reachable
    /// - Returns: True if server responds to health check
    func healthCheck() async -> Bool {
        let url = baseURL.appendingPathComponent("/health")

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 5.0

        do {
            let (_, response) = try await session.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                return false
            }
            return httpResponse.statusCode == 200
        } catch {
            return false
        }
    }
}

// MARK: - Response Models

/// Transcription response from server
struct TranscriptResponse: Codable {
    let text: String
    let language: String
    let confidence: Double?
    let speakers: [Speaker]?
    let translation: Translation?
}

/// Speaker information
struct Speaker: Codable {
    let speakerId: String
    let name: String?
    let start: Double
    let end: Double
}

/// Translation information
struct Translation: Codable {
    let text: String
    let language: String
}

// MARK: - Error Types

enum APIError: LocalizedError {
    case invalidResponse
    case serverError(statusCode: Int)
    case decodingFailed(Error)
    case networkError(Error)

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid server response"
        case .serverError(let statusCode):
            return "Server error with status code: \(statusCode)"
        case .decodingFailed(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        }
    }
}
