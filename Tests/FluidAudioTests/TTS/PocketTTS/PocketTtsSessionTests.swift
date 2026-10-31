import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsSessionTests: XCTestCase {

    // MARK: - Manager Guard Tests

    func testMakeSessionFailsWithoutInitialization() async {
        let manager = PocketTtsManager()

        do {
            _ = try await manager.makeSession()
            XCTFail("Expected error when not initialized")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    func testMakeSessionWithVoiceDataFailsWithoutInitialization() async {
        let manager = PocketTtsManager()
        let fakeVoiceData = PocketTtsVoiceData(audioPrompt: [], promptLength: 0)

        do {
            _ = try await manager.makeSession(voiceData: fakeVoiceData)
            XCTFail("Expected error when not initialized")
        } catch let error as PocketTTSError {
            if case .modelNotFound = error {
                // Expected
            } else {
                XCTFail("Expected modelNotFound error, got: \(error)")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    // MARK: - KV Cache Clone Tests

    func testCloneKVCacheStateProducesIndependentCopy() throws {
        let original = try PocketTtsSynthesizer.emptyKVCacheState(
            layers: PocketTtsLanguage.english.transformerLayers)

        // Write a known value into the original
        let ptr = original.caches[0].dataPointer.bindMemory(to: Float.self, capacity: 1)
        ptr[0] = 42.0
        original.positions[0][0] = NSNumber(value: Float(7.0))

        let clone = try PocketTtsSynthesizer.cloneKVCacheState(original)

        // Clone should have the same values
        let clonePtr = clone.caches[0].dataPointer.bindMemory(to: Float.self, capacity: 1)
        XCTAssertEqual(clonePtr[0], 42.0)
        XCTAssertEqual(clone.positions[0][0].floatValue, 7.0)

        // Mutating clone should not affect original
        clonePtr[0] = 99.0
        clone.positions[0][0] = NSNumber(value: Float(15.0))
        XCTAssertEqual(ptr[0], 42.0, "Original cache should be unaffected by clone mutation")
        XCTAssertEqual(
            original.positions[0][0].floatValue, 7.0,
            "Original position should be unaffected by clone mutation"
        )
    }

    func testCloneKVCacheStatePreservesShape() throws {
        let original = try PocketTtsSynthesizer.emptyKVCacheState(
            layers: PocketTtsLanguage.english.transformerLayers)
        let clone = try PocketTtsSynthesizer.cloneKVCacheState(original)

        XCTAssertEqual(clone.caches.count, original.caches.count)
        XCTAssertEqual(clone.positions.count, original.positions.count)

        for i in 0..<clone.caches.count {
            XCTAssertEqual(clone.caches[i].shape, original.caches[i].shape)
            XCTAssertEqual(clone.caches[i].dataType, original.caches[i].dataType)
        }
    }

    /// Real model test, without an audio device. Explicit opt-in keeps ordinary
    /// CI independent of model downloads and local CoreML hardware.
    func testPerUtteranceStreamsFinishWhileSessionRemainsOpen() async throws {
        guard ProcessInfo.processInfo.environment["FLUID_TEST_POCKET_SESSION"] == "1" else {
            throw XCTSkip("Set FLUID_TEST_POCKET_SESSION=1 to run Pocket model integration")
        }
        let manager = PocketTtsManager()
        try await manager.initialize()
        let session = try await manager.makeSession(voice: "alba", seed: 42)
        let first = session.enqueueStreaming("The first sentence.")
        let second = session.enqueueStreaming("The second sentence.")
        var firstCount = 0
        for try await frame in first {
            XCTAssertEqual(frame.utteranceIndex, 0)
            firstCount += frame.samples.count
        }
        XCTAssertGreaterThan(firstCount, 0)
        // Deliberately consume both streams before finish(): sentence completion
        // must neither wait for another enqueue nor close the voice session.
        var secondCount = 0
        for try await frame in second {
            XCTAssertEqual(frame.utteranceIndex, 1)
            secondCount += frame.samples.count
        }
        XCTAssertGreaterThan(secondCount, 0)
        session.finish()
        let refused = session.enqueueStreaming("Already finished.")
        do {
            for try await _ in refused { XCTFail("Finished session accepted audio") }
            XCTFail("Expected cancellation for enqueue after finish")
        } catch is CancellationError { }
        await session.cancel()
    }

}
