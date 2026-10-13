import CoreML
import XCTest

@testable import FluidAudio

final class CtcLogProbsTests: XCTestCase {

    /// Deterministic logits in a realistic range, independent of any model.
    private func logit(_ t: Int, _ v: Int) -> Float {
        Float((t * 7919 + v * 104_729) % 2003) / 100 - 10
    }

    /// The element-by-element implementation the buffer path replaced, kept as the reference.
    private func referenceLogProbs(
        _ array: MLMultiArray, blankId: Int, temperature: Float, blankBias: Float
    ) -> [[Float]] {
        let rank4 = array.shape.count == 4
        let timeSteps = rank4 ? array.shape[3].intValue : array.shape[1].intValue
        let vocabSize = rank4 ? array.shape[1].intValue : array.shape[2].intValue
        return (0..<timeSteps).map { t in
            let logits = (0..<vocabSize).map { v -> Float in
                let index = (rank4 ? [0, v, 0, t] : [0, t, v]).map { NSNumber(value: $0) }
                return array[index].floatValue / temperature
            }
            let maxLogit = logits.max() ?? 0
            let logSumExp = logf(logits.reduce(0) { $0 + expf($1 - maxLogit) })
            var row = logits.map { ($0 - maxLogit) - logSumExp }
            if blankBias != 0, blankId < row.count { row[blankId] -= blankBias }
            return row
        }
    }

    /// Float16 [1, vocab, 1, time], the CTC head's output layout, with each vocab row padded.
    private func makeFloat16Rank4(vocabSize: Int, timeSteps: Int, paddedTime: Int) throws -> MLMultiArray {
        let storage = UnsafeMutablePointer<Float16>.allocate(capacity: vocabSize * paddedTime)
        storage.initialize(repeating: 0, count: vocabSize * paddedTime)
        for v in 0..<vocabSize {
            for t in 0..<timeSteps { storage[v * paddedTime + t] = Float16(logit(t, v)) }
        }
        return try MLMultiArray(
            dataPointer: storage,
            shape: [1, vocabSize, 1, timeSteps].map { NSNumber(value: $0) },
            dataType: .float16,
            strides: [vocabSize * paddedTime, paddedTime, paddedTime, 1].map { NSNumber(value: $0) },
            deallocator: { $0.deallocate() })
    }

    private func assertClose(
        _ actual: [[Float]], _ expected: [[Float]], file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        var worst: Float = 0
        for (a, e) in zip(actual, expected) {
            XCTAssertEqual(a.count, e.count, file: file, line: line)
            for (x, y) in zip(a, e) { worst = max(worst, abs(x - y)) }
        }
        XCTAssertLessThan(worst, 1e-4, file: file, line: line)
    }

    func testFloat16StridedOutputMatchesReferenceAndIsFaster() throws {
        let array = try makeFloat16Rank4(vocabSize: 1025, timeSteps: 188, paddedTime: 192)

        let referenceStart = Date()
        let expected = referenceLogProbs(array, blankId: 1024, temperature: 1, blankBias: 0)
        let referenceTime = Date().timeIntervalSince(referenceStart)

        let start = Date()
        let actual = try CtcKeywordSpotter.makeLogProbs(from: array, blankId: 1024)
        let time = Date().timeIntervalSince(start)

        assertClose(actual, expected)
        print(
            String(
                format: "CTC log-probs 188x1025: reference %.1f ms, buffer %.1f ms", referenceTime * 1000, time * 1000))
        XCTAssertLessThan(time, referenceTime)
    }

    func testFloat32Rank3AppliesTemperatureAndBlankBias() throws {
        let timeSteps = 12
        let vocabSize = 9
        let array = try MLMultiArray(
            shape: [1, timeSteps, vocabSize].map { NSNumber(value: $0) }, dataType: .float32)
        for t in 0..<timeSteps {
            for v in 0..<vocabSize { array[[0, t, v].map { NSNumber(value: $0) }] = NSNumber(value: logit(t, v)) }
        }

        let actual = try CtcKeywordSpotter.makeLogProbs(from: array, blankId: 8, temperature: 1.7, blankBias: 0.5)

        assertClose(actual, referenceLogProbs(array, blankId: 8, temperature: 1.7, blankBias: 0.5))
    }
}
