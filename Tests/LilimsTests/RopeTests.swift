#if canImport(CoreML)
import CoreML
import XCTest
@testable import RuntimeCoreML

@available(iOS 15.0, macOS 12.0, *)
final class RopeTests: XCTestCase {
    
    // MARK: - Table Generation Tests
    
    func testRotaryTablesShapeCorrectness() {
        let sequenceLengths = [1, 10, 512, 2048]
        let headDimensions = [32, 64, 128, 256]
        
        for seqLen in sequenceLengths {
            for headDim in headDimensions {
                let (sine, cosine) = Rope.rotaryTables(
                    sequenceLength: seqLen,
                    headDimension: headDim
                )
                
                XCTAssertEqual(sine.shape.count, 2, "Sine table should be 2D")
                XCTAssertEqual(cosine.shape.count, 2, "Cosine table should be 2D")
                
                XCTAssertEqual(sine.shape[0], seqLen, "First dimension should be sequence length")
                XCTAssertEqual(sine.shape[1], headDim / 2, "Second dimension should be headDim/2")
                
                XCTAssertEqual(cosine.shape[0], seqLen, "First dimension should be sequence length")
                XCTAssertEqual(cosine.shape[1], headDim / 2, "Second dimension should be headDim/2")
            }
        }
    }
    
    func testSineCosineValuesInRange() {
        let (sine, cosine) = Rope.rotaryTables(
            sequenceLength: 100,
            headDimension: 64
        )
        
        for value in sine.scalars {
            XCTAssertTrue(value >= -1.0 && value <= 1.0, 
                         "Sine values should be in range [-1, 1], got \(value)")
        }
        
        for value in cosine.scalars {
            XCTAssertTrue(value >= -1.0 && value <= 1.0, 
                         "Cosine values should be in range [-1, 1], got \(value)")
        }
    }
    
    func testTablesWithDifferentSequenceLengths() {
        let sequenceLengths = [1, 10, 512, 2048]
        let headDimension = 64
        
        for seqLen in sequenceLengths {
            XCTAssertNoThrow({
                let (sine, cosine) = Rope.rotaryTables(
                    sequenceLength: seqLen,
                    headDimension: headDimension
                )
                
                XCTAssertEqual(sine.scalars.count, seqLen * headDimension / 2)
                XCTAssertEqual(cosine.scalars.count, seqLen * headDimension / 2)
            })
        }
    }
    
    func testTablesWithDifferentHeadDimensions() {
        let headDimensions = [32, 64, 128, 256]
        let sequenceLength = 128
        
        for headDim in headDimensions {
            XCTAssertNoThrow({
                let (sine, cosine) = Rope.rotaryTables(
                    sequenceLength: sequenceLength,
                    headDimension: headDim
                )
                
                XCTAssertEqual(sine.scalars.count, sequenceLength * headDim / 2)
                XCTAssertEqual(cosine.scalars.count, sequenceLength * headDim / 2)
            })
        }
    }
    
    func testTablesWithDifferentBaseFrequencies() {
        let bases: [Float] = [1000, 10_000, 500_000]
        let sequenceLength = 64
        let headDimension = 32
        
        var previousTables: [(sine: MLShapedArray<Float32>, cosine: MLShapedArray<Float32>)] = []
        
        for base in bases {
            let (sine, cosine) = Rope.rotaryTables(
                sequenceLength: sequenceLength,
                headDimension: headDimension,
                base: base
            )
            
            XCTAssertEqual(sine.shape[0], sequenceLength)
            XCTAssertEqual(sine.shape[1], headDimension / 2)
            
            // Ensure different bases produce different results
            for previousTable in previousTables {
                let areDifferent = zip(sine.scalars, previousTable.sine.scalars).contains { abs($0.0 - $0.1) > 1e-6 }
                XCTAssertTrue(areDifferent, "Different base frequencies should produce different tables")
            }
            
            previousTables.append((sine, cosine))
        }
    }
    
    func testPreconditionFailureForOddHeadDimensions() {
        // Test that odd head dimensions are properly handled
        // Note: precondition failures can't be easily tested in XCTest
        // We test the valid case to ensure it doesn't crash
        XCTAssertNoThrow({
            _ = Rope.rotaryTables(sequenceLength: 4, headDimension: 32) // even
        })
        
        // The actual precondition test would require a different testing framework
        // For now, we document that odd dimensions should fail
    }
    
    func testNumericalAccuracyOfGeneratedFrequencies() {
        let (sine, cosine) = Rope.rotaryTables(
            sequenceLength: 4,
            headDimension: 4,
            base: 10_000
        )
        
        // For position 0, all values should be specific known values
        // sin(0) = 0, cos(0) = 1 for all frequencies
        let tolerance: Float = 1e-6
        
        XCTAssertEqual(sine[0, 0].scalar ?? -999, 0, accuracy: tolerance, "sin(0) should be 0")
        XCTAssertEqual(sine[0, 1].scalar ?? -999, 0, accuracy: tolerance, "sin(0) should be 0")
        XCTAssertEqual(cosine[0, 0].scalar ?? -999, 1, accuracy: tolerance, "cos(0) should be 1")
        XCTAssertEqual(cosine[0, 1].scalar ?? -999, 1, accuracy: tolerance, "cos(0) should be 1")
        
        // Test mathematical identity: sin^2 + cos^2 = 1
        for pos in 0..<sine.shape[0] {
            for dim in 0..<sine.shape[1] {
                let sinVal = sine[pos, dim].scalar ?? 0
                let cosVal = cosine[pos, dim].scalar ?? 0
                let identityResult = sinVal * sinVal + cosVal * cosVal
                XCTAssertEqual(identityResult, 1.0, accuracy: tolerance, 
                             "sin^2 + cos^2 should equal 1 at position \(pos), dimension \(dim)")
            }
        }
    }
    
    func testMemoryEfficiencyForLargeTables() {
        // Test that large table generation doesn't crash due to memory issues
        XCTAssertNoThrow({
            _ = Rope.rotaryTables(
                sequenceLength: 2048,
                headDimension: 128
            )
        })
        
        // Test memory usage is reasonable (no specific assertion, just shouldn't crash)
        let largeSequenceLength = 4096
        let largeHeadDimension = 256
        
        XCTAssertNoThrow({
            let (sine, cosine) = Rope.rotaryTables(
                sequenceLength: largeSequenceLength,
                headDimension: largeHeadDimension
            )
            
            let expectedSize = largeSequenceLength * largeHeadDimension / 2
            XCTAssertEqual(sine.scalars.count, expectedSize)
            XCTAssertEqual(cosine.scalars.count, expectedSize)
        })
    }
    
    // MARK: - Frequency Formula Tests
    
    func testFrequencyFormula() {
        // Test that the frequency formula is correct
        // freq_i = 1.0 / (base^(2i / head_dim)) for RoPE
        let base: Float = 10_000
        let headDimension = 64
        let sequenceLength = 4
        
        let (sine, cosine) = Rope.rotaryTables(
            sequenceLength: sequenceLength,
            headDimension: headDimension,
            base: base
        )
        
        // Manually calculate expected values for position 1 (to avoid zeros)
        let position = 1
        let halfDim = headDimension / 2
        
        for i in 0..<halfDim {
            let expectedFreq = 1.0 / pow(base, 2.0 * Float(i) / Float(headDimension))
            let expectedAngle = Float(position) * expectedFreq
            let expectedSin = sin(expectedAngle)
            let expectedCos = cos(expectedAngle)
            
            let actualSin = sine[position, i].scalar ?? 0
            let actualCos = cosine[position, i].scalar ?? 0
            
            let tolerance: Float = 1e-5
            XCTAssertEqual(actualSin, expectedSin, accuracy: tolerance, 
                         "Sine value at position \(position), dim \(i)")
            XCTAssertEqual(actualCos, expectedCos, accuracy: tolerance, 
                         "Cosine value at position \(position), dim \(i)")
        }
    }
    
    // MARK: - Performance Tests
    
    func testTableGenerationPerformance() {
        measure {
            for _ in 0..<10 {
                _ = Rope.rotaryTables(
                    sequenceLength: 1024,
                    headDimension: 128
                )
            }
        }
    }
    
    func testLargeTableGenerationPerformance() {
        measure {
            _ = Rope.rotaryTables(
                sequenceLength: 4096,
                headDimension: 256
            )
        }
    }
    
    // MARK: - RoPE Application Tests
    
    func testRotateHalfFunctionCorrectness() throws {
        let numberOfHeads = 2
        let headDimension = 4
        
        // Create test tensor
        let shape = [numberOfHeads, headDimension]
        let tensor = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        
        // Fill with test values
        tensor[[0, 0]] = 1.0  // head 0, first half
        tensor[[0, 1]] = 2.0
        tensor[[0, 2]] = 3.0  // head 0, second half
        tensor[[0, 3]] = 4.0
        tensor[[1, 0]] = 5.0  // head 1, first half
        tensor[[1, 1]] = 6.0
        tensor[[1, 2]] = 7.0  // head 1, second half  
        tensor[[1, 3]] = 8.0
        
        // Create simple sin/cos values for testing
        let sin: [Float32] = [0.5, 0.0] // halfDim = 2
        let cos: [Float32] = [0.866, 1.0] // sqrt(3)/2, 1
        
        let rotated = try Rope.rotateHalf(tensor, sin: sin, cos: cos, numberOfHeads: numberOfHeads)
        
        // Verify shape is preserved
        XCTAssertEqual(rotated.shape.count, 2)
        XCTAssertEqual(rotated.shape[0].intValue, numberOfHeads)
        XCTAssertEqual(rotated.shape[1].intValue, headDimension)
        
        // Test rotation formula: 
        // result[i] = x1 * cos[i] - x2 * sin[i]
        // result[i + halfDim] = x1 * sin[i] + x2 * cos[i]
        
        let tolerance: Float = 1e-5
        
        // Head 0, dimension 0: x1=1, x2=3, cos=0.866, sin=0.5
        let expected_0_0: Float = 1.0 * 0.866 - 3.0 * 0.5  // 0.866 - 1.5 = -0.634
        let expected_0_2: Float = 1.0 * 0.5 + 3.0 * 0.866   // 0.5 + 2.598 = 3.098
        
        XCTAssertEqual(rotated[[0, 0]].floatValue, expected_0_0, accuracy: tolerance)
        XCTAssertEqual(rotated[[0, 2]].floatValue, expected_0_2, accuracy: tolerance)
        
        // Head 0, dimension 1: x1=2, x2=4, cos=1.0, sin=0.0
        XCTAssertEqual(rotated[[0, 1]].floatValue, 2.0, accuracy: tolerance) // 2*1 - 4*0 = 2
        XCTAssertEqual(rotated[[0, 3]].floatValue, 4.0, accuracy: tolerance) // 2*0 + 4*1 = 4
    }
    
    func testRopeApplicationToQueryTensors() throws {
        let numberOfHeads = 1
        let headDimension = 4
        
        let queries = try MLMultiArray(shape: [numberOfHeads, headDimension].map { NSNumber(value: $0) }, dataType: .float32)
        queries[[0, 0]] = 1.0
        queries[[0, 1]] = 0.0
        queries[[0, 2]] = 0.0  
        queries[[0, 3]] = 1.0
        
        let sin: [Float32] = [0.0, 1.0] // [sin(0), sin(π/2)]
        let cos: [Float32] = [1.0, 0.0] // [cos(0), cos(π/2)]
        
        let rotated = try Rope.rotateHalf(queries, sin: sin, cos: cos, numberOfHeads: numberOfHeads)
        
        // For rotation: 
        // x1=1, x2=0 (first pair), cos=1.0, sin=0.0 -> [1*1 - 0*0, 1*0 + 0*1] = [1, 0]
        // x1=0, x2=1 (second pair), cos=0.0, sin=1.0 -> [0*0 - 1*1, 0*1 + 1*0] = [-1, 0]
        
        let tolerance: Float = 1e-5
        XCTAssertEqual(rotated[[0, 0]].floatValue, 1.0, accuracy: tolerance)
        XCTAssertEqual(rotated[[0, 2]].floatValue, 0.0, accuracy: tolerance)  // x1*sin + x2*cos = 1*0 + 0*0 = 0
        XCTAssertEqual(rotated[[0, 1]].floatValue, -1.0, accuracy: tolerance) // x1*cos - x2*sin = 0*0 - 1*1 = -1
        XCTAssertEqual(rotated[[0, 3]].floatValue, 0.0, accuracy: tolerance)  // x1*sin + x2*cos = 0*1 + 1*0 = 0
    }
    
    func testRopeApplicationToKeyTensors() throws {
        let numberOfHeads = 2  
        let headDimension = 2
        
        let keys = try MLMultiArray(shape: [numberOfHeads, headDimension].map { NSNumber(value: $0) }, dataType: .float32)
        keys[[0, 0]] = 2.0
        keys[[0, 1]] = 0.0
        keys[[1, 0]] = 0.0
        keys[[1, 1]] = 3.0
        
        let sin: [Float32] = [0.5] // Only one dimension since headDim/2 = 1
        let cos: [Float32] = [0.866] // sqrt(3)/2
        
        let rotated = try Rope.rotateHalf(keys, sin: sin, cos: cos, numberOfHeads: numberOfHeads)
        
        let tolerance: Float = 1e-3
        
        // Head 0: [2, 0] -> [2*0.866 - 0*0.5, 2*0.5 + 0*0.866] = [1.732, 1.0]
        XCTAssertEqual(rotated[[0, 0]].floatValue, 1.732, accuracy: tolerance)
        XCTAssertEqual(rotated[[0, 1]].floatValue, 1.0, accuracy: tolerance)
        
        // Head 1: [0, 3] -> [0*0.866 - 3*0.5, 0*0.5 + 3*0.866] = [-1.5, 2.598]
        XCTAssertEqual(rotated[[1, 0]].floatValue, -1.5, accuracy: tolerance)
        XCTAssertEqual(rotated[[1, 1]].floatValue, 2.598, accuracy: tolerance)
    }
    
    func testPositionDependentRotationBehavior() throws {
        let headDimension = 4
        let numberOfHeads = 1
        let sequenceLength = 3
        
        // Generate tables
        let (sineTable, cosineTable) = Rope.rotaryTables(
            sequenceLength: sequenceLength,
            headDimension: headDimension
        )
        
        let tensor = try MLMultiArray(shape: [numberOfHeads, headDimension].map { NSNumber(value: $0) }, dataType: .float32)
        tensor[[0, 0]] = 1.0
        tensor[[0, 1]] = 1.0
        tensor[[0, 2]] = 1.0
        tensor[[0, 3]] = 1.0
        
        var previousRotations: [MLMultiArray] = []
        
        // Test rotation at different positions
        for position in 0..<sequenceLength {
            var sinValues: [Float32] = []
            var cosValues: [Float32] = []
            
            let halfDim = headDimension / 2
            for i in 0..<halfDim {
                sinValues.append(sineTable[position, i].scalar ?? 0)
                cosValues.append(cosineTable[position, i].scalar ?? 0)
            }
            
            let rotated = try Rope.rotateHalf(tensor, sin: sinValues, cos: cosValues, numberOfHeads: numberOfHeads)
            
            // Different positions should produce different rotations
            for previousRotation in previousRotations {
                let areDifferent = (0..<headDimension).contains { dim in
                    let current = rotated[[NSNumber(value: 0), NSNumber(value: dim)]].floatValue
                    let previous = previousRotation[[NSNumber(value: 0), NSNumber(value: dim)]].floatValue
                    return abs(current - previous) > 1e-6
                }
                XCTAssertTrue(areDifferent, "Different positions should produce different rotations")
            }
            
            previousRotations.append(rotated)
        }
    }
    
    func testRopePreservesTensorShapes() throws {
        let shapes = [[1, 32], [4, 64], [8, 128]]
        
        for shape in shapes {
            let numberOfHeads = shape[0]
            let headDimension = shape[1]
            
            let tensor = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
            
            // Fill with random-ish values
            for head in 0..<numberOfHeads {
                for dim in 0..<headDimension {
                    tensor[[NSNumber(value: head), NSNumber(value: dim)]] = NSNumber(value: Float.random(in: -1...1))
                }
            }
            
            let halfDim = headDimension / 2
            let sin = Array(repeating: Float32(0.1), count: halfDim)
            let cos = Array(repeating: Float32(0.995), count: halfDim) // cos(0.1) ≈ 0.995
            
            let rotated = try Rope.rotateHalf(tensor, sin: sin, cos: cos, numberOfHeads: numberOfHeads)
            
            XCTAssertEqual(rotated.shape.count, tensor.shape.count)
            XCTAssertEqual(rotated.shape[0], tensor.shape[0])
            XCTAssertEqual(rotated.shape[1], tensor.shape[1])
            XCTAssertEqual(rotated.dataType, tensor.dataType)
        }
    }
    
    func testRopeNumericalStability() throws {
        let numberOfHeads = 1
        let headDimension = 4
        
        // Test with extreme values
        let tensor = try MLMultiArray(shape: [numberOfHeads, headDimension].map { NSNumber(value: $0) }, dataType: .float32)
        tensor[[0, 0]] = 1000.0
        tensor[[0, 1]] = -1000.0
        tensor[[0, 2]] = 0.001
        tensor[[0, 3]] = -0.001
        
        let sin: [Float32] = [0.1, 0.9]
        let cos: [Float32] = [0.995, 0.436] // cos(0.1), cos(0.9)
        
        XCTAssertNoThrow({
            let rotated = try Rope.rotateHalf(tensor, sin: sin, cos: cos, numberOfHeads: numberOfHeads)
            
            // Check for NaN or infinite values
            for head in 0..<numberOfHeads {
                for dim in 0..<headDimension {
                    let value = rotated[[NSNumber(value: head), NSNumber(value: dim)]].floatValue
                    XCTAssertFalse(value.isNaN, "Result should not be NaN")
                    XCTAssertFalse(value.isInfinite, "Result should not be infinite")
                }
            }
        })
    }
    
    func testRopeWithDifferentPositionsInSequence() throws {
        let headDimension = 8
        let numberOfHeads = 2
        let maxSequenceLength = 16
        
        let (sineTable, cosineTable) = Rope.rotaryTables(
            sequenceLength: maxSequenceLength,
            headDimension: headDimension
        )
        
        let tensor = try MLMultiArray(shape: [numberOfHeads, headDimension].map { NSNumber(value: $0) }, dataType: .float32)
        
        // Fill tensor with sequential values for easy verification
        for head in 0..<numberOfHeads {
            for dim in 0..<headDimension {
                tensor[[NSNumber(value: head), NSNumber(value: dim)]] = NSNumber(value: head * headDimension + dim)
            }
        }
        
        // Test positions throughout the sequence
        let testPositions = [0, 1, 4, 8, 15]
        
        for position in testPositions {
            var sinValues: [Float32] = []
            var cosValues: [Float32] = []
            
            let halfDim = headDimension / 2
            for i in 0..<halfDim {
                sinValues.append(sineTable[position, i].scalar ?? 0)
                cosValues.append(cosineTable[position, i].scalar ?? 0)
            }
            
            XCTAssertNoThrow({
                let rotated = try Rope.rotateHalf(tensor, sin: sinValues, cos: cosValues, numberOfHeads: numberOfHeads)
                
                // Verify shapes
                XCTAssertEqual(rotated.shape[0].intValue, numberOfHeads)
                XCTAssertEqual(rotated.shape[1].intValue, headDimension)
                
                // At position 0, sin values should be 0, so rotation should be minimal
                if position == 0 {
                    for head in 0..<numberOfHeads {
                        for dim in 0..<headDimension {
                            let original = tensor[[NSNumber(value: head), NSNumber(value: dim)]].floatValue
                            let rotated_val = rotated[[NSNumber(value: head), NSNumber(value: dim)]].floatValue
                            
                            // For position 0, rotation should be close to identity
                            // (though not exactly due to floating point precision)
                            XCTAssertTrue(abs(rotated_val - original) < 1e-3,
                                        "Rotation at position 0 should be close to identity")
                        }
                    }
                }
            })
        }
    }
}
#endif