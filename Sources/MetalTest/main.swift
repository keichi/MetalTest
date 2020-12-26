import Foundation
import MetalPerformanceShaders

func gflops(time: Double, size: Int) -> Double {
    return 2.0 * pow(Double(size), 3) / time / 1E9
}

// Prepare some data
/* let N = 4096 */
let N = 8192
let rowsA = N
let columnsA = N

let a = UnsafeMutablePointer<Float>.allocate(capacity: rowsA * columnsA)
let arrayA = UnsafeMutableBufferPointer(start: a, count: rowsA * columnsA)
arrayA.assign(repeating: Float(1.0))
print("Values in input array: \(arrayA[0])")
print()

// Get the device
 if #available(macOS 10.13, *) {
let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!
let commandBuffer = commandQueue.makeCommandBuffer()!
/* let blitEncoder = commandBuffer.makeBlitCommandEncoder()! */

// 1. Prepare managed buffers
let rowBytesA = columnsA * MemoryLayout<Float>.stride
let bufferA = device.makeBuffer(bytes: arrayA.baseAddress!, length: rowsA * rowBytesA, options: [.storageModeManaged])!
let bufferC = device.makeBuffer(length: columnsA * rowBytesA, options: [.storageModeManaged])!

// 2. Encode matrix multiplication
let descrA = MPSMatrixDescriptor(rows: rowsA, columns: columnsA, rowBytes: rowBytesA, dataType: .float32)
let descrC = MPSMatrixDescriptor(rows: columnsA, columns: columnsA, rowBytes: rowBytesA, dataType: .float32)

let matrixA = MPSMatrix(buffer: bufferA, descriptor: descrA)
let matrixC = MPSMatrix(buffer: bufferC, descriptor: descrC)
let matMul = MPSMatrixMultiplication(device: device, resultRows: columnsA, resultColumns: columnsA, interiorColumns: rowsA)

let startTime = CFAbsoluteTimeGetCurrent()
matMul.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixA, resultMatrix: matrixC)

// 3. Get data back from GPU
/* blitEncoder.synchronize(resource: bufferC) */

// 4. Run buffer
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

let elapsed = CFAbsoluteTimeGetCurrent() - startTime
let gf = gflops(time: elapsed / 1.0, size: N)
print("Run at \(Int(gf)) GFlops total")

// Read results
let resultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: columnsA * columnsA)
let result = UnsafeBufferPointer(start: resultPointer, count: columnsA * columnsA)
print("Resulting values: [\(result[0])...\(result[columnsA * columnsA - 1])]")
print("Resulting values: [\(result[0])...\(result[columnsA * columnsA - 1])]")
}
