import Foundation
import CoreVideo
import CoreFoundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CoreImage

/// CoreVideo / CoreFoundation / ImageIO helpers for camera frames.
enum PixelBufferUtils {

    // MARK: - Create a buffer from a CFDictionary of attributes

    /// Builds an empty CVPixelBuffer. The attributes are passed as a CFDictionary,
    /// which is how the C-level CoreVideo API expects them.
    static func makeBuffer(width: Int,
                           height: Int,
                           format: OSType = kCVPixelFormatType_32BGRA) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [CFString: Any]() as CFDictionary,
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]
        var out: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, format,
                                         attrs as CFDictionary, &out)
        return status == kCVReturnSuccess ? out : nil
    }

    // MARK: - Deep copy (lock + read base addresses plane by plane)

    /// Copies a pixel buffer so the original (for example an ARKit frame buffer)
    /// can be released immediately. Works for planar (YCbCr) and packed formats.
    static func clone(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(src)
        let height = CVPixelBufferGetHeight(src)
        let format = CVPixelBufferGetPixelFormatType(src)
        guard let dst = makeBuffer(width: width, height: height, format: format) else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }

        let planes = CVPixelBufferGetPlaneCount(src)
        if planes == 0 {
            guard let s = CVPixelBufferGetBaseAddress(src),
                  let d = CVPixelBufferGetBaseAddress(dst) else { return nil }
            let sRow = CVPixelBufferGetBytesPerRow(src)
            let dRow = CVPixelBufferGetBytesPerRow(dst)
            for r in 0..<height {
                memcpy(d + r * dRow, s + r * sRow, min(sRow, dRow))
            }
        } else {
            for p in 0..<planes {
                guard let s = CVPixelBufferGetBaseAddressOfPlane(src, p),
                      let d = CVPixelBufferGetBaseAddressOfPlane(dst, p) else { continue }
                let sRow = CVPixelBufferGetBytesPerRowOfPlane(src, p)
                let dRow = CVPixelBufferGetBytesPerRowOfPlane(dst, p)
                let rows = CVPixelBufferGetHeightOfPlane(src, p)
                for r in 0..<rows {
                    memcpy(d + r * dRow, s + r * sRow, min(sRow, dRow))
                }
            }
        }
        return dst
    }

    // MARK: - Crop (BGRA buffers)

    /// Crops a 32BGRA buffer to a pixel rect, returning a new buffer.
    static func crop(_ src: CVPixelBuffer, to rect: CGRect) -> CVPixelBuffer? {
        guard CVPixelBufferGetPixelFormatType(src) == kCVPixelFormatType_32BGRA else { return nil }
        let bounds = CGRect(x: 0, y: 0,
                            width: CVPixelBufferGetWidth(src),
                            height: CVPixelBufferGetHeight(src))
        let r = rect.integral.intersection(bounds)
        guard !r.isNull, r.width >= 1, r.height >= 1,
              let dst = makeBuffer(width: Int(r.width), height: Int(r.height)) else { return nil }

        CVPixelBufferLockBaseAddress(src, .readOnly)
        CVPixelBufferLockBaseAddress(dst, [])
        defer {
            CVPixelBufferUnlockBaseAddress(dst, [])
            CVPixelBufferUnlockBaseAddress(src, .readOnly)
        }
        guard let s = CVPixelBufferGetBaseAddress(src),
              let d = CVPixelBufferGetBaseAddress(dst) else { return nil }
        let sRow = CVPixelBufferGetBytesPerRow(src)
        let dRow = CVPixelBufferGetBytesPerRow(dst)
        for row in 0..<Int(r.height) {
            let srcPtr = s + (Int(r.minY) + row) * sRow + Int(r.minX) * 4
            memcpy(d + row * dRow, srcPtr, Int(r.width) * 4)
        }
        return dst
    }

    // MARK: - Convert to CGImage (BGRA)

    static func cgImage(from buffer: CVPixelBuffer) -> CGImage? {
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }

        let info = CGBitmapInfo.byteOrder32Little.rawValue
                 | CGImageAlphaInfo.premultipliedFirst.rawValue
        guard let ctx = CGContext(data: base,
                                  width: CVPixelBufferGetWidth(buffer),
                                  height: CVPixelBufferGetHeight(buffer),
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: info) else { return nil }
        return ctx.makeImage()   // makeImage copies the pixels, safe after unlock
    }

    // MARK: - Save a PNG snapshot with ImageIO (CFMutableData + CGImageDestination)

    /// Encodes the buffer as PNG using ImageIO and writes it to disk.
    /// Returns the file URL on success.
    @discardableResult
    static func saveSnapshot(_ buffer: CVPixelBuffer, to url: URL,
                             note: String? = nil) -> URL? {
        guard let image = cgImage(from: buffer) else { return nil }

        let data = CFDataCreateMutable(kCFAllocatorDefault, 0)!      // CFMutableData
        guard let dest = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil) else { return nil }

        // Metadata passed as a CFDictionary
        var props: [CFString: Any] = [:]
        if let note {
            props[kCGImagePropertyPNGDictionary] = [
                kCGImagePropertyPNGDescription: note
            ] as CFDictionary
        }
        CGImageDestinationAddImage(dest, image, props as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }

        do {
            try (data as Data).write(to: url, options: .atomic)
            return url
        } catch {
            return nil
        }
    }

    /// Snapshot into the app's Documents folder with a timestamped name.
    static func saveSnapshotToDocuments(_ buffer: CVPixelBuffer, note: String? = nil) -> URL? {
        guard let docs = FileManager.default.urls(for: .documentDirectory,
                                                  in: .userDomainMask).first else { return nil }
        let stamp = Int(Date().timeIntervalSince1970)
        return saveSnapshot(buffer, to: docs.appendingPathComponent("snapshot_\(stamp).png"),
                            note: note)
    }

    // MARK: - Read metadata back with ImageIO (CGImageSource)

    static func readPNGDescription(at url: URL) -> String? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let png = props[kCGImagePropertyPNGDictionary] as? [CFString: Any]
        else { return nil }
        return png[kCGImagePropertyPNGDescription] as? String
    }

    // MARK: - Convert a camera YCbCr frame to BGRA (needed to snapshot ARKit frames)

    /// ARKit frames are YCbCr (bi-planar). This converts one to BGRA via CIContext
    /// so it can be cropped and saved.
    static func bgra(from src: CVPixelBuffer, using ciContext: CIContext = CIContext()) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(src), h = CVPixelBufferGetHeight(src)
        guard let dst = makeBuffer(width: w, height: h, format: kCVPixelFormatType_32BGRA) else { return nil }
        ciContext.render(CIImage(cvPixelBuffer: src), to: dst)
        return dst
    }
}
