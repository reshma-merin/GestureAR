import SwiftUI
import RealityKit
import ARKit
import Vision
import Combine
import CoreML

// MARK: - Model

enum Gesture: String {
    case none, openPalm, pinch, fist
}

final class GestureModel: ObservableObject {
    @Published var gesture: Gesture = .none
    @Published var lastSnapshot = ""
    @Published var fps = ""
    @Published var latency = ""
    var snapshotRequested = false
}

// MARK: - UI

struct ContentView: View {
    @StateObject private var model = GestureModel()

    var body: some View {
        ZStack(alignment: .top) {
            ARViewContainer(model: model).ignoresSafeArea()

            VStack(spacing: 6) {
                Text("Gesture: \(model.gesture.rawValue)")
                    .padding(10)
                    .background(.ultraThinMaterial, in: Capsule())
                Text(model.fps)
                    .font(.caption)
                Text(model.latency)
                    .font(.system(size: 11, design: .monospaced))
                    .multilineTextAlignment(.center)
                Text(model.lastSnapshot)
                    .font(.caption)
            }
            .padding(.top, 60)

            Button("Save snapshot") { model.snapshotRequested = true }
                .padding(10)
                .background(.ultraThinMaterial, in: Capsule())
                .frame(maxHeight: .infinity, alignment: .bottom)
                .padding(.bottom, 40)
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    let model: GestureModel

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        arView.environment.background = .cameraFeed()
        arView.renderOptions.insert(.disableMotionBlur)
        arView.renderOptions.insert(.disableDepthOfField)

        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal]
        config.environmentTexturing = .none
        arView.session.delegate = context.coordinator
        context.coordinator.arView = arView
        arView.session.delegateQueue = DispatchQueue(label: "ar.delegate")
        arView.session.run(config)
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(model: model) }
}

// MARK: - Logic

final class Coordinator: NSObject, ARSessionDelegate {
    let model: GestureModel
    weak var arView: ARView?

    private let request: VNDetectHumanHandPoseRequest = {
        let r = VNDetectHumanHandPoseRequest()
        r.maximumHandCount = 1
        return r
    }()
    private var frameCount = 0

    // FPS measurement (frames actually processed by the pipeline)
    private var fpsFrames = 0
    private var fpsStart = CFAbsoluteTimeGetCurrent()

    // Latency measurement (all touched only on the serial AR delegate queue)
    private var visionMs: [Double] = []
    private var predictMs: [Double] = []

    private let classifier: GestureClassifier? = {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        return try? GestureClassifier(configuration: cfg)
    }()

    // debounce state
    private var candidate: Gesture = .none
    private var candidateCount = 0
    private var stable: Gesture = .none

    private var current: ModelEntity?

    init(model: GestureModel) { self.model = model }

    // Median and 95th percentile of a list of samples
    private func stats(_ xs: [Double]) -> (median: Double, p95: Double)? {
        guard !xs.isEmpty else { return nil }
        let s = xs.sorted()
        let median = s[s.count / 2]
        let p95 = s[min(s.count - 1, Int(Double(s.count) * 0.95))]
        return (median, p95)
    }

    // Called by ARKit for every camera frame
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Debug snapshot (CoreVideo / CoreFoundation / ImageIO utilities)
        if model.snapshotRequested {
            model.snapshotRequested = false
            if let bgra = PixelBufferUtils.bgra(from: frame.capturedImage),
               let url = PixelBufferUtils.saveSnapshotToDocuments(bgra, note: "GestureAR debug frame") {
                DispatchQueue.main.async { self.model.lastSnapshot = "Saved \(url.lastPathComponent)" }
            }
        }

        frameCount += 1
        guard frameCount % 3 == 0 else { return }   // process every 3rd frame

        let handler = VNImageRequestHandler(cvPixelBuffer: frame.capturedImage, orientation: .right)
        var result: Gesture = .none
        do {
            let t0 = CFAbsoluteTimeGetCurrent()
            try handler.perform([request])
            visionMs.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)

            if let obs = request.results?.first {
                result = classify(obs)
            }
        } catch {
            print("Vision error: \(error)")
        }

        // Count a frame only after it has been fully processed
        fpsFrames += 1
        let now = CFAbsoluteTimeGetCurrent()
        if now - fpsStart >= 1.0 {
            let fps = Double(fpsFrames) / (now - fpsStart)
            DispatchQueue.main.async { self.model.fps = String(format: "%.1f fps", fps) }
            fpsFrames = 0
            fpsStart = now
        }

        // Every 100 processed frames, publish median / p95 and start a new window
        if visionMs.count >= 100 {
            var text = ""
            if let v = stats(visionMs) {
                text += String(format: "Vision  med %.2f ms  p95 %.2f ms", v.median, v.p95)
            }
            if let p = stats(predictMs) {
                text += String(format: "\nCoreML  med %.3f ms  p95 %.3f ms", p.median, p.p95)
            }
            DispatchQueue.main.async { self.model.latency = text }
            visionMs.removeAll(keepingCapacity: true)
            predictMs.removeAll(keepingCapacity: true)
        }

        DispatchQueue.main.async { self.update(with: result) }
    }

    // CoreML gesture classifier on Vision hand-pose keypoints
    private func classify(_ obs: VNHumanHandPoseObservation) -> Gesture {
        guard let classifier,
              let poses = try? obs.keypointsMultiArray() else { return .none }

        let t0 = CFAbsoluteTimeGetCurrent()
        guard let out = try? classifier.prediction(poses: poses) else { return .none }
        predictMs.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)

        let confidence = out.labelProbabilities[out.label] ?? 0
        guard confidence > 0.85 else { return .none }
        return Gesture(rawValue: out.label) ?? .none
    }

    // Engage after 4 matching frames, release after 2 non-matching frames
    private func update(with g: Gesture) {
        if g == candidate { candidateCount += 1 } else { candidate = g; candidateCount = 1 }

        let needed = (candidate == .none) ? 2 : 4          // release quickly
        if candidateCount >= needed, stable != candidate {
            stable = candidate
            model.gesture = stable
            if stable == .openPalm { spawn() }
        }

        // Only act when the CURRENT frame still matches the stable gesture
        guard g == stable else { return }

        switch stable {
        case .pinch:
            if let e = current {
                let s = min(e.scale.x * 1.02, 3.0)         // cap the size at 3x
                e.scale = [s, s, s]
            }
        case .fist:
            if let e = current {
                e.orientation = simd_quatf(angle: .pi / 45, axis: [0, 1, 0]) * e.orientation
            }
        default: break
        }
    }

    // Place a box on a detected surface, or 40 cm in front of the camera as a fallback
    private func spawn() {
        guard let arView else { return }
        let center = CGPoint(x: arView.bounds.midX, y: arView.bounds.midY)

        var transform: simd_float4x4
        if let query = arView.makeRaycastQuery(from: center,
                                               allowing: .estimatedPlane,
                                               alignment: .any),
           let hit = arView.session.raycast(query).first {
            transform = hit.worldTransform
        } else if let cam = arView.session.currentFrame?.camera.transform {
            var t = matrix_identity_float4x4
            t.columns.3.z = -0.4
            transform = cam * t
        } else {
            return
        }

        let box = ModelEntity(
            mesh: .generateBox(size: 0.1, cornerRadius: 0.01),
            materials: [SimpleMaterial(color: .systemBlue, isMetallic: true)]
        )
        let anchor = AnchorEntity(world: transform)
        anchor.addChild(box)
        arView.scene.addAnchor(anchor)
        current = box
    }
}
