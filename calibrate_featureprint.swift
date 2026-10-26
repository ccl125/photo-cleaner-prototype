import AppKit
import Vision
import Foundation

// 校准 VNFeaturePrint 距离：已知近重复对 vs 语义相似但不同的对
let dir = "/Users/cuichenglong/mayday/photo-cleaner-prototype/images"
let files = ["a1_base.jpg","a2_resize75.jpg","a3_jpeg35.jpg",
             "b1_base.jpg","b2_crop80.jpg","b3_burst_shift_exp.jpg",
             "c1_base.jpg","c2_jpeg25.jpg","c3_resize50.jpg",
             "d1_unrelated.jpg","d2_unrelated.jpg","d3_unrelated.jpg",
             "d4_unrelated.jpg","d5_unrelated.jpg"]

var prints: [String: VNFeaturePrintObservation] = [:]
for f in files {
    let url = URL(fileURLWithPath: "\(dir)/\(f)")
    guard let img = NSImage(contentsOf: url)?.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("加载失败 \(f)"); continue
    }
    let req = VNGenerateImageFeaturePrintRequest()
    let handler = VNImageRequestHandler(cgImage: img, options: [:])
    try? handler.perform([req])
    if let obs = req.results?.first { prints[f] = obs } else { print("无指纹 \(f)") }
}

func dist(_ a: String, _ b: String) -> Float {
    var d: Float = -1
    if let pa = prints[a], let pb = prints[b] { try? pa.computeDistance(&d, to: pb) }
    return d
}

print("== 近重复对（应小）==")
for (a, b) in [("a1_base.jpg","a2_resize75.jpg"), ("a1_base.jpg","a3_jpeg35.jpg"),
               ("b1_base.jpg","b2_crop80.jpg"), ("b1_base.jpg","b3_burst_shift_exp.jpg"),
               ("c1_base.jpg","c2_jpeg25.jpg"), ("c1_base.jpg","c3_resize50.jpg")] {
    print(String(format: "  %@ vs %@: %.2f", a, b, dist(a, b)))
}
print("== 无关/语义相似对（应大）==")
for (a, b) in [("a1_base.jpg","d1_unrelated.jpg"), ("b1_base.jpg","d3_unrelated.jpg"),
               ("d4_unrelated.jpg","d5_unrelated.jpg"), ("a1_base.jpg","c1_base.jpg")] {
    print(String(format: "  %@ vs %@: %.2f", a, b, dist(a, b)))
}
