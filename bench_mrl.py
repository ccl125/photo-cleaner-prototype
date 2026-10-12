"""MRL (Matryoshka) truncation validation: 768d -> 256d / 128d.

Recomputes 768-d embeddings with the existing vision_70 pipeline, truncates
to the first N dims, re-L2-normalizes, and compares similarity-separation
metrics across dims. Does not touch bench_results_b*.json.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from bench_embedding import GROUPS, UNRELATED, IMG_DIR, EmbeddingGemma2CoreML

ROOT = Path(__file__).parent
DIMS = [768, 256, 128]
THRESHOLD = 0.93


def truncate_l2(emb: np.ndarray, dim: int) -> np.ndarray:
    v = emb[:dim].astype(np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def pair_sets(names):
    def group_pairs(members):
        return [(members[i], members[j])
                for i in range(len(members)) for j in range(i + 1, len(members))]

    intra = [p for g in GROUPS.values() for p in group_pairs(g)]
    bases = [g[0] for g in GROUPS.values()] + UNRELATED
    inter = [(bases[i], bases[j])
             for i in range(len(bases)) for j in range(i + 1, len(bases))]
    cross = [(a, b) for g in GROUPS.values() for a in g[1:] for b in UNRELATED]
    return intra, inter, cross


def main():
    names = [n for g in GROUPS.values() for n in g] + UNRELATED
    images = {n: Image.open(IMG_DIR / f"{n}.jpg").convert("RGB") for n in names}
    print(f"loading model (vision_70, ALL), embedding {len(names)} images...")
    model = EmbeddingGemma2CoreML("ALL", 70)
    embs768 = {n: model.embed(img)[0].astype(np.float64) for n, img in images.items()}
    assert all(abs(np.linalg.norm(v) - 1) < 1e-3 for v in embs768.values())

    intra, inter, cross = pair_sets(names)
    all_pairs = intra + inter + cross
    idx = {n: i for i, n in enumerate(names)}

    results = {}
    sims = {}
    for dim in DIMS:
        E = np.stack([truncate_l2(embs768[n], dim) for n in names])
        sim = E @ E.T
        sims[dim] = sim
        s = {(a, b): float(sim[idx[a], idx[b]]) for a, b in all_pairs}
        iv = [s[p] for p in intra]
        ev = [s[p] for p in inter]
        cv = [s[p] for p in cross]
        max_other = max(ev + cv)
        min_intra = min(iv)
        separable = min_intra > max_other
        results[dim] = {
            "intra": {"min": min(iv), "mean": float(np.mean(iv)), "max": max(iv)},
            "inter": {"min": min(ev), "mean": float(np.mean(ev)), "max": max(ev)},
            "variant_vs_unrelated_max": max(cv),
            "margin": min_intra - max_other,
            "separable_at_093": separable and min_intra > THRESHOLD and max_other < THRESHOLD,
            "min_intra_pair": min(intra, key=lambda p: s[p]),
            "max_other_pair": max(inter + cross, key=lambda p: s[p]),
        }
        r = results[dim]
        print(f"\n== {dim}d: intra min/mean/max = {r['intra']['min']:.4f}/{r['intra']['mean']:.4f}/{r['intra']['max']:.4f} "
              f"| inter max = {max(ev):.4f} | variant-vs-unrel max = {max(cv):.4f}")
        print(f"   margin = {r['margin']:.4f} | fully separable @0.93: {r['separable_at_093']}")
        print(f"   weakest similar pair: {r['min_intra_pair']} | strongest unrelated pair: {r['max_other_pair']}")

    # ---- per-pair drift and flips between dims
    print("\n== pair drift (768 -> 256 -> 128) ==")
    drift_rows = []
    for a, b in all_pairs:
        i, j = idx[a], idx[b]
        v768, v256, v128 = sims[768][i, j], sims[256][i, j], sims[128][i, j]
        drift_rows.append((a, b, float(v768), float(v256), float(v128),
                           float(v128) - float(v768)))
    # classification flips under threshold 0.93 (and under per-dim perfect separation)
    flips = []
    for a, b, v768, v256, v128, _ in drift_rows:
        cls = [v > THRESHOLD for v in (v768, v256, v128)]
        if len(set(cls)) > 1:
            flips.append((a, b, v768, v256, v128))
    print(f"threshold-0.93 classification flips across dims: {len(flips)}")
    for a, b, v768, v256, v128 in flips:
        print(f"  FLIP {a} x {b}: {v768:.4f} -> {v256:.4f} -> {v128:.4f}")
    # separation-status change between dims
    sep = {d: results[d]["margin"] > 0 for d in DIMS}
    print(f"perfect-separation status 768/256/128: {sep[768]}/{sep[256]}/{sep[128]}")
    # biggest movers
    drop_intra = sorted([r for r in drift_rows if (r[0], r[1]) in intra],
                        key=lambda r: r[5])[:3]
    rise_other = sorted([r for r in drift_rows if (r[0], r[1]) not in intra],
                        key=lambda r: r[5], reverse=True)[:3]
    print("intra pairs with biggest 768->128 drop:")
    for a, b, v768, v256, v128, d in drop_intra:
        print(f"  {a} x {b}: {v768:.4f} -> {v256:.4f} -> {v128:.4f} ({d:+.4f})")
    print("unrelated pairs with biggest 768->128 rise:")
    for a, b, v768, v256, v128, d in rise_other:
        print(f"  {a} x {b}: {v768:.4f} -> {v256:.4f} -> {v128:.4f} ({d:+.4f})")

    # rank correlation of all pairwise scores across dims
    pair_scores = {d: [float(sims[d][idx[a], idx[b]]) for a, b, *_ in drift_rows]
                   for d in DIMS}
    def kendall_tau(x, y):
        x, y = np.asarray(x), np.asarray(y)
        conc = disc = 0
        for i in range(len(x) - 1):
            prod = np.sign(x[i] - x[i + 1:]) * np.sign(y[i] - y[i + 1:])
            conc += int((prod > 0).sum())
            disc += int((prod < 0).sum())
        return (conc - disc) / (conc + disc)
    tau_256 = kendall_tau(pair_scores[768], pair_scores[256])
    tau_128 = kendall_tau(pair_scores[768], pair_scores[128])
    print(f"\npair-score Kendall tau vs 768d: 256d={tau_256:.4f}, 128d={tau_128:.4f}")

    # ---- storage & search cost (theory + numpy micro-bench)
    print("\n== storage / search cost ==")
    n_lib = 10_000
    rng = np.random.default_rng(0)
    storage = {}
    search_ms = {}
    for dim in DIMS:
        lib = rng.standard_normal((n_lib, dim), dtype=np.float32)
        q = rng.standard_normal(dim, dtype=np.float32)
        lib /= np.linalg.norm(lib, axis=1, keepdims=True)
        q /= np.linalg.norm(q)
        for _ in range(3):
            lib @ q
        t0 = time.perf_counter()
        reps = 50
        for _ in range(reps):
            lib @ q
        search_ms[dim] = (time.perf_counter() - t0) / reps * 1e3
        storage[dim] = {"fp16_bytes_per_vec": dim * 2, "fp32_bytes_per_vec": dim * 4}
        print(f"  {dim}d: {dim*2} B/vec fp16 ({dim*2*n_lib/1e6:.1f} MB per 10k photos), "
              f"brute-force dot vs 10k lib: {search_ms[dim]*1e3:.0f} us")

    report = {
        "threshold": THRESHOLD,
        "dims": {str(d): {k: v for k, v in results[d].items()} for d in DIMS},
        "flips_at_threshold": [
            {"pair": [a, b], "768": v768, "256": v256, "128": v128}
            for a, b, v768, v256, v128 in flips],
        "kendall_tau_vs_768": {"256": tau_256, "128": tau_128},
        "storage": {str(d): storage[d] for d in DIMS},
        "search_ms_per_10k_fp32_numpy": {str(d): search_ms[d] for d in DIMS},
        "pair_scores": [
            {"a": a, "b": b, "intra": (a, b) in intra,
             "768": v768, "256": v256, "128": v128}
            for a, b, v768, v256, v128, _ in drift_rows],
    }
    out = ROOT / "bench_results_mrl.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\njson written: {out}")


if __name__ == "__main__":
    main()
