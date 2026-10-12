"""Benchmark EmbeddingGemma 2 Core ML port for similar-photo detection.

Pipeline (per FluidInference/embeddinggemma-2-coreml README + config.json):
  image -> aspect-preserving resize (sides multiple of 48px, <= budget*9 patches of 16px)
        -> RGB/255 -> row-major 16x16x3 patches
        -> EmbeddingGemma2Vision.mlpackage (vision_70/140/280)
        -> image_tokens [1, T, 512]
        -> <bos> <|image|> tokens <image|> <eos> (special rows from embeddings.bf16 * sqrt(512))
        -> EmbeddingGemma2Text.mlpackage (smallest embed_S that fits)
        -> 768-d embedding, L2-normalized

Usage: python bench_embedding.py [--budget 70] [--runs 10]
"""
import argparse
import io
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models" / "embeddinggemma-2-coreml"
IMG_DIR = ROOT / "images"

PATCH = 16
POOL_K = 3
EMBED_SCALE = 22.627417  # sqrt(512)
BOS, EOS = 2, 1
BOI, EOI = 255999, 258882

GROUPS = {
    "A (resize/jpeg)": ["a1_base", "a2_resize75", "a3_jpeg35"],
    "B (crop/burst)": ["b1_base", "b2_crop80", "b3_burst_shift_exp"],
    "C (jpeg/scale)": ["c1_base", "c2_jpeg25", "c3_resize50"],
}
UNRELATED = ["d1_unrelated", "d2_unrelated", "d3_unrelated", "d4_unrelated", "d5_unrelated"]


# ---------------------------------------------------------------- model wrap
def _spec_dtype(model, fn_name, name):
    from coremltools.proto.FeatureTypes_pb2 import ArrayFeatureType
    desc = model.get_spec().description
    fn = next(f for f in desc.functions if f.name == fn_name)
    for inp in fn.input:
        if inp.name == name:
            t = inp.type.multiArrayType.dataType
            if t == ArrayFeatureType.FLOAT16:
                return np.float16
            if t == ArrayFeatureType.INT32:
                return np.int32
            return np.float32
    raise KeyError(f"input {name} not found in {fn_name}")


def load_token_rows(ids):
    """Read specific rows from embeddings.bf16 (raw bfloat16 [262144, 512]) -> fp32."""
    table = np.memmap(MODEL_DIR / "embeddings.bf16", dtype=np.uint16, mode="r").reshape(-1, 512)
    rows = np.asarray(table[ids], dtype=np.uint32) << 16
    return rows.view(np.float32) * EMBED_SCALE


class EmbeddingGemma2CoreML:
    def __init__(self, compute_units: str, budget: int = 70):
        cu = getattr(ct.ComputeUnit, compute_units)
        # smallest embed_S that fits <bos> <|image|> [budget tokens] <image|> <eos>
        self.text_seq = next(s for s in (32, 48, 64, 128, 256, 512) if s >= budget + 4)
        self.text_fn = f"embed_{self.text_seq}"
        t0 = time.perf_counter()
        self.vision = ct.models.MLModel(
            str(MODEL_DIR / "EmbeddingGemma2Vision.mlpackage"),
            compute_units=cu, function_name=f"vision_{budget}")
        self.vision_load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.text = ct.models.MLModel(
            str(MODEL_DIR / "EmbeddingGemma2Text.mlpackage"),
            compute_units=cu, function_name=self.text_fn)
        self.text_load_s = time.perf_counter() - t0

        self.n_patches = budget * 9
        self.n_tokens = budget
        self.pos_table = np.fromfile(
            MODEL_DIR / "position_embeddings.f16", dtype=np.float16).reshape(2, 1024, 768)
        self.special = load_token_rows([BOS, BOI, EOI, EOS])  # fp32 [4, 512], pre-scaled

        self.in_dtype = {
            name: _spec_dtype(self.vision, f"vision_{budget}", name)
            for name in ["patches", "positions", "position_embeddings", "valid", "pool"]
        }
        self.text_dtype = {
            name: _spec_dtype(self.text, self.text_fn, name)
            for name in ["inputs_embeds", "attention_mask"]
        }

    # ------------------------------------------------------------ preprocessing
    def _target_size(self, W, H):
        scale = min(1.0, (self.n_patches * PATCH * PATCH / (W * H)) ** 0.5)
        w = max(48, round(W * scale / 48) * 48)
        h = max(48, round(H * scale / 48) * 48)
        while (w // PATCH) * (h // PATCH) > self.n_patches:
            if w >= h:
                w -= 48
            else:
                h -= 48
        return w, h

    def preprocess(self, img: Image.Image):
        W, H = img.size
        w, h = self._target_size(W, H)
        img = img.resize((w, h), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        gh, gw = h // PATCH, w // PATCH
        n = gh * gw
        N, T = self.n_patches, self.n_tokens

        patches = np.zeros((1, N, PATCH * PATCH * 3), np.float32)
        p = arr.reshape(gh, PATCH, gw, PATCH, 3).transpose(0, 2, 1, 3, 4).reshape(n, -1)
        patches[0, :n] = p

        ys, xs = np.mgrid[0:gh, 0:gw]
        positions = np.full((1, N, 2), -1.0, np.float32)
        positions[0, :n, 0] = xs.ravel()
        positions[0, :n, 1] = ys.ravel()

        pe = np.zeros((1, N, 768), np.float32)
        xi, yi = xs.ravel().astype(int), ys.ravel().astype(int)
        pe[0, :n] = (self.pos_table[0][xi].astype(np.float32)
                     + self.pos_table[1][yi].astype(np.float32))

        valid = np.zeros((1, N), np.float32)
        valid[0, :n] = 1.0

        pool = np.zeros((T, N), np.float32)
        th, tw = gh // POOL_K, gw // POOL_K
        for tr in range(th):
            for tc in range(tw):
                idx = [((POOL_K * tr + i) * gw + POOL_K * tc + j)
                       for i in range(POOL_K) for j in range(POOL_K)]
                pool[tr * tw + tc, idx] = 1.0 / (POOL_K * POOL_K)

        n_real_tokens = th * tw
        inputs = {
            "patches": patches.astype(self.in_dtype["patches"]),
            "positions": positions.astype(self.in_dtype["positions"]),
            "position_embeddings": pe.astype(self.in_dtype["position_embeddings"]),
            "valid": valid.astype(self.in_dtype["valid"]),
            "pool": pool.astype(self.in_dtype["pool"]),
        }
        return inputs, n_real_tokens

    # ------------------------------------------------------------ inference
    def embed(self, img: Image.Image):
        inputs, n_tok = self.preprocess(img)
        t0 = time.perf_counter()
        out = self.vision.predict(inputs)
        t_vision = time.perf_counter() - t0
        image_tokens = np.asarray(out["image_tokens"], dtype=np.float32)[0, :n_tok]  # [n, 512]

        S = self.text_seq
        seq = np.zeros((1, S, 512), np.float32)
        seq[0, 0] = self.special[0]            # <bos>
        seq[0, 1] = self.special[1]            # <|image|>
        seq[0, 2:2 + n_tok] = image_tokens
        seq[0, 2 + n_tok] = self.special[2]    # <image|>
        seq[0, 3 + n_tok] = self.special[3]    # <eos>
        mask = np.zeros((1, S), np.float32)
        mask[0, :4 + n_tok] = 1.0

        t0 = time.perf_counter()
        out = self.text.predict({
            "inputs_embeds": seq.astype(self.text_dtype["inputs_embeds"]),
            "attention_mask": mask.astype(self.text_dtype["attention_mask"]),
        })
        t_text = time.perf_counter() - t0
        emb = np.asarray(out["embedding"], dtype=np.float32).ravel()
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        return emb, t_vision, t_text


# ---------------------------------------------------------------- bench
def pct(vals, p):
    return float(np.percentile(np.asarray(vals), p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=70, choices=[70, 140, 280])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--compute-units", nargs="*", default=["ALL", "CPU_AND_GPU"])
    args = ap.parse_args()

    names = [n for g in GROUPS.values() for n in g] + UNRELATED
    images = {n: Image.open(IMG_DIR / f"{n}.jpg").convert("RGB") for n in names}
    print(f"images: {len(names)}")

    all_results = {}
    embeddings = {}
    for cu in args.compute_units:
        print(f"\n=== compute_units={cu} budget=vision_{args.budget} ===")
        model = EmbeddingGemma2CoreML(cu, args.budget)
        print(f"load: vision {model.vision_load_s:.1f}s, text {model.text_load_s:.1f}s")

        # warmup
        for _ in range(3):
            model.embed(images[names[0]])

        per_image = {}
        vt_all, tt_all, tot_all = [], [], []
        embs = {}
        for n in names:
            vt, tt = [], []
            for _ in range(args.runs):
                emb, tv, tx = model.embed(images[n])
                vt.append(tv * 1e3)
                tt.append(tx * 1e3)
            embs[n] = emb
            vt_all += vt
            tt_all += tt
            tot_all += [v + t for v, t in zip(vt, tt)]
            per_image[n] = pct([v + t for v, t in zip(vt, tt)], 50)

        stats = {
            "load_s": {"vision": model.vision_load_s, "text": model.text_load_s},
            "vision_ms": {"mean": float(np.mean(vt_all)), "p50": pct(vt_all, 50), "p95": pct(vt_all, 95)},
            "text_ms": {"mean": float(np.mean(tt_all)), "p50": pct(tt_all, 50), "p95": pct(tt_all, 95)},
            "total_ms": {"mean": float(np.mean(tot_all)), "p50": pct(tot_all, 50), "p95": pct(tot_all, 95)},
            "per_image_p50_ms": per_image,
        }
        all_results[cu] = stats
        embeddings[cu] = embs
        print(f"total pipeline ms: mean={stats['total_ms']['mean']:.1f} "
              f"p50={stats['total_ms']['p50']:.1f} p95={stats['total_ms']['p95']:.1f} "
              f"(vision p50={stats['vision_ms']['p50']:.1f}, text p50={stats['text_ms']['p50']:.1f})")

    # similarity from the first compute-unit run (embeddings should be nearly identical across CUs)
    ref_cu = args.compute_units[0]
    embs = embeddings[ref_cu]
    # cross-CU consistency check
    if len(args.compute_units) > 1:
        other = embeddings[args.compute_units[1]]
        diffs = [float(np.dot(embs[n], other[n])) for n in names]
        print(f"\ncross-CU cosine(same image): min={min(diffs):.5f} mean={float(np.mean(diffs)):.5f}")

    M = np.stack([embs[n] for n in names])
    sim = M @ M.T

    print("\ncosine similarity matrix:")
    header = "      " + " ".join(f"{n[:6]:>7}" for n in names)
    print(header)
    for i, n in enumerate(names):
        print(f"{n[:6]:>6}" + " ".join(f"{sim[i, j]:7.3f}" for j in range(len(names))))

    # group stats
    def group_pairs(members):
        return [(members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))]

    intra = []
    for members in GROUPS.values():
        for a, b in group_pairs(members):
            intra.append(float(np.dot(embs[a], embs[b])))
    bases = [g[0] for g in GROUPS.values()] + UNRELATED
    inter = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            inter.append(float(np.dot(embs[bases[i]], embs[bases[j]])))
    cross = [float(np.dot(embs[a], embs[b]))
             for g in GROUPS.values() for a in g[1:] for b in UNRELATED]

    print(f"\nintra-group (similar pairs, n={len(intra)}): "
          f"min={min(intra):.4f} mean={float(np.mean(intra)):.4f} max={max(intra):.4f}")
    print(f"inter-group (base vs base/unrelated, n={len(inter)}): "
          f"min={min(inter):.4f} mean={float(np.mean(inter)):.4f} max={max(inter):.4f}")
    print(f"variant-vs-unrelated (n={len(cross)}): "
          f"min={min(cross):.4f} mean={float(np.mean(cross)):.4f} max={max(cross):.4f}")
    margin = min(intra) - max(inter + cross)
    print(f"separation margin (min intra - max inter): {margin:.4f}")

    report = {
        "budget": args.budget,
        "runs": args.runs,
        "timing": all_results,
        "similarity": {
            "intra": {"min": min(intra), "mean": float(np.mean(intra)), "max": max(intra)},
            "inter": {"min": min(inter), "mean": float(np.mean(inter)), "max": max(inter)},
            "variant_vs_unrelated": {"min": min(cross), "mean": float(np.mean(cross)), "max": max(cross)},
            "margin": margin,
        },
        "matrix": {"names": names, "sim": sim.tolist()},
    }
    out_path = ROOT / f"bench_results_b{args.budget}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\njson written: {out_path}")


if __name__ == "__main__":
    main()
