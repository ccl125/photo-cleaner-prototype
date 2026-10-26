"""One-shot export of iOS-bundle assets for the EmbeddingGemma 2 pipeline.

Outputs (into ../photo-cleaner-ios/PhotoCleaner/Resources/):
  - special_tokens.f16: 4 x 512 Float16 (little-endian), order = [BOS, BOI, EOI, EOS],
    values = embeddings.bf16 row * sqrt(512), pre-scaled exactly as bench_embedding.py does.
  - position_embeddings.f16: raw copy of the vision position table [2, 1024, 768] Float16.
"""
import shutil
from pathlib import Path

import numpy as np

SRC = Path(__file__).parent / "models" / "embeddinggemma-2-coreml"
DST = Path(__file__).parent.parent / "photo-cleaner-ios" / "PhotoCleaner" / "Resources"
DST.mkdir(parents=True, exist_ok=True)

BOS, EOS, BOI, EOI = 2, 1, 255999, 258882
EMBED_SCALE = 22.627417  # sqrt(512)

table = np.memmap(SRC / "embeddings.bf16", dtype=np.uint16, mode="r").reshape(-1, 512)
rows = (np.asarray(table[[BOS, BOI, EOI, EOS]], dtype=np.uint32) << 16).view(np.float32)
rows16 = (rows * EMBED_SCALE).astype(np.float16)
(DST / "special_tokens.f16").write_bytes(rows16.tobytes(order="C"))
print("special_tokens.f16:", rows16.shape, rows16.dtype, "->", DST / "special_tokens.f16")

shutil.copyfile(SRC / "position_embeddings.f16", DST / "position_embeddings.f16")
print("position_embeddings.f16 copied [2, 1024, 768] float16")
