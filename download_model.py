"""Download the EmbeddingGemma 2 Core ML files needed for the image pipeline.

Full image embedding pipeline requires:
  - EmbeddingGemma2Vision.mlpackage  (vision encoder, fp16, ~292MB)
  - EmbeddingGemma2Text.mlpackage    (text tower -> final 768-d embedding, ~271MB)
  - embeddings.bf16                  (token table for <bos>/<eos>/<|image|>/<image|>, ~256MB)
  - position_embeddings.f16          (vision patch position tables, ~3MB)
  - config.json                      (shapes, token ids)
Audio package and tokenizer.json are skipped (not needed for image similarity).
"""
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="FluidInference/embeddinggemma-2-coreml",
    local_dir="models/embeddinggemma-2-coreml",
    allow_patterns=[
        "config.json",
        "embeddings.bf16",
        "position_embeddings.f16",
        "EmbeddingGemma2Vision.mlpackage/*",
        "EmbeddingGemma2Text.mlpackage/*",
    ],
)
print("downloaded to:", path)
