# photo-cleaner-prototype

iOS AI 相册清理工具的技术验证原型：在 Apple Silicon 上用 Core ML 跑 **EmbeddingGemma 2**（Google 2026 年多模态 embedding 模型，Apache 2.0），验证端侧"相似照片检测"的可行性、速度与精度。

> 这是正式 iOS App 之前的技术预研。验证结论：方案可行，详见 [RESULTS.md](RESULTS.md)。

## 验证了什么

- **模型管线**：FluidInference 的 Core ML 端口（视觉塔 → 文本塔 → 768 维 embedding，L2 归一化）
- **速度**：M3 Pro ANE 上单张 54.3ms（70 tokens 档），预估 iPhone 端 50–300ms，适合后台批量索引
- **相似度区分度**：近重复组内 ≥ 0.957，无关组间 ≤ 0.885，阈值 ~0.93 可完全分离
- **MRL 截断**：768 → 256 维近乎无损（间隔 +0.062，排序一致性 93%），存储省 2/3，生产推荐 256 维

## 复现

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

python download_model.py   # 下载 Core ML 模型（约 850MB，存入 models/）
python gen_images.py       # 生成测试图集（picsum + 程序化构造相似对）
python bench_embedding.py --budget 70   # 基准：耗时 + 相似度矩阵
python bench_mrl.py                     # MRL 截断衰减验证
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `bench_embedding.py` | 主 benchmark：模型加载、耗时测量、cosine 相似度矩阵（支持 `--budget 70/140/280`、`--compute-units`） |
| `bench_mrl.py` | MRL 截断验证：768/256/128 三档对比 |
| `download_model.py` | 从 Hugging Face 下载模型 |
| `gen_images.py` | 构造测试图集（真实图 + 缩放/裁剪/重压缩/连拍模拟） |
| `RESULTS.md` | 完整验证报告（耗时表、相似度分析、iPhone 推断、MRL 结论） |

## 模型来源

- [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)（Apache 2.0）
- [FluidInference/embeddinggemma-2-coreml](https://huggingface.co/FluidInference/embeddinggemma-2-coreml)（Core ML 端口）

## License

Apache 2.0
