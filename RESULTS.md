# EmbeddingGemma 2 Core ML — iOS 相册清理（相似照片检测）可行性验证

- 日期：2026-10-10
- 机器：Apple M3 Pro, macOS 26.5, Python 3.12.13 (uv venv), coremltools 9.0
- 模型：[google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2)（740M 多模态 embedding，768 维输出，Apache 2.0）
- Core ML 端口：[FluidInference/embeddinggemma-2-coreml](https://huggingface.co/FluidInference/embeddinggemma-2-coreml)（fp16，已修复 fp16 RMSNorm 溢出问题）

## 结论（先说结果）

**可行。** 首选路径（FluidInference 端口）一次成功，未触发任何 fallback。在 M3 Pro 上 vision_70 配置单张全链路 54.3ms（ANE 调度），相似对 cosine ≥ 0.957，无关对 ≤ 0.885，存在 0.071 的分离间隔（vision_140 时间隔扩大到 0.128）。阈值取 ~0.93 可在测试集上完全分离相似/无关组。推算 iPhone A17 Pro 端侧约 50–200ms/张，适合后台批量索引。

## 模型文件与加载方式

下载的文件（`snapshot_download`，`allow_patterns` 避开 audio 塔和 tokenizer）：

| 文件 | 大小 | 作用 |
|---|---|---|
| `EmbeddingGemma2Vision.mlpackage` | 292 MB | 视觉编码器（Gemma 4 ViT 16 层 + 3×3 pooling + 投影），fp16，multifunction：`vision_70` / `vision_140` / `vision_280` |
| `EmbeddingGemma2Text.mlpackage` | 271 MB | 文本塔，fp16，`embed_32…512` 七个固定形状函数，把 image tokens 转成最终 768 维 embedding |
| `embeddings.bf16` | 256 MB | token embedding 表（262144×512 bf16），只需查 `<bos>=2`、`<eos>=1`、`<\|image\|>=255999`、`<image\|>=258882` 四行（bf16→fp32：uint16 左移 16 位）× √512 |
| `position_embeddings.f16` | 3 MB | 视觉 patch 位置表 [2,1024,768] fp16，`table[0][x]+table[1][y]` |
| `config.json` | 3 KB | 形状、token id、预处理参数 |

**加载**：`coremltools.models.MLModel(path, compute_units=…, function_name="vision_70")`。
macOS 26.5 上 .mlpackage 直接加载运行，无需编译 .mlmodelc（首次加载会做一次 ANE 编译并缓存：ALL 模式 vision 13.2s / text 18.8s；CPU_AND_GPU 模式 2.8s / 7.1s）。注意 **multifunction 模型的输入/输出 spec 在 `spec.description.functions[i]` 下**，顶层 description.input 为空。

**重要架构事实**：视觉编码器输出的是 `image_tokens [1, T, 512]`（T=70/140/280），**不是最终 embedding**。完整链路必须再过文本塔：`<bos> <|image|> [image tokens] <image|> <eos>` → `embed_S`（选能容纳 T+4 的最小 S）→ 768 维 L2 归一化向量。只做 vision 塔的相似度不是该模型的官方用法。

## 预处理管线（按 config.json / Gemma4ImageProcessor 复刻）

1. 保持长宽比缩放：边长取 48px 的倍数，patch（16px）总数 ≤ 9×budget（vision_70 → ≤630 patch）。800×600 → 480×336（30×21 patch）。
2. RGB / 255 归一化到 [0,1]（模型内部再做 2x−1）。
3. 行优先切 patch，每 patch 按 [16 行][16 列][RGB] 展平成 768 维；pad 到 630。
4. 输入五个张量（全 fp16）：`patches [1,630,768]`、`positions [1,630,2]`（x,y，padding=-1）、`position_embeddings [1,630,768]`、`valid [1,630]`、`pool [70,630]`（每个 3×3 patch 组权重 1/9）。

## 测试集（14 张，`images/`）

- **A 组**（应相似）：`a1_base` 800×600 + `a2_resize75`（0.75×缩放）+ `a3_jpeg35`（JPEG q=35 重压缩）
- **B 组**：`b1_base` + `b2_crop80`（中心裁剪 80% 再放大）+ `b3_burst_shift_exp`（平移 12px + 亮度 ×1.15，模拟连拍）
- **C 组**：`c1_base` + `c2_jpeg25`（q=25 重压缩）+ `c3_resize50`（0.5× 缩放）
- **无关**：`d1…d5`（picsum 随机图）

## 耗时（warm up 3 次后每图 10 次，仅推理耗时，不含预处理）

| budget | compute_units | vision p50 | text p50 | 全链路 mean / p50 / p95 |
|---|---|---|---|---|
| 70 | **ALL** | 47.0 ms | 7.3 ms | **54.3 / 54.3 / 54.5 ms** |
| 70 | CPU_AND_GPU | 63.9 ms | 16.6 ms | 80.6 / 80.5 / 80.8 ms |
| 140 | ALL | 135.2 ms | 29.3 ms | 165.0 / 164.5 / 168.8 ms |
| 140 | CPU_AND_GPU | 134.6 ms | 29.1 ms | 163.8 / 163.7 / 164.1 ms |
| 280 | ALL | 1321.3 ms | 51.7 ms | 1354.4 / 1375.1 / 1483.0 ms |
| 280 | CPU_AND_GPU | 320.3 ms | 54.9 ms | 375.2 / 375.2 / 376.1 ms |

- 固定形状函数 → 耗时与图片内容无关，每图完全一致（p95 与 mean 几乎重合），这对端侧耗时预估非常友好。
- M3 Pro 上 vision_70 时 ALL（ANE）比 GPU 快；vision_280 时 GPU 反超 ANE 3.6 倍（全注意力随 patch 数平方增长）。与端口 README 的 M5 Pro 数据趋势一致（M5 Pro GPU vision_70 为 14.6ms，M3 Pro GPU 64ms，GPU 代差明显；ANE 侧 M5 Pro 37ms vs M3 Pro 47ms）。
- ALL 与 CPU_AND_GPU 产出的 embedding 高度一致：同图跨 compute-unit cosine min 0.99974 / mean 0.99991，fp16 无 NaN、无溢出迹象（输出模长 1.0000）。

## 相似度区分度（vision_70，全量矩阵见 `bench_results_b70.json`）

| 指标 | vision_70 | vision_140 | vision_280 |
|---|---|---|---|
| 组内（9 对相似）min / mean / max | **0.9568** / 0.9771 / 0.9966 | **0.9708** / 0.9830 / 0.9945 | **0.9197** / 0.9629 / 0.9977 |
| 组间（28 对无关）min / mean / max | 0.6213 / 0.7108 / **0.8854** | 0.6114 / 0.6966 / **0.8433** | 0.5723 / 0.6859 / **0.8399** |
| 变体 vs 无关（30 对）max | 0.8481 | 0.8416 | 0.8259 |
| **分离间隔（min 组内 − max 组间）** | **+0.0714** | **+0.1275** | **+0.0798** |

关键配对（vision_70）：a1↔a2_resize 0.997、a1↔a3_jpeg 0.994、b1↔b2_crop 0.959、b1↔b3_burst 0.978、c1↔c2_jpeg 0.980、c1↔c3_resize 0.972。

**注意：无关对的绝对值（0.57–0.89）比直觉高**，这不是模型缺陷：
1. embedding 模型的 cosine 分布本身偏正（向量集中在锥体内），0.6–0.7 是"无关"的常见基线；
2. 抽查最高的两对确认是真实语义相似——`d4↔d5`（0.885）分别是金门大桥仰拍和曼哈顿大桥（同为红色钢构悬索桥+蓝天），`b1↔d2`（0.843）同为茂密针叶林场景。

**工程含义**：相似照片检测的阈值不能按"0.85"拍脑袋，应取 ~0.93（本测试集 vision_70 下 0.93 可完全分离）；若要求更大安全边际可用 vision_140（间隔 0.128，代价是 3 倍耗时）。对"连拍/重复"场景（组内均 ≥0.92）绰绰有余；对"同场景不同照片"（如多次旅行拍同一座桥），语义 embedding 会给出 0.8+ 的分数——这对相册清理反而是 feature（可聚类"同主题照片"），但阈值设计要区分"近重复"（>0.93）与"同主题"（0.75–0.9）两档。

## 判定标准逐条对照

| 标准 | 结果 | 判定 |
|---|---|---|
| (a) 模型能加载跑通 | .mlpackage 直接加载，14 图 × 3 budget × 2 compute-unit 全部跑通，无 NaN | ✅ |
| (b) 相似对 >0.85、无关对 <0.6、有清晰间隔 | 组内 min 0.957 ✅；无关对 max 0.885 不满足 <0.6 ❌；但存在 +0.071 ~ +0.128 的正间隔，阈值 0.93 可完全分离 | ⚠️ 部分满足（区分度明确，但绝对阈值需上移到 ~0.93） |
| (c) M3 Pro 单张 <50ms | vision_70 ALL 为 54.3ms，略超 50ms；CPU_AND_GPU 80.6ms | ⚠️ 基本满足（同一量级；vision_70+ANE 接近达标） |

## 对 iPhone 端侧的推断

- **质量**：embedding 由模型权重决定，与设备无关；fp16 端口与 PyTorch fp32 参考 cosine ≥ 0.997（端口 README），跨 compute-unit 一致性 0.9999，iOS 上可复现本测试的相似度结论。
- **速度**：M3 Pro ANE（约 18 TOPS）vision_70 全链路 54ms。A17 Pro ANE 标称 35 TOPS，理论不弱于 M3 Pro，预估 iPhone 15 Pro 级设备 50–200ms/张（取保守 2–4 倍放大也在此区间）。按 100ms/张估算，1000 张照片全量索引约 100 秒，适合充电+后台任务场景；768 维 fp16 向量每张照片仅 1.5KB（Matryoshka 截断到 256 维则 0.5KB，质量损失约 1 个点）。
- **集成注意**：iOS 18+（multifunction 要求）；首次加载需编译（本机 13–19s/模型，README 称全函数编译需数分钟，Core ML 会缓存）；视觉塔之外必须同时打包文本塔（embed_128 即可覆盖 vision_70），总体积 ~820MB（vision 292MB + text 271MB + 表 259MB），按需下载或截断 embedding 表可压缩。

## Fallback 记录

未触发。首选路径（FluidInference .mlpackage 直接加载）一次成功，macOS 26.5 / coremltools 9.0 无兼容问题。备选路径（② 其他端口 ③ onnx 自转）未需要。

## 复现

```bash
uv venv --python python3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python download_model.py     # ~822MB
.venv/bin/python gen_images.py         # 14 张测试图
.venv/bin/python bench_embedding.py --runs 10              # vision_70, ALL + CPU_AND_GPU
.venv/bin/python bench_embedding.py --budget 140 --runs 10 # 可选
.venv/bin/python bench_mrl.py                              # MRL 截断验证
```

原始输出：`bench_output.txt`、`bench_results_b{70,140,280}.json`、`bench_mrl_output.txt`、`bench_results_mrl.json`。

## MRL 截断验证（768 → 256 / 128 维）

方法：复用 vision_70 的 768 维 embedding，取前 N 维 + 重新 L2 归一化（官方要求的用法），重算全量矩阵。脚本 `bench_mrl.py`，结果 `bench_results_mrl.json`，未改动原有 `bench_results_b*.json`。

| 指标 | 768d | 256d | 128d |
|---|---|---|---|
| 组内 min / mean / max | 0.9568 / 0.9771 / 0.9966 | 0.9645 / 0.9807 / 0.9970 | 0.9806 / 0.9886 / 0.9981 |
| 组间 max（最像的无关对 d4↔d5 双桥） | 0.8854 | 0.9029 | 0.9351 |
| 变体 vs 无关 max | 0.8481 | 0.8672 | 0.9045 |
| **分离间隔（min 组内 − max 组间）** | **+0.0714** | **+0.0616** | **+0.0455** |
| 阈值 0.93 完全分离 | ✅ | ✅ | ❌（d4↔d5=0.9351 越过 0.93） |
| 配对分数 Kendall τ vs 768d | 1.0 | 0.929 | 0.751 |

**衰减规律**：截断不是"丢失分数"，而是把所有配对分数整体上抬（丢弃尾部维度再归一化，向量锥变窄）。组内对已经接近 1，抬升有限（+0.02 以内）；无关对抬升明显（最大的一对 a1↔d1 从 0.62 涨到 0.80，+0.18）。净效果是**间隔随截断变窄**：0.071 → 0.062 → 0.046。最弱相似对始终是 b2_crop80↔b3_burst_shift_exp，最强无关对始终是 d4↔d5，极端对身份在三档间不变。

**翻转检查**：0.93 阈值下跨档分类翻转共 1 对——d4↔d5（两座红色悬索桥）在 128d 从"无关"翻成"相似"。**没有任何"相似→不相似"的翻转**，三档间隔均为正（完美分离在三档都成立，只是 128d 下阈值要上移到 ~0.95）。τ 系数显示 256d 保持了 93% 的配对排序一致性，128d 只剩 75%。

**存储与搜索**（10,000 张照片量级）：

| 维度 | fp16 每向量 | 1 万张向量库 | 暴力点积检索（numpy fp32，M3 Pro） |
|---|---|---|---|
| 768 | 1536 B | 15.4 MB | 354 µs/query |
| 256 | 512 B | 5.1 MB | 56 µs/query（6.3× 快） |
| 128 | 256 B | 2.6 MB | 48 µs/query |

检索耗时对 1 万张库本来就不是瓶颈（亚毫秒），256 维的收益主要在存储（省 2/3）和内存驻留。

**结论：生产推荐 256 维。** 间隔仍为正（+0.062）、阈值 0.93 完全可分、排序一致性 93%，体积和搜索成本降为 1/3；128 维间隔收窄到 +0.046 且出现无关对越过 0.93 阈值的翻转（与官方"128d 多模态质量明显下降"的警告一致），不建议用于图像。实际部署时 768/256 可同库双写：默认 256d 检索，对 0.90–0.95 边界对用 768d 复核，成本几乎为零（同一次推理，只是存两份切片）。
