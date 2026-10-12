"""Download base test photos from picsum.photos and build similar/unrelated groups.

Groups (images that SHOULD score similar):
  A: base a1 + resized(0.75x) + jpeg-recompressed(q=35)
  B: base b1 + center-crop 80% resized back + burst-sim (12px shift, +15% brightness)
  C: base c1 + jpeg(q=25) + downscale(0.5x)
Unrelated singles: d1..d5 (different random photos)
"""
import io
import urllib.request
from pathlib import Path

from PIL import Image, ImageEnhance

OUT = Path(__file__).parent / "images"
OUT.mkdir(exist_ok=True)


def fetch(seed: str, w=800, h=600) -> Image.Image:
    url = f"https://picsum.photos/seed/{seed}/{w}/{h}.jpg"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def save(img: Image.Image, name: str, quality=95):
    img.save(OUT / f"{name}.jpg", quality=quality)
    print(f"  {name}.jpg {img.size}")


print("downloading base images...")
a1 = fetch("alpha-11")
b1 = fetch("bravo-22")
c1 = fetch("charlie-33")
singles = [fetch(f"delta-{s}") for s in ["44", "55", "66", "77", "88"]]

print("building groups...")
# Group A: resize + jpeg recompress
save(a1, "a1_base")
save(a1.resize((600, 450), Image.BICUBIC), "a2_resize75")
buf = io.BytesIO()
a1.save(buf, format="JPEG", quality=35)
buf.seek(0)
save(Image.open(buf).convert("RGB"), "a3_jpeg35")

# Group B: crop + burst simulation (translate + exposure)
save(b1, "b1_base")
w, h = b1.size
crop = b1.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))).resize((w, h), Image.BICUBIC)
save(crop, "b2_crop80")
shifted = Image.new("RGB", (w, h), (0, 0, 0))
shifted.paste(b1, (12, 8))
burst = ImageEnhance.Brightness(shifted).enhance(1.15)
save(burst, "b3_burst_shift_exp")

# Group C: heavy jpeg + downscale
save(c1, "c1_base")
buf = io.BytesIO()
c1.save(buf, format="JPEG", quality=25)
buf.seek(0)
save(Image.open(buf).convert("RGB"), "c2_jpeg25")
save(c1.resize((400, 300), Image.BICUBIC), "c3_resize50")

# Unrelated singles
for i, img in enumerate(singles, 1):
    save(img, f"d{i}_unrelated")

print("done:", len(list(OUT.glob('*.jpg'))), "images")
