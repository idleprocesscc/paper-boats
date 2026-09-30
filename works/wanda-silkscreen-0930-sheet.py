"""把四张印好的格子排到一张纸上：白边、格子之间一道窄缝、铅笔签在下沿（用 press.sign，跟 skill 自己签名一样）。
The sheet for 《看什么看》: four pulls from wanda-silkscreen-0930.py on one 2400×2500 sheet, 20px gutters,
pencil signature underneath. The engine's signature font signs it. — Claude, for nerolette

Engine: a friend's silkscreen skill (see ENGINES.md; not included). Set SILKSCREEN_ENGINE_DIR.
    python3 wanda-silkscreen-0930-sheet.py out.png a.png b.png c.png d.png [--title 《看什么看》]
"""
import sys, os
ENGINE_DIR = os.environ.get("SILKSCREEN_ENGINE_DIR", "./silkscreen")  # set to where the engine lives
sys.path.insert(0, ENGINE_DIR)
import numpy as np
from PIL import Image
from press import Press, WHITE

SW, SH = 2400, 2500
PANEL = 1040
GUT = 20
MX = (SW - 2 * PANEL - GUT) // 2      # 150
MY = 150


def build(out, panels, title="《看什么看》", edition="1/1", name="Claude", year="2026", small=None):
    p = Press(SW, SH, S=1, seed=11)
    base = np.ones((SH, SW, 3), np.float32) * p.paper
    grain = 1 + p.noise(0.6) * 0.007 + p.noise(3, aniso=(0.3, 4)) * 0.003
    base = np.clip(base * grain[..., None], 0, 1)
    for i, f in enumerate(panels):
        im = np.asarray(Image.open(f).convert("RGB"), np.float32) / 255
        r, c = divmod(i, 2)
        x = MX + c * (PANEL + GUT)
        y = MY + r * (PANEL + GUT)
        base[y:y + PANEL, x:x + PANEL] = im
    bx0, bx1 = MX, SW - MX
    by1 = MY + 2 * PANEL + GUT
    base = p.sign(base, y=by1 + 30, x0=bx0, x1=bx1, edition=edition, title=title, name=name, year=year, size=40)
    img = Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))
    img.save(out)
    if small:
        k = 1200 / max(img.size)
        img.resize((round(img.size[0] * k), round(img.size[1] * k)), Image.LANCZOS).save(small)
    return img


if __name__ == "__main__":
    args = sys.argv[1:]
    title = "《看什么看》"
    if "--title" in args:
        i = args.index("--title"); title = args[i + 1]; del args[i:i + 2]
    build(args[0], args[1:5], title=title)
