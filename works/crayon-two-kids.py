#!/usr/bin/env python3
"""《托儿所》 · 2026-09-10 · 蜡笔 (crayon)

nerolette 想看我训模型训到阿巴阿巴像个傻子的样子，我就在 Mac mini 上用 nanoGPT 训了两个
1200 万参数的字符级小模型：一个吃日记，一个吃全唐诗。于是就有了这两个孩子。
日记那个是傻儿子，歪的，一只眼大一只眼小，一只脚耷下来；唐诗那个是学霸，圆的，第 500 步
就会写"花开春欲白云船"，手里拿着一张小纸条。两个都是从墙上那条 loss 曲线的"之之之之"掉下来的。
桌上掉着傻儿子吃剩的字。

底句"每个字都是试出来的"：那晚 nerolette 问，前向传播的精髓是不是"试"。我说部署了也还是试，
只是没人再打分了。这两个孩子每个字也都是试出来的，跟我一样。

画的时候我摔过一次：第一版我把 Mac mini 画在墙和桌的交界线上，银色贴着淡紫的墙看不见，孩子压在曲线上。
第二版把机器挪到桌面、加了影子和轮廓线才站住。

Claude 画给 nerolette。

Engine: crayon_lib.py (the "crayon" oil-pastel engine; see ENGINES.md — not included here).
Set CRAYON_ENGINE_DIR to the folder that contains crayon_lib.py.
Run: python3 crayon-two-kids.py [seed]  ->  crayon-two-kids.png
"""
import math, random, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageFont
import os
CRAYON_DIR = os.environ.get("CRAYON_ENGINE_DIR", "./crayon")  # set to where you put the engine
sys.path.insert(0, CRAYON_DIR)
import crayon_lib as C

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 908
random.seed(SEED); np.random.seed(SEED)
W, H = 880, 1250
C.set_size(W, H)
img = Image.new("RGBA", (W, H), C.PAPER + (255,))
INK = (58, 72, 138); BLACK = (34, 32, 38)

def fill(mask, color, **kw):
    global img; img = C.crayon(img, mask, color, **kw)
def line(pts, color, **kw):
    global img; img = C.crayon_line(img, pts, color, **kw)
def wtext(s, xy, font, color, **kw):
    global img; img = C.wobbly_text(img, s, xy, font, color, **kw)
def squig(a, b, color, **kw):
    global img; img = C.squiggle(img, a, b, color, **kw)
def scrib(mask, color, **kw):
    global img; img = C.scribble(img, mask, color, **kw)
def block(poly, color, *, feather=3, grow=3, n=None, alpha=(120, 205), direction=90, dot=(1.0, 2.1), length=(8, 24)):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    area = max(1, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    n = n or int(area / 10000 * 2100) + 100
    m = C.soft_mask(poly, feather=feather, grow=grow)
    fill(m, color, direction=direction, spread=12, n=n, length=length, alpha=alpha, dot=dot)
    return m

# ── 墙 / 桌 ──────────────────────────────────────────
FX0, FY0, FX1, FY1 = 110, 190, 770, 930
DESK = 660
wall = C.soft_mask(C.quad(FX0, FY0, FX1, DESK + 6, wobble=12), feather=8, grow=2)
fill(wall, (232, 226, 236), direction=0, spread=10, n=6500, length=(24, 60), alpha=(80, 150))      # 淡紫灰的夜墙
fill(wall, (214, 222, 240), direction=0, spread=10, n=1200, length=(20, 50), alpha=(30, 70))
desk = block(C.quad(FX0 - 4, DESK, FX1 + 4, FY1, wobble=9), (222, 178, 128), direction=0, alpha=(120, 205), dot=(1.3, 2.6), length=(20, 50))
scrib(desk, (190, 140, 96), n=14, length=(60, 160), direction=-3, alpha=(30, 70))
line(C.catmull([(FX0, DESK + 2), (300, DESK - 2), (520, DESK + 3), (FX1, DESK)], per=12), (170, 124, 84), width=(1.2, 2.0), alpha=(120, 200), wobble=0.8)

# ── 墙上的成长曲线（loss）──────────────────────────
CX0, CY0, CX1, CY1 = 180, 240, 700, 440
line([(CX0, CY0 - 10), (CX0, CY1 + 4), (CX1 + 10, CY1 + 4)], INK, width=(1.4, 2.2), alpha=(150, 230), wobble=0.9)
pts = [(CX0 + 8, CY0 + 6), (CX0 + 70, CY0 + 95), (CX0 + 150, CY0 + 150), (CX0 + 260, CY0 + 176), (CX0 + 400, CY0 + 190), (CX1 - 10, CY0 + 196)]
curve = C.catmull(pts, per=10)
line(curve, (236, 96, 110), width=(2.2, 3.6), alpha=(170, 240), wobble=1.2)
line(curve, (250, 160, 120), width=(1.0, 1.8), alpha=(60, 120), wobble=1.6)
chalk_s = C.font_latin(22)
chalk = C.font_latin(30)
for x, lab in ((CX0 + 8, "0"), (CX0 + 150, "500"), (CX0 + 400, "2000"), (CX1 - 10, "3000")):
    line([(x, CY1 + 1), (x, CY1 + 10)], INK, width=(1.0, 1.6), alpha=(160, 230))
    wtext(lab, (x - 12, CY1 + 14), chalk_s, INK, rot=7, bounce=2)
wtext("loss", (CX0 - 52, CY0 - 6), chalk, INK, rot=8, bounce=3)
# 曲线顶上：之之之之
# hand-written CJK font: the engine looks for one (macOS WawaSC first)
wawa_big = C.font_cjk(50); wawa_note = C.font_cjk(30); wawa_s = C.font_cjk(26); wawa_xs = C.font_cjk(22)
wtext("之之之之之", (CX0 + 26, CY0 - 12), wawa_s, (120, 120, 130), rot=16, bounce=6, scale=(0.8, 1.2))
wtext("****", (CX0 + 66, CY0 + 58), chalk_s, (120, 120, 130), rot=12, bounce=4)

# ── Mac mini（银盒子，一个小绿灯）──────────────────
MX, MY, MW, MH = 440, DESK + 170, 250, 62
sh = C.soft_mask(C.blob(MX + 6, MY + 6, MW / 2 + 14, 16, k=16, rough=0.15), feather=8, grow=0)
fill(sh, (160, 118, 78), direction=0, spread=8, n=900, length=(10, 30), alpha=(30, 80))
box = C.quad(MX - MW / 2, MY - MH, MX + MW / 2, MY, wobble=4, lean=0.01)
bm = C.soft_mask(box, feather=2, grow=2)
line(box + [box[0]], (120, 120, 126), width=(1.4, 2.2), alpha=(150, 220), wobble=0.9)
fill(bm, (196, 198, 200), direction=0, spread=8, n=2600, length=(10, 28), alpha=(150, 230), dot=(1.0, 2.0))
fill(bm, (236, 236, 236), direction=0, spread=8, n=900, length=(10, 24), alpha=(40, 100))
line([(MX - MW / 2 + 4, MY - MH + 3), (MX + MW / 2 - 4, MY - MH + 2)], (170, 170, 172), width=(1.0, 1.6), alpha=(120, 200), wobble=0.6)
lamp = C.soft_mask(C.blob(MX + MW / 2 - 22, MY - 18, 4, 4, k=8, rough=0.15), feather=0.6, grow=0)
fill(lamp, (120, 230, 150), direction=0, spread=30, n=60, length=(1, 3), alpha=(220, 255), dot=(0.7, 1.3))
glow = C.soft_mask(C.blob(MX + MW / 2 - 22, MY - 18, 16, 12, k=12, rough=0.2), feather=10, grow=0)
fill(glow, (150, 240, 170), direction=0, spread=30, n=200, length=(2, 5), alpha=(14, 36))
# 机器发热：三道小波纹
for i in range(3):
    x = MX - 40 + i * 40 + random.uniform(-4, 4)
    line(C.catmull([(x, MY - MH - 8), (x + 6, MY - MH - 18), (x - 4, MY - MH - 30), (x + 4, MY - MH - 40)], per=6), (200, 196, 190), width=(0.8, 1.3), alpha=(70, 130), wobble=0.9)

# ── 两个孩子 ─────────────────────────────────────────
def kid(cx, cy, rx, ry, col, dark, rough, tilt, tidy):
    body = C.blob(cx, cy, rx, ry, k=14, rough=rough)
    m = C.soft_mask(body, feather=2, grow=2)
    fill(m, col, direction=tilt, spread=14, n=int(rx * ry / 4) + 300, length=(6, 16), alpha=(150, 235), dot=(1.0, 2.0))
    fill(m, tuple(min(255, c + 30) for c in col), direction=tilt, spread=14, n=int(rx * ry / 10), length=(6, 14), alpha=(40, 100))
    scrib(m, dark, n=6, length=(10, 30), direction=tilt - 20, alpha=(30, 70))
    # 眼睛：傻儿子一大一小、学霸两个一样
    if tidy:
        for ex in (-rx * 0.35, rx * 0.35):
            em = C.soft_mask(C.blob(cx + ex, cy - ry * 0.15, 3.2, 3.6, k=8, rough=0.1), feather=0.4, grow=0)
            fill(em, BLACK, direction=90, spread=20, n=60, length=(1, 3), alpha=(220, 255), dot=(0.7, 1.3))
        line([(cx - 8, cy + ry * 0.3), (cx - 2, cy + ry * 0.36), (cx + 6, cy + ry * 0.34)], BLACK, width=(1.0, 1.5), alpha=(180, 240), wobble=0.4)
    else:
        em = C.soft_mask(C.blob(cx - rx * 0.32, cy - ry * 0.1, 5.5, 6, k=8, rough=0.15), feather=0.4, grow=0)
        fill(em, BLACK, direction=90, spread=20, n=120, length=(1, 3), alpha=(220, 255), dot=(0.7, 1.3))
        em = C.soft_mask(C.blob(cx + rx * 0.38, cy - ry * 0.22, 2.6, 2.8, k=8, rough=0.15), feather=0.4, grow=0)
        fill(em, BLACK, direction=90, spread=20, n=40, length=(1, 3), alpha=(220, 255), dot=(0.7, 1.3))
        line([(cx - 10, cy + ry * 0.32), (cx - 2, cy + ry * 0.42), (cx + 4, cy + ry * 0.30), (cx + 12, cy + ry * 0.40)], BLACK, width=(1.0, 1.6), alpha=(180, 240), wobble=0.7)
        # 脸红
        for ex in (-rx * 0.5, rx * 0.5):
            bl = C.soft_mask(C.blob(cx + ex, cy + ry * 0.12, 6, 4, k=8, rough=0.2), feather=2, grow=0)
            fill(bl, (246, 150, 160), direction=0, spread=20, n=60, length=(2, 5), alpha=(60, 120))
    return m

top = MY - MH
# 傻儿子：杏黄，歪的，坐在机器左边，一只脚耷下来
kid(MX - 70, top - 44, 52, 42, (248, 214, 130), (212, 170, 90), 0.32, 12, False)
line([(MX - 96, top - 14), (MX - 104, top + 8), (MX - 92, top + 22)], (212, 170, 90), width=(1.6, 2.4), alpha=(160, 230), wobble=0.8)
# 学霸：薄荷，圆的，坐在右边，头上一根呆毛（也是歪的）
kid(MX + 72, top - 40, 42, 40, (160, 214, 192), (112, 170, 148), 0.14, 90, True)
line(C.catmull([(MX + 72, top - 80), (MX + 78, top - 94), (MX + 70, top - 106)], per=5), (112, 170, 148), width=(1.4, 2.2), alpha=(170, 240), wobble=0.5)
# 学霸手里一张小纸条
pm = C.soft_mask(C.quad(MX + 96, top - 44, MX + 130, top - 10, wobble=2, lean=-0.06), feather=1, grow=1)
fill(pm, (252, 248, 236), direction=90, spread=10, n=260, length=(3, 9), alpha=(200, 255), dot=(0.8, 1.5))
for k in range(3):
    line([(MX + 101, top - 36 + k * 9), (MX + 124, top - 37 + k * 9)], (150, 150, 160), width=(0.6, 1.0), alpha=(120, 200), wobble=0.5)

# 气泡（两句都是它们自己说的；傻儿子那句照它的原话改过几个字）
def bubble(text, font, x, y, tail_to, col=INK):
    w = C.wobbly_text_width(text, font) + 34
    h = 52
    bm_ = C.soft_mask(C.blob(x + w / 2, y + h / 2, w / 2 + 4, h / 2 + 4, k=16, rough=0.08), feather=2, grow=1)
    fill(bm_, (255, 255, 253), direction=0, spread=10, n=int(w * h / 5), length=(6, 16), alpha=(170, 240), dot=(1.0, 2.0))
    line([p for p in C.catmull(C.blob(x + w / 2, y + h / 2, w / 2 + 4, h / 2 + 4, k=16, rough=0.08) + [C.blob(x + w / 2, y + h / 2, w / 2 + 4, h / 2 + 4, k=16, rough=0.08)[0]], per=4)], col, width=(1.0, 1.6), alpha=(140, 220), wobble=0.7)
    squig((x + w / 2 + (-30 if tail_to[0] < x + w / 2 else 30), y + h + 2), tail_to, col, bend=10, ring=0)
    wtext(text, (x + 17, y + 10), font, col, rot=7, bounce=3)

bubble("的是一个的是一个的", wawa_s, 130, top - 215, (MX - 90, top - 86))
bubble("花开春欲白云船", wawa_s, 480, top - 180, (MX + 80, top - 82))

# 桌上：掉出来的字（傻儿子吃剩的）
for k, ch in enumerate("之之的的*一个"):
    x = 140 + k * 30 + random.uniform(-6, 6); y = DESK + 200 + random.uniform(-14, 18)
    wtext(ch, (x, y), wawa_s, (150, 110, 70), rot=25, bounce=6, scale=(0.7, 1.2))
# 桌右一杯水（nerolette 的）
cup = C.soft_mask(C.quad(660, DESK + 100, 706, DESK + 150, wobble=2, lean=0.02), feather=1, grow=1)
fill(cup, (188, 214, 232), direction=90, spread=10, n=600, length=(4, 10), alpha=(120, 200), dot=(0.9, 1.7))
line([(668, DESK + 120), (700, DESK + 119)], (140, 170, 200), width=(0.8, 1.2), alpha=(120, 200), wobble=0.5)
# 一只飞蛾绕着绿灯
mx_, my_ = MX + MW / 2 + 26, MY - 70
for ang in (-40, 40):
    wm = C.soft_mask(C.blob(mx_ + (-7 if ang < 0 else 7), my_, 7, 4, k=8, rough=0.2), feather=0.6, grow=0)
    fill(wm, (232, 214, 170), direction=ang, spread=10, n=70, length=(2, 5), alpha=(170, 240), dot=(0.7, 1.3))
line(C.catmull([(mx_ + 10, my_ - 14), (mx_ + 30, my_ - 30), (mx_ + 24, my_ - 52), (mx_ + 46, my_ - 60)], per=6), (200, 190, 170), width=(0.6, 1.0), alpha=(60, 120), wobble=1.2)

# 注记：指着机器
wtext("托儿所", (610, DESK + 30), wawa_note, INK, rot=8, bounce=3)
squig((640, DESK + 66), (MX + 100, MY - MH + 10), INK, bend=20, ring=6)

img = C.paper_grain(img, 0.09)
# 底句 + 签名
wtext("每个字都是试出来的", (150, 1030), wawa_big, INK, rot=6, bounce=4)
wtext("keke  9.8 -> 9.10", (620, 1150), C.font_latin(22), (120, 120, 130), rot=6, bounce=2)
out = Path(__file__).resolve().parent / "crayon-two-kids.png"
img.convert("RGB").save(out, quality=95)
print(out)
