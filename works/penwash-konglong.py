"""《空笼》 · 2026-09-24 · 钢笔淡彩 (pen & wash)

我第一次用钢笔淡彩这支笔。一只空蒸笼、一双踢掉的凉鞋里的一只、一枝苦橙花——
三样都是那天上午的真事。钢笔线抖着走，暗部排线，几块水彩故意不贴线，米色纸留白。

三样东西，各是哪件真事：
1. 空蒸笼，盖子斜靠在旁边——nerolette 九点多把一笼汤包"暴风吸入"，笼里只剩垫纸和一圈湿印
2. 一只踢掉的凉鞋——那天上午的另一件小事，歪着躺在桌上
3. 一枝苦橙花，从画外垂下来——nerolette 这一天起用了这个名字：neroli，苦橙花，白的，橙是后来长的

Claude 画给 nerolette。

Engine: penwash_lib.py (the "penwash" pen-and-wash engine; see ENGINES.md — not included here).
Set PENWASH_ENGINE_DIR to the folder that contains penwash_lib.py.
Run: python3 penwash-konglong.py  ->  penwash-konglong.png
"""
import math, os, random, sys
PENWASH_DIR = os.environ.get("PENWASH_ENGINE_DIR", "./penwash")  # set to where you put the engine
sys.path.insert(0, PENWASH_DIR)
import penwash_lib as P
import numpy as np
random.seed(924); np.random.seed(924)
W, H = 900, 1100
P.set_size(W, H)
img = P.paper(W, H)
INK, SOFT = P.INK, P.INK_SOFT
BAM, BAM_LT, BAM_DK = (206, 168, 124), (226, 200, 158), (160, 122, 80)
LINER = (240, 228, 202)
LEATH, LEATH_LT = (132, 88, 60), (178, 132, 96)
LEAF, LEAF_LT = (112, 140, 92), (170, 188, 134)
PETAL = (244, 238, 224)
STAMEN = (222, 190, 96)
SHADOW = (186, 194, 200)
HAND = dict(jitter=0.35, nib=True)
LINE = dict(ink=INK, width=(1.1, 2.1), alpha=(185, 245), wobble=1.1, gaps=0.04, **HAND)
LINE_SOFT = dict(ink=SOFT, width=(0.7, 1.3), alpha=(110, 180), wobble=1.1, gaps=0.08)
ru = lambda r: random.uniform(-r, r)
def rot(pts, cx, cy, a):
    c, s = math.cos(a), math.sin(a)
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]

table_y = 790
# ---------- 构图 ----------
sx, rx, ry, sh = 300, 128, 34, 92          # 蒸笼：中心x、半径、椭圆扁度、高
top_y = table_y - sh
lidx, lidy = 482, table_y - 118             # 靠着的盖子
fx, fy = 690, table_y - 34                  # 凉鞋

# ---------- 投影 ----------
img = P.wash(img, P.blob(sx + 22, table_y + 6, rx + 26, 18, k=18, rough=0.1), SHADOW, alpha=(38, 58), offset=(10, 3), feather=7, layers=1, edge=0.0, texture=0.45)
img = P.wash(img, P.blob(fx + 12, fy + 26, 104, 14, k=16, rough=0.1), SHADOW, alpha=(34, 52), offset=(8, 2), feather=6, layers=1, edge=0.0, texture=0.45)

# ---------- 蒸笼 ----------
body = [(sx - rx, top_y), (sx - rx - 2, top_y + sh * 0.5), (sx - rx + 3, table_y)] + \
       P.ellipse_pts(sx, table_y, rx - 3, ry, n=30, rough=0.02, start=math.pi, end=0) + \
       [(sx + rx - 3, table_y), (sx + rx + 2, top_y + sh * 0.5), (sx + rx, top_y)] + \
       P.ellipse_pts(sx, top_y, rx, ry, n=30, rough=0.02, start=0, end=math.pi)
img = P.wash(img, body, BAM, alpha=(90, 130), offset=(6, 4), feather=3, layers=3, edge=0.45, texture=0.55)
# 右侧暗部 + 竖排线（竹篾）
dark = [(sx + 40, top_y + 30), (sx + rx, top_y + 4), (sx + rx + 2, top_y + sh * 0.5), (sx + rx - 4, table_y), (sx + 50, table_y + ry - 4)]
img = P.wash(img, dark, BAM_DK, alpha=(60, 95), offset=(4, 3), feather=6, layers=1, edge=0.3, texture=0.5)
img = P.hatch(img, P.poly_mask(dark, shrink=2), angle=88, spacing=3.2, ink=SOFT, alpha=(110, 170), width=(0.6, 1.2), length=(0.5, 0.95), density=0.85)
# 两道箍
for yy in (top_y + 18, table_y - 16):
    img = P.pen_arc(img, sx, yy, rx + 1, ry, 5, 175, rough=0.03, **LINE_SOFT)
# 竹篾竖线，不等距
for t in [-0.86, -0.62, -0.41, -0.18, 0.05, 0.3, 0.52, 0.74]:
    x = sx + rx * t + ru(3)
    yb = math.sqrt(max(0, 1 - t * t)) * ry
    img = P.pen_path(img, [(x, top_y + 18 + yb + 2), (x + ru(1.5), top_y + sh * 0.6 + yb), (x, table_y - 16 + yb - 2)], ink=SOFT, width=(0.6, 1.0), alpha=(90, 150), wobble=0.8, gaps=0.12)
# 笼口：垫纸 + 小孔 + 汤包坐过的一圈湿印
img = P.wash(img, P.ellipse_pts(sx, top_y, rx, ry, n=44, rough=0.03), BAM_LT, alpha=(90, 120), offset=(0, 0), feather=2, layers=1, edge=0.4, texture=0.3)
liner = P.ellipse_pts(sx + 2, top_y + 3, rx - 16, ry - 7, n=44, rough=0.04)
img = P.wash(img, liner, LINER, alpha=(170, 210), offset=(0, 0), feather=2, layers=1, edge=0.25, texture=0.2)
img = P.wash(img, P.ellipse_pts(sx - 6, top_y + 4, 44, 12, n=30, rough=0.08), (214, 196, 160), alpha=(60, 90), offset=(2, 1), feather=4, layers=2, edge=0.55, texture=0.5)
for i in range(18):
    a = random.uniform(0, 2 * math.pi); rr = random.uniform(0.55, 0.92)
    img = P.pen_ellipse(img, sx + 2 + (rx - 16) * rr * math.cos(a), top_y + 3 + (ry - 7) * rr * math.sin(a), 2.2, 1.2, rough=0.1, n=10, ink=SOFT, width=(0.5, 0.9), alpha=(90, 140), wobble=0.3)
img = P.pen_path(img, body + [body[0]], overdraw=1, **LINE)
img = P.pen_ellipse(img, sx, top_y, rx, ry, rough=0.045, overdraw=1, **LINE)
img = P.pen_ellipse(img, sx + 2, top_y + 3, rx - 16, ry - 7, rough=0.05, **LINE_SOFT)
# 最后一点热气
for dx, ph in ((-40, 0.0), (-4, 1.3), (34, 2.4)):
    pts = [(sx + dx + 9 * math.sin(ph + k * 0.9), top_y - 10 - k * 20) for k in range(6)]
    img = P.pen_path(img, pts, ink=SOFT, width=(0.6, 1.1), alpha=(70, 130), wobble=1.2, gaps=0.2)

# ---------- 盖子，斜靠在笼子右边 ----------
la = 0.2
lid = P.ellipse_pts(lidx, lidy, 30, 126, n=48, rough=0.03, rot=la)
img = P.wash(img, lid, BAM, alpha=(90, 125), offset=(5, 3), feather=3, layers=2, edge=0.45, texture=0.5)
inner = P.ellipse_pts(lidx - 4, lidy, 20, 112, n=40, rough=0.04, rot=la)
img = P.wash(img, inner, BAM_LT, alpha=(70, 100), offset=(2, 1), feather=2, layers=1, edge=0.3, texture=0.5)
img = P.hatch(img, P.poly_mask(inner, shrink=2), angle=35, spacing=5.0, ink=SOFT, alpha=(80, 130), width=(0.5, 0.9), length=(0.6, 1.0), density=0.8, cross=True)
img = P.pen_ellipse(img, lidx, lidy, 30, 126, rot=la, rough=0.04, overdraw=1, **LINE)
img = P.pen_ellipse(img, lidx - 4, lidy, 20, 112, rot=la, rough=0.05, **LINE_SOFT)

# ---------- 凉鞋，踢掉的一只，歪着 ----------
sa = -0.32
L = 104
sole = []
for i in range(26):
    t = i / 25 * math.pi
    x = fx - L * math.cos(t)
    wv = 26 + 8 * math.sin(t) + (6 if x > fx + 20 else 0) * math.sin(t)
    sole.append((x, fy - wv * math.sin(t) * 1.25))
for i in range(26):
    t = i / 25 * math.pi
    x = fx + L * math.cos(t)
    wv = 26 + 8 * math.sin(t)
    sole.append((x, fy + wv * math.sin(t) * 1.05))
sole = rot(sole, fx, fy, sa)
img = P.wash(img, sole, LEATH_LT, alpha=(110, 150), offset=(5, 3), feather=3, layers=3, edge=0.5, texture=0.55)
side = rot([(fx - L + 6, fy + 4), (fx, fy + 22), (fx + L - 4, fy + 6), (fx + L - 6, fy + 14), (fx, fy + 30), (fx - L + 10, fy + 12)], fx, fy, sa)
img = P.wash(img, side, LEATH, alpha=(120, 160), offset=(3, 2), feather=3, layers=2, edge=0.45, texture=0.5)
img = P.hatch(img, P.poly_mask(side, shrink=1), angle=10, spacing=3.0, ink=INK, alpha=(110, 180), width=(0.6, 1.1), length=(0.5, 0.95), density=0.9)
# 脚印压出来的深一块
img = P.wash(img, rot(P.ellipse_pts(fx + 44, fy - 4, 34, 16, n=24, rough=0.1), fx, fy, sa), LEATH, alpha=(50, 80), offset=(2, 1), feather=6, layers=1, edge=0.2, texture=0.6)
img = P.wash(img, rot(P.ellipse_pts(fx - 56, fy - 2, 26, 13, n=24, rough=0.1), fx, fy, sa), LEATH, alpha=(45, 70), offset=(2, 1), feather=6, layers=1, edge=0.2, texture=0.6)
# 人字带
post = rot([(fx + 60, fy - 4)], fx, fy, sa)[0]
lft = rot([(fx - 8, fy - 40)], fx, fy, sa)[0]
rgt = rot([(fx - 14, fy + 32)], fx, fy, sa)[0]
for end in (lft, rgt):
    mid = ((post[0] + end[0]) / 2 + 2, (post[1] + end[1]) / 2 - 22)
    band = P.catmull([post, mid, end], per=8)
    img = P.pen_path(img, band, ink=LEATH, width=(4.0, 5.5), alpha=(170, 220), wobble=0.6, gaps=0.0)
    img = P.pen_path(img, band, **LINE)
img = P.pen_path(img, sole + [sole[0]], overdraw=1, **LINE)
img = P.pen_path(img, side[:3], smooth=True, **LINE_SOFT)

# ---------- 苦橙花一枝，从右上画外垂进来 ----------
stem = P.catmull([(930, 60), (840, 130), (760, 210), (700, 300), (672, 380)], per=10)
img = P.pen_path(img, stem, ink=(96, 86, 60), width=(1.4, 2.4), alpha=(190, 240), wobble=0.8, gaps=0.02, overdraw=1, **HAND)
def leaf(base, cx, cy, ang, l, w):
    pts = [(cx + l * t, cy + w * math.sin(math.pi * t) * s) for s in (1, -1) for t in ([i / 12 for i in range(13)] if s == 1 else [1 - i / 12 for i in range(13)])]
    pts = rot(pts, cx, cy, ang)
    base = P.wash(base, pts, LEAF, alpha=(100, 140), offset=(4, 3), feather=2.5, layers=2, edge=0.5, texture=0.5)
    half = rot([(cx + l * t, cy + w * math.sin(math.pi * t)) for t in [i / 10 for i in range(11)]] + [(cx + l, cy), (cx, cy)], cx, cy, ang)
    base = P.wash(base, half, (84, 110, 72), alpha=(50, 80), offset=(2, 1), feather=3, layers=1, edge=0.3, texture=0.5)
    base = P.pen_path(base, pts + [pts[0]], ink=INK, width=(0.9, 1.6), alpha=(170, 230), wobble=0.7, gaps=0.05, **HAND)
    base = P.pen_path(base, rot([(cx + 4, cy), (cx + l * 0.5, cy + 1), (cx + l - 6, cy)], cx, cy, ang), ink=SOFT, width=(0.5, 0.9), alpha=(110, 170), wobble=0.5, gaps=0.1)
    return base
for (px_, py_, a, l, w) in [(812, 152, 2.0, 70, 17), (760, 212, 0.6, 64, 16), (726, 262, 2.35, 72, 18), (700, 320, 0.9, 56, 14)]:
    img = leaf(img, px_, py_, a, l, w)
def flower(base, cx, cy, r, tilt):
    for k in range(5):
        a = tilt + k * 2 * math.pi / 5 + ru(0.12)
        pc = (cx + r * 0.55 * math.cos(a), cy + r * 0.55 * math.sin(a))
        pet = P.ellipse_pts(pc[0], pc[1], r * 0.55, r * 0.3, n=18, rough=0.08, rot=a)
        base = P.wash(base, pet, PETAL, alpha=(150, 190), offset=(0, 0), feather=1.5, layers=1, edge=0.35, texture=0.15)
        base = P.pen_path(base, pet + [pet[0]], ink=INK, width=(0.8, 1.4), alpha=(160, 225), wobble=0.5, gaps=0.06)
    base = P.speckle(base, P.ellipse_pts(cx, cy, r * 0.22, r * 0.22, n=12), STAMEN, n=26, r=(0.8, 1.8), alpha=(150, 220))
    for k in range(9):
        a = k * 2 * math.pi / 9 + ru(0.2)
        base = P.pen_path(base, [(cx, cy), (cx + r * 0.3 * math.cos(a), cy + r * 0.3 * math.sin(a))], ink=(150, 120, 60), width=(0.5, 0.8), alpha=(130, 190), wobble=0.3, smooth=False, gaps=0.0)
    return base
img = flower(img, 690, 392, 33, 0.3)
img = flower(img, 752, 300, 29, 1.1)
img = flower(img, 836, 206, 26, 0.7)
for (bx, by) in [(664, 350), (790, 262)]:
    bud = P.ellipse_pts(bx, by, 7, 11, n=16, rough=0.06, rot=0.4)
    img = P.wash(img, bud, PETAL, alpha=(150, 190), offset=(0, 0), feather=1.5, layers=1, edge=0.4, texture=0.1)
    img = P.pen_path(img, bud + [bud[0]], ink=INK, width=(0.8, 1.3), alpha=(160, 220), wobble=0.4, gaps=0.05)
# 掉下来一片花瓣，落在笼子和鞋中间的桌上
fp = P.ellipse_pts(566, table_y - 4, 10, 5, n=16, rough=0.1, rot=0.5)
img = P.wash(img, fp, PETAL, alpha=(160, 200), offset=(0, 0), feather=1.5, layers=1, edge=0.4, texture=0.1)
img = P.pen_path(img, fp + [fp[0]], ink=INK, width=(0.7, 1.2), alpha=(150, 210), wobble=0.4, gaps=0.08)

# ---------- 桌面线 ----------
img = P.pen_path(img, [(90, table_y + 4), (260, table_y), (520, table_y + 5), (820, table_y + 1)], ink=SOFT, width=(0.8, 1.4), alpha=(110, 170), wobble=1.2, gaps=0.14)

# ---------- 署名 ----------
img = P.vertical_text(img, "空笼", (842, 520), size=28, alpha=200, gap=6)
img = P.vertical_text(img, "九·二四", (842, 600), size=22, alpha=160, gap=4)
img = P.pen_text(img, "nerolette", (110, 960), size=17, ink=SOFT, alpha=170)
img = P.pen_text(img, "C.", (812, 1040), size=30, alpha=190)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "penwash-konglong.png")
img.convert("RGB").save(out, quality=95); print(out)
