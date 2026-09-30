"""《看什么看》 · What Are You Looking At
2026-09-30 · silkscreen print (Python: six plates drawn in code, pulled four times)

nerolette sent a photo of Wanda, the lilac-and-white British Shorthair: upper lids pressed flat, head
tilted, the corners of the mouth turned down. The whole face says "what are you looking at?"
nerolette said it made her laugh.

So Wanda got Warhol's 1967 *Marilyn* recipe: one key plate (lids, slit pupils, nose, mouth, halftone
shadows) plus flat colour blocks, and one extra plate for the pink of the nose and inner ears. Six
plates. The photo was soft-focus and lit from the window, too mushy to posterize, so every shape was
traced by hand against it (the points are in wanda_shapes.py) and every plate is drawn here in code.
The plates are drawn once and printed with four ink sets. Anything can go wild except the eyes: they
stay amber in every colourway. The stare is the constant.
Another Claude cut the plates and pulled the prints that night; set A is the one that was kept.

— Claude, for nerolette

Engine: a friend's silkscreen skill (press.py, brush.py, inks.py and its riso ink card; see ENGINES.md;
not included here). Set SILKSCREEN_ENGINE_DIR to its folder (or edit ENGINE_DIR below).
Needs numpy, scipy, Pillow. Keep wanda_shapes.py next to this file.
    python3 wanda-silkscreen-0930.py A     -> panels/A0-tangerine.png … A3-night.png (+ a plate strip each)
    python3 wanda-silkscreen-0930-sheet.py wanda-silkscreen-0930.png \\
        panels/A0-tangerine.png panels/A1-mint.png panels/A2-hotpink.png panels/A3-night.png
Sets B and C are the two colourways that were not chosen.
The reference photo (wanda_reference.jpg) is not included, and nothing here reads it.

The brief, written before the first plate:

想说什么：旺达那一眼。「你看什么看」。上眼皮压得平平的，头往一边歪，嘴角往下撇一点。
  换四套颜色她还是同一个眼神——颜色怎么闹，她都不为所动。所以眼睛永远是那一路暖黄/琥珀，
  其他的全可以离谱：橙底、薄荷底、荧光粉底、夜蓝底，灰帽子可以是紫的、蓝的、青的。
颜色：每张一个底色情绪。白毛最亮（或者干脆留纸白），灰帽中间，眼睛比灰帽至少亮 15，骨架最深。
版数：六版。沃霍尔 1967 的玛丽莲是「一版照片 + 四版色块」，这里多一版给鼻头和耳朵里的粉。
  1 底色  平涂，整块，挖掉猫（往里少挖一圈，错开时叠出一线深色）
  2 白毛  嘴套、下巴、鼻梁那道倒 V、胸口；有的配色直接留纸白，这版就空着
  3 灰帽  耳朵、脑门、两腮；眼睛挖空
  4 粉    鼻头、耳朵里
  5 眼睛  琥珀那一路，比挖的洞大一圈
  6 骨架  黑。眼皮、竖瞳、鼻孔、人中和往下撇的嘴、胡子根；阴影出网点（照片版的那种网点）
深浅：纸 > 白毛 > 眼睛 > 灰帽 ≈ 底色（有两张故意拉平让它颤）> 骨架。
性格：粗放型，沃霍尔那路。色块跟骨架错开 5–12px，边上露一线纸白或叠出一线深色；骨架自己套得紧。
叠还是盖：平涂挨着印，色块互相挖开，错位处透明相乘出第三色的一线。
纸白：胡子挖成纸白，从嘴边甩出去横穿底色；眉毛上那几根长胡子往上戳。
点睛：眼皮那一笔用毛笔，带一点飞白——整张画最要紧的地方就是那条眼皮线。

形是对着照片描的（描图纸在 wanda_shapes.py），照片虚焦又逆光，没法直接拿来做照片版，
所以骨架版也是手画的。
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.environ.get("SILKSCREEN_ENGINE_DIR", "./silkscreen")  # set to where the engine lives
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, HERE)

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from press import Press, WHITE
from brush import Pens
import wanda_shapes as W

P = 1040                       # 一格的成品像素
VX, VY, VS = 60, 40, 960       # 取景：描图纸上这块方（裁紧一点，脸撑满，像玛丽莲那样）
K = P / VS                     # 描图纸 → 成品像素


def sc(pts):
    return [((x - VX) * K, (y - VY) * K) for x, y in pts]


def closed_smooth(pts, per=8):
    Pn = np.asarray(pts, float)
    Pn = np.vstack([Pn[-1], Pn, Pn[0], Pn[1]])
    out = []
    for i in range(1, len(Pn) - 2):
        p0, p1, p2, p3 = Pn[i - 1], Pn[i], Pn[i + 1], Pn[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return out


def shape(p, pts, smooth=True, sharp=()):
    """描图纸上的多边形 → 版像素 0/1。smooth 就过一遍圆滑（耳尖、V 尖这种保留尖角的，走 sharp 直接连）。"""
    pts = sc(pts)
    if smooth:
        pts = closed_smooth(pts)
    im = p.canvas()
    ImageDraw.Draw(im).polygon([(x * p.S, y * p.S) for x, y in pts], fill=255)
    return p.arr(im)


def grow(p, a, px):
    """往外长 px 成品像素（负数就往里缩）。"""
    k = int(abs(px) * 2 * p.S) | 1
    if px > 0:
        return ndimage.grey_dilation(a, size=(k, k))
    return ndimage.grey_erosion(a, size=(k, k))


def soft(p, a, px):
    return ndimage.gaussian_filter(a, px * p.S)


def make_plates(seed=7):
    """画一套版（一次），之后每套墨都用这同一套。"""
    p = Press(P, P, seed=seed)
    pen = Pens(p)
    sil = shape(p, W.SIL)
    # 耳尖要尖：耳朵那一段不圆滑，单独再盖一个尖的
    ear_tip_l = shape(p, [(126, 130), (140, 66), (172, 92), (214, 146), (180, 200), (130, 210)], smooth=False)
    ear_tip_r = shape(p, [(770, 316), (838, 330), (912, 346), (890, 392), (864, 432), (800, 400)], smooth=False)
    sil = np.maximum(sil, np.maximum(ear_tip_l, ear_tip_r))
    white = shape(p, W.WHITE) * sil
    v_tip = shape(p, [(662, 478), (690, 560), (636, 560)], smooth=False)       # 倒 V 的尖
    white = np.maximum(white, v_tip * sil)
    eyes = np.maximum(shape(p, W.EYE_L), shape(p, W.EYE_R))
    ears = np.maximum(shape(p, W.EAR_L), shape(p, W.EAR_R))
    nose = shape(p, W.NOSE)

    # 纸白的胡子：一根根手画（粗细、长短、弯度都拉开，别像扫帚）
    wh = np.zeros((p.h, p.w), np.float32)
    rng = np.random.default_rng(seed + 1)
    for line in W.WHISKERS_L + W.WHISKERS_R:
        pen.brush(sc(line), width=rng.uniform(3.0, 4.4), dry=0.0, taper=(0.02, 0.55), wobble=0.4, bristles=3, acc=wh)
    wh = np.clip(wh, 0, 1)
    brow_w = np.zeros((p.h, p.w), np.float32)
    for line in W.BROW_WHISKERS:
        pen.brush(sc(line), width=rng.uniform(2.6, 3.4), dry=0.0, taper=(0.02, 0.7), wobble=0.4, bristles=3, acc=brow_w)
    wh = np.maximum(wh, np.clip(brow_w, 0, 1) * (1 - grow(p, sil, 2)))    # 眉毛上的长胡子只在头外面看得见

    plates = {}
    # 1 底色：整块，挖掉猫（往里少挖 4px，错开时和灰帽叠出一线深色）
    plates["ground"] = np.clip(1 - grow(p, sil, -4), 0, 1)
    # 2 白毛：往灰帽底下多钻一圈
    plates["fur"] = np.clip(grow(p, white, 4) * sil, 0, 1)
    # 3 灰帽：挖掉白、眼、耳朵里
    coat = np.clip(sil - white, 0, 1)
    coat = coat * (1 - eyes) * (1 - grow(p, ears, -3))
    plates["coat"] = coat
    # 4 粉：鼻头和耳朵里
    plates["pink"] = np.clip(np.maximum(grow(p, nose, 2), ears), 0, 1)
    # 5 眼睛：比洞大一圈
    plates["eyes"] = grow(p, eyes, 3)

    # 6 骨架 —— 实线 1.0，阴影出网点（0.2–0.55）
    key = np.zeros((p.h, p.w), np.float32)
    # 阴影：眼窝一圈，上头（眉骨压下来）和内眼角重一点
    ring = np.clip(soft(p, grow(p, eyes, 16), 8) - eyes, 0, 1)
    brow = shape(p, [(400, 470), (560, 470), (600, 520), (560, 520), (420, 505)]) + \
           shape(p, [(720, 540), (840, 552), (850, 590), (740, 560)])
    tone = 0.36 * ring + 0.30 * soft(p, np.clip(brow, 0, 1), 10)
    # 右半边背光：沿右脸边一条影子；下巴底下脖子一片影子
    right = shape(p, [(868, 430), (912, 540), (912, 690), (878, 760), (834, 740), (858, 640), (852, 520)])
    # 下巴底下一道月牙影：头和胸口靠它分开
    jaw = shape(p, [(470, 870), (560, 896), (650, 906), (760, 890), (850, 850), (858, 874), (780, 924), (650, 948), (540, 934), (464, 892)])
    tone = tone + 0.30 * soft(p, right, 14) + 0.19 * soft(p, jaw, 12)
    # 左腮那块深一点的灰（照片里眼睛左下）
    cheek = shape(p, [(200, 640), (330, 640), (420, 700), (400, 760), (300, 800), (200, 760)])
    tone = tone + 0.16 * soft(p, cheek, 26)
    # 耳朵里下半截是暗的
    ear_dark = shape(p, [(200, 350), (290, 330), (296, 400), (256, 432), (214, 424)])
    tone = tone + 0.42 * soft(p, ear_dark, 8)
    # 下巴底下一小块影
    chin = shape(p, [(604, 850), (704, 848), (716, 870), (660, 884), (606, 872)])
    tone = tone + 0.35 * soft(p, chin, 6)
    # 脑门上几道淡淡的虎斑（网点里的道子）
    fur_marks = np.zeros((p.h, p.w), np.float32)
    # 从两眼中间往上散开，长短粗细都不一样（一样的会像条形码）
    for a, b, wd in [((566, 420), (556, 300), 20), ((526, 424), (500, 318), 15), ((606, 428), (628, 336), 14),
                     ((490, 440), (452, 372), 11)]:
        pen.brush(sc([a, ((a[0] + b[0]) / 2 + rng.uniform(-5, 5), (a[1] + b[1]) / 2), b]),
                  width=wd, dry=0.0, taper=(0.2, 0.5), wobble=0.3, acc=fur_marks)
    tone = tone + 0.22 * soft(p, fur_marks, 5) * (1 - white)
    key = np.clip(tone, 0, 0.6)

    lines = np.zeros((p.h, p.w), np.float32)
    # 上眼皮：整张画最要紧的一笔。毛笔，平平地压下来，带一点飞白
    pen.brush(sc([(402, 524), (440, 508), (484, 504), (524, 507), (552, 518), (560, 540)]),
              width=21, dry=0.2, taper=(0.14, 0.28), wobble=0.3, press=[0.6, 1, 1, 0.95, 0.6], acc=lines)
    pen.brush(sc([(726, 584), (760, 569), (796, 568), (824, 577), (840, 596)]),
              width=16, dry=0.2, taper=(0.14, 0.3), wobble=0.3, press=[0.6, 1, 1, 0.8], acc=lines)
    # 下眼皮：细、断开，让纸白透进来
    pen.pen(sc([(420, 548), (432, 572), (456, 591), (486, 599)]), width=4.5, acc=lines)
    pen.pen(sc([(506, 596), (530, 584), (548, 562)]), width=4.0, acc=lines)
    pen.pen(sc([(742, 612), (760, 632), (784, 640)]), width=3.8, acc=lines)
    pen.pen(sc([(800, 636), (822, 620)]), width=3.4, acc=lines)
    # 内眼角往鼻子那边拖一小道
    pen.brush(sc([(554, 542), (564, 566), (568, 584)]), width=6, dry=0.1, taper=(0.1, 0.9), wobble=0.2, acc=lines)
    pen.brush(sc([(732, 588), (724, 612), (722, 636)]), width=5, dry=0.1, taper=(0.1, 0.8), wobble=0.2, acc=lines)
    # 竖瞳：细缝，上头被眼皮吃掉
    pen.brush(sc(W.PUPIL_L), width=13, dry=0.0, taper=(0.05, 0.45), wobble=0.1, bristles=8, acc=lines)
    pen.brush(sc(W.PUPIL_R), width=10, dry=0.0, taper=(0.05, 0.45), wobble=0.1, bristles=8, acc=lines)
    # 鼻孔、鼻头下沿、人中、往下撇的嘴
    for (x, y), r in [(W.NOSTRIL_L, 9), (W.NOSTRIL_R, 8)]:
        (cx, cy), = sc([(x, y)])
        pen.blob(cx, cy, r, rough=0.18, acc=lines)
    pen.pen(sc([(650, 744), (664, 762), (672, 772), (682, 760), (694, 746)]), width=5, acc=lines)
    pen.brush(sc(W.PHILTRUM), width=6, dry=0.0, taper=(0.05, 0.1), wobble=0.2, acc=lines)
    pen.brush(sc(W.MOUTH_L), width=6, dry=0.1, taper=(0.05, 0.6), wobble=0.3, acc=lines)
    pen.brush(sc(W.MOUTH_R), width=6, dry=0.1, taper=(0.05, 0.6), wobble=0.3, acc=lines)
    # 胡子根的小点
    for (cx, cy) in sc(W.PADS_L + W.PADS_R):
        pen.blob(cx, cy, rng.uniform(3.4, 4.8), rough=0.2, acc=lines)
    # 轮廓只在背光那边和耳朵里沿出现，断断续续
    pen.brush(sc([(172, 96), (214, 148), (262, 200), (306, 240), (340, 262)]), width=5, dry=0.3, taper=(0.1, 0.4), wobble=0.4, acc=lines)
    pen.brush(sc([(890, 482), (908, 552), (910, 626)]), width=5, dry=0.3, taper=(0.2, 0.45), wobble=0.4, acc=lines)
    key = np.maximum(key, np.clip(lines, 0, 1))
    # 眼睛里的一点高光：纸白（骨架和眼睛都挖掉），眼神才活
    hl = np.zeros((p.h, p.w), np.float32)
    for (x, y), r in [((466, 530), 6.5), ((770, 590), 4.8)]:
        (cx, cy), = sc([(x, y)])
        pen.blob(cx, cy, r, rough=0.15, acc=hl)
    key = key * (1 - hl)
    plates["key"] = key

    # 胡子挖成纸白：穿过所有版
    for n in plates:
        plates[n] = plates[n] * (1 - wh)
    plates["eyes"] = plates["eyes"] * (1 - hl)
    return plates


# 一套版，多套墨。rgb 用色卡上的名字。fur=None 就留纸白。
import inks as INKS
C = INKS.INK

SETS = {
    "A": [
        dict(name="tangerine", ground=C["Orange"], fur=None, coat=C["Dark Mauve"], pink=C["Fluorescent Pink"], eyes=C["Sunflower"], key=C["Black"]),
        dict(name="mint", ground=C["Mint"], fur=C["Bisque"], coat=C["Violet"], pink=C["Bubble Gum"], eyes=C["Sunflower"], key=C["Black"]),
        dict(name="hotpink", ground=C["Fluorescent Pink"], fur=C["Yellow"], coat=C["Medium Blue"], pink=C["Fluorescent Red"], eyes=C["Sunflower"], key=C["Black"]),
        dict(name="night", ground=C["RisoFederal Blue"], fur=None, coat=C["Aqua"], pink=C["Bubble Gum"], eyes=C["Orange"], key=C["Black"]),
    ],
    # 深帽子：灰帽压成深色，眼睛成了整张最亮的地方；套色放得更开
    "B": [
        dict(name="grape", ground=C["Fluorescent Yellow"], fur=None, coat=C["Grape"], pink=C["Fluorescent Pink"], eyes=C["Sunflower"], key=C["Black"]),
        dict(name="teal", ground=C["Fluorescent Orange"], fur=C["Mist"], coat=C["Teal"], pink=C["Bubble Gum"], eyes=C["Sunflower"], key=C["Black"]),
        dict(name="burgundy", ground=C["Aqua"], fur=C["Bisque"], coat=C["Burgundy"], pink=C["Fluorescent Red"], eyes=C["Melon"], key=C["Black"]),
        dict(name="purple", ground=C["Light Lime"], fur=None, coat=C["Purple"], pink=C["Fluorescent Pink"], eyes=C["Orange"], key=C["Black"]),
    ],
    # 旺达不变，世界在变：猫永远是自己调的丁香灰 + 白 + 琥珀，只换底色（沃霍尔的《花》那种）
    "C": [
        dict(name="lilac-orange", ground=C["Orange"], fur=None, coat=(176, 160, 176), pink=C["Bubble Gum"], eyes=C["Sunflower"], key=C["Black"]),
        dict(name="lilac-turq", ground=C["Turquoise"], fur=None, coat=(176, 160, 176), pink=C["Bubble Gum"], eyes=C["Sunflower"], key=C["Black"]),
        dict(name="lilac-pink", ground=C["Fluorescent Pink"], fur=None, coat=(176, 160, 176), pink=C["Bubble Gum"], eyes=C["Sunflower"], key=C["Black"]),
        dict(name="lilac-yellow", ground=C["Yellow"], fur=None, coat=(176, 160, 176), pink=C["Bubble Gum"], eyes=C["Sunflower"], key=C["Black"]),
    ],
}
BOLD = {"A": 1.3, "B": 1.5, "C": 1.0}


def print_panel(plates, cw, seed, out, bold=1.0):
    """把同一套版用一套墨印一次。bold 放大套色错位。"""
    p = Press(P, P, seed=seed)
    r = np.random.default_rng(seed)

    def sh(m):
        a = r.uniform(0, 2 * np.pi)
        d = r.uniform(0.55, 1.0) * m * bold
        return (round(float(np.cos(a) * d), 1), round(float(np.sin(a) * d), 1))

    p.ink("ground", cw["ground"], angle=15, shift=sh(9), skips=0.25, starve=0.03, edge=0.9)
    if cw["fur"] is not None:
        p.ink("fur", cw["fur"], angle=75, shift=sh(7), skips=0.5, edge=0.9)
    p.ink("coat", cw["coat"], angle=0, shift=sh(9), skips=0.25, starve=0.025, edge=0.9)
    p.ink("pink", cw["pink"], angle=60, shift=sh(5), skips=0.3, edge=0.8)
    p.ink("eyes", cw["eyes"], angle=30, shift=sh(5), skips=0.2, edge=0.7)
    p.ink("key", cw["key"], angle=45, cell=8, shift=sh(2), skips=0.4, edge=0.45, opacity=0.95)
    for n in p.inks:
        p.plates[n][:] = plates[n]
    return p.run(out, strip=True)


if __name__ == "__main__":
    import time
    which = sys.argv[1] if len(sys.argv) > 1 else "A"
    only = int(sys.argv[2]) if len(sys.argv) > 2 else None
    t = time.time()
    plates = make_plates()
    print(f"plates {time.time() - t:.1f}s")
    os.makedirs("panels", exist_ok=True)
    for i, cw in enumerate(SETS[which]):
        if only is not None and i != only:
            continue
        t = time.time()
        print_panel(plates, cw, seed=100 + i, bold=BOLD[which], out=os.path.join("panels", f"{which}{i}-{cw['name']}"))
        print(cw["name"], f"{time.time() - t:.1f}s")
