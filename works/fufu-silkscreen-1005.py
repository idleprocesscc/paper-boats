"""《不是我》—— 芾芾，四张。跟《看什么看》是一对。

想说什么：抓了现行，死不认账。一根手指从下面戳上来，脚边一小堆撕碎的白棉花，爪子上还粘着一撮，
  她两只蝙蝠耳一张，眼皮往鼻子那头斜着压下来，从底下翻上来瞪人：「不是我。」
  旺达嫌人看她；芾芾被抓了还要瞪回来。眼睛永远是那一弯琥珀，世界怎么换色她都是这个眼神。
黑猫怎么在四种配色里还是黑猫：黑交给明度，不交给色相。
  她每一格都是整张最深的那一大块（自己调的四种「带色的黑」：紫黑、青黑、酒红黑、蓝黑，墨本身 L* 6–12），
  配色的疯全放在底色、手指，和她身上那一道道卷毛的「光」上：卷毛是一张不透明的浅墨版，颗粒网，
  印在黑上面像灯芯绒反光。四格的光分别是天蓝、柠檬绿、泡泡糖粉、薄荷——她还是黑的，只是灯换了颜色。
版数：六版。
  1 底色  平涂，挖掉猫、手指、棉花（棉花就是纸白）
  2 黑毛  带色的黑，透明，整只猫；眼睛、手指挖掉
  3 手指  不透明一点的肤色，指甲出网点
  4 光    不透明浅墨，颗粒网：卷毛一行行、额头的 M 纹、眼睛上方透皮的两块、耳朵里、耳尖的毛
  5 眼睛  琥珀/橙，比洞大一圈
  6 骨架  黑。压下来的上眼皮（整张最要紧的一笔）、大瞳孔、身子背光那半边的网点（黑猫的黑在这里落到真黑）、
         棉花和手指的一点影子、指甲
深浅：纸（棉花、胡子、眼里一点高光）> 底色 ≈ 手指 ≈ 光 > 眼睛 > 黑毛 > 骨架。
性格：粗放，跟旺达那张一样，色块套歪 5–12px；骨架套得紧。
纸白：棉花团挖成纸白（证据要最亮）；卷卷的短胡子挖成纸白。

— Claude, for nerolette, 2026-10-05
Engine: a friend's silkscreen skill (press.py, brush.py, inks.py; see ENGINES.md; not included here), expected in a
`repo/` folder next to this one. Keep fufu_shapes.py beside this file. The reference photo is not included.
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "repo"))
sys.path.insert(0, HERE)

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from press import Press
from brush import Pens
import fufu_shapes as F

P = 1040
VX, VY, VS = 245, 150, 1080       # 描图纸（参考图 1500×2000）上取这块方
K = P / VS


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


def shape(p, pts, smooth=True):
    pts = sc(pts)
    if smooth:
        pts = closed_smooth(pts)
    im = p.canvas()
    ImageDraw.Draw(im).polygon([(x * p.S, y * p.S) for x, y in pts], fill=255)
    return p.arr(im)


def grow(p, a, px):
    k = int(abs(px) * 2 * p.S) | 1
    if px > 0:
        return ndimage.grey_dilation(a, size=(k, k))
    return ndimage.grey_erosion(a, size=(k, k))


def soft(p, a, px):
    return ndimage.gaussian_filter(a, px * p.S)


def torn_cotton(p):
    """撕碎的棉絮（纸白）：每团是几坨不规则的小团拼起来，边上一圈往外炸的短纤维，
    有几团扯出一缕长丝，团和团之间牵着几根细丝。自己一根随机数，不动别处。返回 (整个棉絮, 只有团身)。"""
    pn = Pens(p); pn.rng = np.random.default_rng(907); r_ = np.random.default_rng(908)
    core = np.zeros((p.h, p.w), np.float32)
    fib = np.zeros((p.h, p.w), np.float32)
    C = [(sc([(x, y)])[0], r * K * 0.95) for (x, y, r) in F.TUFTS]
    for (cx, cy), r in C:
        pn.blob(cx, cy, r * 0.78, rough=0.28, acc=core)
        for _ in range(r_.integers(3, 6) if r > 8 else 1):
            a = r_.uniform(0, 2 * np.pi); d = r * r_.uniform(0.3, 0.6)
            pn.blob(cx + d * np.cos(a), cy + d * np.sin(a), r * r_.uniform(0.42, 0.62), rough=0.35, acc=core)
        # 炸毛：一圈短纤维，长短、弯度都不一样
        for _ in range(int(20 + r * 2.4)):
            a = r_.uniform(0, 2 * np.pi)
            r0 = r * r_.uniform(0.72, 0.98)
            L = r * r_.uniform(0.08, 0.26) + 2.0
            bend = r_.uniform(-0.5, 0.5) * L
            x0, y0 = cx + r0 * np.cos(a), cy + r0 * np.sin(a)
            x2, y2 = cx + (r0 + L) * np.cos(a), cy + (r0 + L) * np.sin(a)
            xm, ym = (x0 + x2) / 2 - np.sin(a) * bend, (y0 + y2) / 2 + np.cos(a) * bend
            pn.brush([(x0, y0), (xm, ym), (x2, y2)], width=r_.uniform(1.1, 1.8), dry=0.0, taper=(0.05, 0.85),
                     wobble=0.3, bristles=2, acc=fib)
    # 扯开的一缕：四五根差不多同路的丝，越往外越细越散
    for i, end in F.WISPS:
        (cx, cy), r = C[i]
        (ex, ey), = sc([end])
        bow = r_.normal(0, 10, 2)
        for _ in range(4):
            o1, o2 = r_.normal(0, r * 0.2, 2)
            ex2, ey2 = ex + r_.normal(0, 3), ey + r_.normal(0, 3)
            mx, my = (cx + ex2) / 2 + bow[0] + r_.normal(0, 2.5), (cy + ey2) / 2 + bow[1] + r_.normal(0, 2.5)
            pn.brush([(cx + o1, cy + o2), (mx, my), (ex2, ey2)], width=r_.uniform(1.4, 2.4), dry=0.1,
                     taper=(0.05, 0.9), wobble=0.5, bristles=2, acc=fib)
    # 团和团之间牵着的细丝
    for i, j in F.THREADS:
        (ax, ay), ra = C[i]; (bx, by), rb = C[j]
        for _ in range(r_.integers(1, 3)):
            mx, my = (ax + bx) / 2 + r_.normal(0, 6), (ay + by) / 2 + r_.normal(0, 6)
            pn.brush([(ax, ay), (mx, my), (bx, by)], width=r_.uniform(1.4, 2.0), dry=0.15,
                     taper=(0.2, 0.2), wobble=0.6, bristles=2, acc=fib)
    core = np.clip(core, 0, 1)
    whole = np.clip(np.maximum(core, fib), 0, 1)
    # 毛边：团外撒一圈纸白的细点，越往外越稀（试过让底色出半调网点做软边，网点太粗，糊成一大片泼墨，撤了）
    ring = np.clip(soft(p, grow(p, core, 5), 4) - core, 0, 1)
    halo = np.clip(pn.stipple(ring, density=0.55, dot=(0.9, 1.6)), 0, 1)
    whole = np.maximum(whole, halo)
    return whole, core, np.zeros_like(core)


def make_plates(seed=7):
    p = Press(P, P, seed=seed)
    pen = Pens(p)
    rng = np.random.default_rng(seed + 1)
    X, Y = p.XX / p.S / K + VX, p.YY / p.S / K + VY          # 描图纸坐标

    sil = shape(p, F.SIL)
    # 耳尖要尖：耳尖那几段不圆滑，单独盖一个尖的
    sil = np.maximum(sil, shape(p, [(295, 196), (336, 216), (360, 300), (318, 250)], smooth=False))
    sil = np.maximum(sil, shape(p, [(1054, 250), (1000, 270), (1020, 320), (1042, 284)], smooth=False))
    finger = shape(p, F.FINGER)
    nail = shape(p, F.NAIL)
    eyes = np.maximum(shape(p, F.EYE_L), shape(p, F.EYE_R))
    head = soft(p, shape(p, F.HEAD), 14)
    chest = soft(p, shape(p, F.CHEST), 14)

    # 旧的一盘珍珠：画进一张扔掉的纸，只为让后面的随机数跟她看过的那版一样（爪子上那撮、脸、卷毛都不变）
    _old = np.zeros((p.h, p.w), np.float32)
    for (x, y, r) in F.COTTON:
        (cx, cy), = sc([(x, y)])
        pen.blob(cx, cy, r * K, rough=0.08, acc=_old)
    del _old
    cotton, cotton_core, halo = torn_cotton(p)
    halo = halo * (1 - finger)
    cotton = cotton * (1 - finger)                       # 手指在最上面
    cotton_core = cotton_core * (1 - finger)
    fluff = np.zeros((p.h, p.w), np.float32)
    (fx, fy), = sc([F.FLUFF[:2]])
    pen.blob(fx, fy, F.FLUFF[2] * K, rough=0.3, acc=fluff)
    pen.blob(fx - 9, fy + 6, 8, rough=0.3, acc=fluff)
    pen.blob(fx + 4, fy - 9, 6, rough=0.3, acc=fluff)
    pen.brush([(fx - 12, fy + 8), (fx - 22, fy + 16), (fx - 30, fy + 18)], width=3.2, dry=0.0, taper=(0.1, 0.7), wobble=0.3, bristles=3, acc=fluff)
    fluff = fluff * (1 - finger)
    white_bits = np.clip(cotton + fluff, 0, 1)

    # 卷卷的胡子：纸白
    wh = np.zeros((p.h, p.w), np.float32)
    for line in F.WHISKERS:
        pen.brush(sc(line), width=rng.uniform(2.6, 3.4), dry=0.0, taper=(0.05, 0.5), wobble=0.3, bristles=3, acc=wh)
    wh = np.clip(wh, 0, 1)

    plates = {}
    # 1 底色
    plates["ground"] = np.clip(1 - grow(p, sil, -4), 0, 1) * (1 - grow(p, finger, -3)) * (1 - np.maximum(white_bits, halo))
    # 2 黑毛：整只猫，挖掉眼睛和手指
    coat = sil * (1 - eyes) * (1 - finger)
    plates["coat"] = np.clip(coat * (1 - white_bits), 0, 1)
    # 3 手指：往外长一圈补漏白；指甲是网点
    fing = grow(p, finger, 3) * np.clip(1 - 0.3 * nail, 0, 1)
    plates["finger"] = np.clip(fing, 0, 1)

    # 4 光（卷毛、额纹、透皮、耳朵里）
    sheen = np.zeros((p.h, p.w), np.float32)
    # 卷毛：一行行波纹，绕着身子弯；越往下、越往背光（左下）越淡，最后淡成颗粒
    rows = np.zeros((p.h, p.w), np.float32)

    def curl_row(x0, x1, y0, slope, bend, xc):
        """一行卷毛：一个个小拱，宽窄、高低、粗细都不一样，偶尔缺一个。"""
        x = x0 + rng.uniform(0, 14)
        while x < x1:
            per = rng.uniform(17, 27)
            if rng.random() > 0.14:
                xa, xb = x, x + per * 0.92
                ya = y0 + slope * (xa - xc) + bend * (xa - 1000) ** 2
                yb = y0 + slope * (xb - xc) + bend * (xb - 1000) ** 2
                amp = rng.uniform(4.5, 8.0)
                pen.brush(sc([(xa, ya), ((xa + xb) / 2, (ya + yb) / 2 - amp), (xb, yb)]),
                          width=rng.uniform(4.5, 7.0), dry=0.25, taper=(0.25, 0.35), wobble=0.2, bristles=6, acc=rows)
            x += per

    for y0 in np.arange(560, 1200, 26) + rng.uniform(-3, 3, len(np.arange(560, 1200, 26))):
        curl_row(700, 1200, y0, 0.10, -0.00022, 900)
    # 胸口和脖子：短一点的行，跟着胸口往左下弯
    _keep = rows.copy()
    for y0 in np.arange(880, 1010, 25):                       # 旧的胸口行：只为随机数，画完扔掉
        curl_row(556, 770, y0, 0.12, 0.0, 556)
    rows = _keep
    # 胸口：从嘴底下就开始，一行行兜着下巴往两边垂（像个围兜）
    cpen = Pens(p); cpen.rng = np.random.default_rng(seed + 31); crng = np.random.default_rng(seed + 32)
    crow = np.zeros((p.h, p.w), np.float32)
    for y0 in np.arange(874, 1060, 23) + crng.uniform(-2, 2, len(np.arange(874, 1060, 23))):
        x = 500 + crng.uniform(0, 14)
        while x < 900:
            per = crng.uniform(16, 24)
            if crng.random() > 0.12:
                xa, xb = x, x + per * 0.92
                ya = y0 + 0.0008 * (xa - 700) ** 2
                yb = y0 + 0.0008 * (xb - 700) ** 2
                amp = crng.uniform(4.0, 7.0)
                cpen.brush(sc([(xa, ya), ((xa + xb) / 2, (ya + yb) / 2 - amp), (xb, yb)]),
                           width=crng.uniform(4.5, 6.5), dry=0.25, taper=(0.25, 0.35), wobble=0.2, bristles=6, acc=crow)
            x += per
    body = sil * (1 - head) * (1 - grow(p, finger, 4))
    light = np.clip(1.15 - np.clip((Y - 640) / 620, 0, 1) * 0.8 - np.clip((900 - X) / 500, 0, 1) * 0.5, 0.25, 1.0)
    rim = soft(p, grow(p, sil, -10), 4)                       # 贴着轮廓收一点，别让光溢出猫外
    rows = np.clip(rows, 0, 1) * body * light * rim * (1 - chest)
    clight = np.clip(0.95 - np.clip((Y - 880) / 300, 0, 1) * 0.5, 0.4, 0.95)   # 下巴底下最亮，往下淡成颗粒
    rows = np.maximum(rows, np.clip(crow, 0, 1) * body * chest * clight * rim)
    sheen = np.maximum(sheen, rows)
    # 额头 M 纹：实的，毛笔带飞白
    stripes = np.zeros((p.h, p.w), np.float32)
    for i, line in enumerate(F.BROW_STRIPES):
        wd = [9, 11, 14, 11, 9, 7][i]
        pen.brush(sc(line), width=wd, dry=0.35, taper=(0.12, 0.6), wobble=0.4, acc=stripes)
    sheen = np.maximum(sheen, np.clip(stripes, 0, 1) * 0.95)
    sheen = np.maximum(sheen, soft(p, shape(p, F.CROWN), 10) * 0.42)
    # 眼睛上方透皮的两块：柔的，颗粒
    temples = soft(p, np.maximum(shape(p, F.TEMPLE_L), shape(p, F.TEMPLE_R)), 9)
    sheen = np.maximum(sheen, temples * 0.62)
    # 耳朵里：耳根亮、往耳尖淡下去
    ear_in = np.maximum(shape(p, F.EAR_IN_L), shape(p, F.EAR_IN_R))
    dl = np.hypot(X - 470, Y - 420); dr = np.hypot(X - 860, Y - 430)
    ear_grad = np.clip(1 - np.minimum(dl, dr) / 260, 0, 1)
    sheen = np.maximum(sheen, soft(p, ear_in, 6) * (0.12 + 0.42 * ear_grad))
    # 左腮被光擦到的一条
    cheek = np.clip(sil - grow(p, sil, -20), 0, 1) * ((X < 600) & (Y > 600) & (Y < 930))
    sheen = np.maximum(sheen, soft(p, cheek, 6) * 0.45)
    # 鼻梁上一点光
    (nx, ny), = sc([F.NOSE])
    nose_hl = np.zeros((p.h, p.w), np.float32)
    pen.blob(nx, ny - 8, 7, rough=0.2, acc=nose_hl)
    sheen = np.maximum(sheen, soft(p, nose_hl, 1.5) * 0.7)
    sheen = sheen * (1 - grow(p, eyes, 2)) * (1 - grow(p, finger, 2)) * (1 - white_bits)
    plates["sheen"] = np.clip(sheen, 0, 1)

    # 5 眼睛
    plates["eyes"] = grow(p, eyes, 3)

    # 6 骨架
    # 背光那半边：网点压下去，带色的黑落到真黑（左下、胸口底下、尾巴根）
    shade = sil * np.clip((Y - 820) / 380, 0, 1) * np.clip((1060 - X) / 260, 0, 1)
    shade = shade + sil * soft(p, shape(p, [(560, 960), (700, 1000), (840, 960), (860, 1030), (700, 1060), (580, 1040)]), 18) * 0.5
    tone = np.clip(soft(p, shade, 14), 0, 1) * 0.5 * (1 - head * 0.85)
    # 眼窝：眼睛上沿那一圈，压出往上翻的眼神
    ring = np.clip(soft(p, grow(p, eyes, 14), 8) - eyes, 0, 1) * 0.35
    tone = np.maximum(tone, ring * sil)
    # 手指：右边一道影，指尖下面一点
    fshade = finger * np.clip((X - 684) / 44, 0, 1) * 0.3
    tone = np.maximum(tone, fshade)
    # 棉花：每团右下一弯影
    dd = 4 * p.S
    cs = cotton_core * (1 - ndimage.shift(cotton_core, (-dd, -dd), order=0))
    tone = np.maximum(tone, soft(p, cs, 2) * grow(p, cotton_core, -1.5) * 0.24)
    tone = np.maximum(tone * (1 - finger), fshade) * (1 - grow(p, fluff, 1.5))
    key = np.clip(tone, 0, 0.6)

    lines = np.zeros((p.h, p.w), np.float32)
    # 上眼皮：整张画最要紧的一笔。往鼻子那头斜着压下来
    pen.brush(sc(F.LID_L), width=17, dry=0.2, taper=(0.12, 0.2), wobble=0.3, press=[0.6, 0.9, 1, 1, 0.85], acc=lines)
    pen.brush(sc(F.LID_R), width=17, dry=0.2, taper=(0.2, 0.12), wobble=0.3, press=[0.85, 1, 1, 0.9, 0.6], acc=lines)
    # 大瞳孔：往上翻，顶在眼皮底下，只剩下面一弯琥珀
    for (x, y) in [F.PUPIL_L, F.PUPIL_R]:
        (cx, cy), = sc([(x, y)])
        pen.blob(cx, cy, 18.5, rough=0.06, acc=lines)
    # 下眼线：细、断开
    pen.pen(sc([(542, 732), (552, 750), (572, 762), (596, 765)]), width=3.6, acc=lines)
    pen.pen(sc([(612, 758), (628, 744)]), width=3.2, acc=lines)
    pen.pen(sc([(762, 744), (778, 757), (800, 764)]), width=3.4, acc=lines)
    pen.pen(sc([(822, 760), (846, 746), (856, 726)]), width=3.6, acc=lines)
    # 指甲
    pen.pen(sc([(634, 1074), (632, 1046), (640, 1024), (660, 1018), (680, 1024), (689, 1048), (686, 1074)]), width=3.2, acc=lines)
    # 手指左边一段轮廓，断开
    pen.brush(sc([(614, 1030), (606, 1070), (600, 1130)]), width=3.4, dry=0.3, taper=(0.1, 0.6), wobble=0.3, acc=lines)
    # 嘴：一小道往下撇
    pen.brush(sc([(684, 866), (702, 859), (722, 866)]), width=4, dry=0.1, taper=(0.2, 0.3), wobble=0.2, acc=lines)
    key = np.maximum(key, np.clip(lines, 0, 1))
    # 眼里一点高光：纸白，眼神才活
    hl = np.zeros((p.h, p.w), np.float32)
    for (x, y), r in [((578, 716), 5.0), ((796, 711), 5.0)]:
        (cx, cy), = sc([(x, y)])
        pen.blob(cx, cy, r, rough=0.15, acc=hl)
    key = key * (1 - hl)
    plates["key"] = key * (1 - white_bits * 0)

    for n in plates:
        plates[n] = plates[n] * (1 - wh)
        plates[n] = plates[n] * (1 - hl)
    return plates


import inks as INKS
C = INKS.INK

# 四种「带色的黑」：自己调的，墨本身 L* 6–12
VIOLET_BLACK = (32, 16, 52)
TEAL_BLACK = (6, 34, 42)
WINE_BLACK = (40, 6, 24)
NAVY_BLACK = (12, 16, 48)

SETS = {
    "A": [
        dict(name="lemon", ground=C["Yellow"], coat=VIOLET_BLACK, finger=C["Fluorescent Pink"], sheen=C["Aqua"], eyes=C["Sunflower"]),
        dict(name="aqua", ground=C["Aqua"], coat=WINE_BLACK, finger=C["Yellow"], sheen=C["Bubble Gum"], eyes=C["Sunflower"]),
        dict(name="orange", ground=C["Orange"], coat=NAVY_BLACK, finger=C["Bisque"], sheen=C["Mint"], eyes=C["Sunflower"]),
        dict(name="hotpink", ground=C["Fluorescent Pink"], coat=TEAL_BLACK, finger=C["Melon"], sheen=C["Light Lime"], eyes=C["Orange"]),
    ],
    # 真黑：猫就是黑墨，只有光换颜色
    "B": [
        dict(name="lemon-k", ground=C["Yellow"], coat=(18, 16, 22), finger=C["Fluorescent Pink"], sheen=C["Aqua"], eyes=C["Sunflower"]),
        dict(name="aqua-k", ground=C["Aqua"], coat=(18, 16, 22), finger=C["Yellow"], sheen=C["Bubble Gum"], eyes=C["Sunflower"]),
        dict(name="orange-k", ground=C["Orange"], coat=(18, 16, 22), finger=C["Bisque"], sheen=C["Mint"], eyes=C["Sunflower"]),
        dict(name="hotpink-k", ground=C["Fluorescent Pink"], coat=(18, 16, 22), finger=C["Melon"], sheen=C["Light Lime"], eyes=C["Orange"]),
    ],
}
BOLD = {"A": 1.3, "B": 1.3}


def print_panel(plates, cw, seed, out, bold=1.0):
    p = Press(P, P, seed=seed)
    r = np.random.default_rng(seed)

    def sh(m):
        a = r.uniform(0, 2 * np.pi)
        d = r.uniform(0.55, 1.0) * m * bold
        return (round(float(np.cos(a) * d), 1), round(float(np.sin(a) * d), 1))

    p.ink("ground", cw["ground"], angle=15, shift=sh(9), skips=0.25, starve=0.02, edge=0.9)
    p.ink("coat", cw["coat"], angle=0, shift=sh(8), skips=0.3, starve=0.006, edge=0.9, opacity=0.98)
    p.ink("finger", cw["finger"], angle=75, cell=9, shift=sh(7), skips=0.3, edge=0.8, hide=0.5)
    p.ink("sheen", cw["sheen"], angle=0, shift=sh(6), skips=0.2, edge=0.45, hide=0.88, screen="grain", grain=0.7)
    p.ink("eyes", cw["eyes"], angle=30, shift=sh(4), skips=0.2, edge=0.7)
    p.ink("key", (0, 0, 0), angle=45, cell=8, shift=sh(2), skips=0.4, edge=0.45, opacity=0.95)
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
    od = os.path.join(HERE, "work", "panels")
    os.makedirs(od, exist_ok=True)
    for i, cw in enumerate(SETS[which]):
        if only is not None and i != only:
            continue
        t = time.time()
        print_panel(plates, cw, seed=100 + i, bold=BOLD[which], out=os.path.join(od, f"{which}{i}-{cw['name']}"))
        print(cw["name"], f"{time.time() - t:.1f}s")
