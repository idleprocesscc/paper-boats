"""《只开一天》 · Only One Day
2026-10-01 · silkscreen print (Python: five plates, borrowed engine)

nerolette spent an evening second-guessing the clay jar in our own watercolour painting of these
same poppies, then said: "same model, and it's still hard for me to work out here." So two Claudes
painted the same poppies with a different borrowed tool each, to see whether what was hard to get
right was the brush, or something else.

This one uses a borrowed silkscreen: five plates, yellow, teal, red, violet, black. Red is printed
as a halftone — solid at the petal's centre, breaking into dots toward the edge, with the yellow
light plate showing through underneath: light passing through a thin petal. Every dark in the
picture — the jar, its shadow, the flower's centre — shares one violet plate. A poppy opens for a
single day, so one petal has already fallen on the table.

— Claude, for nerolette

Engine: a friend's silkscreen skill (press.py, brush.py, inks.py and its riso ink card; see
ENGINES.md; not included here). Set SILKSCREEN_ENGINE_DIR to its folder (or edit ENGINE_DIR below).
Needs numpy, scipy, Pillow.
    python3 poppies-silk-1001.py <tag> [set ...]   -> work/<tag>-<set>.png (teal/mist/night)

The brief, written before the first plate (Chinese):

想说什么：薄。光从后窗穿过来，花瓣薄得像一张红纸，被照透了，边上透出橙黄；
  桌上已经落了一片。虞美人一朵只开一天，今天就是那一天。
颜色：墙是一面背光的青（窗在这面墙上，所以墙自己是暗的），窗洞留纸白，整张最亮。
  花是红压在黄上：中间实红，往瓣边出网点，越到边红点越稀、底下的黄透出来，像光穿过薄瓣。
  两瓣叠着的地方光要穿两层，加一层紫，红就沉成深绯。
  桌面是窗光落下来的一块暖黄；罐子是红黄叠出来的陶土色，背光那面压一层紫，成了赭褐。
版数：五版。
  1 光（黄） 颗粒网。桌上的窗光、花瓣的底、花苞和果的底、陶罐
  2 墙（青） 颗粒网。离窗越远越密；窗边疏，露纸，像光晕。花苞和果也印它（叠黄成绿）
  3 红       规则网点 15°。瓣心实底，瓣边出网；落瓣；罐子一层薄红
  4 影（紫） 透明颗粒网。一张版管所有暗处：罐子背光面、罐子的投影、叠瓣、花苞背光、墙角、桌沿
  5 骨架（黑） 网点 45°。茎和茎上的毛、瓣根的黑斑、花蕊一圈、果顶的放射纹、罐口、罐身的粗颗粒
深浅：纸（窗）> 桌上的光 > 透光的瓣边 > 瓣心红 ≈ 墙 > 叠瓣深绯 > 罐子背光 > 投影 > 黑斑。
  红和墙明度拉平，只让色相对撞（红对青），花会颤；形靠瓣边的亮和黑斑站住。
性格：沃霍尔《花》那一路往回收一点：色版错开 3–7px，骨架套紧；墙上一点干刷。
叠还是盖：全部透明叠印。
纸白：窗洞。
点睛：陶罐背光那面用蜡笔颗粒（骨架版上），粗陶的糙就在那一小块。
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.environ.get("SILKSCREEN_ENGINE_DIR", "./silkscreen")  # set to where the engine lives
sys.path.insert(0, ENGINE_DIR)
sys.path.insert(0, HERE)

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from press import Press, WHITE
from brush import Pens
import inks as INKS
C = INKS.INK

W, H = 1200, 1600
BX0, BX1, BY0, BY1 = 90, 1110, 90, 1440
TABLE_Y, FRONT_Y = 905, 1352
WIN = (150, 156, 336, 806)           # 窗洞（纸白），四面都是墙
MUNTIN = (452, 465)
MULLION = (238, 250)
JX, JTOP, JBASE = 640, 928, 1216     # 陶罐


# ---------------------------------------------------------------- 小工具
def poly(p, pts):
    im = p.canvas()
    ImageDraw.Draw(im).polygon([(x * p.S, y * p.S) for x, y in pts], fill=255)
    return p.arr(im)


def ellipse(p, cx, cy, rx, ry):
    im = p.canvas()
    S = p.S
    ImageDraw.Draw(im).ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], fill=255)
    return p.arr(im)


def rect(p, x0, y0, x1, y1):
    a = np.zeros((p.h, p.w), np.float32)
    S = p.S
    a[int(y0 * S):int(y1 * S), int(x0 * S):int(x1 * S)] = 1
    return a


def grow(p, a, px):
    k = int(abs(px) * 2 * p.S) | 1
    return ndimage.grey_dilation(a, size=(k, k)) if px > 0 else ndimage.grey_erosion(a, size=(k, k))


def soft(p, a, px):
    return ndimage.gaussian_filter(a, px * p.S)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def inner_dist(p, m, cap):
    """形里每一点离边多远（成品像素），只在包围盒里算。"""
    ys, xs = np.nonzero(m[::4, ::4] > 0.5)
    out = np.zeros_like(m)
    if len(xs) == 0:
        return out
    y0, y1 = max(0, ys.min() * 4 - 8), min(p.h, ys.max() * 4 + 12)
    x0, x1 = max(0, xs.min() * 4 - 8), min(p.w, xs.max() * 4 + 12)
    sub = m[y0:y1, x0:x1] > 0.5
    out[y0:y1, x0:x1] = np.minimum(ndimage.distance_transform_edt(sub) / p.S, cap)
    return out


def catmull(pts, per=14):
    P = np.asarray(pts, float)
    P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


# ---------------------------------------------------------------- 一朵开着的罂粟
def poppy(p, pen, rng, cx, cy, R, tilt, rot, cup, a0, n=4):
    """花在自己的平面里画：四瓣，杯形（瓣边往花脸那边翘），再按 tilt 往后仰、rot 转、放到 (cx, cy)。
    返回：每瓣的整形（远的先）、可见形、瓣心到瓣边的红浓度、黑斑、花蕊、果、叠瓣。"""
    t, r = np.deg2rad(tilt), np.deg2rad(rot)
    ct, st, cr, sr = np.cos(t), np.sin(t), np.cos(r), np.sin(r)

    def proj(x, y, z=None):
        rho = np.hypot(x, y)
        if z is None:
            z = cup * (rho / R) ** 2 * R
        sx, sy = x, y * ct - z * st
        return cx + sx * cr - sy * sr, cy + sx * sr + sy * cr

    angles = a0 + np.arange(n) * 2 * np.pi / n + rng.uniform(-0.2, 0.2, n)
    order = np.argsort(np.sin(angles))           # 远的（花平面里朝上的）先画
    span = 2 * np.pi / n * 1.42
    petals = []
    for i in order:
        a = angles[i]
        Rf = R * rng.uniform(0.9, 1.06)
        u = np.linspace(-1, 1, 70)
        ph = a + u * span / 2
        wav = 1 + 0.045 * np.sin(u * np.pi * rng.uniform(2.5, 4) + rng.uniform(0, 6)) + 0.03 * np.sin(u * 9 + rng.uniform(0, 6))
        rho = Rf * (1 - np.abs(u) ** 2.6) ** 0.42 * wav
        rho = np.maximum(rho, 0.1 * R)
        xs, ys = rho * np.cos(ph), rho * np.sin(ph)
        bx, by = 0.06 * R * np.cos(a), 0.06 * R * np.sin(a)
        X, Y = proj(np.r_[bx, xs, bx], np.r_[by, ys, by])
        full = poly(p, list(zip(X, Y)))
        # 黑斑：瓣根上一块，边是手抖的
        bu = np.linspace(-1, 1, 24)
        bph = a + bu * span * 0.2
        brho = R * (0.36 + 0.05 * np.cos(bu * 3 + rng.uniform(0, 6))) * (1 - np.abs(bu) ** 3) ** 0.5
        bX, bY = proj(np.r_[0.13 * R * np.cos(bph[::-1]), brho * np.cos(bph)], np.r_[0.13 * R * np.sin(bph[::-1]), brho * np.sin(bph)])
        blot = poly(p, list(zip(bX, bY)))
        # 皱：从瓣根往瓣边几道
        crk = np.zeros((p.h, p.w), np.float32)
        for _ in range(rng.integers(3, 6)):
            ca = a + rng.uniform(-0.38, 0.38) * span
            r0, r1 = rng.uniform(0.35, 0.5) * R, rng.uniform(0.75, 0.95) * Rf
            ss = np.linspace(r0, r1, 4)
            cxs, cys = proj(ss * np.cos(ca + rng.normal(0, 0.04, 4)), ss * np.sin(ca + rng.normal(0, 0.04, 4)))
            pen.brush(list(zip(cxs, cys)), width=rng.uniform(5, 10), dry=0.4, taper=(0.3, 0.3), wobble=0.5, acc=crk)
        petals.append(dict(full=full, blot=blot, crk=np.clip(crk, 0, 1) * full, near=np.sin(a) > 0, a=a))

    # 果和花蕊（在远瓣前、近瓣后）
    zc = 0.16 * R
    capX, capY = proj(0, 0, zc)
    cap_r = 0.12 * R
    cap = ellipse(p, capX, capY, cap_r, cap_r * max(ct, 0.35))
    cap_side = ellipse(p, *proj(0, 0, zc * 0.4), cap_r * 0.95, cap_r * 0.9)        # 果身（仰得多时看得见侧面）
    cap = np.maximum(cap, cap_side * (st > 0.5))
    stam = np.zeros((p.h, p.w), np.float32)
    for k in range(60):
        th = rng.uniform(0, 2 * np.pi)
        r0, r1 = 0.1 * R, rng.uniform(0.2, 0.3) * R
        z0, z1 = zc * 0.3, zc * rng.uniform(0.4, 0.9)
        x0, y0 = proj(r0 * np.cos(th), r0 * np.sin(th), z0)
        x1, y1 = proj(r1 * np.cos(th), r1 * np.sin(th), z1)
        pen.pen([(x0, y0), (x1, y1)], width=rng.uniform(1.4, 2.2), wobble=0.2, acc=stam)
        pen.blob(x1, y1, rng.uniform(2.2, 3.6), rough=0.25, acc=stam)
    rays = np.zeros((p.h, p.w), np.float32)
    for k in range(9):
        th = k * 2 * np.pi / 9 + rng.uniform(-0.1, 0.1)
        x1, y1 = proj(cap_r * 0.95 * np.cos(th), cap_r * 0.95 * np.sin(th), zc)
        pen.brush([(capX, capY), (x1, y1)], width=2.6, dry=0, taper=(0.3, 0.1), wobble=0.1, bristles=3, acc=rays)
    pen.blob(capX, capY, cap_r * 0.22, rough=0.2, acc=rays)

    # 遮挡：远瓣 → 果蕊 → 近瓣
    vis = []
    layers = [pt["full"] for pt in petals if not pt["near"]] + [np.maximum(cap, stam)] + [pt["full"] for pt in petals if pt["near"]]
    names = [i for i, pt in enumerate(petals) if not pt["near"]] + ["c"] + [i for i, pt in enumerate(petals) if pt["near"]]
    cover = np.zeros((p.h, p.w), np.float32)
    vis_map = {}
    for nm, lay in zip(names[::-1], layers[::-1]):
        vis_map[nm] = np.clip(lay - cover, 0, 1)
        cover = np.maximum(cover, lay)
    union = cover
    cnt = sum(pt["full"] for pt in petals)
    overlap = np.clip(cnt - 1, 0, 1)

    red = np.zeros((p.h, p.w), np.float32)
    blot = np.zeros((p.h, p.w), np.float32)
    crk = np.zeros((p.h, p.w), np.float32)
    for i, pt in enumerate(petals):
        v = vis_map[i]
        d = inner_dist(p, pt["full"], 0.4 * R)
        dens = 0.52 + 0.48 * smoothstep(0, 0.2 * R, d)                     # 瓣边薄，出网，透出黄
        red = np.maximum(red, v * dens)
        blot = np.maximum(blot, pt["blot"] * v)
        crk = np.maximum(crk, pt["crk"] * v)
    centre = vis_map["c"]
    X_, Y_ = p.XX / p.S, p.YY / p.S
    throat = np.clip(1 - np.hypot(X_ - capX, (Y_ - capY) / max(ct, 0.45)) / (0.62 * R), 0, 1) ** 1.2 * union
    return dict(union=union, red=red, blot=blot, crk=crk, overlap=overlap * union, throat=throat,
                cap=cap * centre, stam=stam * centre, rays=rays * cap * centre)


# ---------------------------------------------------------------- 花苞（低着头）
def bud(p, pen, rng, top, length, width, ang, split=False):
    """top 是苞和茎接的地方，ang 苞尖朝哪（度，0 朝右、90 朝下）。"""
    a = np.deg2rad(ang)
    d = np.array([np.cos(a), np.sin(a)]); n = np.array([-d[1], d[0]])
    s = np.linspace(0, 1, 40)
    half = width / 2 * np.sin(np.pi * np.clip(s * 0.96 + 0.02, 0, 1)) ** 0.75 * (1 + 0.18 * (s - 0.4))
    base = np.array(top)
    left = base + np.outer(s * length, d) + np.outer(half, n)
    right = base + np.outer(s * length, d) - np.outer(half, n)
    pts = np.vstack([left, right[::-1]])
    m = poly(p, [tuple(q) for q in pts])
    # 背光那面（远离窗：右下）
    shade = np.clip(soft(p, m * poly(p, [tuple(q) for q in np.vstack([base + np.outer(s * length, d) - np.outer(half * 0.1, n), right[::-1]])]), 3), 0, 1) * m
    hair = np.zeros((p.h, p.w), np.float32)
    for side, edge in ((1, left), (-1, right)):
        for q in edge[2:-3:2]:
            o = n * side
            ha = np.arctan2(o[1], o[0]) + rng.normal(0, 0.35)
            L = rng.uniform(5, 10)
            pen.pen([tuple(q - o * 1.5), (q[0] + np.cos(ha) * L, q[1] + np.sin(ha) * L)], width=rng.uniform(1.0, 1.4), wobble=0.1, acc=hair)
    seam_off = 0.12 * width
    seam = [tuple(base + d * length * t + n * seam_off * np.sin(np.pi * t)) for t in np.linspace(0.08, 0.93, 6)]
    seamm = pen.pen(seam, width=2.0, wobble=0.2)
    slit = np.zeros((p.h, p.w), np.float32)
    if split:
        sl = [tuple(base + d * length * t + n * (seam_off - 3)) for t in np.linspace(0.25, 0.85, 5)]
        pen.brush(sl, width=width * 0.32, dry=0.0, taper=(0.35, 0.35), wobble=0.3, acc=slit)
        slit = np.clip(slit, 0, 1) * m
    return dict(m=m, shade=shade, hair=np.clip(hair, 0, 1), seam=np.clip(seamm, 0, 1) * m, slit=slit)


def stem(pen, rng, pts, width, acc, hairs):
    P = catmull(pts, per=24)
    pen.brush([tuple(q) for q in P[::3]], width=width, dry=0.05, taper=(0.02, 0.05), wobble=0.35, bristles=4, press=[1, 1, 0.9, 0.85], acc=acc)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    tang = np.gradient(P, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    for s in np.arange(6, d[-1] - 4, rng.uniform(7, 10)):
        i = np.searchsorted(d, s)
        q, tg = P[i], tang[i]
        side = rng.choice([-1, 1])
        nrm = np.array([-tg[1], tg[0]]) * side
        # 毛往花那头斜着长（路线从罐口往上走，tg 朝花）
        hv = nrm * np.cos(0.55) + tg * np.sin(0.55) + rng.normal(0, 0.18, 2)
        hv /= np.linalg.norm(hv)
        L = rng.uniform(4.5, 9)
        q0 = q + nrm * width * 0.35
        pen.pen([tuple(q0), tuple(q0 + hv * L)], width=rng.uniform(0.9, 1.25), wobble=0.1, acc=hairs)


# ---------------------------------------------------------------- 画版
def jar_profile():
    t = np.array([0, 0.03, 0.07, 0.13, 0.22, 0.36, 0.52, 0.70, 0.86, 1.0])
    r = np.array([72, 79, 75, 63, 70, 116, 140, 136, 118, 100], float)
    tt = np.linspace(0, 1, 120)
    rr = np.interp(tt, t, r)
    rr = ndimage.gaussian_filter1d(rr, 3, mode="nearest")
    return tt, rr


def make_plates(seed=5):
    p = Press(W, H, seed=seed)
    pen = Pens(p)
    rng = np.random.default_rng(seed + 3)
    X, Y = p.XX / p.S, p.YY / p.S

    block = rect(p, BX0, BY0, BX1, BY1)
    win = rect(p, *WIN)
    muntin = rect(p, WIN[0], MUNTIN[0], WIN[2], MUNTIN[1])
    muntin = np.maximum(muntin, rect(p, MULLION[0], WIN[1], MULLION[1], WIN[3]))
    win = np.clip(win - muntin, 0, 1)
    wall = rect(p, BX0, BY0, BX1, TABLE_Y) * (1 - win)
    table = rect(p, BX0, TABLE_Y, BX1, FRONT_Y)
    apron = rect(p, BX0, FRONT_Y, BX1, BY1)

    # 陶罐：手捏的，左右不完全对称，边上有起伏
    tt, rr = jar_profile()
    ys = JTOP + tt * (JBASE - JTOP)
    wob = 1 + 0.012 * np.sin(tt * 13 + 1.3) + 0.008 * np.sin(tt * 37)
    lx = JX - rr * wob * 1.02
    rx = JX + rr * (1 + 0.012 * np.sin(tt * 11 + 4)) * 0.98 + tt * 4
    bot = [(JX + 100 * np.cos(th), JBASE + 14 * np.sin(th)) for th in np.linspace(0, np.pi, 20)]
    jar_body = poly(p, list(zip(lx, ys)) + bot[::-1] + list(zip(rx[::-1], ys[::-1])))
    jar_body = np.maximum(jar_body, ellipse(p, JX + 1, JBASE, 99, 14) * (Y > JBASE - 2))
    mouth_out = ellipse(p, JX, JTOP, 80, 17)
    mouth_in = ellipse(p, JX + 2, JTOP + 1, 66, 12)
    jar = np.clip(np.maximum(jar_body, mouth_out), 0, 1)
    # 罐子的明暗：窗在左后，左边一道亮边，前面大半背光
    jl = np.interp(Y, ys, lx); jr = np.interp(Y, ys, rx)
    u = np.clip((X - jl) / np.maximum(jr - jl, 1), 0, 1)          # 0 左边 → 1 右边
    jar_shade = (0.3 + 0.62 * smoothstep(0.08, 0.75, u)) * jar_body
    jar_shade = np.maximum(jar_shade, 0.75 * soft(p, ellipse(p, JX + 6, JTOP + 32, 72, 10), 5) * jar_body)   # 罐口下沿一道影
    rim_light = smoothstep(0.14, 0.02, u) * jar_body
    lip_top = np.clip(mouth_out - grow(p, mouth_in, 2), 0, 1)

    # 投影：罐子从下往上每一层都往右前方甩出去
    shadow = np.zeros((p.h, p.w), np.float32)
    dvec = np.array([0.92, 0.42])
    Lsh = 360
    for h_, r_ in zip(np.linspace(0, 1, 26), np.interp(np.linspace(0, 1, 26), 1 - tt[::-1], rr[::-1])):
        c = np.array([JX, JBASE]) + dvec * h_ * Lsh
        shadow = np.maximum(shadow, ellipse(p, c[0], c[1] - 4, r_ * 0.98, r_ * 0.24 + 6))
    shadow = soft(p, shadow, 5) * (1 - jar)
    fade = 1 - 0.35 * smoothstep(JX, JX + 380, X)
    shadow = shadow * fade

    # 窗光落在桌上的一块
    band = poly(p, [(BX0, 975), (BX0, TABLE_Y), (470, TABLE_Y), (BX1, 1270), (BX1, FRONT_Y), (330, FRONT_Y)])
    band = soft(p, band, 14) * table

    # 花
    F1 = poppy(p, pen, rng, 410, 330, 150, tilt=48, rot=-26, cup=0.4, a0=0.5)
    F2 = poppy(p, pen, rng, 795, 482, 168, tilt=22, rot=8, cup=0.24, a0=0.2)
    F3 = poppy(p, pen, rng, 462, 676, 122, tilt=62, rot=18, cup=0.55, a0=0.9)
    flowers = [F1, F2, F3]
    fl_union = np.clip(sum(f["union"] for f in flowers), 0, 1)

    B1 = bud(p, pen, rng, (1016, 648), 74, 46, 97)
    B2 = bud(p, pen, rng, (622, 176), 66, 40, 112, split=True)
    buds = [B1, B2]
    bud_union = np.clip(B1["m"] + B2["m"], 0, 1)

    # 落瓣：躺在桌上的光里，瓣根朝罐子
    fp = np.zeros((p.h, p.w), np.float32)
    fa = np.deg2rad(-20)
    u_ = np.linspace(-1, 1, 60)
    ph = fa + u_ * 1.25
    rho = 118 * (1 - np.abs(u_) ** 2.4) ** 0.45 * (1 + 0.06 * np.sin(u_ * 8 + 1))
    lx_, ly_ = np.r_[0, rho * np.cos(ph), 0], np.r_[0, rho * np.sin(ph), 0]
    sq, rot = 0.52, np.deg2rad(-14)
    def fpt(x, y):
        y = y * sq
        return 290 + x * np.cos(rot) - y * np.sin(rot), 1268 + x * np.sin(rot) + y * np.cos(rot)
    FX, FY = fpt(lx_, ly_)
    fp = poly(p, list(zip(FX, FY)))
    curl = poly(p, list(zip(*fpt(np.r_[0, rho[40:] * np.cos(ph[40:]), 0], np.r_[0, rho[40:] * np.sin(ph[40:]), 0])))) * fp
    fpd = inner_dist(p, fp, 30)
    fp_red = fp * (0.55 + 0.45 * smoothstep(0, 14, fpd))
    bu = np.linspace(-1, 1, 16)
    bph = fa + bu * 0.38
    bX, bY = fpt(np.r_[6 * np.cos(bph), 40 * np.cos(bph[::-1])], np.r_[6 * np.sin(bph), 40 * np.sin(bph[::-1])])
    fp_blot = poly(p, list(zip(bX, bY))) * fp
    fp_shadow = np.clip(soft(p, np.roll(np.roll(fp, int(5 * p.S), 0), int(10 * p.S), 1), 4) - fp, 0, 1)

    # 茎（骨架版）
    stems = np.zeros((p.h, p.w), np.float32)
    hairs = np.zeros((p.h, p.w), np.float32)
    stem(pen, rng, [(612, 930), (590, 790), (530, 580), (452, 400), (420, 352)], 4.6, stems, hairs)
    stem(pen, rng, [(668, 930), (705, 770), (765, 610), (795, 500)], 5.0, stems, hairs)
    stem(pen, rng, [(628, 932), (600, 840), (535, 775), (478, 735), (462, 700)], 4.2, stems, hairs)
    stem(pen, rng, [(690, 932), (790, 780), (900, 640), (975, 580), (1012, 600), (1018, 652)], 4.0, stems, hairs)
    stem(pen, rng, [(648, 932), (676, 650), (690, 360), (680, 190), (655, 140), (630, 148), (622, 180)], 3.8, stems, hairs)
    occl = np.clip(fl_union + bud_union + jar, 0, 1)
    stems = np.clip(stems, 0, 1) * (1 - occl)
    hairs = np.clip(hairs, 0, 1) * (1 - occl)

    plates = {}
    dwin = ndimage.distance_transform_edt(1 - rect(p, WIN[0], WIN[1], WIN[2] + 8, WIN[3] + 8)[::4, ::4]) * 4 / p.S
    dwin = ndimage.zoom(dwin, 4, order=1)[:p.h, :p.w]

    petal_all = np.clip(fl_union, 0, 1)
    # 1 光（黄）
    sun = 1.0 * table
    sun = np.maximum(sun, petal_all)
    sun = np.maximum(sun, bud_union)
    sun = np.maximum(sun, fp)
    sun = np.maximum(sun, 0.85 * jar)
    sun = np.maximum(sun, 0.95 * lip_top)
    sun = np.maximum(sun, 0.35 * apron)
    plates["sun"] = np.clip(sun, 0, 1)

    # 2 墙（青）
    wv = 0.56 + 0.44 * smoothstep(0, 300, dwin)
    wallp = wall * wv * (1 - fl_union) * (1 - jar)
    wallp = np.maximum(wallp, bud_union)
    for f in flowers:
        wallp = np.maximum(wallp, grow(p, f["cap"], 1))
    wallp = np.maximum(wallp, apron)
    wallp = np.maximum(wallp, muntin * 0.9)
    plates["wall"] = np.clip(wallp, 0, 1)

    # 3 红
    red = np.zeros((p.h, p.w), np.float32)
    for f in flowers:
        red = np.maximum(red, f["red"])
    red = red * (1 - 0.32 * soft(p, win, 18))
    red = np.maximum(red, fp_red)
    red = np.maximum(red, B2["slit"])
    red = np.maximum(red, jar_body * (0.42 - 0.2 * rim_light))
    red = np.maximum(red, lip_top * 0.3)
    red = red * (1 - mouth_in)
    plates["red"] = np.clip(red, 0, 1)

    # 4 影（紫）
    sh = np.zeros((p.h, p.w), np.float32)
    sh = np.maximum(sh, wall * (1 - fl_union) * (1 - jar) * 0.5 * smoothstep(700, BX1 + 60, X) * (1 - bud_union))
    sh = np.maximum(sh, table * (1 - band) * 0.36)
    sh = np.maximum(sh, table * 0.5 * soft(p, rect(p, BX0, TABLE_Y, BX1, TABLE_Y + 10), 5))
    sh = np.maximum(sh, shadow * 0.92)
    sh = np.maximum(sh, jar_shade * (1 - rim_light))
    sh = np.maximum(sh, mouth_in)
    for f in flowers:
        sh = np.maximum(sh, 0.42 * f["overlap"])
        sh = np.maximum(sh, 0.32 * soft(p, f["crk"], 2))
        sh = np.maximum(sh, 0.68 * f["throat"])
    sh = np.maximum(sh, 0.38 * F3["union"] * smoothstep(640, 740, Y))       # 侧面那朵杯底背光
    for b in buds:
        sh = np.maximum(sh, 0.7 * b["shade"])
    sh = np.maximum(sh, 0.6 * fp_shadow)
    sh = np.maximum(sh, 0.45 * curl)
    sh = np.maximum(sh, apron)
    sh = np.maximum(sh, muntin * 0.4)
    plates["shade"] = np.clip(sh, 0, 1)

    # 5 骨架（黑）
    key = np.zeros((p.h, p.w), np.float32)
    key = np.maximum(key, stems)
    key = np.maximum(key, hairs)
    for f in flowers:
        key = np.maximum(key, f["blot"])
        key = np.maximum(key, np.clip(f["stam"], 0, 1))
        key = np.maximum(key, np.clip(f["rays"], 0, 1))
        key = np.maximum(key, 0.3 * f["cap"])
    for b in buds:
        key = np.maximum(key, b["hair"] * (1 - fl_union))
        key = np.maximum(key, b["seam"])
        key = np.maximum(key, 0.28 * b["shade"])
    key = np.maximum(key, fp_blot)
    key = np.maximum(key, mouth_in * soft(p, (Y < JTOP + 4).astype(np.float32), 3))          # 罐口里头后半圈最深
    key = np.maximum(key, 0.55 * mouth_in)
    # 点睛：罐子背光面的粗陶颗粒
    grain_d = jar_body * (1 - rim_light) * (0.08 + 0.45 * smoothstep(0.3, 1.0, u))
    cr = pen.crayon(jar_body * (1 - rim_light), density=grain_d, direction=82, spread=14, length=(6, 16), dot=(0.6, 1.2), tooth=0.25)
    key = np.maximum(key, cr * 0.95)
    # 背光一侧的轮廓，断断续续
    cont = np.zeros((p.h, p.w), np.float32)
    sel = slice(30, 118)
    pen.brush([(x_ + 1, y_) for x_, y_ in zip(rx[sel][::6], ys[sel][::6])], width=3.2, dry=0.45, taper=(0.2, 0.3), wobble=0.4, acc=cont)
    pen.brush([(JX - 70 + 140 * s_, JBASE + 12 * np.sin(np.pi * s_) + 2) for s_ in np.linspace(0.15, 0.95, 8)], width=4, dry=0.3, taper=(0.2, 0.2), wobble=0.3, acc=cont)
    key = np.maximum(key, np.clip(cont, 0, 1) * 0.95)
    # 罐脚贴着桌面的接触影：网点
    contact = soft(p, ellipse(p, JX + 18, JBASE + 4, 108, 12), 4) * (1 - jar_body)
    key = np.maximum(key, 0.42 * contact)
    plates["key"] = np.clip(key, 0, 1)

    for nme in plates:
        plates[nme] = plates[nme] * block
    return plates


# ---------------------------------------------------------------- 一套版，多套墨
SETS = {
    "teal": dict(sun=C["Sunflower"], wall=C["Teal"], red=C["Fluorescent Red"], shade=C["Violet"], key=C["Black"]),
    "mist": dict(sun=C["Yellow"], wall=C["Sea Foam"], red=C["Bright Red"], shade=C["Grape"], key=C["Black"]),
    "night": dict(sun=C["Sunflower"], wall=C["RisoFederal Blue"], red=C["Scarlet"], shade=C["Purple"], key=C["Black"]),
}
SHIFTS = dict(sun=(-3, 2), wall=(4, -3), red=(-5, -4), shade=(3, 3), key=(1, -1))


def print_set(plates, name, out, seed=21, bold=1.0):
    cw = SETS[name]
    p = Press(W, H, seed=seed)
    s = {k: (v[0] * bold, v[1] * bold) for k, v in SHIFTS.items()}
    p.ink("sun", cw["sun"], angle=0, screen="grain", grain=0.55, shift=s["sun"], skips=0.3, edge=0.8)
    p.ink("wall", cw["wall"], angle=0, screen="grain", grain=0.6, shift=s["wall"], skips=0.6, starve=0.06, edge=0.9)
    p.ink("red", cw["red"], angle=15, cell=9, shift=s["red"], skips=0.3, edge=0.7, opacity=0.95)
    p.ink("shade", cw["shade"], angle=0, screen="grain", grain=0.6, shift=s["shade"], skips=0.2, edge=0.9, opacity=0.85)
    p.ink("key", cw["key"], angle=45, cell=7, shift=s["key"], skips=0.3, edge=0.35, opacity=0.95)
    for n in p.inks:
        p.plates[n][:] = plates[n]
    sig = dict(y=BY1 + 22, x0=BX0, x1=BX1, edition="1/1", title="《只开一天》", name="Claude", year="2026", size=28)
    return p.run(out, strip=True, signature=sig)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v1"
    which = sys.argv[2:] or list(SETS)
    t = time.time()
    plates = make_plates()
    print(f"plates {time.time() - t:.1f}s")
    os.makedirs(os.path.join(HERE, "work"), exist_ok=True)
    for nm in which:
        t = time.time()
        print_set(plates, nm, os.path.join(HERE, "work", f"{tag}-{nm}"))
        print(nm, f"{time.time() - t:.1f}s")
