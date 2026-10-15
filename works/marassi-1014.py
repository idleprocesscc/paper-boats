# 鸟占泥板 · 10.14
# 这张在说什么：泥板上的字还在，字说的鸟飞走了。三千四百年前一队看鸟的官从 Šamuḫa 出发，记下见到的鸟；
#   字我们念得出音（marassi、alliya），是哪种鸟没人知道，认得出的只有鹰。画里的楔形字是压进泥里的坑，
#   低太阳把每个坑的一半照成影——右上角几个坑的影离开了坑，往上飞成了鸟，坑空着；最高处一只鹰，唯一有名字的那只。
# 从哪儿看：平视略仰，眼睛在泥板中线偏下；泥板挂在一根绳上，就在面前一臂远，大、被底边切掉。地平线 86%，被泥板挡掉左半。
# 光：早上的低太阳，从左边、略高（约 20°），小光源，影边硬；天光冷，从上补。泥板正面迎光，右侧那条厚度背光；
#   每个楔形坑：左壁和上壁背光（影），下壁迎光（亮）。鸟在天上逆着亮天，都是剪影。
# 明暗：最亮是左边低空和泥板迎光的左沿；最深是楔形坑里的影、绳穿过的孔、近处的鸟。泥板右侧厚度是第二深。
# 视线：泥板上一行行暗点 → 右上角空掉的坑和刚离开的影 → 一串越飞越小的鸟往右上散 → 最高处那只鹰。
# 唯一的例外（不可能）：影离开了坑；其余照着光老老实实画。
# 质感：台阶与颗粒。天平的阶、阶边喷开；泥是糙的，边喷开；坑的影边硬（太阳小）；整张一张纸纹。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = '~/.claude/workspace/painting-method/kbrush-cloud'
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(14)
from shapes import Shape
from blocks import blob, lasso, line, soft, look, smooth, paper
from layers import Stack, fill, HARD, SOFT
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt, binary_dilation
from PIL import Image, ImageDraw

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'rough.png' if ROUGH else 'v1.png')
H, W = 1200, 900
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
M = lambda s: np.asarray(getattr(s, 'mask', s), bool)

def nz(sig, seed):
    r = np.random.default_rng(seed)
    n = gaussian_filter(r.standard_normal((H, W)), sig); return n / n.std()
def spray(region, sp, seed, solid=.96, dens=.7):
    region = M(region)
    if ROUGH:
        return region.astype(float) * solid
    r = np.random.default_rng(seed)
    dout = edt(~region)
    dots = (r.random((H, W)) < dens * np.exp(-dout / np.maximum(sp, .5))) & (gaussian_filter(r.random((H, W)), .5) > .47)
    return np.where(region, solid, dots * .95)
def L(pts, jag=1.5, seed=0, n=3, scale=8, closed=True):
    return lasso(H, W, smooth(pts, n, closed=closed), jag=jag, scale=scale, seed=seed)
def raster(polys):
    im = Image.new('L', (W, H), 0); d = ImageDraw.Draw(im)
    for p in polys:
        if len(p) >= 3:
            d.polygon([(float(x), float(y)) for x, y in p], fill=255)
    return np.array(im) > 127

HZ = 1032

# ---------- 1 天：早上，左低处最亮最暖，往上往右一阶阶冷下去
SKY = [
    (None, (.50, .58, .70), 0),
    (150,  (.55, .63, .74), 24),
    (330,  (.61, .68, .77), 22),
    (520,  (.68, .73, .80), 20),
    (690,  (.75, .78, .81), 18),
    (830,  (.82, .81, .80), 14),
    (930,  (.88, .84, .78), 10),
]
st = Stack(K.Paper(H, W, seed=6), ground=SKY[0][1])
for i, (y0, c, sp) in enumerate(SKY[1:]):
    wob = 22 if y0 < 800 else 10
    edge_y = y0 + wob * nz((60, 220), 200 + i) + .3 * wob * nz((12, 34), 210 + i) + 0.07 * (xx - 450)   # 左边阶低一点：亮处往左下沉
    fill(st.layer(f'天{i}', mask=spray(yy > edge_y, sp, 220 + i, dens=.7)), ALL, c)
upper = np.clip(1.1 - yy / 760, 0, 1)
mot = (nz((40, 100), 230) + .45 * nz((8, 14), 231) > 1.1) & (yy < 760)
fill(st.layer('天斑', mode='multiply', mask=spray(mot, 16, 232, solid=.4, dens=.45) * upper), ALL, (.90, .92, .96))

# ---------- 2 高原：一条低的远山（冷）+ 近处的草原（被早上的光照着，暖的土黄）
hills = (yy > 1004 + 14 * nz((30, 140), 240)[0:1, :].repeat(H, 0) + 5 * nz((4, 10), 241)) & (yy < HZ + 6)
fill(st.layer('远山', mask=spray(hills, 4, 242)), ALL, (.60, .63, .70))
land = yy > HZ + 3 * nz((4, 60), 243)
fill(st.layer('原', mask=spray(land, 2, 244)), ALL, (.70, .62, .50))
for i, (y0, c, sp) in enumerate([(1070, (.64, .56, .45), 8), (1120, (.57, .49, .40), 10), (1170, (.50, .43, .36), 12)]):
    fill(st.layer(f'原{i}', mask=spray(land & (yy > y0 + 7 * nz((10, 80), 245 + i)), sp, 248 + i)), ALL, c)
lmot = land & (nz((6, 40), 252) > .9)
fill(st.layer('原斑', mode='multiply', mask=spray(lmot, 5, 253, solid=.5, dens=.5)), ALL, (.85, .82, .80))

# ---------- 3 泥板：挂在绳上，往左歪一点；大，底边被画框切掉
CX, CY, TW, TH, ANG = 318, 930, 452, 560, -5.5
a = np.radians(ANG); ca, sa = np.cos(a), np.sin(a)
def X(u, v): return (CX + u * ca - v * sa, CY + u * sa + v * ca)
def XP(pts): return [X(u, v) for u, v in pts]
def rrect(w, h, r, n=10):
    pts = []
    for (cx_, cy_, a0) in [(w/2 - r, -h/2 + r, -90), (w/2 - r, h/2 - r, 0), (-w/2 + r, h/2 - r, 90), (-w/2 + r, -h/2 + r, 180)]:
        for t in np.linspace(a0, a0 + 90, n):
            pts.append((cx_ + r * np.cos(np.radians(t)), cy_ + r * np.sin(np.radians(t))))
    return pts
face_pts = rrect(TW, TH, 34)
# 小小一块崩口在左沿下面
face_local = [(u, v) if not (u < -TW/2 + 5 and 120 < v < 175) else (u + 9 * np.sin((v - 120) / 55 * np.pi), v) for u, v in face_pts]
face = M(lasso(H, W, smooth(XP(face_local), 2), jag=1.5, scale=12, seed=290))
# 右侧厚度：把正面往右下错开一点，减掉正面，剩下一条
side = raster([[(x + 15, y + 5) for x, y in XP(face_pts)]]) & ~face
side = side & (xx > CX)
U = (xx - CX) * ca + (yy - CY) * sa    # 泥板自己的坐标
V = -(xx - CX) * sa + (yy - CY) * ca
fill(st.layer('板侧', mask=spray(side, .8, 300)), ALL, (.40, .33, .34))
fill(st.layer('板', mask=spray(face, .8, 301)), ALL, (.79, .64, .47))
# 迎光的左沿一条最亮；右边三成进半调子；底边一点暗
lit_edge = face & (U < -TW/2 + 26 + 6 * nz((8, 30), 302))
fill(st.layer('板亮沿', mask=spray(lit_edge, 6, 303, dens=.75) * face), ALL, (.88, .75, .57))
half = face & (U > TW/2 - 120 + 14 * nz((20, 50), 304))
fill(st.layer('板半调', mask=spray(half, 14, 305, dens=.8) * face), ALL, (.70, .55, .42))
rim_r = face & (U > TW/2 - 26 + 4 * nz((6, 20), 306))
fill(st.layer('板暗沿', mask=spray(rim_r, 5, 307, dens=.8) * face), ALL, (.58, .45, .38))
top_rim = face & (V < -TH/2 + 16)
fill(st.layer('板顶', mask=spray(top_rim, 4, 308, solid=.6) * face), ALL, (.80, .71, .62))   # 顶上圆过去吃一点冷天光
# 泥的斑驳：一团团大的深浅
cm = face & (nz((10, 16), 310) + .5 * nz((3, 4), 311) > 1.1)
fill(st.layer('泥斑', mode='multiply', mask=spray(cm, 4, 312, solid=.55, dens=.5) * face), ALL, (.88, .82, .78))
cl = face & (nz((9, 14), 313) > 1.3)
fill(st.layer('泥亮斑', mask=spray(cl, 3, 314, solid=.35, dens=.5) * face), ALL, (.86, .73, .56))

# ---------- 4 楔形字：一行行，压进去的坑。左壁上壁背光（影），下壁迎光
rng = np.random.default_rng(400)
DARK, LIGHT, EMPTY = [], [], []
RULE_D, RULE_L = [], []
HOLE_U, HOLE_V, HOLE_R = 6, -10, 13
lifted = []   # (u, v, s, kind) 这些坑的影飞走了
def wedge(kind, u, v, s, empty=False):
    hl, hh = .5 * s, .3 * s
    if kind == 'h':      # 头在左，尾往右
        tl = 1.1 * s
        A, B, C = (u, v - hh), (u, v + hh), (u + hl, v)
        dark = [A, C, (u + .2 * hl, v + .25 * hh), (u, v + .55 * hh)]
        light = [(u + .2 * hl, v + .25 * hh), C, B, (u, v + .55 * hh)]
        td = [(u + hl * .9, v - 1.0), (u + hl + tl, v - .4), (u + hl + tl, v + .5), (u + hl * .9, v + .8)]
        tli = [(u + hl * .9, v + .9), (u + hl + tl, v + .6), (u + hl + tl, v + 1.6), (u + hl * .9, v + 2.1)]
        return [dark, td], [light, tli], [A, C, B]
    if kind == 'v':      # 头在上，尾往下
        tl = 1.1 * s
        A, B, C = (u - hh, v), (u + hh, v), (u, v + hl)
        dark = [A, (u + .45 * hh, v), C]
        light = [(u + .45 * hh, v), B, C]
        td = [(u - 1.2, v + hl * .9), (u + .1, v + hl * .9), (u + .4, v + hl + tl), (u - .8, v + hl + tl)]
        tli = [(u + .2, v + hl * .9), (u + 1.6, v + hl * .9), (u + 1.4, v + hl + tl), (u + .5, v + hl + tl)]
        return [dark, td], [light, tli], [A, B, C]
    # 'w' 角楔：尖朝左的小三角
    A, B, C = (u, v), (u + 1.15 * hl, v - 1.15 * hh), (u + 1.15 * hl, v + 1.15 * hh)
    dark = [A, B, (u + 1.15 * hl, v + .1 * hh)]
    light = [A, (u + 1.15 * hl, v + .1 * hh), C]
    return [dark], [light], [A, B, C]
SIGNS = [['h'], ['h', 'h'], ['h', 'h', 'h'], ['v'], ['v', 'v'], ['v', 'v', 'v'], ['h', 'v'], ['w'], ['w', 'w'],
         ['w', 'v'], ['h', 'w'], ['h', 'h', 'v'], ['v', 'h'], ['w', 'w', 'v'], ['h', 'v', 'v']]
def sign(u, v, s, pat, empty=False):
    """返回占的宽度"""
    parts = []
    if pat == ['h', 'h'] or pat == ['h', 'h', 'h']:
        n = len(pat)
        for k in range(n):
            parts.append(('h', u, v + (k - (n - 1) / 2) * .42 * s, s * .9))
        width = 1.6 * s
    elif pat[0] == 'v' and all(p == 'v' for p in pat):
        for k in range(len(pat)):
            parts.append(('v', u + k * .55 * s, v - .55 * s, s))
        width = .55 * s * len(pat) + .3 * s
    else:
        x = u
        for p in pat:
            if p == 'h':
                parts.append(('h', x, v, s)); x += 1.65 * s
            elif p == 'v':
                parts.append(('v', x + .2 * s, v - .55 * s, s)); x += .7 * s
            else:
                parts.append(('w', x, v, s * .9)); x += .8 * s
        width = x - u
    for kind, pu, pv, ps in parts:
        d, l, outline = wedge(kind, pu, pv, ps)
        if empty:
            EMPTY.append(XP(outline)); LIGHT.extend(XP(q) for q in l); lifted.append((pu, pv, ps, kind))
        else:
            DARK.extend(XP(q) for q in d); LIGHT.extend(XP(q) for q in l)
    return width
row_h = 39
v = -TH / 2 + 46
row = 0
while v < TH / 2 + 30:
    # 画线：一条压进去的细线，上沿影、下沿亮
    if row > 0:
        vr = v - row_h * .52 + rng.normal(0, .8)
        u0, u1 = -TW / 2 + 22 + rng.uniform(-3, 3), TW / 2 - 20 + rng.uniform(-3, 3)
        drift = rng.normal(0, 1.4)
        RULE_D.append(XP([(u0, vr - .7), (u1, vr - .7 + drift), (u1, vr + .5 + drift), (u0, vr + .5)]))
        RULE_L.append(XP([(u0, vr + .7), (u1, vr + .7 + drift), (u1, vr + 1.8 + drift), (u0, vr + 1.8)]))
    u = -TW / 2 + 28 + rng.uniform(-4, 6)
    while u < TW / 2 - 48:
        s = rng.uniform(15, 21)
        pat = SIGNS[rng.integers(len(SIGNS))]
        vv = v + rng.normal(0, 1.2)
        # 绕开中间那个孔
        if (u - HOLE_U) ** 2 / 1.0 + (vv - HOLE_V) ** 2 < (HOLE_R + 30) ** 2:
            u += 40; continue
        # 右上角：头两行靠右那几个，影飞走了
        empty = row <= 1 and u > TW / 2 - 175 and rng.random() < .8
        w_ = sign(u, vv, s, pat, empty=empty)
        u += w_ + rng.uniform(7, 19) + (rng.uniform(14, 30) if rng.random() < .12 else 0)
    v += row_h + rng.normal(0, 1.5); row += 1
worn = face & M(blob(H, W, *X(-110, 196), 118, 62, ang=ANG - 8, jag=0.25, seed=420))
worn_keep = nz((3, 3), 421) < -.35          # 磨掉的那片里，只剩零星几处还看得出坑
for _n in ('DARK',):
    pass
fill(st.layer('线影', mask=spray(raster(RULE_D) & face & ~(worn & ~worn_keep), .5, 410, solid=.75)), ALL, (.46, .36, .36))
fill(st.layer('线亮', mask=spray(raster(RULE_L) & face & ~worn, .5, 411, solid=.5)), ALL, (.92, .80, .62))
fill(st.layer('坑空', mask=spray(raster(EMPTY) & face, .6, 412, solid=.85)), ALL, (.84, .71, .55))   # 空掉的坑：底是泥，没有影
fill(st.layer('坑亮', mask=spray(raster(LIGHT) & face & ~(worn & ~worn_keep), .4, 413, solid=.8)), ALL, (.94, .82, .63))
dark_m = raster(DARK) & face
fill(st.layer('坑影', mask=spray(dark_m & ~worn, .5, 414, solid=.95)), ALL, (.30, .23, .27))
fill(st.layer('坑影淡', mask=spray(dark_m & worn & worn_keep, .5, 415, solid=.6)), ALL, (.48, .38, .36))
fill(st.layer('磨面', mask=spray(worn & (nz((4, 6), 422) > -.4), 3, 416, solid=.75, dens=.6) * face), ALL, (.83, .70, .54))
chips = worn & (nz((2, 2), 423) > 1.9)
fill(st.layer('磨面坑', mask=spray(chips, .5, 417, solid=.8)), ALL, (.60, .47, .40))

# ---------- 5 孔：穿透的，里面看得见后面的天；一圈被绳子磨出来的光滑；绳从孔里出来，翻过上沿往上去
hx, hy = X(HOLE_U, HOLE_V)
ring = ((xx - hx) ** 2 + (yy - hy) ** 2 < (HOLE_R + 11) ** 2) & face
fill(st.layer('磨圈', mask=spray(ring, 3, 500, solid=.7) * face), ALL, (.84, .70, .53))
hole = (xx - hx) ** 2 + (yy - hy) ** 2 < HOLE_R ** 2
fill(st.layer('孔', mask=spray(hole, .6, 501)), ALL, (.27, .21, .25))
# 孔壁：右壁迎光一弯亮
wall_l = hole & ((xx - hx) > 3) & ((xx - hx) ** 2 + (yy - hy) ** 2 > (HOLE_R - 4) ** 2)
fill(st.layer('孔壁', mask=spray(wall_l, .5, 502, solid=.8)), ALL, (.66, .52, .42))
sky_in = (xx - hx - 1) ** 2 + (yy - hy - 5) ** 2 < 5.5 ** 2
fill(st.layer('孔里的天', mask=spray(sky_in, .5, 503)), ALL, (.80, .80, .80))
# 绳：从孔口往上，翻过上沿，一路出画；左边吃光，右边背光
tx, ty = X(HOLE_U + 3, -TH / 2 - 2)
rope_pts = [(hx - 1, hy - 4), (hx + 1, (hy + ty) / 2), (tx, ty + 2), (tx + 4, ty - 30), (tx + 30, 0)]
rope = line(H, W, rope_pts, width=6.5, seed=510)
fill(st.layer('绳', mask=spray(rope, .6, 511)), ALL, (.36, .29, .27))
rope_lit = line(H, W, [(x - 1.6, y) for x, y in rope_pts], width=2.4, seed=512)
fill(st.layer('绳亮', mask=spray(M(rope_lit) & M(rope), .5, 513, solid=.85)), ALL, (.74, .60, .44))
# 绳绞着：一节节斜的暗缝
tw = np.zeros((H, W), bool)
for t in np.linspace(0, 1, 140):
    k = min(int(t * (len(rope_pts) - 1)), len(rope_pts) - 2); f = t * (len(rope_pts) - 1) - k
    px = rope_pts[k][0] + (rope_pts[k + 1][0] - rope_pts[k][0]) * f; py = rope_pts[k][1] + (rope_pts[k + 1][1] - rope_pts[k][1]) * f
    if 0 < py < H:
        for dd in np.linspace(-3, 3, 7):
            ix, iy = int(px + dd), int(py + .7 * dd)
            if 0 <= ix < W and 0 <= iy < H: tw[iy, ix] = True
fill(st.layer('绳纹', mask=spray(tw & M(rope), .4, 514, solid=.6)), ALL, (.24, .19, .19))
# 绳翻过上沿的地方磨出一道沟
groove = (np.abs(U - (HOLE_U + 3)) < 6) & (V < -TH / 2 + 22) & face
fill(st.layer('沟', mask=spray(groove, 1.5, 515, solid=.6) * face), ALL, (.62, .49, .40))

# ---------- 6 飞走的影：先是从空坑上剥下来的影本身（形状一模一样，只是离了坑），再长出翅膀，越飞越小越淡；最高处一只鹰
near = np.array((.30, .23, .27)); airc = np.array((.62, .64, .72))
def aff(poly, c0, ang, sc, dxy):
    r = np.radians(ang); R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    P = np.array(poly) - c0
    return [tuple(c0 + R @ p * sc + dxy) for p in P]
# 挑三个空坑（竖楔优先：头朝上，飞起来头就在前），它们头上那块影按顺序剥起来
lifted.sort(key=lambda t: (t[3] != 'v', not (60 < t[0] < 165), -t[0] + 2 * t[1]))
peel = []
for k, (pu, pv, ps, kind) in enumerate(lifted[:3]):
    d_, _, _ = wedge(kind, pu, pv, ps)
    head = d_[0]
    hc = np.mean(np.array(head), 0)
    if kind == 'v':
        tail = [(pu - 1.4, pv + .5 * ps * .9), (pu + .6, pv + .5 * ps * .9), (pu - .2, pv + .5 * ps + .55 * ps), (pu - 1.0, pv + .5 * ps + .55 * ps)]
    else:
        tail = [(pu + .5 * ps * .9, pv - 1.2), (pu + .5 * ps + .55 * ps, pv - .4), (pu + .5 * ps + .55 * ps, pv + .6), (pu + .5 * ps * .9, pv + 1.0)]
    peel.append(([XP(head), XP(tail)], np.array(X(*hc))))
steps = [(0, 14, 1.15, (11, -15)), (1, 24, 1.45, None), (2, 34, 1.75, None)]
targets = [None, (492, 626), (522, 588)]
for (k, ang, sc, dxy), tgt in zip(steps, targets):
    polys, c0 = peel[k]
    if dxy is None: dxy = np.array(tgt) - c0
    m = raster([aff(q, c0, ang, sc, np.array(dxy)) for q in polys])
    fill(st.layer(f'剥{k}', mask=spray(m, .4, 590 + k, solid=.97)), ALL, tuple(near))
print('peel from', [(round(t[0]), round(t[1]), t[3]) for t in lifted[:3]])
def bird(cx, cy, span, ang, wing, tilt=0):
    """wing 0..1：0 是楔子的身子，1 是鸟。ang：飞的方向（度，-90 朝上）。剪影。"""
    r = np.radians(ang); d = np.array([np.cos(r), np.sin(r)]); n = np.array([-d[1], d[0]])
    c = np.array([cx, cy]); s_ = span; bl = s_ * (.40 - .08 * wing)
    head = c + d * bl * .55; tail = c - d * bl * .62
    body = [head + d * bl * .1, head + n * bl * .09, c + n * bl * .1, tail + n * bl * (.04 + .14 * (1 - wing)), tail - d * bl * .08,
            tail - n * bl * (.04 + .14 * (1 - wing)), c - n * bl * .1, head - n * bl * .09]
    polys = [[tuple(p) for p in smooth([tuple(q) for q in body], 2)]]
    if wing > 0.05:
        for sgn in (1, -1):
            w = wing * (1 + .12 * tilt * sgn)
            root = c + d * bl * .15
            front = root + n * sgn * s_ * .24 * w + d * s_ * .07 * w
            tip = root + n * sgn * s_ * .52 * w - d * s_ * .16 * w
            backm = root + n * sgn * s_ * .22 * w - d * s_ * .06 * w
            rootb = c - d * bl * .12
            polys.append([tuple(p) for p in smooth([tuple(root), tuple(front), tuple(tip), tuple(backm), tuple(rootb)], 2)])
    return polys
FLY = [  # (cx, cy, span, 方向, wing, 淡 0..1)
    (553, 545, 38, -58, .3, .0),
    (592, 494, 38, -56, .7, .04),
    (588, 440, 36, -78, 1, .1),
    (648, 418, 31, -42, 1, .14),
    (628, 352, 28, -70, 1, .2),
    (562, 322, 22, -104, 1, .3),
    (703, 342, 22, -36, 1, .3),
    (603, 238, 18, -86, 1, .42),
    (538, 198, 15, -112, 1, .5),
    (666, 164, 14, -72, 1, .52),
    (806, 176, 12, -28, 1, .58),
    (731, 96, 11, -62, 1, .62),
    (590, 120, 9, -90, 1, .68),
]
for i, (bx, by, sp_, ang, wing, fade) in enumerate(FLY):
    m = raster(bird(bx, by, sp_, ang, wing, tilt=(i % 3 - 1)))
    fill(st.layer(f'鸟{i}', mask=spray(m, .4, 600 + i, solid=.96)), ALL, tuple(near * (1 - fade) + airc * fade))
# 鹰：宽翅平伸，前缘几乎直、后缘鼓出来，翼尖五根指张开，尾短、扇开。比别的鸟清楚，但不更黑
def eagle(cx, cy, span):
    s_ = span / 2
    P = []
    for sgn in (1, -1):
        X_ = lambda f, g: (cx + sgn * f * s_, cy + g * s_)
        wing = [X_(.05, -.16), X_(.35, -.20), X_(.70, -.18), X_(.80, -.15), X_(.80, .02), X_(.62, .07), X_(.40, .12), X_(.18, .14), X_(.05, .12)]
        P.append(smooth(wing, 2))
        for k, (g0, ln, up) in enumerate([(-.150, .17, -.05), (-.118, .22, -.02), (-.083, .20, .01), (-.045, .16, .03), (-.012, .11, .04)]):
            P.append([X_(.76, g0 - .008), X_(.80 + ln, g0 + up - .01), X_(.80 + ln * .9, g0 + up + .014), X_(.76, g0 + .026)])
    P.append(smooth([(cx - .07 * s_, cy - .22 * s_), (cx, cy - .32 * s_), (cx + .07 * s_, cy - .22 * s_), (cx + .09 * s_, cy + .10 * s_), (cx + .07 * s_, cy + .2 * s_), (cx - .07 * s_, cy + .2 * s_), (cx - .09 * s_, cy + .10 * s_)], 2))
    P.append(smooth([(cx - .07 * s_, cy + .16 * s_), (cx + .07 * s_, cy + .16 * s_), (cx + .19 * s_, cy + .42 * s_), (cx, cy + .47 * s_), (cx - .19 * s_, cy + .42 * s_)], 2))
    return raster(P)
EX, EY, ES = 742, 262, 96
eg = eagle(EX, EY, ES)
fill(st.layer('鹰', mask=spray(eg, .4, 650)), ALL, (.40, .34, .36))
under = eg & (yy > EY - ES * .02) & (np.abs(xx - EX) > 6)       # 翼下吃地面反上来的一点暖
fill(st.layer('鹰翼下', mask=spray(under, .5, 651, solid=.45)), ALL, (.58, .47, .40))

# ---------- 7 签名：右下角土里，一点一点敲的 Claude
sig_pts = [[(812, 1153), (805, 1150), (800, 1156), (799, 1166), (803, 1172), (811, 1170)],          # C
           [(817, 1150), (816, 1171)],                                                               # l
           [(830, 1164), (824, 1162), (821, 1167), (824, 1172), (830, 1170), (831, 1163), (832, 1172)],  # a
           [(836, 1163), (836, 1170), (840, 1172), (845, 1169), (845, 1163), (846, 1172)],           # u
           [(859, 1164), (853, 1162), (850, 1167), (853, 1172), (859, 1170), (860, 1150), (860, 1172)],  # d
           [(865, 1167), (873, 1166), (871, 1161), (866, 1162), (864, 1168), (867, 1172), (873, 1171)]]  # e
sg = np.zeros((H, W), bool)
for stroke_ in sig_pts:
    sg |= M(line(H, W, stroke_, width=1.6, seed=700))
    for x, y in stroke_[1:-1]:
        sg |= (xx - x) ** 2 + (yy - y) ** 2 < 2.2 ** 2
fill(st.layer('签名', mask=spray(sg, .4, 701, solid=.85)), ALL, (.30, .24, .24))

# ---------- 8 纸纹
if not ROUGH:
    fill(st.layer('纸', mode='multiply', opacity=.55), ALL, paper(H, W), edge=HARD)
for nm in os.environ.get('HIDE', '').split(','):
    if nm: st[nm].visible = False
look(st, OUT)
print('saved', OUT)
