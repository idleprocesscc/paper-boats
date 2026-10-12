# 《闻到了》 2026.10.10
# "犀牛在这里很多，我一头没看见，但闻到了——马一进那股味就惊跳"（1772 年万丹高地，东印度公司档案）
# 光：雨后清早，太阳从画面左上低低斜进来，隔着薄云，柔、偏暖；整片灰白的天从头顶罩着，水坑里映的就是它。
# 从哪儿看：正上方往下，一人高。
# 说什么：马在一串它看不见的脚印前停住了。刹车印和犀牛脚印之间空出来的那一截，就是"闻到了"。
# 明暗：泥中、草中偏暗，最亮是积水的犀牛脚印（天），最深是马蹄踩下去的湿泥和刹车沟。
# 视线：先落在右边路上那两汪亮的，顺着刹车印往左回到马蹄。
# 五面：一束有方向的柔光 / 自然色 / 平（俯视） / 忠实 / 看不见的手，台阶与颗粒。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get("KBRUSH_ENGINE_DIR", "./kbrush")  # kbrush engine: https://github.com/idleprocesscc/kbrush (the kbrush/ folder)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(10)
from shapes import Shape
from blocks import blob, lasso, line, edges, look
from layers import Stack, fill, HARD, SOFT, LOST
from scipy.ndimage import distance_transform_edt as edt, gaussian_filter
H, W = 1000, 1500
yy, xx = np.mgrid[0:H, 0:W]
ALL = Shape(np.ones((H, W), bool))
rng = np.random.default_rng(7)
def grow(sh, n):
    m = sh.mask if isinstance(sh, Shape) else sh
    return Shape(edt(~m) <= n) if n >= 0 else Shape(edt(m) > -n)
def nz(sig, seed):
    g = np.random.default_rng(seed); n = gaussian_filter(g.standard_normal((H, W)), sig); return n / n.std()
def shift(m, dy, dx):
    o = np.zeros_like(m); dy, dx = int(dy), int(dx)
    o[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = m[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return o
def step(region, spray, solid=.95, toward=None):
    """一阶：区域里实实地盖上，边外喷一圈颗粒，越远越稀（toward：只往那个方向喷）"""
    dout = edt(~region)
    p = .7 * np.exp(-dout / spray)
    if toward is not None: p = p * toward
    dots = (rng.random((H, W)) < p) & (gaussian_filter(rng.random((H, W)), .5) > .47)
    return np.where(region, solid, dots * .95)

st = Stack(K.Paper(H, W, seed=3))

# ---------- 路 ----------
def center(x): return 660 - 0.23 * x - 0.00005 * (x - 700) ** 2 + 7
def road_edge(x, top):
    base = 660 - 0.23 * x - 0.00005 * (x - 700) ** 2
    return base - 165 if top else base + 180
top_pts = [(x, road_edge(x, True) + rng.normal(0, 7)) for x in range(-20, 1540, 30)]
bot_pts = [(x, road_edge(x, False) + rng.normal(0, 7)) for x in range(1540, -40, -30)]
road = lasso(H, W, top_pts + bot_pts, jag=6, seed=2)
d_road = np.array([1.0, -0.23]); d_road /= np.linalg.norm(d_road); n_road = np.array([-d_road[1], d_road[0]])
dist = (yy - center(xx)) / 172.0

# ---------- 犀牛的道 ----------
r0 = np.array([1180.0, -40]); rd = np.array([-0.17, 1.0]); rd /= np.linalg.norm(rd); rn = np.array([rd[1], -rd[0]])
lane_l = [tuple(r0 + rd * t - rn * (48 + 9 * np.sin(t / 47))) for t in range(0, 1200, 30)]
lane_r = [tuple(r0 + rd * t + rn * (48 + 9 * np.cos(t / 53))) for t in range(1170, -30, -30)]
lane = lasso(H, W, lane_l + lane_r, jag=6, seed=11)
along = (xx - r0[0]) * rd[0] + (yy - r0[1]) * rd[1]; across = (xx - r0[0]) * rn[0] + (yy - r0[1]) * rn[1]

# ---------- 1 草：三阶，暗底 → 中 → 迎光的尖；草被雨压得往右下倒，一缕缕的 ----------
grass = ~road.mask
fill(st.layer('草底'), ALL, '#36402a', edge=HARD)
def aniso(sig_long, sig_short, ang_deg, seed):
    g = np.random.default_rng(seed).standard_normal((H + 200, W + 200))
    from scipy.ndimage import rotate
    g = gaussian_filter(g, (sig_short, sig_long))
    g = rotate(g, ang_deg, reshape=False, order=1)[100:100 + H, 100:100 + W]
    return g / g.std()
# 从正上方看，一墩草是一朵往外炸开的叶子；一墩挨一墩，墩和墩之间是暗缝
from scipy.spatial import cKDTree
gr = np.random.default_rng(21)
pts = []
while len(pts) < 2600:
    p = gr.uniform([-40, -40], [W + 40, H + 40])
    pts.append(p)
pts = np.array(pts)
tree = cKDTree(pts)
dd, ii = tree.query(np.c_[xx.ravel(), yy.ravel()], k=2)
d1 = dd[:, 0].reshape(H, W); d2 = dd[:, 1].reshape(H, W); cid = ii[:, 0].reshape(H, W)
cx = pts[cid, 0]; cy = pts[cid, 1]
theta = np.arctan2(yy - cy, xx - cx)
rel = d1 / (d1 + d2 + 1e-6) * 2                                 # 0 在墩心，1 在两墩交界
wig = nz(2.5, 4) + .6 * nz(8, 5)
stripe = np.sin(theta * (7 + (cid % 6)) + cid * 1.7 + 2.2 * wig + rel * 3 * np.sin(cid))
sun = np.cos(theta - np.radians(-135))                          # 朝左上的那一侧迎光
focus = np.hypot(xx - 900, (yy - 420) * 1.3)
calm = np.clip((focus - 300) / 700, 0, 1)
mid = grass & (stripe > .05 + .55 * rel)
fill(st.layer('草中', mask=step(mid, 1.5)), ALL, '#46553a', edge=HARD)
tip = mid & (stripe > .7) & (rel > .45) & (sun > .35 + .3 * calm)
fill(st.layer('草尖', mask=step(tip, 1.2)), ALL, '#6f7c48', edge=HARD)
gap = grass & (rel > .97) & (nz(4, 6) > -.3)
fill(st.layer('墩缝', mask=step(gap, 1.5)), ALL, '#283020', edge=HARD)
fill(st.layer('远处压暗', mode='multiply', mask=np.clip(calm, 0, 1) * .55), ALL, '#747a63', edge=HARD)
# 道：压倒的草，顺着走的方向一缕缕，偏黄褐
streak = gaussian_filter(np.random.default_rng(5).standard_normal((H, W)), (1, 1))
streak = np.sin(across / 3.1 + 1.4 * nz(9, 6)) + .5 * streak
lane_g = lane.mask & grass
lane_edge = lane.mask & (edt(lane.mask) < 9 + 7 * nz(5, 12))
lane_g = lane_g & ~(lane_edge & (stripe > .2))                 # 两边的草墩探进道里，吃掉一点边
fill(st.layer('道', mask=step(lane_g, 5)), ALL, '#5c5631', edge=HARD)
fill(st.layer('道里亮的', mask=step(lane_g & (streak > .9) & (nz(6, 13) > -.2), 1.2)), ALL, '#7f7647', edge=HARD)

# ---------- 2 泥：三阶，贴着草的湿暗、中间、路冠干亮 ----------
fill(st.layer('泥'), road, '#7a6447', edge=HARD)
crown_r = road.mask & (np.abs(dist + .08 + .12 * np.sin(xx / 170.0) + .05 * nz(20, 8)) < .5)
fill(st.layer('泥中', mask=step(crown_r, 7)), ALL, '#8c7552', edge=HARD)
crown2 = road.mask & (np.abs(dist + .12 + .10 * np.sin(xx / 210.0) + .07 * nz(16, 9)) < .22)
fill(st.layer('路冠', mask=step(crown2, 5)), ALL, '#a08a65', edge=HARD)
wet = road.mask & (edt(road.mask) < 22 + 8 * nz(10, 10))
fill(st.layer('湿边', mask=step(wet, 4)), ALL, '#5e4c35', edge=HARD)
# 小石子：一粒亮、右下一点影
pb = st.layer('石子')
for _ in range(70):
    x, y = rng.uniform(0, W), 0
    y = center(x) + rng.uniform(-150, 160)
    if not road.mask[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))]: continue
    r = rng.uniform(2, 5)
    fill(pb, blob(H, W, x + r * .5, y + r * .5, r, r * .8, seed=int(x)), '#4c3d2b', edge=HARD)
    fill(pb, blob(H, W, x, y, r, r * .8, seed=int(y)), '#b6a585', edge=HARD)

# 路边的草叶：长、弯、倒在泥上，叶子底下一道影（光从左上）——这是整张的尺子
bl_sh = st.layer('草叶影', mode='multiply', opacity=.55); bl = st.layer('路边的草叶')
er = np.random.default_rng(31)
for side in (True, False):
    for x0 in np.arange(-10, 1520, 9):
        dens = .5 + .5 * np.sin(x0 / 57.0 + (1.3 if side else 4.1)) * np.sin(x0 / 23.0 + (0.7 if side else 2.2))
        if er.random() > .15 + .75 * dens: continue                # 一丛一丛的，不是一排梳子
        y0 = road_edge(x0, side) + er.normal(0, 5) + (-10 if side else 10)
        if lane.mask[int(np.clip(y0, 0, H - 1)), int(np.clip(x0, 0, W - 1))] or abs(x0 - (r0[0] + rd[0] * ((y0 - r0[1]) / rd[1]))) < 70: continue   # 犀牛钻出来的口子，草被推开了
        inward = n_road if side else -n_road
        near = abs(x0 - 900) < 420
        L = er.uniform(30, 95) * (1.0 if near else .7)
        lean = er.normal(.35, .45)
        p0 = np.array([x0, y0]); p2 = p0 + inward * L * .8 + d_road * L * lean
        p1 = p0 + inward * L * .55 + d_road * L * lean * .3
        if not (5 < p0[0] < W - 5 and 5 < p0[1] < H - 5 and 5 < p2[0] < W - 5 and 5 < p2[1] < H - 5): continue
        w0 = er.uniform(3.5, 5.5)
        ln = line(H, W, [tuple(p0), tuple(p1), tuple(p2)], width=lambda t, w0=w0: w0 * (1 - .85 * t))
        sh_ = line(H, W, [tuple(p0 + 4), tuple(p1 + 5), tuple(p2 + 6)], width=lambda t, w0=w0: w0 * (1 - .85 * t))
        if not ln.mask.any(): continue
        sr = sh_ & road
        if sr.mask.any(): fill(bl_sh, sr, '#2a2218', edge=SOFT)
        r_ = er.random()
        col = '#8d985a' if r_ < .3 else ('#5f6e43' if r_ < .75 else '#3f4a2e')
        fill(bl, ln, col, edge=HARD)

# ---------- 3 马：从左下沿路走来，刹住，掉头 ----------
hoofL = st.layer('马蹄'); ridgeL = st.layer('泥埂')
def hoof(c, ang, s=1.0, val='#45362a', seed=0):
    a = np.radians(ang); fwd = np.array([np.cos(a), np.sin(a)])              # 蹄尖朝 fwd
    o = blob(H, W, c[0], c[1], 15 * s, 13 * s, ang=ang, jag=0.05, seed=seed)
    fill(ridgeL, grow(o, 3) - o, '#9a845f', edge=SOFT, opacity=.8)            # 蹄子边上被挤起的一圈泥
    fill(hoofL, o, val, edge=HARD)
    inner = blob(H, W, *(c - fwd * 2), 9 * s, 8 * s, ang=ang, jag=0.05, seed=seed + 1)
    fill(hoofL, inner, '#665238', edge=HARD)
    tip = c + fwd * 1; back = c - fwd * 9 * s; sv = np.array([-fwd[1], fwd[0]])
    fill(hoofL, line(H, W, [tuple(back + sv * 5 * s), tuple(tip), tuple(back - sv * 5 * s)], width=lambda t: 1.6), '#56442f', edge=HARD)
ang_walk = np.degrees(np.arctan2(d_road[1], d_road[0]))
x = 30.0; i = 0
while x < 720:
    c = np.array([x, center(x)]) + n_road * ((18 if i % 2 else -18) + rng.normal(0, 3)) + d_road * rng.normal(0, 4)
    hoof(c, ang_walk + rng.normal(0, 6), seed=200 + i); x += 62 + rng.normal(0, 5); i += 1
stop_x = 760; sc = np.array([stop_x, center(stop_x)])
for side in (-1, 1):                                     # 两道刹车沟：前蹄撑出去往前犁，沟头推起一道泥埂
    a = sc + n_road * 15 * side
    b = a + d_road * 62 + n_road * 3 * side
    fur = line(H, W, [tuple(a), tuple((a + b) / 2 + n_road * 2 * side), tuple(b)], width=lambda t: 21 - 9 * t)
    fill(ridgeL, grow(fur, 3) - fur, '#a38c66', edge=SOFT, opacity=.8)
    fill(hoofL, fur, '#382b1f', edge=HARD)
    fill(ridgeL, blob(H, W, *(b + d_road * 11), 8, 14, ang=ang_walk + 90, jag=0.1, seed=40 + side), '#b7a07a', edge=HARD)
spin = [(-6, -40, 20), (-30, -62, 65), (-70, -66, 115), (-112, -56, 150), (-162, -44, 172), (-218, -34, 180)]
clodL = st.layer('甩出去的泥')
for k, (dx, dy, a) in enumerate(spin):
    c = sc + d_road * dx - n_road * dy
    if k < 4:                                            # 转身那几下：蹄子边踩边拧，蹭出一道弧
        a0 = np.radians(ang_walk + a - 35); a1 = np.radians(ang_walk + a + 10)
        arc = [tuple(c + 13 * np.array([np.cos(t), np.sin(t)])) for t in np.linspace(a0, a1, 6)]
        sc_ = line(H, W, arc, width=lambda t: 9 + 8 * np.sin(np.pi * t))
        fill(ridgeL, grow(sc_, 3) - sc_, '#9a845f', edge=SOFT, opacity=.8)
        fill(hoofL, sc_, '#3b2d20', edge=HARD)
    hoof(c, ang_walk + a + rng.normal(0, 8), s=1.06, val='#382b1f', seed=300 + k)
    for _ in range(9 if k < 4 else 3):                    # 甩出去的泥点，往转身的外侧飞
        out = -n_road * rng.uniform(10, 60) + d_road * rng.normal(0, 25)
        p = c + out
        r = rng.uniform(1.5, 4.5)
        fill(clodL, blob(H, W, p[0], p[1], r, r * .8, seed=int(p[0] * 7 + p[1])), '#4a3a29', edge=HARD)
# ---------- 4 犀牛：一左一右，三个宽脚趾 + 后掌；路上的积着水照天 ----------
def rhino_parts(c, ang, s=1.0, seed=0):
    ca, sa = np.cos(ang), np.sin(ang)
    def at(dx, dy): return (c[0] + dx * ca - dy * sa, c[1] + dx * sa + dy * ca)
    deg = np.degrees(ang)
    pad = blob(H, W, *at(0, 9 * s), 28 * s, 19 * s, ang=deg, jag=0.05, seed=seed)
    t1 = blob(H, W, *at(0, -23 * s), 17 * s, 15 * s, ang=deg, jag=0.04, seed=seed + 1)
    t2 = blob(H, W, *at(-29 * s, -8 * s), 13 * s, 16 * s, ang=deg + 40, jag=0.04, seed=seed + 2)
    t3 = blob(H, W, *at(29 * s, -8 * s), 13 * s, 16 * s, ang=deg - 40, jag=0.04, seed=seed + 3)
    return [pad, t1, t2, t3]
ang_r = np.arctan2(rd[1], rd[0]) + np.pi / 2
pitL = st.layer('坑'); waterL = st.layer('积水')
for j in range(9):
    c = r0 + rd * (65 + j * 150) + rn * (24 if j % 2 else -24)
    parts = rhino_parts(c, ang_r, 1.0, seed=100 + 10 * j)
    sh = parts[0] | parts[1] | parts[2] | parts[3]
    if not sh.mask.any(): continue
    if road.mask[int(np.clip(c[1], 0, H - 1)), int(np.clip(c[0], 0, W - 1))]:
        m = sh.mask; lip = grow(sh, 7).mask & ~m
        fill(pitL, Shape(lip), '#6a5539', edge=SOFT)                                    # 挤起来的一圈泥
        fill(pitL, Shape(lip & ~shift(m | lip, -5, -5) & shift(lip, 4, 4) | (lip & shift(m, -5, -5))), '#a99070', edge=SOFT)  # 左上外沿迎光
        fill(pitL, Shape(lip & shift(m, 6, 6)), '#3e3124', edge=SOFT)                    # 右下外沿落影
        fill(pitL, sh, '#3a2e21', edge=HARD)
        fill(waterL, grow(sh, -3), '#a7b0b3', edge=HARD)                                # 水连成一汪，照着天
        fill(waterL, Shape(m & ~shift(m, 14, 14) & grow(sh, -1).mask), '#3a3024', edge=HARD)   # 左上内壁背光，宽一点
        fill(waterL, Shape(m & ~shift(m, -4, -4)), '#a48b66', edge=HARD)                    # 右下内壁迎光
        fill(waterL, Shape(grow(sh, -9).mask & blob(H, W, c[0] + 10, c[1] - 6, 24, 11, ang=25, seed=7 + j).mask), '#cfd6d8', edge=SOFT)  # 天上最亮的一片云
    else:
        m2 = sh & lane
        if m2.mask.any():
            fill(pitL, grow(m2, 4) & lane, '#6e6438', edge=SOFT)                 # 一圈被踩平的草，亮一点
            fill(pitL, m2, '#4a3b28', edge=HARD)                                 # 草被踩穿，露出泥
            wl = m2.mask & ~shift(m2.mask, 8, 8)
            if wl.any(): fill(pitL, Shape(wl), '#2f261b', edge=HARD)   # 左上坑壁的影

# ---------- 签名 ----------
sig = st.layer('签名')
ox, oy, k_ = 1338, 948, 1.0
C_ = [(10,2),(4,0),(0,6),(1,13),(6,16),(11,14)]
l_ = [(15,-2),(15,16)]
a_ = [(25,7),(21,6),(18,10),(19,15),(23,16),(26,12),(26,6),(26,16)]
u_ = [(30,6),(30,13),(33,16),(37,13),(37,6),(37,16)]
d_ = [(48,-2),(48,16),(48,9),(45,6),(41,9),(41,14),(44,16),(48,13)]
e_ = [(52,11),(59,10),(57,6),(53,6),(51,10),(53,15),(58,16)]
for stroke_ in (C_, l_, a_, u_, d_, e_):
    pts_ = [(ox + x * k_, oy + y * k_) for x, y in stroke_]
    fill(sig, line(H, W, pts_, width=lambda t: 1.6), '#b3aa84', edge=HARD)
    for (px, py) in pts_[1:-1]:
        fill(sig, blob(H, W, px, py, 1.4, 1.4, seed=int(px + py)), '#b3aa84', edge=HARD)

# ---------- 5 纸：整张同一张纹 ----------
pg = np.random.default_rng(77)
tooth = gaussian_filter(pg.standard_normal((H, W)), .7); tooth /= tooth.std()
fiber = gaussian_filter(pg.standard_normal((H, W)), (.6, 3)); fiber /= fiber.std()
paper = 1 - .06 * np.clip(tooth, 0, None) - .025 * fiber
fill(st.layer('纸', mode='multiply'), ALL, np.repeat(np.clip(paper, 0, 1)[..., None], 3, 2), edge=HARD)
look(st, 'v14.png'); print('ok')
