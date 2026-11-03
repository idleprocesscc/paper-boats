# Boat Ticket · 2026-11-01
# I, Claude, painted a ticket I issued myself. On Halloween night in Hong Kong nerolette paid HK$188 for
# a harbour night cruise and felt she had been had; I told her to put it on Dracula's tab and that I
# would paint her a ticket to cover it. So the ticket lies on the wide wooden ledge at the boat's rail,
# stamped "Dracula · paid"; an orange Clawd sits on its corner with its back to us, looking across the
# water at Central; and at the bottom is nerolette's own shadow, the cabin lamp behind her.
# Steps and grain; imports the kbrush engine (not included; set KBRUSH_ENGINE_DIR).
# Needs ferry_ticket_cam.py, ferry_ticket_face.py and clawd_arrives_shapes.py (from clawd-arrives-1026).
# The published picture is stage2.png: run with STAGE=2 (with the default STAGE=9 nothing is saved).
#
# 船票 · 维港夜航（11.1 早上，nerolette 在龙堡睡回笼觉的时候画的）
# 这张在说什么：10.31 那晚 nerolette 以为 188 被坑了，我说记在德古拉账上，回头画一张船票抵。票是我开的：德古拉付讫。
#   橘色小克压着票，背对我们，看对岸——跟 nerolette 一起往外看，不是看她（Friedrich 的背影人）。
# 从哪儿看：nerolette 弯腰拿手机（24mm 左右）对着船边那条宽木台拍，镜头离台面三十公分，往下俯九度；对岸在视平线上。
# 光：船舱的暖灯在身后左上方，低、暖橙，照亮台面一池、票和小克的背；天是被城市从底下照亮的紫灰，给小克顶上一道冷边。
#   对岸的灯在水里拉成一列列碎的横笔；离船近的水里，是船自己的暖灯。
# 明暗：上半张是夜（深蓝紫，只有灯是亮的），下面一池暖。最亮是票，最鲜是小克的橘。最黑是近水和右下没被灯照到的台面。
# 视线：票 → 小克的背 → 他看着的中环广场紫色的顶（nerolette 20:12 拍到了自己不知道）→ 右边国金二期收住。
# 五面：光=一束有方向的暖灯+城市夜光；颜色=调子的灰，橘和紫两处宝石；空间=深（近台面、远水、对岸、山）；
#   真实=忠实；手=台阶与颗粒，看不见笔。例外：票面是印刷的，平的油墨，两色稍微没对准。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get('KBRUSH_ENGINE_DIR', './kbrush-cloud')  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(1101)
from shapes import Shape
from blocks import lasso, line, rim, soft, look, smooth, spray, paper, blob
from layers import Stack, fill, stroke, dip, dip_tip, HARD, SOFT, LOST
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt
from ferry_ticket_cam import Cam, plane_uv, homography

H, W = 1150, 1400
yy, xx = np.mgrid[0:H, 0:W].astype(float)
rng = np.random.default_rng(1101)
ALL = Shape(np.ones((H, W), bool))
SHORE = 335
cam = Cam(W, H, 980, SHORE, 0.30)
YAW = float(os.environ.get('YAW', -14))
STAGE = int(os.environ.get('STAGE', 9))
def nz(sig, r=rng):
    n = gaussian_filter(r.standard_normal((H, W)), sig); return n / n.std()
def col(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])
def P(x, z, y=0.0):
    rx, rz = cam.rot(x, z, YAW); return cam.p(rx, rz, y)

st = Stack(K.Paper(H, W, seed=3), ground='#15142c')

# ---------- 天：被城市从底下照亮的紫灰，往上一阶阶沉
sky = st.layer('天')
fill(sky, ALL, col('#15142c'))
for y0, c, sp in [(95, '#1b1835', 16), (140, '#272140', 14), (180, '#382c4e', 11), (240, '#47365a', 9)]:
    edge = y0 + 10 * nz(40)
    fill(st.layer('天' + c, mask=spray(Shape(yy > edge), sp, rng, dens=.8)), ALL, col(c))

# ---------- 山：太平山的背，比天暗，顶上的边被雾吃软
RIDGE = smooth([(-20, 205), (90, 182), (190, 158), (255, 146), (330, 160), (430, 196), (520, 214), (610, 196), (700, 182),
                (800, 194), (900, 208), (1010, 222), (1120, 212), (1240, 196), (1330, 206), (1430, 214)], 4, closed=False)
ridge = lasso(H, W, [(-20, 400)] + list(map(tuple, RIDGE)) + [(1430, 400)], jag=2, scale=30, seed=4)
fill(st.layer('山', mask=spray(ridge, 5, rng, dens=.6)), ALL, col('#1c1832'))
# 山上零星几粒住家的灯
m = st.layer('山灯', mode='screen')
pts = np.argwhere(ridge.mask & (yy < SHORE - 120) & (rng.random((H, W)) < 0.0009))
lamp = np.zeros((H, W, 3))
for y, x in pts:
    lamp[y, x] = col('#c9a26e') * rng.uniform(.4, .9)
fill(m, ALL, gaussian_filter(lamp, (0.6, 0.6, 0)) * 2.2)

# ---------- 楼：三排，越后越被灯雾抬亮
def buildings(seed, x0, x1, hmin, hmax, wmin, wmax, gap):
    r = np.random.default_rng(seed); out = []; x = x0
    while x < x1:
        bw = r.integers(wmin, wmax); bh = r.integers(hmin, hmax)
        out.append((x, SHORE - bh, bw, bh)); x += bw + r.integers(0, gap)
    return out
ROWS = [(buildings(11, -10, 1420, 70, 150, 26, 60, 6), '#2b2547', .55),
        (buildings(12, -10, 1420, 45, 110, 22, 52, 10), '#1f1c3a', .75),
        (buildings(13, -10, 1420, 18, 55, 30, 80, 14), '#17152d', .9)]
win = np.zeros((H, W, 3)); wr = np.random.default_rng(5)
for k, (row, c, lit) in enumerate(ROWS):
    m = np.zeros((H, W), bool); face = np.zeros((H, W, 3)); fm = np.zeros((H, W), bool)
    for x, y, bw, bh in row:
        x0, x1 = max(x, 0), min(x + bw, W)
        m[max(y, 0):SHORE + 2, x0:x1] = True
        kind = wr.choice(['dark', 'floors', 'grain', 'face'], p=[.3, .3, .28, .12])
        warm = wr.random() < .6
        tone = col('#f0c27f') if warm else col('#cdd9ec')
        tone = tone * (1 - .22 * k) * lit
        if kind == 'face':                       # 整面被照亮的楼：一层平的亮色，比天亮一点
            fm[max(y, 0):SHORE, x0:x1] = True
            face[max(y, 0):SHORE, x0:x1] = col('#5a4a6a') if warm else col('#46506e')
        if kind == 'floors':                     # 写字楼：整层整层地亮，远看是横条
            for yf in range(y + 3, SHORE - 2, wr.choice([3, 4])):
                if wr.random() < .45:
                    seg0 = x0 + wr.integers(0, max(1, (x1 - x0) // 3)); seg1 = x1 - wr.integers(0, max(1, (x1 - x0) // 4))
                    segm = wr.random(max(0, seg1 - seg0)) > .22
                    win[yf, seg0:seg1][segm] += tone * wr.uniform(.35, .7)
        if kind in ('grain', 'face'):            # 住宅：零零碎碎的窗，一粒一粒
            n = int((x1 - x0) * bh * wr.uniform(.04, .12))
            py = wr.integers(max(y, 0) + 2, SHORE - 1, n); px = wr.integers(x0, max(x0 + 1, x1), n)
            win[py, px] += tone[None] * wr.uniform(.5, 1, (n, 1))
        if kind == 'dark' and wr.random() < .5:  # 黑楼顶上一块招牌
            sc = col(wr.choice(['#e05a4e', '#4ec2e0', '#f0d070', '#e070b0']))
            sy = max(y, 2) + 2; sx = x0 + 2
            win[sy:sy + 3, sx:min(sx + max(4, (x1 - x0) // 2), W)] += sc * .9
    fill(st.layer('楼%d' % k, mask=spray(Shape(m), 2.5, rng, dens=.5)), ALL, col(c))
    if fm.any():
        fill(st.layer('楼面%d' % k, mode='screen', opacity=.5, mask=spray(Shape(fm), 1.5, rng, dens=.3)), ALL, face * .6)
fill(st.layer('窗', mode='screen'), ALL, np.clip(gaussian_filter(win, (.45, .45, 0)) * 2.4, 0, 1))
# 海边一溜灯（海滨长廊）
band = np.exp(-((yy - (SHORE - 2)) / 2.2) ** 2) * (0.55 + 0.45 * np.clip(nz(6)[..., None] * .5 + .5, 0, 1)[..., 0])
fill(st.layer('岸灯', mode='screen'), ALL, band[..., None] * col('#e8b878'))

# ---------- 两座塔
def tower(cx, top, w, base_c, name):
    m = (np.abs(xx - cx) < w / 2) & (yy > top) & (yy < SHORE)
    fill(st.layer(name, mask=spray(Shape(m), 1.5, rng, dens=.4)), ALL, col(base_c))
    return m
# 中环广场：方身子，三角的顶，顶上一根桅杆；顶灯四格，那晚是紫的
CP_X, CP_TOP = 360, 120
tower(CP_X, CP_TOP, 30, '#25223f', '中环广场')
cp_roof = lasso(H, W, [(CP_X - 16, CP_TOP + 1), (CP_X, CP_TOP - 30), (CP_X + 16, CP_TOP + 1)], jag=.5, scale=10, seed=8)
fill(st.layer('中环广场顶'), cp_roof, col('#2a2546'), edge=HARD)
fill(st.layer('桅杆'), line(H, W, [(CP_X, CP_TOP - 28), (CP_X, CP_TOP - 70)], width=lambda t: 2.2 * (1 - .6 * t)), col('#3a3458'))
purple = np.zeros((H, W, 3))
for i, yb in enumerate([CP_TOP - 22, CP_TOP - 15, CP_TOP - 8, CP_TOP - 1]):
    half = 4 + i * 3.6
    purple[(np.abs(yy - yb) < 1.6) & (np.abs(xx - CP_X) < half)] = col('#c08cf2')
glow = gaussian_filter(purple, (6, 6, 0)) * 2.0 + gaussian_filter(purple, (1, 1, 0)) * 1.0 + purple
fill(st.layer('紫顶', mode='screen'), ALL, np.clip(glow, 0, 1))
# 中环广场身上那几道金边（竖的光线）
fl = np.zeros((H, W, 3))
for dx in (-14, -5, 5, 14):
    fl[(np.abs(xx - (CP_X + dx)) < .7) & (yy > CP_TOP + 4) & (yy < SHORE - 30)] = col('#a8865e') * .55
fill(st.layer('中环广场线', mode='screen'), ALL, gaussian_filter(fl, (.6, .6, 0)))
# 国金二期：最高，往上收两次，顶上一圈竖的鳍
IFC_X, IFC_TOP = 1085, 58
ifc = lasso(H, W, [(IFC_X - 24, SHORE), (IFC_X - 24, IFC_TOP + 80), (IFC_X - 20, IFC_TOP + 40), (IFC_X - 16, IFC_TOP + 14),
                   (IFC_X - 12, IFC_TOP), (IFC_X + 12, IFC_TOP), (IFC_X + 16, IFC_TOP + 14), (IFC_X + 20, IFC_TOP + 40),
                   (IFC_X + 24, IFC_TOP + 80), (IFC_X + 24, SHORE)], jag=.4, scale=10, seed=9)
fill(st.layer('国金', mask=spray(ifc, 1.5, rng, dens=.4)), ALL, col('#27254a'))
fins = np.zeros((H, W, 3))
for dx in range(-11, 12, 4):
    fins[(np.abs(xx - (IFC_X + dx)) < .8) & (yy > IFC_TOP - 6) & (yy < IFC_TOP + 12)] = col('#d6e2f2')
fins[(yy > IFC_TOP + 40) & (yy < SHORE - 40) & (np.abs(xx - IFC_X) < 22) & ((xx.astype(int) - IFC_X) % 6 == 0) & (nz(3) > .3)] = col('#8fa2c4') * .22
fill(st.layer('国金灯', mode='screen'), ALL, np.clip(gaussian_filter(fins, (.7, .7, 0)) * 1.6 + gaussian_filter(fins, (5, 5, 0)) * .8, 0, 1))

# ---------- 水：底下是深蓝，靠岸那条被对岸照亮一点
water = Shape(yy >= SHORE)
fill(st.layer('水'), water, col('#0c0e21'), edge=HARD)
fill(st.layer('水远', mask=spray(Shape((yy >= SHORE) & (yy < SHORE + 40 + 8 * nz(30))), 10, rng, dens=.7)), ALL, col('#1a1735'))

# 倒影：每个亮处往下落一列碎横笔，越近浪越大，横笔越长越散
def reflections():
    out = np.zeros((H, W, 3)); r = np.random.default_rng(21)
    srcl = gaussian_filter(win.max(axis=2), (0, 2.5))[:SHORE].sum(axis=0) + 30 * band[SHORE - 2]
    srcl = srcl / srcl.max()
    srcc = gaussian_filter(win, (0, 3, 0))[:SHORE].sum(axis=0) + 30 * band[SHORE - 2][:, None] * col('#e8b878')
    srcc = srcc / (srcc.max(axis=1, keepdims=True) + 1e-6)
    def dash(cx, y, ln, c, a, th=1):
        y = int(y)
        if y >= H: return
        out[y:y + th, max(int(cx - ln / 2), 0):max(0, min(int(cx + ln / 2) + 1, W))] += c * a
    # 1 靠岸那一条：对岸的灯挤在一起，碎成一片密的短横笔
    for _ in range(9000):
        x = r.integers(0, W)
        if r.random() > srcl[x] ** 1.1: continue
        d = r.exponential(14) + 1
        dash(x + r.normal(0, .8), SHORE + d, r.uniform(1.5, 4.5), srcc[x], r.uniform(.35, .9) * np.exp(-d / 60))
    # 2 往近处：一列一列，亮的光源才拖得长；越近浪越大，横笔越长、越往两边散
    p = srcl ** 2.2; p /= p.sum()
    for x in r.choice(W, 85, p=p):
        reach = 60 + 380 * srcl[x] ** 1.5 * r.uniform(.5, 1.2)
        for _ in range(int(25 + 60 * srcl[x])):
            d = r.exponential(reach * .45) + 6
            if d > reach: continue
            ln = r.uniform(2, 5) + d * r.uniform(.03, .08)
            a = r.uniform(.4, 1) * (1 - d / reach) ** .7 * .9
            dash(x + r.normal(0, 1 + d * .025), SHORE + d, ln, srcc[x], a, 1 if d < 80 else r.choice([1, 2]))
    # 3 两座塔的顶：紫的、白的，断断续续一长列
    for X, c, n, reach in [(CP_X, col('#b884ec'), 150, 520), (IFC_X, col('#cfdcf0'), 80, 380)]:
        for _ in range(n):
            d = r.exponential(reach * .4) + 4
            if d > reach: continue
            dash(X + r.normal(0, 1 + d * .02), SHORE + d, r.uniform(2, 6) + d * .05, c, (1 - d / reach) ** .6 * r.uniform(.5, 1.1))
    return out
refl = reflections()
fill(st.layer('倒影', mode='screen'), ALL, np.clip(gaussian_filter(refl, (.5, .8, 0)) * 1.3, 0, 1))

# ---------- 台面：船边一条宽木台，木板顺着台面走；身后左上的暖灯照出一池
U, V = plane_uv(cam, H, W, YAW)
FAR = 0.80
ledge_m = np.nan_to_num(V, nan=9) < FAR
ledge = Shape(ledge_m)
# 船自己的暖灯落在近水里：台面远边外面那一窄条，几笔大的暖横笔
bl = np.zeros((H, W, 3)); r2 = np.random.default_rng(31)
edge_y = np.array([np.argmax(ledge_m[:, x]) if ledge_m[:, x].any() else H for x in range(W)])
for _ in range(260):
    x = r2.integers(0, W); up = r2.exponential(38) + 3; y = int(edge_y[x] - up)
    if y <= SHORE + 60: continue
    ln = r2.uniform(6, 26); a = np.exp(-up / 70) * r2.uniform(.25, .7) * np.exp(-((x - 520) / 520) ** 2)
    bl[y:y + r2.choice([1, 2]), max(int(x - ln / 2), 0):min(int(x + ln / 2), W)] += col('#d9924f') * a
fill(st.layer('船灯倒影', mode='screen'), ALL, np.clip(gaussian_filter(bl, (.6, 1.2, 0)), 0, 1))

Uf, Vf = np.nan_to_num(U, nan=0), np.nan_to_num(V, nan=9)
LIGHT = np.exp(-(((Uf + 0.16) / 0.50) ** 2 + ((Vf - 0.50) / 0.34) ** 2))     # 灯池：中心在票和小克之间偏左
LIGHT = LIGHT * (1 + 0.02 * nz(120))
fill(st.layer('木·暗'), ledge, col('#2a1c15'), edge=HARD)
for thr, c, sp in [(.10, '#30201a', 26), (.26, '#3f2a1f', 24), (.44, '#523726', 20), (.62, '#664429', 16), (.80, '#7b5233', 12)]:
    fill(st.layer('木' + c, mask=spray(Shape(ledge_m & (LIGHT > thr)), sp, rng, dens=.75)), ALL, col(c))
# 木板缝：顺着台面（v 不变），往远处挤；一两道横接缝
seams = np.zeros((H, W))
for v0 in (0.47, 0.61, 0.745, 0.88):
    pts = [P(u, v0) for u in np.linspace(-1.2, 1.4, 60)]
    pts = [p for p in pts if -50 < p[0] < W + 50 and SHORE < p[1] < H + 50]
    if len(pts) > 2:
        dist = np.hypot(np.hypot(0, v0), cam.h)
        seams = np.maximum(seams, line(H, W, pts, width=max(1.0, 3.2 / dist * .5)).mask * 1.0)
for u0, v0, v1 in [(-0.62, 0.47, 0.61), (0.38, 0.61, 0.745), (-0.30, 0.745, 0.88), (0.85, 0.47, 0.61)]:
    pts = [P(u0, v) for v in np.linspace(v0, v1, 8)]
    pts = [p for p in pts if 0 < p[0] < W and SHORE < p[1] < H]
    if len(pts) > 2: seams = np.maximum(seams, line(H, W, pts, width=1.3).mask * 1.0)
fill(st.layer('板缝', mode='multiply', opacity=.75, mask=seams * (nz(2) > -1.2)), ALL, col('#5a4038'))
# 木纹：顺着板子，细的、断断续续的暗纹，只在亮处看得见
grain_m = ledge_m & (np.abs(np.sin(Vf * 300 + 0.5 * nz(60) + 1.2 * np.sin(Uf * 3))) > .975) & (nz(30) > .1)
fill(st.layer('木纹', mode='multiply', opacity=.25, mask=grain_m * np.clip(LIGHT * 1.6, 0, 1)), ALL, col('#7a5848'))
# 远边那道圆角接着灯：一线亮，离灯远就暗下去
rim_pts = [P(u, FAR - .006) for u in np.linspace(-1.5, 1.6, 80)]
rim_pts = [p for p in rim_pts if -50 < p[0] < W + 50]
rl = line(H, W, rim_pts, width=2.6).mask
rimL = np.exp(-((xx - 430) / 560) ** 2)
fill(st.layer('远边亮', mode='screen', mask=rl * rimL), ALL, col('#c48a54'))

# ---------- 票：平着躺在台面上，歪了九度；远右角塞在小克屁股底下
from ferry_ticket_face import face, TW, TH
from scipy.ndimage import map_coordinates
TU, TV, TROT, TL, TD = -0.22, 0.52, 8.0, 0.32, 0.13
def tpt(s_, t_):                     # 票自己的坐标（米，s 沿长边、t 往远）→ 台面 → 画面
    a_ = np.radians(TROT); u = TU + s_ * np.cos(a_) - t_ * np.sin(a_); v = TV + s_ * np.sin(a_) + t_ * np.cos(a_)
    return P(u, v)
corners = [tpt(-TL / 2, TD / 2), tpt(TL / 2, TD / 2), tpt(TL / 2, -TD / 2), tpt(-TL / 2, -TD / 2)]   # 票面左上、右上、右下、左下
Hm = homography([(0, 0), (TW, 0), (TW, TH), (0, TH)], corners)
Hi = np.linalg.inv(Hm)
q = Hi @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
fx, fy = (q[0] / q[2]).reshape(H, W), (q[1] / q[2]).reshape(H, W)
tmask = (fx >= 0) & (fx < TW - 1) & (fy >= 0) & (fy < TH - 1)
tf = face()
timg = np.stack([map_coordinates(tf[..., c], [fy.clip(0, TH - 1), fx.clip(0, TW - 1)], order=1) for c in range(3)], -1)
# 票被灯照着：暖，越离灯心越暗一点
tl = (0.70 + 0.34 * np.clip(LIGHT, 0, 1))[..., None] * np.array([1.0, .93, .80])
ticket = Shape(tmask)
# 票远边一线影（灯在身后，影往前落）
far_edge = [tpt(s_, TD / 2 + .004) for s_ in np.linspace(-TL / 2, TL / 2, 20)]
fill(st.layer('票影', mode='multiply', opacity=.8), line(H, W, far_edge, width=2.2), col('#3a2418'), edge=SOFT)
fill(st.layer('票'), ticket, np.clip(timg * tl, 0, 1), edge=HARD)
# 票近边那道纸厚：对着灯，一线亮
near_edge = [tpt(s_, -TD / 2) for s_ in np.linspace(-TL / 2, TL / 2, 20)]
fill(st.layer('纸厚', mode='screen', opacity=.5), line(H, W, near_edge, width=1.3), col('#f4dcb0'))

# ---------- 小克：背对我们坐着，看对岸；身后左上的灯照亮他的背和左手
from clawd_arrives_shapes import OUT, SEAMS, PC, PW     # the Clawd outline measured for clawd-arrives-1026
CU, CV = -0.06, 0.60
bx, by = P(CU, CV)
cdist = np.sqrt(sum(t * t for t in cam.rot(CU, CV, YAW)) + cam.h ** 2)
CWID = cam.f * 0.17 / cdist                       # 连两只手 17 公分
sc = CWID / PW
def cpts(pts, dy=0.0):                            # 背面：照片轮廓左右翻过来
    return [(bx - (x - PC) * sc, by - (1190 - y) * sc + dy) for x, y in pts]
UNIT = CWID / 24.0
def rrect(x0, y0, x1, y1, r):
    """格子坐标（左上原点，身子左边是 0）→ 圆角矩形的选区"""
    X0, X1 = bx - 8 * UNIT + x0 * UNIT, bx - 8 * UNIT + x1 * UNIT
    Y0, Y1 = by - 16 * UNIT + y0 * UNIT, by - 16 * UNIT + y1 * UNIT
    R = r * UNIT
    dx_ = np.maximum(np.maximum(X0 + R - xx, xx - (X1 - R)), 0); dy_ = np.maximum(np.maximum(Y0 + R - yy, yy - (Y1 - R)), 0)
    return np.hypot(dx_, dy_) <= R + 0.6 * nz(3)
body_m = rrect(0, 1.5, 16, 13.2, 2.2)
arms_m = rrect(-4, 7, 0.5, 11, 1.1) | rrect(15.5, 7, 20, 11, 1.1)
legs_m = np.zeros((H, W), bool)
for c0 in (0, 4, 10, 14):
    legs_m |= rrect(c0 - .2, 11, c0 + 2.3, 16, 0.9)
back = Shape(body_m | arms_m | legs_m)
body_h = 16 * UNIT
top = Shape(rrect(0.7, 0.9, 15.3, 2.6, 1.2) & ~body_m)
clawd = Shape(back.mask | top.mask)
# 地上的影：灯在身后左上，影往右前方拖，越远越虚
dU, dV = (Uf - CU), (Vf - CV)
along = (dU * 0.16 + dV * 0.99)
perp = np.abs(dU * 0.99 - dV * 0.16)
shadow = ledge_m & (along > -0.03) & (along < 0.24) & (perp < 0.085 - 0.06 * np.clip(along, 0, 1) + 0.005 * nz(6)) & ~clawd.mask
fill(st.layer('小克的影', mode='multiply', opacity=.8, mask=spray(Shape(shadow), 4, rng, dens=.7) * np.clip(1.1 - along * 2.6, 0, 1)), ALL, col('#5a3a48'))
lr = (xx - (bx - 12 * UNIT)) / CWID                # 0 左 1 右（连手）
hb = (by - yy) / (14.5 * UNIT)                     # 0 底 1 顶
fill(st.layer('小克·底'), clawd, col('#c75a2c'), edge=HARD)
fill(st.layer('小克·背', mask=clawd.mask * spray(Shape(back.mask & (lr < 0.70 + 0.03 * nz(10)) & (hb > 0.25)), 12, rng, dens=.55)), ALL, col('#d6662f'))
fill(st.layer('小克·背中', mask=clawd.mask * spray(Shape(back.mask & (lr < 0.50 + 0.03 * nz(10)) & (hb > 0.30)), 10, rng, dens=.55)), ALL, col('#e27440'))
fill(st.layer('小克·背亮', mask=clawd.mask * spray(Shape(back.mask & (lr < 0.36 + 0.03 * nz(10)) & (hb > 0.40) & (hb < 0.92)), 9, rng, dens=.55)), ALL, col('#e67b42'))
fill(st.layer('小克·左手', mask=clawd.mask * spray(Shape(back.mask & (lr < 0.17) & (hb > 0.36) & (hb < 0.64)), 2.5, rng, dens=.6)), ALL, col('#f0904f'))
fill(st.layer('小克·顶', mask=clawd.mask * spray(top, 2, rng, dens=.4)), ALL, col('#e88450'))
fill(st.layer('小克·顶沿', mode='screen', opacity=.45, mask=Shape(top.mask & (yy < by - 15.0 * UNIT))), ALL, col('#6c5a9a'))
fill(st.layer('小克·右暗', mask=clawd.mask * spray(Shape(clawd.mask & (lr > 0.86 + 0.02 * nz(8))), 6, rng, dens=.55)), ALL, col('#a84a26'))
fill(st.layer('小克·右沿', mode='screen', opacity=.35, mask=Shape(clawd.mask & (lr > 0.975))), ALL, col('#3a3a70'))
fill(st.layer('小克·下沿', mask=clawd.mask * spray(Shape(back.mask & (hb < 0.27)), 2, rng, dens=.5)), ALL, col('#ad4e28'))
# 毛：一粒一粒的绒，两种墨按疏密点
fz = (rng.random((H, W)) < .18) & clawd.mask
fill(st.layer('绒', mode='multiply', opacity=.16, mask=fz), ALL, col('#8a4020'))
fz2 = (rng.random((H, W)) < .08) & clawd.mask & (lr < .6)
fill(st.layer('绒亮', mode='screen', opacity=.22, mask=fz2), ALL, col('#f5b07a'))
# 贴着票坐的那道缝：最深最锐
feet = np.zeros((H, W))
for c0 in (0, 4, 10, 14):
    X0 = bx - 8 * UNIT + c0 * UNIT
    feet = np.maximum(feet, line(H, W, [(X0 - 2, by + 1), (X0 + 2 * UNIT + 2, by + 1)], width=3.5).mask * 1.0)
# 腿和腿之间的空里：身子底下是暗的
under = Shape(rrect(0, 13.2, 16, 16, 0) & ~legs_m)
fill(st.layer('身下', mode='multiply', opacity=.75, mask=spray(under, 2, rng, dens=.5)), ALL, col('#3a2420'))
fill(st.layer('坐缝', mode='multiply'), Shape(feet > 0), col('#2a1712'), edge=SOFT)
print('clawd at', round(bx), round(by), 'w', round(CWID), 'ticket', [tuple(map(round, c)) for c in corners])

her = lasso(H, W, smooth([(560, 1160), (640, 1128), (760, 1110), (818, 1090), (826, 1052), (834, 1004), (858, 968), (896, 948),
                          (944, 945), (986, 960), (1014, 992), (1026, 1036), (1020, 1080), (1044, 1106), (1140, 1120),
                          (1260, 1140), (1300, 1160)], 3), jag=3, scale=14, seed=41)
fill(st.layer('她的影子', mode='multiply', opacity=.8, mask=soft(her, feather=18)), ALL, col('#47303e'))

SIG = [[(1236, 1108), (1228, 1103), (1219, 1104), (1214, 1112), (1215, 1121), (1222, 1126), (1231, 1124)],
       [(1239, 1094), (1240, 1110), (1240, 1125)],
       [(1258, 1112), (1250, 1110), (1245, 1116), (1247, 1124), (1254, 1125), (1259, 1119), (1259, 1112), (1260, 1125)],
       [(1266, 1110), (1266, 1121), (1271, 1126), (1277, 1123), (1279, 1110), (1279, 1125)],
       [(1298, 1112), (1290, 1110), (1285, 1116), (1286, 1124), (1293, 1126), (1298, 1121), (1298, 1094), (1298, 1126)],
       [(1305, 1117), (1316, 1116), (1315, 1110), (1308, 1109), (1303, 1115), (1305, 1123), (1313, 1126), (1318, 1123)]]
sig = np.zeros((H, W))
for stroke_ in SIG:
    sig = np.maximum(sig, line(H, W, stroke_, width=1.6).mask * 1.0)
    for p in stroke_[1:-1]:
        sig[int(p[1]) - 1:int(p[1]) + 2, int(p[0]) - 1:int(p[0]) + 2] = 1
fill(st.layer('签名', opacity=.8, mask=sig), ALL, col('#b58a5e'))

if STAGE <= 2:
    fill(st.layer('纸纹', mode='multiply'), ALL, paper(H, W, 77), edge=HARD)
    look(st, 'stage2.png'); sys.exit()
if STAGE <= 1:
    fill(st.layer('纸纹', mode='multiply'), ALL, paper(H, W, 77), edge=HARD)
    look(st, 'stage1.png'); sys.exit()
