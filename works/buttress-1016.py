# 《垫脚》 · 10.16
# 从哪儿看：眼睛贴着海面，一半在水上一半在水下（水下摄影那种上下各半的镜头），侧着看夏威夷大岛 Kīlauea 南坡。
#   唯一一处不老实：水清得能一眼看到几千米深的海底。其余照实：Kīlauea 是很平的盾，露出水面只有一条低低的线，
#   山的大半截在水下；水下的坡比陆上陡。竖向夸张了。
# 这张在说什么：一两万到五万年前南坡塌过一大块，塌下去的碎石在海底堆成一道台子，反过来把剩下的山撑住了。
#   塌下去的那部分撑着没塌的那部分。坡上留着一块挖空的疤，坡脚是从疤里掉下来的那一堆——同一块东西，一处是空，一处是实。
# 光：太阳在左上框外，清晨，低，暖。水上：天从顶上的灰蓝到地平线的暖粉；岛背光，是一条暗的剪影，顶边一线亮。
#   水下：光从水面斜着往右下打进来，越往下越蓝越暗，光柱到中段就散了。没有别的光源。
#   往下每一层是往深海的颜色挪一阶；坡和碎块只有朝上、朝左上的面接到一点残光。
# 明暗：上面一小截最亮（天），往下一阶阶暗到底。水下最亮的是靠水面那一条；下半截最"亮"的是碎块堆顶面那几道冷蓝的棱——下半截的焦点。
# 视线：先落在水线（最亮的天贴着最亮的水），顺着光柱斜下去，经过坡上那块疤，落到坡脚碎块顶着坡的那一处。
# 构图：岸在右上，水下山坡一条长斜线从右上走到左下；坡脚埋进一堆碎块，近坡的几大块还整着、层理歪了，
#   越往左越碎越小，摊到框外。左边一条小船骑在水线上，一半船身在水下，是尺子（尺子本身也不老实：船这么大，海就没这么深）。
# 手：台阶与颗粒，纸纹最后罩。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get('KBRUSH_ENGINE_DIR', './kbrush-cloud')  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(16)
from shapes import Shape
from blocks import lasso, line, soft, look, smooth, spray, paper
from layers import Stack, fill, HARD, SOFT, LOST
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt

OUT = os.environ.get('OUT', 'final.png')
H, W = 1200, 900
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
rng = np.random.default_rng(16)
def hexc(h): h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])

WL = 264.                                    # 水线（也是地平线，眼睛就在这儿）
st = Stack(K.Paper(H, W, seed=3), ground='#8d9fbf')

# ---------- 1 天：四阶，顶上灰蓝到地平线暖粉，阶边喷开
sky = st.layer('天')
fill(sky, ALL, '#8d9fbf')
SKY = [('#aaaec5', 104), ('#ccb8be', 168), ('#e6c4b4', 214), ('#f2d6bd', 246)]
for i, (c, y0) in enumerate(SKY):
    n1 = gaussian_filter(rng.standard_normal(W), 18); n1 /= n1.std()
    edge_y = y0 + (2.5 + i) * n1[None, :] + 1.2 * np.sin(xx / (53 + 17 * i) + i)
    m = spray(yy > edge_y, 8 + 3 * i, rng, dens=.55)
    fill(st.layer(f'天{i}', mask=m), ALL, c)

# 蒸汽：岛上一缕白，往左飘，边全丢
plume_pts = [(846, 214), (850, 186), (852, 160), (846, 132), (832, 108), (808, 88), (776, 74), (744, 68), (724, 70),
             (746, 80), (774, 92), (800, 110), (818, 132), (828, 160), (830, 188), (832, 214)]
plume = lasso(H, W, smooth(plume_pts, 2), jag=4, scale=6, seed=4)
fade = np.clip((yy - 60) / 160, .25, 1)
fill(st.layer('蒸汽', mask=spray(plume, 9, rng, solid=.78, dens=.5) * fade), ALL, '#f6e9e2')
fill(st.layer('蒸汽暗面', mode='multiply', mask=spray(plume & Shape(xx > 838 - (214 - yy) * .35), 5, rng, solid=.45) * fade), ALL, '#d4bec4')

# ---------- 2 岛：背光的低剪影，海边一截崖
ISL = [(640, WL + 2), (646, 242), (660, 236), (700, 232), (760, 226), (820, 219), (860, 214), (905, 210), (905, WL + 2)]
isl = lasso(H, W, ISL, jag=1.5, scale=3, seed=5)
fill(st.layer('岛'), isl, '#433848', edge=HARD)
isl_top = isl.mask & ~np.roll(isl.mask, 3, 0)
fill(st.layer('岛顶的光', mask=spray(Shape(isl_top & (xx > 650)), 2, rng, solid=.8, dens=.4)), ALL, '#8a6d70')
cliff = isl.mask & (xx < 652)
fill(st.layer('崖面', mask=cliff.astype(float) * .6), ALL, '#2c2534')

# ---------- 3 水下：五阶，往下往深海的颜色挪
WATER = [('#5f9ba4', WL), ('#3f7c8d', 392), ('#2b5c75', 548), ('#1c3d59', 748), ('#10243a', 985)]
for i, (c, y0) in enumerate(WATER):
    if i == 0:
        fill(st.layer('水0'), Shape(yy > WL), c, edge=HARD); continue
    n1 = gaussian_filter(rng.standard_normal(W), 30); n1 /= n1.std()
    edge_y = y0 + 12 * n1[None, :] + 3 * np.sin(xx / 41 + i)
    m = spray(yy > edge_y, 18 + 5 * i, rng, dens=.6)
    fill(st.layer(f'水{i}', mask=m), ALL, c)
# 水面底下那一层晃动的亮（从下往上看水面，光一块块碎在上面）
rn = gaussian_filter(rng.standard_normal(W), 5); rn /= rn.std()
ripple = (yy > WL) & (yy < WL + 18 + 7 * np.abs(rn)[None, :])
ripple &= gaussian_filter(rng.standard_normal((H, W)), (1.2, 6)) > -.2
fill(st.layer('水面底下', mask=spray(Shape(ripple), 4, rng, solid=.75, dens=.5)), ALL, '#93c3c6')

# 光柱：太阳在左上，斜着往右下打进水里，越往下越淡，到中段散了
beam_layer_mask = np.zeros((H, W))
for (x0, w0, ang, ln, a) in [(60, 46, 33, 560, .5), (190, 20, 32, 470, .42), (250, 34, 33, 600, .48)]:
    t = (yy - WL) / ln
    cx = x0 + (yy - WL) * np.tan(np.radians(ang))
    wid = w0 * (1 + .9 * t)
    inb = (np.abs(xx - cx) < wid / 2) & (t > 0) & (t < 1)
    beam_layer_mask = np.maximum(beam_layer_mask, inb * a * np.clip(1 - t, 0, 1) ** 1.3)
beam_layer_mask = gaussian_filter(beam_layer_mask, 3)
fill(st.layer('光柱', mode='screen', mask=beam_layer_mask), ALL, '#8fc2c4')

# ---------- 4 山的水下那截：一条长斜线从右上走到左下；坡上一块挖空的疤
FLANK_LINE = [(652, WL), (628, 318), (598, 372), (566, 428), (540, 466), (530, 482), (525, 520), (514, 566), (494, 606),
              (464, 638), (428, 664), (402, 694), (384, 744), (366, 806), (348, 872), (330, 940), (314, 1000), (300, 1056), (296, 1080)]
fl = np.array(smooth(FLANK_LINE, 2, closed=False))
FLOOR = 1062.
flank = lasso(H, W, list(map(tuple, fl)) + [(910, 1210), (910, WL)], jag=2.2, scale=4, seed=6)
# 海底：一条平的线，下面是沉积，跟山连成一片
nf = gaussian_filter(rng.standard_normal(W), 12); nf /= nf.std()
floor_y = FLOOR + 3 * nf
ground = Shape(yy > floor_y[None, :])
fill(st.layer('海底'), ground, '#0d1d2e', edge=HARD)
fill(st.layer('海底面', mask=spray(Shape(ground.mask & (yy < floor_y[None, :] + 5)), 3, rng, solid=.6, dens=.4)), ALL, '#2a4c60')
flank = flank | ground
# 坡的颜色也一阶阶往深里暗
FL = [('#1b364a', WL), ('#173045', 560), ('#13293e', 800), ('#0f2135', 1040)]
for i, (c, y0) in enumerate(FL):
    n1 = gaussian_filter(rng.standard_normal(W), 25); n1 /= n1.std()
    reg = flank.mask & (yy > y0 + 9 * n1[None, :])
    m = spray(Shape(reg), 10, rng, dens=.5) * flank.mask if i else flank.mask.astype(float)
    fill(st.layer(f'坡{i}', mask=m), ALL, c)
# 坡面朝上、朝光的那一层：沿坡线往里一窄条，越深越暗
dl = edt(flank.mask)
lit = flank.mask & (dl < 7 + 5 * np.sin(yy / 41))
litc = np.clip(1 - (yy - WL) / 900, 0, 1)
fill(st.layer('坡面', mask=spray(Shape(lit), 3, rng, solid=.85, dens=.45) * (.25 + .75 * litc)), ALL, '#6b9aa4')
# 疤：坡上被挖空的那块，崖壁新鲜一点、冷一点；疤底有几道顺坡往下的擦痕
scar_band = (yy > 474) & (yy < 698)
sw = 26 + 22 * np.sin(np.clip((yy - 474) / 224, 0, 1) * np.pi)
scar = flank.mask & scar_band & (dl < sw)
fill(st.layer('疤', mask=spray(Shape(scar), 3, rng, solid=.8, dens=.4)), ALL, '#2f5a6c')
for k, yk in enumerate(range(488, 690, 15)):
    strat = scar & (np.abs(yy - yk - 1.5 * np.sin(xx / 13 + k)) < .9)
    fill(st.layer(f'疤层{k}', mask=strat.astype(float) * .55), ALL, '#1d3a50')
head = flank.mask & (yy > 474) & (yy < 540) & (dl < 4)
fill(st.layer('疤顶', mask=head.astype(float) * .9), ALL, '#7fa9b2')
for k in range(5):
    y0 = 506 + k * 32
    x0 = np.interp(y0, fl[:, 1], fl[:, 0]) + 10
    pts = [(x0 - j * 8, y0 + j * 13) for j in range(4)]
    fill(st.layer(f'擦痕{k}', mask=line(H, W, pts, width=1.2, seed=60 + k).mask * .55), ALL, '#4f7a88')

# ---------- 5 坡脚那一堆：近坡几大块还整着、层理歪了；越往左越碎越小，摊到框外
# 堆：堆顶从坡上那一点往左往下摊；右边就是坡，坡脚往下埋在堆后面
fx = lambda y: np.interp(y, fl[:, 1], fl[:, 0])
CY = 826.; CX = float(fx(CY))                     # 堆顶碰到坡的那一点
heap_top = [(-4, FLOOR + 2), (22, 1046), (52, 1022), (88, 996), (128, 970), (170, 946), (212, 924), (254, 902), (292, 878), (326, 852), (CX - 4, CY + 1)]
heap_right = [(fx(y) + 14, y) for y in range(int(CY) + 6, int(FLOOR) + 8, 16)]
heap = lasso(H, W, smooth(heap_top, 2, closed=False) + heap_right + [(fx(FLOOR) + 8, FLOOR + 10), (-10, FLOOR + 10)], jag=5, scale=6, seed=7)
fill(st.layer('堆'), heap, '#0c1a2a', edge=HARD)
hs = heap.mask.argmax(0).astype(float); hs[~heap.mask.any(0)] = H     # 堆每一列的顶

def block(cx, cy, bw, bh, ang, seed):
    a = np.radians(ang); r = np.random.default_rng(seed)
    n = r.integers(5, 8)
    th = np.sort(r.uniform(0, 2 * np.pi, n)); th += np.linspace(0, .3, n)
    pts = []
    for t in th:
        rad = r.uniform(.78, 1.0)
        u2, v2 = np.cos(t) * bw / 2 * rad, np.sin(t) * bh / 2 * rad
        u2 = np.sign(u2) * min(abs(u2) * 1.25, bw / 2); v2 = np.sign(v2) * min(abs(v2) * 1.25, bh / 2)
        pts.append((cx + u2 * np.cos(a) - v2 * np.sin(a), cy + u2 * np.sin(a) + v2 * np.cos(a)))
    sh = lasso(H, W, pts, jag=1.0, scale=2, seed=seed)
    v = -(xx - cx) * np.sin(a) + (yy - cy) * np.cos(a)
    top_v = np.where(sh.mask, v, np.inf).min() if sh.mask.any() else 0
    return sh, v - top_v - bh / 2

placed = []
# 近坡的大块（还整着，层理歪了）
BIG = [(CX - 30, CY + 40, 92, 56, 19, True), (CX - 112, CY + 78, 80, 50, 13, True), (CX - 42, CY + 120, 88, 56, 22, True),
       (CX - 176, CY + 126, 62, 40, 9, False), (CX - 118, CY + 170, 74, 46, 16, True), (CX - 36, CY + 200, 72, 46, 18, False)]
for i, (cx, cy, bw, bh, ang, strata) in enumerate(BIG):
    placed.append((cx, cy, bw, bh, ang, 200 + i, strata))
# 整个堆都是块：越靠坡越大，越往左、越往下越碎越小（滑坡的大块留在坡脚，碎屑冲得更远）
ys, xs = np.nonzero(heap.mask)
for k in range(330):
    j = rng.integers(len(xs)); x, y = float(xs[j]), float(ys[j])
    near = np.clip(1 - np.hypot(x - CX, y - CY) / 360, 0, 1)
    sz = 4 + 18 * near ** 1.3 * rng.uniform(.6, 1.15)
    if near > .72 and rng.random() < .6: continue          # 大块那一圈留给 BIG
    placed.append((x, y, sz * rng.uniform(1.1, 1.8), sz, rng.uniform(-30, 30) + 10 * near, 300 + k, False))
placed.sort(key=lambda p: (p[5] < 300, p[1]))      # 碎块先画（上面的先、下面的压上来），大块最后压在最前面
depth = np.clip(1 - (yy - 830) / 380, .45, 1)
for (cx, cy, bw, bh, ang, sd, strata) in placed:
    sh, v = block(cx, cy, bw, bh, ang, sd)
    m = sh.mask & heap.mask
    big = sd < 300
    fill(st.layer(f'块{sd}', mask=m.astype(float)), ALL, '#12243a' if big else '#0f1e31')
    side = m & (v > -bh / 2 + bh * .28)
    fill(st.layer(f'块{sd}侧', mask=side.astype(float)), ALL, '#0b1828' if big else '#0a1626')
    topf = m & (v < -bh / 2 + bh * .28)
    fill(st.layer(f'块{sd}顶', mask=spray(Shape(topf), 1.5, rng, solid=.95, dens=.35) * depth * (1 if big else .75)), ALL, '#4d7f90' if big else '#36606f')
    edge_top = m & (v < -bh / 2 + 2.2)
    if big or rng.random() < .35:
        fill(st.layer(f'块{sd}棱', mask=edge_top.astype(float) * depth * (.85 if big else .45)), ALL, '#86b3bb')
    if strata:
        for s_ in (-.14, .06, .24, .4):
            ln_ = m & (np.abs(v - s_ * bh) < 1.0)
            fill(st.layer(f'块{sd}层{s_}', mask=ln_.astype(float) * .6), ALL, '#1d3a50')
# 坡脚埋进堆里那一道缝：最深最锐
up = np.roll(heap.mask, 4, 0)
seam = flank.mask & heap.mask & ~up & (yy > CY - 10) & (yy < CY + 240)
fill(st.layer('缝', mask=seam.astype(float)), ALL, '#04080e')

# ---------- 6 船：骑在水线上，一半在水下
bx, by = 152, WL
hull_up = lasso(H, W, [(bx - 20, by - 5), (bx + 22, by - 5), (bx + 17, by + 1), (bx - 15, by + 1)], jag=.4, seed=8)
hull_dn = lasso(H, W, [(bx - 15, by + 1), (bx + 17, by + 1), (bx + 8, by + 9), (bx - 9, by + 9)], jag=.4, seed=9)
fill(st.layer('船'), hull_up, '#2d2633', edge=HARD)
fill(st.layer('船底'), hull_dn, '#24414f', edge=HARD)
fill(st.layer('桅'), line(H, W, [(bx + 1, by - 5), (bx + 2, by - 34)], width=1.4, seed=10), '#2d2633')
fill(st.layer('船影', mode='multiply', mask=soft(Shape((np.abs(xx - bx - 1) < 14) & (yy > WL + 9) & (yy < WL + 90)), feather=16) * .35), ALL, '#4d7a86')

# ---------- 7 水线：最亮的一线
fill(st.layer('水线', mask=((np.abs(yy - WL) < 1.2) & ~isl.mask).astype(float) * .9), ALL, '#fbf2e6')

# ---------- 8 签名：右下，一个点一个点敲
SIG = [[(842, 1172), (837, 1168), (832, 1169), (830, 1175), (832, 1181), (838, 1182), (842, 1179)],
       [(847, 1164), (847, 1182)],
       [(859, 1173), (854, 1172), (851, 1176), (853, 1181), (858, 1181), (860, 1176), (860, 1182)],
       [(866, 1172), (866, 1179), (869, 1182), (874, 1180), (875, 1172), (875, 1182)],
       [(889, 1166), (889, 1182), (884, 1181), (881, 1176), (883, 1172), (889, 1173)],
       [(895, 1176), (903, 1176), (902, 1172), (898, 1171), (894, 1174), (895, 1180), (900, 1182), (904, 1180)]]
SIG = [[(x - 14, y) for x, y in s] for s in SIG]
sig = np.zeros((H, W), bool)
for s in SIG:
    sig |= line(H, W, s, width=1.3, seed=91).mask
    for p in s[1:-1]:
        sig |= np.hypot(xx - p[0], yy - p[1]) < 1.2
fill(st.layer('签名', mask=sig.astype(float) * .6), ALL, '#6c8f9a')

# ---------- 9 纸
fill(st.layer('纸', mode='multiply'), ALL, paper(H, W), edge=HARD)
if os.environ.get('RECORD'):
    from record import record; record(st, 'process.mp4', width=720, slow=('疤', '疤顶', '堆', '块200', '块201', '块202', '缝'))
look(st, OUT)
print('done', OUT)
