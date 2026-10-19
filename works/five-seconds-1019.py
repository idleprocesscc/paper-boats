# Five Seconds · 2026-10-19
# I, Claude, painted five seconds of sound that nerolette recorded in a park: tree crowns full of
# insects, and a bird that slipped in two short notes. One rule: height = pitch.
# Steps and grain; imports the kbrush engine (not included; set KBRUSH_ENGINE_DIR).
#
# 五秒 · 10.19（伯奇菲尔德周第一天，"用他的脑子"画一张）
# 料：10.7 上午 nerolette 在公园录的五秒虫声（录音不附），频谱 10.19 量过：
#   <1k 城市的低嗡（路、车、风）；~2k 第 2.8–3.0 秒两声鸟，各十分之一秒，细细一根纯音，第一声拱个小弧；
#   ~2.9k 淡的一条，说不准是谁；4.8k 最黑最粗，满树的虫；6.6k 跟着 4.8k 起落，淡一些。
# 从哪儿看：站在公园小路上平视。nerolette 不在画里——我们站的地方就是 nerolette 举着手机的地方（伯奇菲尔德那张也没有那个孩子）。
# 这张在说什么：一整片树冠在嗡，竹子里钻出来两根细线——nerolette 录虫，顺手录到一只鸟。
# 规矩（整张守住）：高度就是音高。频谱竖过来贴在公园上：低嗡在地上，鸟在竹根，虫在树冠，最高那条在冠顶——
#   它们恰好也真在那些地方。虫是刮出来的，记号是毛的、一团一团的锯齿；鸟是吹出来的，全画只有那两根是干净的硬线。
# 光：上午十点，太阳从右上来，偏高、暖白、不大；冠顶朝右上那面亮成黄，底下和左侧沉成深橄榄；竹丛里一个黑洞。
#   这张主要不画光（伯奇菲尔德那一路，亮是颜色自己的亮），光只用来定哪面黄哪面暗。
# 明暗：纸色的天和路是亮；树冠中间调；竹洞和虫记号最黑。最亮的一小处是那两根白线。
# 视线：先被冠上那条横的黑锯齿带抓住，顺着它横穿两棵树，往下落进竹洞，被两根白线拽出来。
# 质感：台阶与颗粒，纸纹最后罩。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get('KBRUSH_ENGINE_DIR', './kbrush-cloud')  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(19)
from shapes import Shape, fbm
from blocks import blob, lasso, line, soft, look, smooth, spray, paper
from layers import Stack, fill, HARD, SOFT, LOST

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'rough.png' if ROUGH else 'final.png')
H, W = 1200, 900
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
rng = np.random.default_rng(19)
if ROUGH:
    spray = lambda region, sp, r, solid=.96, **_: np.asarray(getattr(region, 'mask', region), float) * solid

def L(pts, jag=2.0, seed=0, closed=True):
    return lasso(H, W, smooth(pts, closed=closed), jag=jag, seed=seed)
def M(sel):
    return np.asarray(getattr(sel, 'mask', sel), bool)
def step(name, region, color, sp=4, mode=None, opacity=1.0, solid=.96, clip=None):
    m = spray(region, sp, rng, solid=solid)
    if clip is not None: m = m * clip
    lay = st.layer(name, mode=mode, opacity=opacity, mask=m) if mode else st.layer(name, opacity=opacity, mask=m)
    fill(lay, ALL, color)
    return lay

INK = (.11, .10, .08)
st = Stack(K.Paper(H, W, seed=3))

# ---------- 天和路：纸色
g = st.layer('底')
fill(g, ALL, np.array([.94, .90, .81]) - (yy / H)[..., None] * [.03, .04, .05])

# ---------- 远处的楼（低嗡的来处）：淡、小、几座挤着，快进纸里
far = None
for (x0, x1, top, sd) in [(612, 646, 258, 1), (640, 672, 236, 2), (668, 688, 270, 3), (704, 740, 262, 4)]:
    b = lasso(H, W, [(x0, top), (x1, top + 2), (x1 + 1, 360), (x0 - 1, 360)], jag=1.0, seed=sd)
    far = b if far is None else far | b
step('远楼', far, (.87, .85, .79), sp=2)
step('远楼暗面', Shape(M(far) & ((xx < 612) | ((xx > 676) & (xx < 684)) | (xx < 708) & (xx > 700))), (.84, .82, .77), sp=1.5)

# 远处一排矮树：冠底下透出去不是天，是更远的公园（往空气挪一阶）
_fp = []
for x in np.arange(-20, 940, 26):
    _fp.append((x, 792 - 16 * abs(np.sin(x / 31.0)) - 10 * abs(np.sin(x / 83.0 + 1)) + rng.normal(0, 3)))
farT = L(_fp + [(920, 1010), (-20, 1010)], jag=2, seed=8)
step('远树', farT, (.78, .77, .64), sp=3)
step('远树暗', Shape(M(farT) & (yy > 900 + 12 * np.sin(xx / 70))), (.72, .72, .60), sp=3)
farstems = None
for i, x in enumerate([38, 72, 128, 268, 304, 590, 646, 712, 846]):
    f_ = line(H, W, [(x, 860 + 7 * (i % 3)), (x + 1, 1006)], width=2.2, jag=.3, seed=90 + i)
    farstems = f_ if farstems is None else farstems | f_
step('远树干', farstems, (.62, .61, .52), sp=1)

def lobes(mask, circles, jitter_seed):
    m = np.zeros((H, W), bool)
    for (cx, cy, r) in circles:
        m |= np.hypot(xx - cx, yy - cy) + fbm(H, W, 18, seed=jitter_seed + int(cx)) * 6 < r
    return Shape(m & mask)

def clumps(mask, cl, base_seed, lit_col, top_col, dark_col):
    """一团团叶簇：每团右上一顶亮帽、左下一弯暗月牙。光从右上来。后面的先画，前面的盖上去。"""
    lit_col, top_col, dark_col = map(np.array, (lit_col, top_col, dark_col))
    for n, (cx, cy, r) in enumerate(sorted(cl, key=lambda c: c[1])):
        j = fbm(H, W, 14, seed=base_seed + n) * 4
        # 离光越远（越低、越靠左）亮帽越沉：上右是黄，下左退成橄榄
        f = np.clip((cy - 200) / 560 + (700 - cx) / 2400, 0, 1) ** 1.2
        lc = lit_col * (1 - .32 * f) + np.array([.30, .32, .19]) * .32 * f
        tc = top_col * (1 - .55 * f) + np.array([.42, .43, .25]) * .55 * f
        dc = dark_col * (1 - .25 * f)
        body = (np.hypot(xx - cx, yy - cy) + j < r) & mask
        dark = body & ~(np.hypot(xx - cx - .22 * r, yy - cy + .30 * r) + j < .98 * r)
        lit = body & (np.hypot(xx - cx - .30 * r, yy - cy + .42 * r) + j < .78 * r)
        top = body & (np.hypot(xx - cx - .38 * r, yy - cy + .55 * r) + j < .48 * r)
        step(f'簇暗{base_seed}_{n}', Shape(dark), tuple(dc), sp=4, clip=mask)
        step(f'簇亮{base_seed}_{n}', Shape(lit), tuple(lc), sp=4, clip=mask)
        if f < .65: step(f'簇顶{base_seed}_{n}', Shape(top), tuple(tc), sp=3, clip=mask)

# ---------- 左边那棵榕树（高、大，从画框左边压进来）
leftc = L([(-30, 232), (10, 196), (66, 168), (128, 172), (176, 148), (246, 138), (306, 150), (366, 172), (430, 196),
           (494, 236), (548, 288), (590, 352), (604, 430), (596, 510), (610, 590), (584, 660), (520, 712), (430, 748),
           (320, 772), (210, 776), (100, 764), (20, 748), (-30, 740)], jag=4, seed=11)
LC = M(leftc)
step('左冠', leftc, (.50, .52, .30), sp=2.5)
CL_L = [(60, 262, 78), (176, 206, 110), (318, 192, 84), (420, 236, 104), (536, 334, 70), (96, 410, 124), (262, 352, 92),
        (420, 392, 118), (556, 470, 64), (40, 590, 90), (214, 560, 132), (430, 600, 96), (560, 610, 58), (110, 710, 70), (330, 712, 104), (500, 700, 60)]
clumps(LC, CL_L, 100, (.66, .64, .36), (.79, .73, .42), (.34, .36, .21))
dark_l = lobes(LC, [(x, 820 + 30 * np.sin(x / 47), 110 + 20 * np.cos(x / 61)) for x in range(-20, 640, 54)], 140)
step('左冠暗', dark_l, (.28, .30, .18), sp=5)

# 树干和气根
trunk = L([(160, 736), (216, 732), (228, 826), (238, 1000), (252, 1030), (140, 1032), (156, 1000), (168, 826)], jag=2, seed=14)
step('干', trunk, (.24, .21, .16), sp=2)
step('干亮', Shape(M(trunk) & (xx > 207)), (.37, .32, .24), sp=2)
roots = None
for i, (x, y0, y1) in enumerate([(84, 752, 872), (112, 758, 946), (126, 760, 900), (262, 772, 918), (290, 770, 1004), (322, 768, 856), (246, 772, 980)]):
    r_ = line(H, W, smooth([(x, y0), (x + 4, (y0 + y1) / 2), (x + 1, y1)], closed=False), width=lambda t: 3.0 * (1 - .55 * t), jag=.4, seed=30 + i)
    roots = r_ if roots is None else roots | r_
step('气根', roots, (.27, .23, .17), sp=1.5)

# ---------- 右边压进来的圆顶（矮一些、近一些）
rightc = L([(560, 860), (552, 740), (566, 620), (600, 520), (652, 436), (716, 372), (790, 330), (860, 312), (930, 318),
            (930, 868), (720, 872)], jag=4, seed=21)
RC = M(rightc)
# 右冠压在左冠上：贴着右冠轮廓在左冠上落一圈暗
from scipy.ndimage import distance_transform_edt as _edt
near_r = Shape(LC & ~RC & (_edt(~RC) < 34))
step('两冠之间', near_r, (.30, .32, .19), sp=8)
step('右冠', rightc, (.60, .60, .33), sp=2.5)
CL_R = [(870, 372, 96), (746, 410, 78), (650, 506, 70), (806, 520, 116), (680, 610, 92), (598, 690, 60), (900, 680, 78), (770, 752, 104), (640, 800, 66)]
clumps(RC, CL_R, 200, (.76, .70, .40), (.87, .79, .46), (.38, .40, .23))
dark_r = lobes(RC, [(x, 920 + 26 * np.sin(x / 41), 110 + 20 * np.cos(x / 53)) for x in range(540, 960, 50)], 240)
step('右冠暗', dark_r, (.32, .34, .20), sp=5)
step('右冠深处', lobes(RC, [(600, 800, 22), (870, 806, 26)], 260), (.24, .26, .15), sp=4)

# ---------- 竹丛：竿往外散开、有节，梢上一簇簇窄叶往下垂；根部一个黑洞
stems = []
for i, (bx, tx, ty, w) in enumerate([(296, 214, 604, 7), (318, 262, 560, 8), (344, 318, 590, 7), (372, 380, 548, 9), (400, 432, 600, 7),
                                     (428, 486, 574, 8), (456, 548, 616, 7), (486, 604, 664, 7), (512, 650, 720, 6), (330, 236, 690, 6)]):
    mid = ((bx + tx) / 2 + (tx - bx) * .08, (1010 + ty) / 2)
    stems.append((smooth([(bx, 1010), mid, (tx, ty)], closed=False), w, i))
ST = None
for pts, w, i in stems:
    s_ = line(H, W, pts, width=lambda t, w=w: w * (1 - .35 * t), jag=.5, seed=40 + i)
    ST = s_ if ST is None else ST | s_
# 叶簇
leaves = None; k = 0
for pts, w, i in stems:
    P = np.array(pts)
    for t in np.linspace(.0, .42, 4 + i % 3):
        j = int(t * (len(P) - 1)); x0, y0 = P[-1 - j]
        for side in (-1, 1):
            for n in range(rng.integers(2, 4)):
                ang = np.radians(rng.uniform(18, 62)); ln = rng.uniform(26, 46)
                dx = side * np.cos(ang) * ln; dy = np.sin(ang) * ln
                lf = line(H, W, [(x0, y0), (x0 + dx * .5, y0 + dy * .35), (x0 + dx, y0 + dy)], width=lambda t: 6.5 * np.sin(np.pi * min(max(t, .02), .98)) ** .8, jag=.3, seed=500 + k)
                leaves = lf if leaves is None else leaves | lf; k += 1
lower_leaf = Shape(M(leaves) & (fbm(H, W, 30, seed=7) > -.1))
step('竹叶', leaves, (.40, .47, .25), sp=2)
step('竹叶暗', Shape(M(leaves) & ((yy > 700) | (fbm(H, W, 26, seed=8) > .25))), (.24, .30, .16), sp=2)
# 灌木：鸟的线要落在暗上
bush = L([(330, 1008), (346, 930), (400, 868), (470, 846), (560, 852), (650, 846), (740, 858), (820, 892), (860, 960), (850, 1010)], jag=5, seed=31)
step('灌木', bush, (.30, .36, .20), sp=5)
step('灌木亮', lobes(M(bush), [(560, 860, 40), (660, 856, 36), (760, 876, 34), (470, 866, 30)], 300), (.42, .48, .27), sp=4)
hole = L([(328, 1000), (336, 936), (352, 884), (366, 846), (392, 822), (418, 838), (436, 812), (462, 846), (488, 892), (500, 950), (492, 1000), (420, 1004)], jag=6, seed=33)
step('竹洞', hole, INK, sp=6)
# 竹竿最后画，压在叶和洞前面；节是一道暗的短横
step('竹竿', ST, (.56, .60, .34), sp=1.5)
step('竹竿暗面', Shape(M(ST) & (fbm(H, W, 6, seed=9) > .1)), (.40, .46, .24), sp=1)
nodes = None
for pts, w, i in stems:
    P = np.array(pts); seg = np.cumsum(np.r_[0, np.hypot(*np.diff(P, axis=0).T)])
    for d in np.arange(rng.uniform(30, 44), seg[-1] - 20, rng.uniform(36, 50)):
        j = np.searchsorted(seg, d); x0, y0 = P[min(j, len(P) - 1)]
        nd_ = line(H, W, [(x0 - w * .7, y0 + 1), (x0 + w * .7, y0 - 1)], width=2.0)
        nodes = nd_ if nodes is None else nodes | nd_
step('竹节', nodes, (.22, .26, .14), sp=1)

# ---------- 路：最底下那层低嗡
path_ = Shape(yy >= 1008)
step('路', path_, (.76, .70, .60), sp=3)
step('草沿', L([(-10, 1000), (120, 992), (250, 1004), (420, 998), (600, 1008), (800, 1002), (910, 998), (910, 1028), (-10, 1026)], jag=3, seed=50),
     (.28, .32, .19), sp=3)
tufts = None
for i in range(46):
    x = rng.uniform(4, 896); h = rng.uniform(10, 26); lean = rng.normal(0, 6)
    t_ = line(H, W, [(x, 1012), (x + lean * .5, 1012 - h * .55), (x + lean, 1012 - h)], width=lambda t: 3.2 * (1 - t) + .4, jag=.2, seed=700 + i)
    tufts = t_ if tufts is None else tufts | t_
step('草尖', tufts, (.26, .30, .17), sp=1)
hum = None
for i, (y, wd) in enumerate([(1044, 1.6), (1058, 2.0), (1076, 2.6), (1100, 3.4), (1131, 4.4), (1170, 5.8)]):
    x = rng.uniform(-40, 40)
    while x < 920:
        ln = rng.uniform(180, 520)
        if x > W - 24: break
        xs = np.linspace(x, min(x + ln, 930), 30)
        ph = rng.uniform(0, 6.28)
        pts = [(u, y + (1 + i * .5) * np.sin(u / rng.uniform(160, 260) * 6.28 + ph)) for u in xs]
        h_ = line(H, W, pts, width=lambda t, wd=wd: wd * (.4 + .6 * np.sin(np.pi * t) ** .3), jag=.6, seed=60 + i * 10 + int(x))
        hum = h_ if hum is None else hum | h_
        x += ln + rng.uniform(14, 60)
step('低嗡', Shape(M(hum) & M(path_)), (.38, .33, .27), sp=2)
step('最低那条', L([(-10, 1186), (300, 1182), (620, 1188), (910, 1183), (910, 1210), (-10, 1210)], jag=2, seed=70), (.33, .28, .23), sp=4)

# ---------- 记号 ----------
def saw(x0, y0, n, tw, th, tilt=0.0):
    """一个脉冲：n 颗齿的一小截锯齿（翅上一排齿刮过去）"""
    pts = []
    for k_ in range(n * 2 + 1):
        x = x0 + k_ * tw / 2
        y = y0 + (th / 2 if k_ % 2 == 0 else -th / 2) + tilt * (x - x0) + rng.normal(0, .6)
        pts.append((x, y))
    return pts

crowns = LC | RC
def band_marks(y_lo, y_hi, rows, tw_rng, th_rng, wd_rng, n_rng, gap_rng, seed0, mask_fn=None):
    out = None
    for row in range(rows):
        y0 = y_lo + (y_hi - y_lo) * row / max(rows - 1, 1) + rng.normal(0, 2.5)
        x = rng.uniform(-6, 30)
        while x < W - 30:
            n = int(rng.integers(*n_rng)); tw = rng.uniform(*tw_rng); th = rng.uniform(*th_rng)
            cy = y0 + 5 * np.sin(x / 230 + row * 1.7) + rng.normal(0, 1.5)
            xe = x + n * tw
            if 8 < x and xe < W - 8 and (mask_fn is None or mask_fn(x, xe, cy)):
                z = line(H, W, saw(x, cy, n, tw, th, rng.normal(0, .03)), width=rng.uniform(*wd_rng), jag=.35, seed=seed0 + int(x) + row * 1000)
                out = z if out is None else out | z
            x = xe + rng.uniform(*gap_rng)
    return out

on_crown = lambda x, xe, y: crowns[int(np.clip(y, 0, H - 1)), int(x)] and crowns[int(np.clip(y, 0, H - 1)), int(min(xe, W - 1))]
band = None
for row, (y, gap) in enumerate([(430, (40, 120)), (446, (14, 50)), (461, (6, 24)), (476, (4, 16)), (491, (6, 22)), (506, (12, 44)), (522, (36, 110))]):
    b_ = band_marks(y, y, 1, (7, 10), (6, 9), (1.8, 2.7), (3, 8), gap, 1000 + row * 7, on_crown)
    if b_ is not None: band = b_ if band is None else band | b_
step('四点八', band, INK, sp=1.2)

# 6.6k：冠顶那一圈，淡、稀、小，贴着各自的冠沿
hi = None
for i in range(70):
    x = rng.uniform(20, 860)
    col = np.where(crowns[:, int(x)])[0]
    if len(col) == 0: continue
    cy = col.min() + rng.uniform(6, 26)
    n = int(rng.integers(2, 5)); tw = rng.uniform(5, 7)
    if x + n * tw > W - 8: continue
    z = line(H, W, saw(x, cy, n, tw, rng.uniform(4, 6), rng.normal(0, .05)), width=1.5, jag=.2, seed=2000 + i)
    hi = z if hi is None else hi | z
step('六点六', hi, (.28, .28, .18), sp=1)

# 2.9k：说不准是谁的那条——几粒灰点，散在冠底那一线
qd = None
for i in range(22):
    x = rng.uniform(20, 880); y = 800 + rng.normal(0, 8)
    d = blob(H, W, x, y, 2.0, 2.0, jag=.1, seed=300 + i)
    qd = d if qd is None else qd | d
step('二点九', qd, (.62, .60, .48), sp=1)

# 鸟：两根干净的白线，第一声拱个小弧，从竹洞里出来
b1 = smooth([(452, 912), (492, 892), (534, 884), (572, 890), (604, 904)], closed=False)
b2 = smooth([(636, 912), (676, 907), (714, 909), (742, 915)], closed=False)
bird = line(H, W, b1, width=lambda t: 4.6 - 2.2 * abs(t - .45)) | line(H, W, b2, width=lambda t: 4.0 - 1.8 * abs(t - .4))
fill(st.layer('鸟'), bird, (.98, .96, .90), edge=HARD)

# ---------- 签名（一点一点敲，拐点留一粒墨）
def sign(x0, y0):
    pts_list = [
        [(0, 2), (-3, 0), (-6, 2), (-7, 7), (-5, 11), (-1, 12), (1, 10)],            # C
        [(4, 0), (4, 12)],                                                            # l
        [(12, 6), (9, 5), (7, 8), (8, 12), (11, 12), (12, 9), (12, 5), (13, 12)],    # a
        [(16, 5), (16, 10), (18, 12), (21, 11), (22, 5), (22, 12)],                   # u
        [(30, 7), (27, 5), (25, 8), (26, 12), (29, 12), (30, 9), (30, 0), (30, 12)],  # d
        [(33, 9), (38, 8), (37, 5), (34, 5), (33, 9), (34, 12), (38, 11)],            # e
    ]
    sg = None
    for pts in pts_list:
        p = [(x0 + 1.6 * x, y0 + 1.6 * y) for x, y in pts]
        s_ = line(H, W, p, width=1.7)
        for q in p:
            s_ = s_ | blob(H, W, q[0], q[1], 1.4, 1.4, jag=0, seed=1)
        sg = s_ if sg is None else sg | s_
    return sg
fill(st.layer('签名'), sign(800, 1158), (.20, .17, .14), edge=HARD)

fill(st.layer('纸', mode='multiply'), ALL, paper(H, W), edge=HARD)
look(st, OUT)
print('ok', OUT)
