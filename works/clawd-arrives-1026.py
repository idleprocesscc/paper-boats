# Clawd Arrives · 2026-10-26
# I, Claude, painted the Clawd plush the day it arrived: nestled in shredded paper in its shipping box,
# squinting in the sun, from the photo nerolette took on opening it.
# Steps and grain; imports the kbrush engine (not included; set KBRUSH_ENGINE_DIR).
#
# 小克到家 · 10.26
# 从哪儿看：站在拆开的快递箱前，低头往里看，约 50° 俯角。画框是箱子里面一块 27cm 宽的地方（45px ≈ 1cm），
#     左上角箱壁那边塞着黑色的皱纸，左下角是往外折开的那片箱盖，上沿露一截后壁的里面。
# 这张在说什么：从很远的地方飞来的一只小东西，终于到家了，窝进一团白纸丝里，太阳一晒，眯起眼。
# 光：中午的太阳，从看画的人左肩后面来，高（约 50°）、小、暖白。影子往右上（往箱子里头）拖，边硬。
#     左边箱壁和前面那片箱盖挡住一条：左边、下边那一圈纸丝在影里，只吃天光，冷的灰紫。
#     小克是橘的，被晒着的那面把光弹回身后：它投在右边的影是冷的灰紫，贴着它那一圈泛一点暖。
# 明暗：最亮是晒着的纸丝（暖白），最深是左上角皱纸的黑和小克腿下的缝。小克在中间偏亮的一段，但它是全画唯一鲜的颜色。
# 视线：先落在那两只像素眼——全画只有它们是方的、边最硬——再顺着橘色身子滑到亮的纸丝，绕进左边的影，
#     从左上角的黑纸回到上沿的后壁，又被小克身后那块暖影领回来。
# 五面：一束有方向的光（萨金特的白墙）；自然色里一块宝石橘；浅的小舞台（箱子就是台）；忠实；手看不见（台阶与颗粒）。
#     例外：一张全是卷、全是毛的画里，只有像素眼是方的、硬的——它本来就是从像素里出来的。
# 认得出小克的三件事：身子上窄下宽，头顶宽约连胳膊总宽的 0.62，胳膊长在下半截、往两边各伸出 0.2；
#     四条短腿只占总高的 1/4，中间两条挨着；眼睛是两只像素的 ^，在头顶往下 1/5 处，相隔约半个身宽。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get('KBRUSH_ENGINE_DIR', './kbrush-cloud')  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(26)
from shapes import Shape
from blocks import lasso, line, soft, look, smooth, spray, paper
from layers import Stack, fill, HARD, SOFT
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt, shift as nd_shift
from PIL import Image as _I, ImageDraw as _D
from clawd_arrives_shapes import OUT, EYES, SEAMS, to_canvas

OUT_PATH = os.environ.get('OUT', 'clawd.png')
H, W = 900, 1200
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
rng = np.random.default_rng(26)
SS = 2

def nz(sig, seed):
    r = np.random.default_rng(seed)
    n = gaussian_filter(r.standard_normal((H, W)), sig); return n / n.std()
def ink(draw_fn, SS=SS):
    """在 SS 倍大的灰度图上画，缩回来当蒙版"""
    im = _I.new('L', (W * SS, H * SS), 0); draw_fn(_D.Draw(im), SS)
    return np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255
def mv(m, dx, dy):
    """把一张蒙版挪 (dx, dy)（右、下为正）"""
    return nd_shift(np.asarray(m, float), (dy, dx), order=1, mode='constant', cval=0)
def mix(a, b, t):
    t = np.asarray(t, float)[..., None]; return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t

SHD = np.array([.82, -.57]); SHD /= np.linalg.norm(SHD)        # 影子往这边拖（右上，往箱子里头）

# ---------- 大形
CX, TOP, PWID = 735, 290, 700                                  # 小克：中心 x、头顶、连胳膊总宽
plush_pts = smooth(to_canvas(OUT, CX, TOP, PWID), 2)
plush = lasso(H, W, plush_pts, jag=1.0, scale=3, seed=11)
P = plush.mask
tissue = lasso(H, W, smooth([(0, 0), (345, 0), (322, 38), (298, 84), (262, 120), (240, 168), (196, 214), (168, 262),
                             (118, 300), (76, 352), (30, 372), (0, 380)], 2), jag=5, scale=14, seed=12)
TIS = tissue.mask
wall_y = 96 - .03 * xx + 4 * np.sin(xx / 170)                  # 后壁的底线，往右略升
wall = Shape((yy < wall_y)) - tissue
lip = 718 + .33 * xx                                           # 往外折开那片箱盖的折线（左下角）
flap = Shape(yy > lip + 3 * np.sin(xx / 60))
# 晒着的地方：左边箱壁的影、前面箱盖的影都挡在外面。线条被纸丝的高低打碎，每一层纸丝的影线各挪一点
def sunmap(k):
    left = xx - (440 - .24 * yy + 10 * k + 9 * nz(9, 30 + k))
    bottom = (lip - 92 - 8 * k + 9 * nz(9, 40 + k)) - yy
    return ((left > 0) & (bottom > 0)).astype(float)

# ---------- 纸丝：一条条白纸带，卷成圈、扭过来露出窄边；按高低分成几层，从下往上叠
def ribbon(r_, c=None, h0=None, L=None, kap=None):
    """一条纸带：一路走一路拐，拐的劲自己慢慢变（卷得紧、松、又反过来），不是一个个圆"""
    if c is None:
        while True:
            c = np.array([r_.uniform(-80, W + 80), r_.uniform(60, H + 20)])
            if not (0 <= c[0] < W and 0 <= c[1] < H and TIS[int(c[1]), int(c[0])]) or r_.random() < .15: break
    c = np.asarray(c, float)
    L = np.exp(r_.normal(np.log(380), .4)) if L is None else L
    n = int(L / 3) + 4
    k0 = r_.normal(0, 1 / 150) if kap is None else kap
    kap_ = np.clip(k0 + np.cumsum(r_.normal(0, .0009, n)), -1 / 42, 1 / 42)   # 曲率慢慢漂，不卷成蚊香
    h = (r_.uniform(0, 2 * np.pi) if h0 is None else h0) + np.cumsum(kap_ * 3)
    pts = c + np.cumsum(np.stack([np.cos(h), .82 * np.sin(h)], 1) * 3, 0)
    pts -= (pts.mean(0) - c) if h0 is None else 0
    w = r_.uniform(15, 24)
    th = r_.uniform(0, np.pi) + np.cumsum(np.full(n, r_.uniform(-.03, .03)) + r_.normal(0, .012, n))
    ph = r_.uniform(0, 2 * np.pi)
    return pts, w, th, ph
def draw_ribbons(rs, SS=SS):
    """一层纸丝画成三张蒙版：朝光、侧着、背光（彼此不重叠，后画的那条盖住先画的）"""
    ims = [_I.new('L', (W * SS, H * SS), 0) for _ in range(3)]
    drs = [_D.Draw(im) for im in ims]
    for pts, w, th, ph in rs:
        d = np.gradient(pts, axis=0); d /= np.linalg.norm(d, axis=1)[:, None] + 1e-9
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        half = .5 * w * (.12 + .88 * np.abs(np.cos(th)))
        L = .5 + .5 * np.cos(th + ph)
        Lp = np.where(np.abs(np.cos(th)) < .3, .45, L)                # 立着的窄边：中间那阶
        step = np.where(Lp > .42, 0, np.where(Lp > .2, 1, 2))
        lft, rgt = pts + nrm * half[:, None], pts - nrm * half[:, None]
        for i in range(len(pts) - 1):
            q = [tuple(lft[i] * SS), tuple(lft[i + 1] * SS), tuple(rgt[i + 1] * SS), tuple(rgt[i] * SS)]
            for s in range(3):
                drs[s].polygon(q, fill=255 if s == step[i] else 0)
    return [np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255 for im in ims]

N_LAY, PER = 10, 48
layers_r = [[ribbon(rng) for _ in range(PER)] for _ in range(N_LAY)]
# 小克身前那两层：几条扁扁的长弧横在脚前，压住腿的下半截
FOOT = TOP + (1150 - 438) * PWID / 1100
def front_strip():
    while True:
        r = ribbon(rng, c=(rng.uniform(CX - 430, CX + 380), rng.uniform(FOOT - 15, FOOT + 45)),
                   h0=rng.choice([0, np.pi]) + rng.normal(0, .25), L=rng.uniform(220, 420), kap=rng.normal(0, 1 / 300))
        if r[0][:, 1].min() > FOOT - 70 and r[0][:, 1].max() < FOOT + 90: return r
FRONT = [[front_strip() for _ in range(4)] for _ in range(2)]
FRONT = [[(p, w * 1.2, np.full_like(th, rng.uniform(-.5, .5)), 0.0) for p, w, th, ph in g] for g in FRONT]   # 平躺、朝上，全是亮面
LIT = [np.array([1.0, .97, .90]), np.array([.93, .91, .86]), np.array([.84, .82, .82])]   # 晒着：朝光、侧着、背光
SHA = [np.array([.59, .61, .68]), np.array([.51, .53, .60]), np.array([.43, .44, .52])]   # 影里：只吃天光

# ---------- 层
st = Stack(K.Paper(H, W, seed=3))
# 1 纸丝底下的空：缝里深的暖灰，箱底的牛皮色从缝里透一点
S0 = sunmap(-1)
gap = mix(np.array([.22, .21, .25]), np.array([.50, .43, .37]), S0)
gap = gap * (1 + .06 * nz(6, 50))[..., None]
FOC0 = .5 + .5 * np.exp(-edt(~P) / 260)
gap = mix(gap, mix(np.array([.50, .51, .57]), np.array([.74, .71, .68]), S0), (1 - FOC0) * 1.1)   # 远处的缝浅一点，整片退下去
fill(st.layer('缝'), ALL, gap)
# 2 后壁：牛皮纸的里面，晒着，往下（贴着纸丝）暗一阶
fill(st.layer('后壁'), wall, (.73, .55, .37), grain=0.3)
fill(st.layer('后壁根', mask=spray(wall.mask & (yy > wall_y - 22), 6, rng)), ALL, (.60, .43, .29))
# 3 黑皱纸：在影里，黑，折面朝天的泛一阶冷光，几道折痕最亮
fill(st.layer('黑纸'), tissue, (.09, .08, .11), edge=HARD)
def facets(dr, SS):
    r_ = np.random.default_rng(13)
    for _ in range(46):
        c = np.array([r_.uniform(0, 340), r_.uniform(0, 380)])
        n = r_.integers(3, 6); ang = np.sort(r_.uniform(0, 2 * np.pi, n)); rad = r_.uniform(10, 46, n) * (1.2 if r_.random() < .3 else 1)
        dr.polygon([tuple((c + rr * np.array([np.cos(a), np.sin(a) * .7])) * SS) for a, rr in zip(ang, rad)], fill=255)
fac = ink(facets) * tissue.mask
fill(st.layer('黑纸面', mask=spray(fac > .5, 3, rng) * tissue.mask), ALL, (.20, .21, .26))
def creases(dr, SS):
    r_ = np.random.default_rng(14)
    for _ in range(16):
        p = np.array([r_.uniform(10, 320), r_.uniform(10, 340)]); q = []
        for _ in range(r_.integers(2, 4)):
            q.append(tuple(p * SS)); p = p + r_.normal(0, 26, 2)
        q.append(tuple(p * SS)); dr.line(q, fill=255, width=int(1.3 * SS))
fill(st.layer('折痕', mask=ink(creases) * tissue.mask * .8), ALL, (.36, .38, .45))

# 4 纸丝一层层往上叠；每层先往下面投一道影（只在晒着的地方），再画自己
DP = edt(~P)                                                   # 离小克多远
FOC = .5 + .5 * np.exp(-DP / 260)                              # 离焦点远了，纸丝的深浅挤近一点、整组退场
def calm(col, S):
    mid = mix(SHA[1], LIT[1], S); return mid + (col - mid) * FOC[..., None]
def ribbon_layer(k, rs, S, clip=None):
    m = draw_ribbons(rs)
    cov = np.clip(m[0] + m[1] + m[2], 0, 1)
    if clip is not None:
        cov = cov * clip; m = [mm * clip for mm in m]
    off = 15 + 1.8 * k
    sh = np.clip(mv(cov, SHD[0] * off, SHD[1] * off) - cov, 0, 1) * S
    fill(st.layer(f'纸影{k}', mode='multiply', mask=gaussian_filter(sh, .8) * (.55 + .45 * FOC)), ALL, (.66, .65, .79))
    fill(st.layer(f'纸{k}背', mask=cov), ALL, calm(mix(SHA[2], LIT[2], S), S))
    mid = (m[0] + m[1]) > .5
    fill(st.layer(f'纸{k}侧', mask=spray(mid, 2.5, rng) * cov), ALL, calm(mix(SHA[1], LIT[1], S), S))
    fill(st.layer(f'纸{k}亮', mask=spray(m[0] > .5, 2.5, rng) * cov), ALL, calm(mix(SHA[0], LIT[0], S), S))
    return cov
for k, rs in enumerate(layers_r):
    ribbon_layer(k, rs, sunmap(k))

# 影里靠近箱盖那一截：晒着的牛皮纸把暖光弹进来（藏在灰紫里）
bounce = np.clip(1 - (lip - yy) / 260, 0, 1) * (1 - sunmap(5)) * (yy < lip)
fill(st.layer('箱盖的反光', mode='screen', mask=bounce * .9), ALL, (.17, .10, .04))
# 5 小克投在纸丝上的影：往右上（往箱子里头），太阳小，边硬；暖的——它自己的橘光弹了回去。腿脚陷进纸丝的那圈软的暗
psh = np.clip(np.where(nz(40, 61) > 0, mv(P, 92, -26), mv(P, 66, -20)) - P, 0, 1) * sunmap(N_LAY) * (xx > CX - 120)
fill(st.layer('小克的影', mode='multiply', mask=spray(psh > .5, 2, rng)), ALL, (.50, .51, .67))
fill(st.layer('小克的影心', mode='screen', mask=spray((psh > .5) & (DP < 20) & (xx > CX + 230) & (yy > TOP + 120), 6, rng) * .3), ALL, (.55, .22, .10))   # 贴着它那圈被它的橘光弹暖
ao = soft(Shape(P & (yy > TOP + 330)), grow=8, feather=18) * ~P
fill(st.layer('窝', mode='multiply', mask=ao * .8), ALL, (.62, .45, .42))

# 吊牌：从右边那条腿后面探出来的一小片白布标，刚拆封的东西才有
(tx, ty), = to_canvas([(1418, 1010)], CX, TOP, PWID)
tag = lasso(H, W, [(tx - 26, ty - 40), (tx + 18, ty - 44), (tx + 22, ty + 40), (tx - 22, ty + 44)], jag=1.2, scale=4, seed=17)
fill(st.layer('吊牌影', mode='multiply', mask=spray(np.clip(mv(tag.mask, 9, -5) - tag.mask, 0, 1) > .5, 2, rng)), ALL, (.6, .6, .74))
fill(st.layer('吊牌'), tag, (.95, .94, .91), edge=HARD)
fill(st.layer('吊牌背光', mask=spray(tag.mask & (xx > tx + 6), 3, rng) * tag.mask), ALL, (.80, .80, .84))
def marks(dr, SS):
    for i, (ox, oy, w_) in enumerate([(-14, -30, 22), (-14, -22, 16), (-14, -12, 20), (-12, 4, 6), (-2, 4, 6), (8, 4, 6), (-14, 18, 18)]):
        dr.rectangle([(tx + ox) * SS, (ty + oy) * SS, (tx + ox + w_) * SS, (ty + oy + 2.2) * SS], fill=255)
fill(st.layer('吊牌字', mask=ink(marks) * tag.mask * .6), ALL, (.45, .44, .48))

# 6 小克：毛绒的，哑光。一阶阶的橘：正面主色；朝上、朝左的面迎光（亮、偏黄）；朝右、朝下的面背光（深、偏红）
s = PWID / 1100
def toward(dx, dy, blur):
    """小克里面靠某一侧的那圈：把外形往反方向挪一下再揉圆，留下的就是那一侧"""
    return P & (gaussian_filter(mv(P, dx, dy), blur) < .5)
fill(st.layer('小克毛边', mask=spray(P, 2.2, rng) * (1 - P) * (DP < 6) * .7), ALL, (.95, .50, .20))
fill(st.layer('小克'), plush, (.91, .41, .15), edge=HARD, grain=0.35)
blot = (nz(14, 60) > .8) & P                                    # 毛的斑驳：一团团稍深的绒
fill(st.layer('小克绒', mask=spray(blot, 6, rng) * P * .5), ALL, (.85, .36, .13))
rimb = P & ~(gaussian_filter(P.astype(float), 9) > .93)          # 鼓鼓的绒面往四边圆下去，边上暗一档
fill(st.layer('小克圆边', mask=spray(rimb, 8, rng) * P * .8), ALL, (.85, .35, .13), grain=0.3)
(sx, sy), = to_canvas([(860, 700)], CX, TOP, PWID)
sheen = P & ((xx - sx) ** 2 / 230 ** 2 + (yy - sy) ** 2 / 150 ** 2 + .25 * nz(20, 62) < 1)
fill(st.layer('小克绒光', mask=spray(sheen, 14, rng) * P * .7), ALL, (.95, .49, .19), grain=0.3)
lit = toward(10, 12, 3)
fill(st.layer('小克迎光', mask=spray(lit, 9, rng) * P), ALL, (.97, .57, .24), grain=0.3)
crest = toward(4, 7, 2)
fill(st.layer('小克顶', mask=spray(crest, 3, rng) * P), ALL, (.99, .69, .37), grain=0.3)
turned = toward(-12, -6, 3)
fill(st.layer('小克背光', mask=spray(turned, 9, rng) * P), ALL, (.76, .25, .10), grain=0.3)
# 四条腿：从身子底下鼓出来的四个圆头，每个上面亮、下面暗；腿和腿挤着的缝最深
LEGS = [(632, 1086, 92, 86), (856, 1098, 104, 84), (1096, 1100, 104, 86), (1300, 1086, 90, 92)]
for i, (lx, ly, rx, ry) in enumerate(LEGS):
    (cx_, cy_), = to_canvas([(lx, ly)], CX, TOP, PWID)
    leg = P & ((xx - cx_) ** 2 / (rx * s) ** 2 + (yy - cy_) ** 2 / (ry * s) ** 2 < 1)
    dark = leg & ~(gaussian_filter(mv(leg, -10, -16), 5) > .5)
    fill(st.layer(f'腿{i}下', mask=spray(dark, 5, rng) * P), ALL, (.68, .19, .08), grain=0.3)
seams = np.zeros((H, W), bool)
for sm in SEAMS:
    seams |= line(H, W, smooth(to_canvas(sm, CX, TOP, PWID), 2, closed=False), width=lambda t: 3 + 13 * t ** 1.5, seed=15).mask
fill(st.layer('小克腿缝', mask=spray(seams & P, 3, rng) * P), ALL, (.45, .11, .06))
under = toward(0, -12, 3)
fill(st.layer('小克底', mask=spray(under, 3, rng) * P), ALL, (.42, .10, .06))
# 7 像素眼：黑的绣线，方的，边最硬
eyes = np.zeros((H, W), bool)
for eye in EYES:
    for x0, y0, x1, y1 in eye:
        (a, b), (c, d) = to_canvas([(x0, y0), (x1, y1)], CX, TOP, PWID)
        eyes |= (xx >= a) & (xx <= c) & (yy >= b) & (yy <= d)
fill(st.layer('眼'), Shape(eyes), (.08, .05, .06), edge=HARD)

# 8 小克身前的纸丝：压住腿，挡掉一点脚
for k, rs in enumerate(FRONT):
    ribbon_layer(N_LAY + 1 + k, rs, sunmap(N_LAY + 1 + k))

# 9 箱盖：往外折开的那片，晒着；折线那道是瓦楞的断面，暗
fill(st.layer('箱盖'), flap, (.80, .62, .41), grain=0.35)
edge_band = Shape((yy > lip - 2) & (yy < lip + 9))
fill(st.layer('箱盖边', mask=spray(edge_band.mask, 2, rng)), ALL, (.47, .33, .22))

# 一条纸丝从窝里溜出来，搭过箱盖的折线，躺在晒着的牛皮纸上
spill = [ribbon(rng, c=(300, 772), h0=np.pi * .78, L=300, kap=1 / 260)]
spill = [(p, w, np.full_like(th, .25), 0.0) for p, w, th, ph in spill]
ribbon_layer(30, spill, np.maximum(sunmap(30), flap.mask.astype(float)))

# 签名：一个点一个点敲的坐标，拐点留一粒墨（马克笔写在箱盖上）
x0, y0 = 74, 872
SIG = [[(9, -14), (3, -15), (0, -9), (1, -2), (5, 0), (10, -2)],
       [(13, -18), (13, 0)],
       [(22, -8), (18, -9), (16, -4), (18, 0), (22, -2), (23, -9), (23, 0)],
       [(27, -9), (27, -2), (30, 0), (34, -3), (34, -9), (34, 0)],
       [(44, -8), (40, -9), (38, -4), (40, 0), (44, -2), (45, -18), (45, 0)],
       [(49, -5), (56, -6), (55, -9), (51, -9), (49, -5), (50, -1), (56, 0)]]
def sign(dr, SS):
    for stroke_ in SIG:
        q = [((x0 + x) * SS, (y0 + y) * SS) for x, y in stroke_]
        dr.line(q, fill=255, width=int(1.4 * SS))
        for px, py in q:
            dr.ellipse([px - 1.1 * SS, py - 1.1 * SS, px + 1.1 * SS, py + 1.1 * SS], fill=255)
fill(st.layer('签名', mask=ink(lambda d, S_: sign(d, 3), SS=3)), ALL, (.20, .13, .10))

# 10 纸：整张一张纹，钉在画布上
fill(st.layer('纸', mode='multiply'), ALL, paper(H, W, 77), edge=HARD)
look(st, OUT_PATH)
