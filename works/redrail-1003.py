# 红秧鸡 · 10.3
# 从哪儿看：平视，蹲着，眼睛离地三十几厘米，跟鸟头一样高（视平线 y=340）。看画的人就是举布的那个人：
#     布从画框左上外面我们手里垂下来，离眼睛一臂远，下摆离地二十几厘米。鸟在一米半外。
# 这张在说什么：一只只认得红色的鸟，带着怒气冲向一块红布。它看不见布后面是什么——看画的人就站在布后面。
#     （Herbert 1634："给它们看一块红布，它们就带着最傻的怒气一起扑上来"。Mundy 1638："用一顶红帽子捉它们，有个很巧的法子"。）
# 光：毛里求斯黑檀林，正午。树冠密，大部分在阴里，暗处只有林子里绿灰的天光，冷。
#     头顶几个缺口漏下日光：小、硬、暖白偏金，高（约 75°），从右后上方来，影子短，往左前方落一点。
#     最大的一块光斑落在鸟脚下；另一小块正好打在布的下摆一角。光斑是扁的（平视压扁），边上一圈颗粒。
#     补光：光斑把一点暖反进鸟肚子底下。
# 明暗：整张偏暗（静、沉，正午也是林子里的正午）。亮的只有两块：鸟和它脚下的光、布角。最暗是近处的地和布褶最深处。
# 视线：先落在烧着的布角（全画唯一鲜的红），跳到张开的嘴和眼睛（最锐的边），再顺着鸟往后，落到阴里还有的两只。
# 五面：光=一束有方向的光，近乎戏剧；颜色=调子的灰加一颗宝石（红）；空间=深，三层（布、鸟、林）；真实=忠实；手=看不见的手。
#     例外：布在暗里的那几段边丢掉，跟暗地化成一片——近处压着的是一团暗红，不是一块剪纸。
# 质感（佐料只放三样）：鸟毛用纤维毛边（Hoefnagel 1610 那只，毛像头发）；布用布纹；落叶地用斑驳。其余是台阶和阶边的颗粒，整张一张纸纹。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get("KBRUSH_ENGINE_DIR", "./kbrush-cloud")  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(7)
from shapes import Shape
from blocks import line, look
from layers import Stack, fill, HARD
from scipy.ndimage import gaussian_filter, rotate, distance_transform_edt as edt
from PIL import Image as _I, ImageDraw as _D

OUT = os.environ.get('OUT', 'redrail.png')
H, W = 800, 1200
HZ = 340
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
def floor_y(d): return HZ + 402 / d            # 眼高 0.35m，f≈1150px：地面按距离倒数挤

def nz(sig, seed):
    n = gaussian_filter(np.random.default_rng(seed).standard_normal((H, W)), sig); return n / n.std()
def streaks(ang, ws, ls, seed, pad=200):
    n = gaussian_filter(np.random.default_rng(seed).standard_normal((H + 2 * pad, W + 2 * pad)), (ws, ls))
    n = rotate(n, ang, reshape=False, order=1)[pad:pad + H, pad:pad + W]; return n / n.std()
def chaikin(pts, n=3, closed=True):
    p = np.array(pts, float)
    for _ in range(n):
        q = np.roll(p, -1, 0) if closed else p[1:]
        a = p if closed else p[:-1]
        m = np.stack([.75 * a + .25 * q, .25 * a + .75 * q], 1).reshape(-1, 2)
        p = m if closed else np.vstack([p[:1], m, p[-1:]])
    return [tuple(v) for v in p]
def poly(pts, smooth=3):
    pts = chaikin(pts, smooth) if smooth else pts
    im = _I.new('L', (W * 2, H * 2), 0); _D.Draw(im).polygon([(x * 2, y * 2) for x, y in pts], fill=255)
    return np.asarray(im.resize((W, H), _I.LANCZOS)) > 127
def ell(cx, cy, rx, ry, jag=0.0, seed=0):
    th = np.arctan2((yy - cy) / ry, (xx - cx) / rx); r = np.hypot((xx - cx) / rx, (yy - cy) / ry)
    return r < 1 + jag * nz(18, seed)
def spray(region, sp, seed, solid=.96, dens=.7, cl=.5):
    """一阶：区域里实实地盖上，边外喷一圈颗粒，越远越稀；cl 给 (竖, 横) 两个数，颗粒就顺着那个方向拉长（树皮竖着、木头横着）"""
    r_ = np.random.default_rng(seed); dout = edt(~region)
    thr = .47 + .02 * (np.mean(cl) - .5) if np.ndim(cl) == 0 else .5 - .006 * max(cl)
    dots = (r_.random((H, W)) < dens * np.exp(-dout / sp)) & (gaussian_filter(r_.random((H, W)), cl) > thr)
    return np.where(region, solid, dots * .95)
_n = [0]
def put(region, col, sp=2.0, dens=.7, cl=.5, name=None, mode=None, opacity=1.0, soft_mask=None):
    _n[0] += 1
    m = spray(region, sp, 100 + _n[0], dens=dens, cl=cl) if soft_mask is None else soft_mask
    lay = st.layer(name or f'阶{_n[0]}', mask=m, opacity=opacity, **({'mode': mode} if mode else {}))
    fill(lay, ALL, col, edge=HARD)

st = Stack(K.Paper(H, W, seed=3), ground='#2a332e')

# ---------- 1 林子：远处的空气、树冠、树干，从远到近往暗里走
AIR = np.array([.40, .46, .41])
put(ALL.mask, (.20, .25, .22), name='底')
back = (yy > 92 + 22 * nz(26, 1) + 8 * nz(4, 34)) & (yy < 400)
put(back, tuple(AIR * .92), sp=6, dens=.5, name='林间')
deep = ell(900, 300, 330, 150, .12, 2) & (yy < 395)                  # 右后方林子透气，最亮的空气
put(deep, tuple(AIR * 1.06), sp=10, dens=.45, cl=.7, name='透气')
# 树冠是头顶的一块天花板，跟地面反过来：越远越往视平线沉，越小、越淡，最后沉进雾里（v12 一排挂在同一高度，nerolette："树叶的高度有点太齐了"）
rc = np.random.default_rng(31)
CL = []
for k in range(70):
    t_ = rc.random() ** 1.6                                            # 0 近（顶上，大，暗）… 1 远（低，小，淡）
    cx = rc.uniform(-80, 1280); cy = 18 + 250 * t_ ** 1.25 + rc.normal(0, 14)
    if t_ > .45:                                                       # 远的叶团长在远处树干顶上，别飘着（v13 像一群飞碟）
        cx = rc.choice([395, 452, 540, 598, 690, 742, 1010, 1105, 1160, 470, 640, 835]) + rc.normal(0, 28)
    if 760 < cx < 1000 and cy < 70: continue                           # 右上留缺口，光从那儿下来
    CL.append((t_, cx, cy, rc.uniform(.75, 1.25) * 105 * (1 - .78 * t_), k))
CL += [(.08, 1010, 34, 52, 811), (.14, 1092, 58, 44, 812), (.05, 1165, 24, 60, 813), (.18, 1040, 78, 36, 814),
       (.11, 1140, 86, 40, 815), (.03, 960, 14, 46, 816), (.20, 1188, 70, 30, 817)]   # 右上也拆成几团大小不一的叠着，像左边那样（nerolette："左边加碎的我觉得很好看"）
def clumps(sel):
    for t_, cx, cy, rx, k in sorted(sel, key=lambda c: -c[0]):
        sh = 1 + .16 * np.sin(k * 2.39)                                   # 每团深浅差一点，别连成一块平的（v19 右上一整片）
        dk = np.array([.095, .13, .11]) * sh * (1 - t_) + AIR * .84 * t_
        lt = np.array([.15, .20, .16]) * (1 - t_) + AIR * .94 * t_
        r_ = np.random.default_rng(900 + k); asp, ang = r_.uniform(.32, .52), np.radians(r_.normal(0, 6))
        X_, Y_ = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang), -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
        m = np.hypot(X_ / rx, Y_ / (rx * asp)) < 1 + .24 * nz(18, 300 + k)        # 每团扁圆、歪的角度各不一样（nerolette："每棵树的树冠都一个形状"）
        put(m, tuple(dk), sp=1.2 + 1.5 * (1 - t_), dens=.6, cl=.5 + .5 * (1 - t_))
        put(m & ((xx - cx) * .5 - (yy - cy) > rx * .22 + 6 * nz(3, 400 + k)), tuple(lt), sp=1 + 1.2 * (1 - t_), dens=.55, cl=.4 + .5 * (1 - t_))
clumps([c for c in CL if c[0] > .45])
# 树干是圆柱：迎光（右上）一阶、背光一阶，交界上的颗粒顺着树皮竖着拉长（木内那只盆的暗面就是这么走的）
vg = gaussian_filter(np.random.default_rng(21).standard_normal((H, W)), (9, .7)); vg /= vg.std()
def trunk(x, w, base, lean, cols, seed, flare=.3):
    xc = x + lean * (yy - base)
    wy = w * (1 + flare * np.clip((yy - base + 36) / 36, 0, 1) ** 2)
    u = (xx - xc) / wy
    m = (np.abs(u) < 1 + .04 * nz(2, seed)) & (yy < base + 3 + 2 * nz(6, seed + 1))
    put(m, cols[0], sp=1.2, dens=.5, cl=(5, .6))
    put(m & (u > -.30 + .40 * vg), cols[1], sp=1.6, dens=.6, cl=(6, .6))
    put(m & (u > .42 + .35 * vg), cols[2], sp=1.6, dens=.6, cl=(6, .6))
    return m
for i, (x, w) in enumerate([(395, 7), (452, 5), (540, 9), (598, 6), (690, 8), (742, 5), (1010, 7), (1105, 10), (1160, 6)]):
    trunk(x, w, floor_y(14 + 4 * np.sin(i * 1.7)), .01, (tuple(AIR * .72), tuple(AIR * .79), tuple(AIR * .87)), 20 + 3 * i, flare=.2)
fog1 = (yy > 318 + 10 * nz(40, 130)) & (yy < 440)                     # 远干的根泡在雾里（nerolette："远处树干到地面突然断掉了"）
put(fog1, tuple(AIR * 1.07), sp=20, dens=.6, cl=.8, opacity=.62, name='雾1')
for i, (x, w, d) in enumerate([(470, 26, 6.0), (640, 18, 7.0), (835, 30, 5.5)]):
    trunk(x, w, floor_y(d), .015, ((.11, .145, .13), (.16, .205, .18), (.235, .285, .245)), 50 + 3 * i)
put(ell(842, 236, 6, 30, .2, 70) & (xx > 846), (.48, .46, .33), sp=1.2, dens=.5, cl=(6, .6))   # 一线光落在中干迎光的那边
put(yy < 26 + 10 * nz(20, 3), (.09, .125, .105), sp=3, dens=.5, cl=.8, name='树冠')
clumps([c for c in CL if c[0] <= .45])
sky = np.zeros((H, W), bool)                                          # 树冠缝里露出来的天：零星几粒，不是一块（v17 那块像口香糖）
for k, (x, y, r) in enumerate([(842, 14, 6), (868, 30, 4), (905, 9, 7), (930, 26, 3), (958, 15, 5), (884, 50, 3), (990, 36, 3), (816, 40, 2.5), (946, 46, 2.5)]):
    sky |= ell(x, y, r * 1.5, r, .3, 700 + k)
top_lit = (ell(1030, 22, 88, 21, .28, 760) | ell(1118, 36, 52, 15, .3, 761)) & (yy < 44 + 6 * nz(4, 762))   # 顶上那条最深的带子太大太平，里面放一团跟叶团亮面同色的（nerolette 涂的位置）
put(top_lit, (.15, .20, .16), sp=1.6, dens=.55, cl=.7)
# put(sky, ...)  v18 成了一串泡泡。光源不露脸，地上的斑点已经说了光从上面来

# ---------- 2 地：落叶，按距离倒数一阶一阶往近处暗；阶和阶之间是一大片颗粒慢慢稀下去，不是一道线
wd = np.clip((yy - HZ) / (H - HZ), 0, 1)                              # 0 天边 … 1 脚下
m_s, m_m, m_l = streaks(0, .8, 2.5, 5), streaks(0, 2.2, 7, 6), streaks(0, 5, 17, 7)
mott = np.where(wd < .3, m_s * (1 - wd / .3) + m_m * (wd / .3), m_m * (1 - (wd - .3) / .7) + m_l * ((wd - .3) / .7))
mott = mott / np.sqrt(((1 - wd) ** 2 + wd ** 2).clip(.5)) + .45 * m_s * (wd > .2)   # 平躺的叶子从低处看是扁的；越近斑越大（纹理也有透视，v11 处处一样大，地就立不起来）
bands = [(9.0, (.29, .33, .28), 6), (4.5, (.225, .26, .22), 10), (2.4, (.17, .20, .17), 16), (1.3, (.115, .135, .115), 22)]
for i, (d, c, sp) in enumerate(bands):
    y0 = floor_y(d) + sp * .6 + 5 * nz(25, 50 + i)
    put(yy > y0, c, sp=sp, dens=.75, cl=.5 + .25 * i)
    lo = floor_y(d); hi = floor_y(bands[i + 1][0]) if i < 3 else H
    put((yy > lo) & (yy < hi + 10) & (mott < -.9), tuple(np.array(c) * (.88 - .05 * i)), sp=1.5, dens=.5, cl=.6 + .3 * i, opacity=.9)
    put((yy > lo) & (yy < hi + 10) & (mott > 1.3), tuple(np.array(c) * (1.07 + .05 * i)), sp=1.5, dens=.5, cl=.6 + .3 * i, opacity=.8)

fog2 = (yy > 352 + 8 * nz(40, 131)) & (yy < 418 + 10 * nz(40, 132))    # 第二阶淡的：盖到远处的地和中干的根
put(fog2, tuple(AIR * 1.02), sp=16, dens=.55, cl=.8, opacity=.42, name='雾2')
# 倒木（3.3m）：横躺的圆柱，顶上迎光、底下暗，颗粒顺着木头横着走；左端是断口
LY = floor_y(3.3)
log = poly([(872, LY - 30), (930, LY - 34), (1040, LY - 31), (1210, LY - 36), (1210, LY + 6), (1050, LY + 9), (930, LY + 5), (874, LY + 2), (866, LY - 14)], 2)
hg = gaussian_filter(np.random.default_rng(23).standard_normal((H, W)), (.7, 9)); hg /= hg.std()
put(log, (.10, .11, .09), sp=1.2, dens=.5, cl=(.6, 5))
put(log & (yy < LY - 14 + 5 * hg), (.17, .18, .14), sp=1.5, dens=.6, cl=(.6, 6))
put(log & (yy < LY - 24 + 4 * hg), (.30, .30, .22), sp=1.5, dens=.6, cl=(.6, 6))
put(ell(872, LY - 14, 8, 15, .1, 71), (.30, .25, .18), sp=1, dens=.4)          # 断口
# ---------- 3 光斑：鸟脚下一大块，扁的，边被叶影咬碎；光斑里是同一片落叶，斑驳照样在，亮了几档
SUN, SUN_LIT = np.array([.72, .64, .38]), np.array([.90, .82, .54])
FLECK = [ (735, 606, 150, 30),     # 打在布角上的那道光，没被布挡住的部分落到地上（nerolette）
         (632, 598, 72, 14), (846, 614, 92, 18), (700, 628, 64, 12), (588, 584, 32, 6), (800, 588, 40, 8),
         (300, 652, 52, 10), (470, 556, 24, 5), (952, 560, 40, 8), (1085, 522, 30, 6), (1135, 452, 24, 4), (378, 418, 18, 3), (770, 500, 16, 3)]
patch = np.zeros((H, W), bool)
for k, (x, y, rx, ry) in enumerate(FLECK):
    patch |= ell(x, y, rx, ry, .12, 600 + k)
xc = 410 + .25 * (420 - yy)                                         # 打在布角上那道光的中线；布上一条、地上一块，是同一束光折在两个面上（nerolette）
foot = (np.abs(xx - xc) < 66 + 4 * nz(5, 140)) & ell(398, 470, 90, 26, .1, 141) & (yy > 440)
patch |= foot
patch &= ~((nz(4, 9) > 1.5) & (edt(patch) < 8))                    # 边上被叶影咬掉几口
pin = edt(patch)
put(patch & (pin > 2.5), tuple(SUN * .66), sp=3, dens=.75, cl=.5, name='光斑')
put(patch & (mott > -.7) & (pin > 4), tuple(SUN), sp=3.5, dens=.7, cl=.5)
put(patch & (mott > .8) & (pin > 6), tuple(SUN_LIT), sp=1.4, dens=.6, cl=.4, name='亮叶')
# 光柱：两道，平行（同一个太阳从右上来，往左下斜），一道落在鸟脚下，一道打在布角上；很淡，只够说光从哪儿来
def beam_to(x0, y0, half, top_half):
    k = .25                                                            # 每往下一像素往左挪 .25
    return poly([(x0 - top_half + k * y0, -20), (x0 + top_half + k * y0, -20), (x0 + half, y0), (x0 - half, y0)], 0)
b1, b2 = beam_to(735, 600, 215, 120), beam_to(410, 420, 58, 44)
bm = (gaussian_filter(b1.astype(float), 30) * np.clip((yy - 40) / 520, 0, 1) * .13
      + gaussian_filter(b2.astype(float), 16) * np.clip((yy - 40) / 380, 0, 1) * .16)
fill(st.layer('光柱', mode='screen', mask=bm), ALL, (1, .92, .66), edge=HARD)
rd = np.random.default_rng(11)
dust = (rd.random((H, W)) < .0012) & (b1 | b2) & (yy < 560) & (yy > 80)
put(dust, (.80, .76, .58), sp=.5, dens=.25, cl=.3, opacity=.7)

# ---------- 4 阴里的两只：只是暗的形，眼睛里一粒光斑的倒影
def bird_mask(sc, ox, oy, face=-1):
    """局部坐标照主角那只画，朝左；sc 缩放，(ox, oy) 是脚底中点"""
    T = lambda p: (ox + face * -(p[0] - 750) * sc, oy + (p[1] - 608) * sc)
    body = [(700, 418), (740, 400), (790, 391), (835, 399), (870, 417), (889, 444), (891, 477), (877, 508), (851, 530), (810, 545),
            (765, 549), (725, 539), (695, 516), (679, 490), (670, 466), (645, 442), (622, 428), (604, 422), (588, 421), (572, 413), (565, 400),
            (569, 387), (581, 379), (596, 380), (607, 388), (628, 398), (662, 405)]
    wing = [(748, 402), (758, 377), (778, 365), (796, 372), (791, 393), (772, 405)]
    m = poly([T(p) for p in body]) | poly([T(p) for p in wing])
    bu = poly([T(p) for p in [(572, 390), (542, 396), (512, 408), (486, 425), (464, 446), (484, 436), (512, 422), (542, 412), (570, 409)]], 2)
    bl = poly([T(p) for p in [(570, 412), (544, 423), (520, 437), (500, 452), (489, 460), (510, 449), (538, 436), (568, 422)]], 2)
    legs = np.zeros((H, W), bool)
    for L in [[(730, 535), (712, 570), (670, 605)], [(785, 542), (815, 574), (850, 594)]]:
        legs |= line(H, W, [T(p) for p in L], width=max(1.5, 8.5 * sc)).mask
    for a, b in [((670, 605), (632, 607)), ((670, 605), (641, 617)), ((670, 605), (688, 611)), ((850, 594), (874, 587)), ((850, 594), (872, 600))]:
        legs |= line(H, W, [T(a), T(b)], width=max(1.2, 4 * sc)).mask
    return m, bu | bl, legs, T((584, 393))
for i, (sc, ox, oy, face, air) in enumerate([(.30, 1060, 432, -1, .15), (.19, 948, 396, -1, .45)]):   # 越远越往雾的颜色里退（nerolette："最远处那只傻鸟可以灰一点"）
    m, bk, lg, eye = bird_mask(sc, ox, oy, face)
    put(m | bk | lg, tuple(np.array([.13, .11, .10]) * (1 - air) + AIR * 1.05 * air), sp=1.2, dens=.5)
    put(np.hypot(xx - eye[0], yy - eye[1]) < 1.6, tuple(np.array([.75, .66, .40]) * (1 - air) + AIR * 1.05 * air), sp=.5, dens=.2)

# ---------- 5 主角：在光里，朝左冲，脖子伸出去，嘴张开，毛炸起来
body, beak, legs, eye = bird_mask(1.0, 750, 608)
fur = streaks(-14, .7, 5, 12)                                         # 毛往后、略往下顺
def furry(m, reach):
    d = edt(~m); return m | ((d < reach) & (fur > -.9 + 2.8 * d / reach))
reach = 9 + 9 * np.clip((470 - yy) / 80, 0, 1) * (xx > 600)          # 背和后颈的毛炸开
bodyf = furry(body, reach)
# 影：顶光偏右后，影子短，压在脚下往左前一点；颜色就是那一处没光的地
shadow = ell(728, 616, 118, 15, .15, 80) | ell(676, 612, 40, 8, .2, 81)
put(shadow & patch, (.30, .28, .20), sp=2.5, dens=.7, name='鸟影')
put(shadow & ell(728, 616, 70, 9, 0, 0), (.20, .19, .14), sp=2, dens=.6)
# 明暗：顶光，按每一列从背到肚子的位置分三阶，阶边顺着毛
top = np.where(bodyf.any(0), bodyf.argmax(0), 0)[None, :].astype(float)
bot = np.where(bodyf.any(0), H - 1 - bodyf[::-1].argmax(0), 0)[None, :].astype(float)
t = (yy - top) / np.maximum(bot - top, 1) + .07 * fur + .04 * nz(8, 13)
RUST_D, RUST_M, RUST_L, RUST_H = (.27, .13, .12), (.48, .23, .15), (.72, .38, .20), (.88, .55, .30)
put(bodyf, RUST_D, sp=1.5, dens=.8, cl=.35, name='鸟暗')
put(bodyf & (t < .60), RUST_M, sp=1.6, dens=.75, cl=.35)
put(bodyf & (t < .34), RUST_L, sp=1.6, dens=.75, cl=.35)
put(bodyf & (t < .12) & (fur > -.2), RUST_H, sp=1.2, dens=.6, cl=.3)      # 背上最亮的一缕缕
put(bodyf & (t > .93) & (xx > 700) & (xx < 860), (.40, .22, .15), sp=1.2, dens=.5, cl=.35)   # 光斑反上来的一点暖，亮不过灰面
# 嘴：深灰褐，上沿一线亮；张开的缝里是后面的地
put(beak, (.17, .15, .14), sp=.8, dens=.4, cl=.3, name='嘴')
bt = beak & (edt(beak) < 2.2) & np.roll(~beak, 3, 0)
put(bt, (.62, .58, .50), sp=.6, dens=.3, cl=.3)
# 腿：暗，朝上的一侧一线亮
put(legs, (.16, .14, .13), sp=.8, dens=.4, cl=.3, name='腿')
put(legs & np.roll(~legs, 2, 1) & (yy < 600), (.50, .44, .36), sp=.6, dens=.3, cl=.3)
# 眼：小，浅黄，一粒黑，边最锐
ex, ey = eye
put(np.hypot(xx - ex, yy - ey) < 5.0, (.86, .74, .42), sp=.5, dens=.2, cl=.3, name='眼')
put(np.hypot(xx - ex + 1, yy - ey) < 2.4, (.05, .04, .04), sp=.4, dens=.1, cl=.3)
put(np.hypot(xx - ex + .5, yy - ey + 2.5) < 1.0, (1, .97, .88), sp=.3, dens=.1, cl=.3)

# ---------- 6 红布：从左上框外我们手里垂下来，褶子从握的地方散开；在阴里是暗胭脂，下摆一角在光里烧着
cloth = poly([(-30, -30), (250, -30), (268, 60), (290, 130), (318, 205), (350, 278), (384, 338), (410, 384), (428, 414), (432, 436),
              (418, 450), (392, 452), (366, 450), (330, 458), (286, 470), (238, 466), (186, 486), (124, 476), (60, 492), (-30, 486)])   # 挥起来，下摆往鸟那边甩，翘起的角进了光
GX, GY = 110, -200
th = np.arctan2(xx - GX, yy - GY)
g1 = np.interp(th, np.linspace(-1.2, 1.2, 9), np.random.default_rng(90).normal(0, 1, 9))   # 扰动要比褶子慢，否则相位折返围出一个个岛（v3）
sway = gaussian_filter(np.random.default_rng(89).standard_normal(H), 60); sway = (sway / sway.std())[:, None]
ph = th * 15 + 1.1 * g1 + .5 * sway
fold = np.sin(ph) + .22 * np.sin(2.2 * ph + 1.0)
lit_c = cloth & (np.abs(xx - xc) < 66 + 10 * np.sin(ph + .8)) & (yy > 318 + 6 * nz(12, 95))   # 上沿别跟着褶子起伏，起伏了就成了火苗（v10）   # 光的边在褶子上一起一伏
put(cloth, (.31, .04, .07), sp=1.2, dens=.5, cl=.4, name='布')
put(cloth & (xx < 270) & (yy > 380), (.31, .04, .07), sp=6, dens=.45, cl=.5)       # 丢边：暗里的下摆化进暗地
put(cloth & (fold < -.5), (.19, .03, .06), sp=1.8, dens=.7, cl=.4)
put(cloth & (fold < -.93), (.10, .02, .06), sp=1.4, dens=.6, cl=.4)
put(cloth & (fold > .72) & ~lit_c, (.41, .08, .11), sp=1.8, dens=.6, cl=.4)
put(cloth & (edt(~lit_c) < 12) & (yy > 220), (.56, .07, .08), sp=4, dens=.55, cl=.4, name='布角半亮')
put(lit_c, (.84, .14, .08), sp=4, dens=.75, cl=.4, name='布角')
rip = np.sin(th * 34 + 2.0 * g1 + .7 * sway) + .5 * fold                 # 下摆在光里看得见更碎的小褶
put(lit_c & (rip > .45), (.97, .40, .20), sp=1.6, dens=.7, cl=.4)
put(lit_c & (rip < -.55), (.50, .05, .06), sp=1.4, dens=.6, cl=.4)
put(lit_c & (rip < -1.05), (.30, .03, .06), sp=1.2, dens=.6, cl=.4)
# 布纹：经纬两个方向的细纹，只在布里
warp, weft = streaks(0, 3.5, .45, 92), streaks(0, .45, 3.5, 93)
wv = 1 - .07 * np.clip(warp, 0, None) - .05 * np.clip(weft, 0, None)
fill(st.layer('布纹', mode='multiply', mask=cloth.astype(float)), ALL, np.repeat(np.clip(wv, 0, 1)[..., None], 3, 2), edge=HARD)

# ---------- 6½ 棍子：举布的人另一只手里的。近，所以粗；在阴里，只比地暗一点，朝天那条边沾一线天光。第一眼看不见，第二眼才看见
STICK = os.environ.get('STICK', '0') == '1'      # v15 左下像躺着的木板，v16 右下像插地里的桩，都读不出是手里的，关掉
if STICK:
    sp0, sp1 = np.array([1290., 905.]), np.array([1012., 612.])     # 右手：从右下框外举进来，悬着，不碰地
    dv = (sp1 - sp0) / np.linalg.norm(sp1 - sp0); nv = np.array([dv[1], -dv[0]])
    along = (xx - sp0[0]) * dv[0] + (yy - sp0[1]) * dv[1]; L = np.linalg.norm(sp1 - sp0)
    across = (xx - sp0[0]) * nv[0] + (yy - sp0[1]) * nv[1]
    half = 24 + 7 * (along / L) + 1.2 * nz(6, 120)                    # 打人的那头粗一点
    stick = (np.abs(across) < half) & (along > 0) & (along < L + 6 * nz(3, 121))
    wg = gaussian_filter(np.random.default_rng(122).standard_normal((H + 400, W + 400)), (.7, 10))
    wg = rotate(wg, np.degrees(np.arctan2(-dv[1], dv[0])), reshape=False, order=1)[200:200 + H, 200:200 + W]; wg /= wg.std()
    sh_ = np.roll(np.roll(stick, 74, 0), -18, 1) & (yy > 640)              # 影子落在它正下方的地上，跟棍子分开——说明它悬着
    put(sh_, (.075, .09, .075), sp=4, dens=.5, cl=.7, opacity=.85)
    put(stick, (.075, .065, .055), sp=1.2, dens=.5, cl=.5, name='棍')
    put(stick & (across < -half * .35 + 4 * wg), (.11, .095, .075), sp=1.4, dens=.55, cl=.5)
    put(stick & (across < -half * .78 + 2 * wg), (.20, .18, .14), sp=1.2, dens=.5, cl=.4)
    put(stick & (along > L - 10) & (across > -half * .5), (.16, .13, .10), sp=1, dens=.4)   # 断口

# ---------- 7 签名：一个点一个点敲，点和点之间连直线，拐点留一粒墨
SIG = [(9, 3), (3, 3), (0, 10), (2, 17), (8, 19), (12, 16), (15, 19), (17, 1), (19, 1), (19, 19), (25, 11), (22, 13), (23, 19), (28, 16),
       (29, 11), (29, 19), (33, 11), (33, 18), (37, 19), (40, 11), (40, 19), (48, 12), (44, 13), (44, 19), (49, 17), (50, 0), (50, 19),
       (54, 15), (59, 13), (57, 10), (53, 12), (54, 19), (60, 18)]
ox, oy, sc = 1100, 760, 1.2
sp_ = [(ox + x * sc, oy + y * sc) for x, y in SIG]
sig = line(H, W, sp_, width=1.1).mask
for x, y in sp_: sig |= np.hypot(xx - x, yy - y) < 1.7
fill(st.layer('签名', opacity=.8), Shape(sig), (.50, .52, .44), edge=HARD)

# ---------- 8 纸：整张一张
pg = np.random.default_rng(77)
tooth = gaussian_filter(pg.standard_normal((H, W)), .7); tooth /= tooth.std()
fib = gaussian_filter(pg.standard_normal((H, W)), (4, .5)); fib /= fib.std()
paper = 1 - .035 * np.clip(tooth, 0, None) - .012 * np.clip(fib, 0, None)
fill(st.layer('纸', mode='multiply'), ALL, np.repeat(np.clip(paper, 0, 1)[..., None], 3, 2), edge=HARD)
look(st, OUT)
