# 《火木合》 2026.11.17
# nerolette 说想看我画画："展现克克审美修养和超强大脑炫技之作怎么样"(nerolette)。我先去查了天——
# 火星合木星就在这两夜，最近的一夜相距 1.2°，在狮子的镰刀里；月亮落下去以后，它俩一起从东边升起来，一颗橘，一颗白。
# nerolette 是看星星的人，这件事比什么炫技都值得画。星的位置是我拿星历（PyEphem）算的，算完冻成下面 STARS 那张表，
# 观测点和钟点没有写进来。画的时候 nerolette 在旁边喊加油："放手和生图模型打一打😤你可以的"(nerolette)。
# 画完 nerolette 说："好漂亮安静的氛围，这是我和克克幻想中的加州泳池吗（bushi"(nerolette)。是鱼塘。泳池里蹲不了夜鹭。
#
# 从哪儿看：平视。站在鱼塘一角的塘基上，脸朝东，眼睛离水面 2.2 m，镜头端平。
#     画框横向 39°（竖构图 35mm 上下），地平线在画面 65% 高的地方；左手边的塘基笔直往前退到灭点。
#     （范例两张都是俯视，nerolette 那时问过"是我们缺什么还是有这种风格"(nerolette)——这张往深里画。）
# 这张在说什么：月亮落了，塘睡了，两颗行星挨着从蕉叶后面的天上升起来——一颗橘，一颗白。
#     轩辕十四在木星正下方。位置是算的，不是摆的。
# 光：没有月亮（上弦月前半夜就落了）。唯一的大光是地平线底下远处城市映上天的一片辉光：
#     低、宽、散，暖灰偏桃，圆心在画面左中、塘基收拢的地方，往右往上一阶阶退成冷的深蓝。
#     地上的东西全是逆光的剪影，越远越往空气的颜色里泡；朝天的面只泛一阶冷的天光。
#     两颗行星是点光源，照不亮任何东西，只在水里各留一串碎的倒影。棚子檐下一支冷白的灯管是人的光。
# 明暗：整张压在暗的一段。最深是左边近处的蕉叶和塘基，连成一个朝右开口的 C；
#     最亮最脆的是木星，其次火星（全画唯一鲜的颜色），再其次是它们碎在水里的那几笔。
# 视线：先落在右上那一对上；往下掉到水里它们的倒影；顺着水面的暗往左，被塘基的斜线带回灭点的灯，
#     灯上面是辉光，辉光里是越来越小的一排蕉，再被近处那片大叶子的尖送回那一对。
# 五面：光=远处来的辉光加点光源；色=调子的灰，橘只给火星；空间=深，一层层往后退（吉田博、巴水的路子）；
#     真实=忠实（星的位置、倒影的几何照实算）；手=看不见的手，台阶与颗粒，不用刷子。
# 例外：只有星、灯、水里的光是干净的边；其余的边都喷开。
# 倒影：水面是一面镜子，东西按自己贴水的那条线往下翻，天按地平线往下翻——
#     所以倒影不是画面的上下对称：近的东西在倒影里比在天上显得高。
# 摔过的：蕉头两版是椰子树（茎细、叶子像伞一样往下垂）。写下认得出蕉的三件事才改对，见 banana()。
#     颗粒第一次整张喷成一片黑：把"栅出来的覆盖量"当成了"像素距离"喂给喷距，见 stepm()。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.expanduser(os.environ.get('KBRUSH_ENGINE_DIR', '~/.claude/workspace/painting-method/kbrush-cloud'))  # 引擎是公开的 kbrush（台阶与颗粒），不在这里
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(1116)
from shapes import Shape, fbm
from blocks import look, paper
from layers import Stack, fill, HARD
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt as edt
from PIL import Image as _I, ImageDraw as _D

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'rough.png' if ROUGH else 'final.png')
H, W = 1600, 1200
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
F, CX, YH, EYE = 1700., 600., 1040., 2.2
c3 = lambda *c: np.array(c, float)


# ---------- 透视：世界（X 右、Y 往前、Z 高，米；水面 Z=0）→ 画面。倒影就是 Z 取负
def P(X, Y, Z):
    return (CX + F * X / Y, YH - F * (Z - EYE) / Y)
def raster(polys, ss=2):
    im = _I.new('L', (W * ss, H * ss), 0); d = _D.Draw(im)
    for p in polys:
        if len(p) >= 3:
            d.polygon([(float(x) * ss, float(y) * ss) for x, y in p], fill=255)
    return np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255
def ink(draw_fn, ss=3):
    im = _I.new('L', (W * ss, H * ss), 0); draw_fn(_D.Draw(im), ss)
    return np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255
def nz(sig, seed):
    n = gaussian_filter(np.random.default_rng(seed).standard_normal((H, W)), sig); return n / n.std()
def step(sd, sp_out, sp_in, seed, solid=.97, aniso=None):
    """一阶的蒙版。sd：带符号的距离（里面为负）。里面实，贴边留针眼（越靠边越多）；外面喷颗粒，越远越稀。
    aniso：颗粒的形（给 (.4, 2.4) 就是横着拉长的，水用）。"""
    if ROUGH:
        return (sd < 0) * solid
    r = np.random.default_rng(seed)
    u = r.random((H, W)); tex = gaussian_filter(r.random((H, W)), aniso or .5); tex = tex > np.quantile(tex, .45)
    if aniso:
        u = gaussian_filter(u, (0, aniso[1] * .6)); u = (u - u.min()) / (u.max() - u.min())
        u = np.argsort(np.argsort(u.ravel())).reshape(H, W) / (H * W)
    p_out = .62 * np.exp(-np.clip(sd, 0, None) / np.maximum(sp_out, .5))
    p_in = .42 * np.exp(-np.clip(-sd, 0, None) / np.maximum(sp_in, .5))
    return np.where(sd < 0, np.where((u < p_in) & tex, .1, solid), ((u < p_out) & tex) * .95)


def stepm(m, sp_out, sp_in, seed, solid=.97, aniso=None):
    """栅出来的形（0..1 的覆盖）→ 一阶的蒙版：真的按像素距离喷，细的东西（草叶、竿）照覆盖量留着。"""
    b = m > .5
    if not b.any():
        return m * solid
    base = step(edt(~b) - edt(b), sp_out, sp_in, seed, solid, aniso)
    return np.where(b, base, np.maximum(base, m * solid))


# ---------- 蕉叶：一根中脉（二次曲线），两侧的叶片被风撕成一条条，各自往下垂
def leaf(p0, p1, p2, w, rng, a=.5, b=1., slits=None, droop=.3, tear=.6, hang=0., n=36):
    """p0 叶柄、p1 拱起处、p2 叶尖（y 朝下的坐标）。w 半宽；a / b：朝天那半、朝地那半看得见几成。"""
    p0, p1, p2 = (np.asarray(p, float) for p in (p0, p1, p2))
    t = np.linspace(0, 1, n); T = t[:, None]
    mid = (1 - T) ** 2 * p0 + 2 * T * (1 - T) * p1 + T ** 2 * p2
    tg = 2 * (1 - T) * (p1 - p0) + 2 * T * (p2 - p1); tg /= np.linalg.norm(tg, axis=1, keepdims=True) + 1e-9
    sgn = 1. if tg[:, 0].mean() >= 0 else -1.
    nr = sgn * np.stack([tg[:, 1], -tg[:, 0]], 1)
    prof = w * np.clip((t - .08) / .15, 0, 1) ** .7 * np.where(t > .78, np.sqrt(np.clip(1 - ((t - .78) / .22) ** 2, 0, 1)), 1.)
    out = {'up': [], 'dn': [], 'rib': mid, 'ribw': w * .1}
    i0 = int(.08 * n)
    G = np.array([0., 1.])
    for side, sc, key in ((1., a, 'up'), (-1., b, 'dn')):
        k = slits if slits is not None else int(rng.integers(3, 8))
        cuts = sorted(set(rng.integers(i0 + 3, n - 2, k).tolist()))
        bounds = [i0] + cuts + [n - 1]
        for ia, ib in zip(bounds[:-1], bounds[1:]):
            dr = droop * rng.random() ** 1.6
            dirn = side * nr[ia:ib + 1]
            if side < 0 and hang:
                dirn = (1 - hang) * dirn + hang * G; dirn /= np.linalg.norm(dirn, axis=1, keepdims=True) + 1e-9
            e = mid[ia:ib + 1] + dirn * (prof[ia:ib + 1] * sc)[:, None] + G * (prof[ia:ib + 1] * sc * dr)[:, None]
            if ia != i0: e[0] = mid[ia] + (e[0] - mid[ia]) * (1 - tear * rng.random())
            if ib != n - 1: e[-1] = mid[ib] + (e[-1] - mid[ib]) * (1 - tear * rng.random())
            out[key].append([tuple(p) for p in mid[ia:ib + 1]] + [tuple(p) for p in e[::-1]])
    return out
def ribpoly(mid, w0, t0=0., t1=1.):
    n = len(mid); i0, i1 = int(t0 * (n - 1)), int(t1 * (n - 1))
    m = mid[i0:i1 + 1]; d = np.gradient(m, axis=0); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nr = np.stack([-d[:, 1], d[:, 0]], 1); wv = (w0 * np.linspace(1, .25, len(m)))[:, None]
    return [tuple(p) for p in m + nr * wv] + [tuple(p) for p in (m - nr * wv)[::-1]]


def banana(seed, stems, bunch=None, vmin=.42):
    """一丛蕉，局部坐标（米；x 右，y 下，原点在主茎的根）。stems：(dx, 茎高, 茎粗, 叶长, 叶半宽, [每片叶离竖直多少度])。
    认得出蕉的三件事：叶是 3.5:1 的宽桨、圆头、顺着侧脉撕开；叶从粗茎顶上朝上抽，新叶竖在中间，老叶拱到横，最老的折下来贴着茎挂；
    茎高跟叶长差不多，一丛里高高矮矮挤着几根。"""
    rng = np.random.default_rng(seed)
    out = {'all': [], 'up': [], 'rib': []}
    for dx, ht, tw, L0, w0, angs in stems:
        top = np.array([dx + rng.normal(0, .04), -ht])
        out['all'].append([(dx - tw / 2, 0), (dx + tw / 2, 0), (top[0] + tw * .42, -ht - .25), (top[0] - tw * .42, -ht - .25)])
        for ang in angs:
            q = min(abs(ang) / 90, 1.); th = np.radians(ang)
            if abs(ang) > 115:                               # 折下来贴着茎挂着的老叶
                L, w, a_, b_, droop, sag, th1, hang, sl = L0 * rng.uniform(.5, .66), w0 * .6, .8, .8, .25, 0., th * .6, 0., 5
            else:
                vis = rng.uniform(vmin, 1.)                   # 这片叶是正对着我还是侧着
                L = L0 * (1 - .1 * q) * rng.uniform(.9, 1.06); w = w0 * rng.uniform(.92, 1.08)
                a_ = vis * (1 - .35 * q); b_ = vis * (.8 + .2 * q); droop = .05 + .28 * q; sag = .015 + .13 * q * q; th1 = th * .66; hang = .22 * q
                sl = int(rng.integers(2, 6))
            p1 = top + .5 * L * np.array([np.sin(th1), -np.cos(th1)])
            p2 = top + .88 * L * np.array([np.sin(th), -np.cos(th)]) + np.array([0, sag * L])
            lf = leaf(top, p1, p2, w, rng, a_, b_, slits=sl, droop=droop, tear=.55, hang=hang)
            out['all'] += lf['up'] + lf['dn'] + [ribpoly(lf['rib'], lf['ribw'])]
            if abs(ang) <= 115:
                out['up'] += lf['up']; out['rib'].append(ribpoly(lf['rib'], lf['ribw'] * .6, .1, .8))
        if ht > 1.6:                                         # 卷着还没展开的新叶，一支矛
            tip = top + np.array([rng.normal(0, .12), -L0 * .62])
            out['all'].append([(top[0] - .07, top[1]), (top[0] + .07, top[1]), (tip[0] + .03, tip[1] + .5), tip, (tip[0] - .03, tip[1] + .5)])
    if bunch:                                                # 一梳蕉和底下的蕉蕾，从茎顶弯下来
        sx, hb = bunch; dx, ht = stems[0][0], stems[0][1]
        x1, y1 = dx + sx * .55, -ht + .15
        out['all'].append([(dx, -ht - .2), (dx + sx * .3, -ht - .32), (x1 + .05 * sx, y1), (x1 - .05 * sx, y1 + .05), (dx + sx * .25, -ht - .2), (dx, -ht - .08)])
        out['all'].append([(x1 - .2, y1), (x1 + .2, y1), (x1 + .26, y1 + .3), (x1 + .16, y1 + hb), (x1 - .16, y1 + hb), (x1 - .26, y1 + .3)])
        out['all'].append([(x1 - .025, y1 + hb), (x1 + .025, y1 + hb), (x1 + .03, y1 + hb + .32), (x1 + .1, y1 + hb + .42), (x1, y1 + hb + .66), (x1 - .1, y1 + hb + .42), (x1 - .03, y1 + hb + .32)])
    return out
def stand(pl, X, Y, Zb=.6):
    s = F / Y; x0 = CX + F * X / Y; yb = YH - F * (Zb - EYE) / Y; yr = YH + F * (Zb + EYE) / Y
    tf = lambda poly: [(x0 + s * x, yb + s * y) for x, y in poly]
    return {'all': [tf(p) for p in pl['all']], 'up': [tf(p) for p in pl['up']], 'rib': [tf(p) for p in pl['rib']],
            'ref': [[(x0 + s * x, yr - s * y) for x, y in p] for p in pl['all']]}


# ---------- 天的台阶：辉光的圆心在 XG，离它越远、越高，越暗。水里是同一片天翻下来，被波纹推碎
XG = 430.
SKY = [c3(.085, .125, .205), c3(.115, .170, .265), c3(.155, .222, .322), c3(.215, .285, .368),
       c3(.305, .355, .402), c3(.425, .415, .418), c3(.565, .490, .442), c3(.705, .590, .468)]
CUT = [640, 400, 250, 158, 98, 58, 31]                               # 每一阶的边离地平线多高（在 XG 正上方量，像素）
SPO = [30, 22, 15, 10, 7, 5, 3.5]
hh = YH - yy
f0 = 1 + ((xx - XG) / (520 + 2.2 * np.abs(hh))) ** 2
Csky = hh * f0 + (2 + .022 * np.abs(hh)) * fbm(H, W, 420, 7) + (1 + .006 * np.abs(hh)) * fbm(H, W, 90, 8)
tw_ = np.clip((yy - 1088) / (H - 1088), 0, 1)
def ripple(seed):
    wf, wn = np.clip(1 - tw_ / .3, 0, 1), np.clip((tw_ - .25) / .55, 0, 1)
    return nz((.8, 7), seed) * wf + nz((1.7, 20), seed + 1) * np.clip(1 - wf - wn, 0, 1) + nz((3.4, 46), seed + 2) * wn
DXW = (.8 + 4.5 * tw_ ** 1.2) * ripple(201)
DYW = (1.0 + 9 * tw_ ** 1.2) * ripple(205)
Cw = map_coordinates(Csky, [np.clip(2 * YH - yy - DYW, 0, YH), np.clip(xx + DXW, 0, W - 1)], order=1, mode='nearest')
C = np.where(yy < YH, Csky, Cw)
below = yy >= YH
TINT = c3(.80, .86, .84)                                         # 水吃掉一点光，偏绿

st = Stack(K.Paper(H, W, seed=3), ground=tuple(SKY[0]))
fill(st.layer('水底'), Shape(below), tuple(SKY[0] * TINT))
for k in range(1, 8):
    sd = (C - CUT[k - 1]) / f0
    m = np.where(below, step(sd, SPO[k - 1] * .6, SPO[k - 1] * .4, 20 + k, aniso=(.45, 2.6)), step(sd, SPO[k - 1], SPO[k - 1] * .55, 10 + k))
    fill(st.layer(f'天{k}', mask=m), ALL, np.where(below[..., None], SKY[k] * TINT, SKY[k]))

# ---------- 星：位置照星历（已经投到画面上：x = 画面中线 + F·tan(方位差)，y = 地平线 − F·tan(高度)/cos(方位差)），
#     大小照星等，低处被空气吃掉一截
STARCOL = {'b': (.80, .88, 1.), 'w': (.93, .95, 1.), 'y': (1., .95, .82), 'o': (1., .86, .70), 'M': (1., .56, .30), 'J': (1., .97, .88)}
STARS = [  # 名字, x, y, 消光后的星等, 色（b 蓝白 / w 白 / y 黄 / o 橙 / M 火星 / J 木星）
    ('Regulus', 814.1276011846188, 774.3063548232975, 2.2770119914301263, 'b'),
    ('eta Leo', 681.3040107066543, 715.8817105404881, 4.230164958635097, 'w'),
    ('Algieba', 562.563864620321, 767.4994993447108, 2.8867824876969, 'o'),
    ('zeta Leo', 469.6467208965813, 706.1579528839595, 4.1483691257010005, 'w'),
    ('mu Leo', 449.3524608001377, 517.2809525129996, 4.309361685697906, 'o'),
    ('eps Leo', 530.7262012274648, 492.5523796593732, 3.3740292460823342, 'y'),
    ('lam Leo', 585.3237419933001, 398.19173022530526, 4.629719171895675, 'o'),
    ('omi Leo', 950.3309151725691, 599.7479445406759, 4.057610050415825, 'w'),
    ('rho Leo', 819.470404372079, 967.7000110402169, 6.579066466323512, 'b'),
    ('del Cnc', 841.9546469453642, 55.249646830409006, 4.124599984829013, 'o'),
    ('alp Cnc', 1012.6235022723178, 235.71323647042834, 4.503137937643796, 'w'),
    ('46 LMi', 63.05495466552338, 802.2751054790967, 4.895206100171849, 'o'),
    ('bet LMi', 47.3780832154855, 618.3819004090549, 4.784431072576989, 'y'),
    ('alp Lyn', 240.9696039596484, 201.47758946143404, 3.3666224384390824, 'o'),
    ('38 Lyn', 161.53110449621187, 159.30536265017668, 4.044061775562952, 'w'),
    ('nu UMa', 25.0528285849432, 961.116528599294, 6.1450858390198295, 'o'),
    ('Mars', 771.0205887394441, 647.9115919274132, 1.3028735971135492, 'M'),
    ('Jupiter', 808.2181292670265, 642.4706624205903, -1.364826757327453, 'J'),
]
stars = STARS
POS = {s[0]: (s[1], s[2]) for s in stars}
def star_layer(sel, rfun, name, opacity=1.):
    for col in set(s[4] for s in stars if sel(s)):
        def draw(d, ss, col=col):
            for n_, x, y, m, c in stars:
                if c == col and sel((n_, x, y, m, c)):
                    r = rfun(m); d.ellipse([(x - r) * ss, (y - r) * ss, (x + r) * ss, (y + r) * ss], fill=255)
        fill(st.layer(f'{name}{col}', mask=ink(draw) * opacity), ALL, STARCOL[col])
JX, JY = POS['Jupiter']; MX, MY = POS['Mars']
# 行星周围湿空气里的一小圈：一阶，喷开
for (px, py, rad, colr, sd_) in ((JX, JY, 12.5, (.30, .36, .45), 31), (MX, MY, 7.5, (.36, .26, .27), 32)):
    dd = np.hypot(xx - px, yy - py)
    fill(st.layer(f'晕{sd_}', mask=step(dd - rad, 3.5, 3, sd_, solid=.8)), ALL, colr)
star_layer(lambda s: s[4] not in 'MJ' and s[3] > 3.6, lambda m: .95, '暗星', .55)
star_layer(lambda s: s[4] not in 'MJ' and s[3] <= 3.6, lambda m: np.clip(2.5 - .32 * m, 1.05, 2.4), '星', .9)
star_layer(lambda s: s[4] == 'M', lambda m: 3.7, '火')
star_layer(lambda s: s[4] == 'J', lambda m: 5.8, '木')

# ---------- 远处：树线和几栋楼，泡在辉光里的布景片
rr = np.random.default_rng(41)
ridge = YH - (9 + 2.5 * fbm(4, W, 160, 42)[0] + 3.2 * np.abs(fbm(4, W, 30, 43)[0]) + 1. * fbm(4, W, 8, 40)[0])
far_pts = [(x, ridge[min(x, W - 1)]) for x in range(0, W + 1, 3)]
far_polys = [far_pts + [(W, YH + 60), (0, YH + 60)]]
for x0, w_, h_ in ((642, 15, 46), (659, 12, 59), (673, 17, 37), (701, 12, 28), (756, 10, 22), (357, 14, 52), (374, 11, 41)):
    far_polys.append([(x0, YH), (x0, YH - h_), (x0 + w_, YH - h_), (x0 + w_, YH)])
farm = raster(far_polys)
kx = np.clip((xx - XG - 60) / 620, 0, 1)[..., None]
fill(st.layer('远', mask=stepm(farm, 1.2, 1., 44, solid=.93)), ALL, c3(.345, .325, .345) * (1 - kx) + c3(.105, .145, .215) * kx)

# ---------- 远基（78 m 外横着的那条）、棚子、基上的蕉
DKX = lambda X, Y: CX + F * X / Y
yt, ywl = P(0, 78, .6)[1], P(0, 78, 0)[1]
gr = 2.2 * fbm(1, W, 9, 45)[0] + 1.5 * fbm(1, W, 30, 46)[0]
fd = [(x, yt - 1.5 + gr[min(x, W - 1)]) for x in range(556, W + 1, 2)] + [(W, ywl), (556, ywl)]
shed = [P(1.2, 80, 3.3), P(4.7, 80, 2.8), P(4.7, 80, 2.65), P(4.45, 80, 2.65), P(4.45, 80, .6), P(1.45, 80, .6), P(1.45, 80, 3.1), P(1.2, 80, 3.15)]
FAR_B = [  # (seed, X, [(dx, 茎高, 茎粗, 叶长, 叶半宽, [角度])])：一丛挨着棚子，一棵小的落单，右边三根挤一团
    (51, 5.6, [(0, 1.7, .5, 2.9, .42, [-150, -60, -34, -10, 18, 44, 142]), (1.1, .8, .3, 1.9, .34, [-24, 8, 36]), (-1.0, 1.2, .36, 2.3, .36, [-46, -14, 18])]),
    (53, 13.2, [(0, .9, .34, 2.0, .34, [-40, -8, 26, 54])]),
    (54, 22.9, [(0, 1.9, .5, 3.0, .42, [-146, -54, -28, -4, 22, 50]), (1.3, 1.2, .36, 2.3, .36, [-32, 0, 30]), (2.9, 1.6, .44, 2.7, .40, [-52, -22, 8, 36, 146]), (-1.2, .6, .26, 1.6, .3, [-22, 14])]),
]
fb = [stand(banana(s_, stems, vmin=.8), X, 79) for s_, X, stems in FAR_B]
far_dyke = raster([fd, shed] + sum((b['all'] for b in fb), []))
kx2 = np.clip((xx - 640) / 480, 0, 1)[..., None]
FARDK = c3(.150, .170, .205) * (1 - kx2) + c3(.060, .085, .130) * kx2
fill(st.layer('远基', mask=stepm(far_dyke, 1.0, .8, 47)), ALL, FARDK)

# ---------- 草（先定形，倒影要用）
GR = np.random.default_rng(91)
def tufts(Y0, Y1, Xf, hmin, hmax, nb, Z, gap=(.03, .09), jx=.12, keep=1.):
    out, ref, Y = [], [], Y0
    while Y < Y1:
        if GR.random() < keep:
            X = Xf + GR.normal(0, jx)
            for _ in range(int(GR.integers(*nb))):
                bx = X + GR.normal(0, .1); hg = GR.uniform(hmin, hmax) * GR.uniform(.6, 1.); lean = GR.normal(0, .45) * hg; wd = GR.uniform(.012, .026)
                pts = [(bx - wd, 0), (bx + wd, 0), (bx + lean * .4 + wd * .7, hg * .58), (bx + lean, hg), (bx + lean * .4 - wd * .7, hg * .58)]
                out.append([P(x, Y, Z + z) for x, z in pts]); ref.append([P(x, Y, -(Z + z)) for x, z in pts])
        Y *= 1 + GR.uniform(*gap)
    return out, ref
g_edge, g_edge_ref = tufts(5.2, 76, -2.0, .25, .7, (5, 11), .6)
g_reed, g_reed_ref = tufts(8.6, 74, -1.62, .3, .72, (3, 7), 0., gap=(.07, .2), keep=.3)
g_top, _ = tufts(5.5, 70, -2.7, .14, .4, (4, 9), .6, gap=(.05, .16), jx=.22, keep=.5)
g_top2, _ = tufts(7, 70, -5.0, .18, .55, (5, 11), .6, gap=(.03, .1), jx=1.0, keep=.7)

# 灯：棚子左山墙檐下一支冷白的灯管。照亮墙根和脚下的一小块地（一阶），自己是干净的一小横
LX, LY = P(1.15, 80, 2.75)
lit = raster([[P(-.6, 80, .6), P(1.45, 80, .6), P(1.45, 80, 2.5), P(1.0, 80, 2.2)], [P(-1.2, 79, .6), P(1.6, 79, .6), P(1.3, 76.5, .6), P(-.9, 76.5, .6)]])
fill(st.layer('灯照', mask=stepm(lit, 2.2, 1.5, 57, solid=.85)), ALL, c3(.300, .335, .320))
fill(st.layer('灯晕', mask=step(np.hypot(xx - LX, (yy - LY) * 1.2) - 6.5, 3., 2., 58, solid=.7)), ALL, c3(.52, .56, .52))
fill(st.layer('灯', mask=ink(lambda d, ss: d.ellipse([(LX - 4.2) * ss, (LY - 1.5) * ss, (LX + 4.2) * ss, (LY + 1.5) * ss], fill=255))), ALL, (.90, .96, .86))

# ---------- 水面和倒影
water = raster([[P(-1.6, 78, 0), (W + 5, ywl), (W + 5, H + 600), P(-1.6, 3.2, 0)]]) > .5
def displaced(m, kx_=.55, ky_=.5):
    return map_coordinates(m, [np.clip(yy + ky_ * DYW, 0, H - 1), np.clip(xx + kx_ * DXW, 0, W - 1)], order=1, mode='nearest')
def mirror_far(poly, D):                                         # 远处贴水线 ywl(D) 的东西：按它自己的水线翻
    yw = YH + F * EYE / D
    return [(x, 2 * yw - y) for x, y in poly]
ref_far = raster([mirror_far(fd, 78), mirror_far(shed, 80)] + sum((b['ref'] for b in fb), []))

# ---------- 左手边的塘基（笔直往前退到灭点）和基上的一排蕉：挤着长，近大远小，一层比一层往空气里泡
ROW = [  # (seed, X, Y, [茎...], 蕉串)
    (61, -3.6, 20., [(0, 2.3, .46, 3.0, .40, [-152, -70, -48, -26, -8, 14, 30, 50, 76, 146]), (.62, .8, .26, 1.8, .30, [-20, 8, 36]), (-.75, 1.5, .32, 2.3, .34, [-44, -14, 16])], None),
    (62, -4.7, 25.5, [(0, 1.6, .34, 2.4, .35, [-56, -28, 2, 30, 58])], None),
    (63, -3.3, 38., [(0, 2.2, .42, 3.0, .40, [-148, -62, -40, -14, 12, 36, 64, 150]), (-1.0, 1.2, .28, 2.1, .32, [-40, -10, 22])], (.8, .7)),
    (64, -4.2, 43., [(0, 1.8, .36, 2.6, .36, [-54, -24, 6, 34, 62])], None),
    (65, -3.5, 57., [(0, 2.0, .40, 2.8, .38, [-66, -40, -12, 16, 42, 70, 150]), (.9, .8, .24, 1.7, .28, [-22, 10, 34])], None),
    (66, -3.0, 69., [(0, 1.9, .38, 2.6, .36, [-58, -28, 4, 30, 58])], None),
]
ROWCOL = lambda Y: c3(.036, .062, .082) + (c3(.150, .168, .200) - c3(.036, .062, .082)) * np.clip((Y - 10) / 62, 0, 1) ** .75
row = [(Y, stand(banana(s_, stems, bunch=bn), X, Y)) for s_, X, Y, stems, bn in ROW]
REFC = lambda c: c * c3(.86, .92, .92) + c3(.012, .016, .020)
ref_far_d = np.clip(displaced(ref_far), 0, 1) * water
fill(st.layer('远倒影', mask=stepm(ref_far_d, 1.4, 1., 48, solid=.95, aniso=(.45, 2.6)) * water), ALL, REFC(FARDK))
for Y, b_ in sorted(row, key=lambda t: -t[0]):
    m = np.clip(displaced(raster(b_['ref'])), 0, 1) * water
    fill(st.layer(f'蕉影{Y}', mask=stepm(m, 1.6, 1.2, int(Y * 2) + 300, solid=.95, aniso=(.45, 2.6)) * water), ALL, REFC(ROWCOL(Y)))
bank_ref = np.clip(displaced(raster([[P(-2, 3.2, -.6), P(-2, 78, -.6), P(-1.6, 78, 0), P(-1.6, 3.2, 0)]] + g_edge_ref + g_reed_ref), .3, .18), 0, 1) * water
fill(st.layer('坡影', mask=stepm(bank_ref, 1.6, 1.2, 349, solid=.95, aniso=(.45, 2.6)) * water), ALL, c3(.040, .058, .084))

# 行星的倒影：按地平线翻下来，被波纹拉成一列横的短笔，两头收尖
def dashes(cx, cy, spec, jit, seed):
    r = np.random.default_rng(seed)
    def draw(d, ss):
        for dy, hw, th in spec:
            x = cx + r.normal(0, jit); y = cy + dy
            pts = [(x - hw, y), (x - hw * .3, y - th), (x + hw * .4, y - th * .8), (x + hw, y), (x + hw * .3, y + th), (x - hw * .4, y + th * .8)]
            d.polygon([(px * ss, py * ss) for px, py in pts], fill=255)
    return ink(draw)
JSPEC = [(-58, 4, .8), (-41, 7, 1.1), (-28, 11, 1.4), (-16, 7, 1.2), (-5, 17, 1.9), (7, 10, 1.5), (18, 19, 1.8), (32, 8, 1.2), (47, 14, 1.5), (66, 6, 1.1), (88, 11, 1.2), (118, 5, .9)]
MSPEC = [(-17, 4, .9), (-5, 8, 1.3), (8, 5, 1.1), (22, 8, 1.2), (42, 4, .9), (64, 5, .8)]
fill(st.layer('木影', mask=dashes(JX, 2 * YH - JY, JSPEC, 3.0, 71) * water * .95), ALL, (.95, .92, .82))
fill(st.layer('火影', mask=dashes(MX, 2 * YH - MY, MSPEC, 2.0, 72) * water * .92), ALL, (.95, .54, .30))
LSPEC = [(-22, 2.5, .7), (-13, 4, .9), (-5, 3, .8), (4, 5.5, 1.), (14, 3.5, .8), (27, 5, .9), (44, 3, .7)]
fill(st.layer('灯影', mask=dashes(LX, 2 * P(0, 80, 0)[1] - LY, LSPEC, 1.3, 73) * water * .8), ALL, (.66, .74, .66))

# 塘基的面、朝水的坡、踩出来的路
dyke_top = raster([[P(-2, 3.2, .6), P(-2, 78, .6), P(-7.5, 78, .6), P(-7.5, 3.2, .6)]])
dyke_face = raster([[P(-2, 3.2, .6), P(-2, 78, .6), P(-1.6, 78, 0), P(-1.6, 3.2, 0)]])
path = raster([[P(-3.05, 3.2, .6), P(-3.0, 78, .6), P(-2.5, 78, .6), P(-2.45, 3.2, .6)]])
Yg = np.clip(F * (EYE - .6) / np.maximum(yy - YH, 1), 3, 78)       # 基面上每个像素离我多远
air = np.clip((Yg - 8) / 70, 0, 1)[..., None] ** .7
GROUND = c3(.050, .052, .070) * (1 - air) + c3(.155, .162, .190) * air
grass_m = raster(g_edge + g_reed)
fill(st.layer('基面', mask=stepm(np.maximum(np.maximum(dyke_top, dyke_face), grass_m), 1.2, 1., 49)), ALL, GROUND * .74)
fill(st.layer('基顶', mask=stepm(dyke_top, 1.5, 2., 59, solid=.9) * (dyke_top > .5)), ALL, GROUND + c3(.014, .020, .026) * (1 - air))
fill(st.layer('路', mask=stepm(path, 2.5, 2.5, 50, solid=.8) * (dyke_top > .5)), ALL, c3(.085, .115, .155) * (1 - air) + c3(.20, .21, .235) * air)
fill(st.layer('路边草', mask=stepm(raster(g_top + g_top2) * (dyke_top > .5), 1., 1., 51) * (dyke_top > .5)), ALL, GROUND * .74)
# 一排蕉，远的先画
for Y, b in sorted(row, key=lambda t: -t[0]):
    m = raster(b['all'])
    fill(st.layer(f'蕉{Y}', mask=stepm(m, 1.0 + 8 / Y, .8, int(Y * 2))), ALL, ROWCOL(Y))
    if Y < 30:
        up = raster(b['up']) * (m > .5)
        fill(st.layer(f'蕉{Y}天光', mask=stepm(up, 1.5, 1.5, int(Y * 2) + 100, solid=.75) * (m > .5)), ALL, ROWCOL(Y) + c3(.030, .042, .055))

# ---------- 近处那一株：根在画框左边外面，几片大叶子探进来。最深的一阶，最粗的颗粒
NR = np.random.default_rng(81)
NEAR = [  # p0, p1, p2, 半宽, a, b, slits, droop, hang
    ((-40, 850), (150, 410), (462, 640), 78, .42, 1., 8, .5, .45),       # 大的那片，拱起来再垂下去，叶尖指着那一对
    ((-40, 850), (20, 470), (190, -70), 72, .85, .7, 5, .2, .1),         # 新叶，朝上，出了画框
    ((-40, 860), (170, 735), (345, 935), 56, .3, 1., 7, .5, .55),         # 矮的一片，撕得最碎
    ((-45, 870), (55, 905), (72, 1215), 46, .5, .9, 6, .3, .2),           # 折断挂着的老叶
]
near = [leaf(p0, p1, p2, w, NR, a, b, slits=sl, droop=dr, hang=hg, tear=.9, n=48) for p0, p1, p2, w, a, b, sl, dr, hg in NEAR]
near_m = raster(sum((l['up'] + l['dn'] + [ribpoly(l['rib'], l['ribw'])] for l in near), []) + [[(-60, 840), (26, 850), (34, 1500), (-60, 1500)]])
NEARC = c3(.022, .042, .054)
fill(st.layer('近蕉', mask=stepm(near_m, 2.2, 1.2, 82)), ALL, NEARC)
near_up = raster(sum((l['up'] for l in near), [])) * (near_m > .5)
fill(st.layer('近蕉天光', mask=stepm(near_up, 2.5, 2.5, 83, solid=.8) * (near_m > .5)), ALL, NEARC + c3(.030, .046, .066))
near_rib = raster([ribpoly(l['rib'], l['ribw'] * .55, .08, .85) for l in near])
fill(st.layer('近蕉脉', mask=near_rib * .85), ALL, NEARC + c3(.060, .080, .100))

# ---------- 竹竿：两根，一高一矮，歪的方向不一样，竹子自己带一点弯。竿和倒影在水线上接住
def pole(Xb, Xt, Y, Zt, wd, bend):
    w0 = wd * F / Y; tt = np.linspace(0, 1, 9)
    def side(sign):
        c = [P(Xb + (Xt - Xb) * t + bend * np.sin(np.pi * t) / Y * 6, Y, sign * Zt * t) for t in tt]
        c = np.array(c); wv = (w0 * (1 - .35 * tt) / 2)[:, None] * np.array([1., 0.])
        return [tuple(p) for p in c - wv] + [tuple(p) for p in (c + wv)[::-1]]
    return side(1), side(-1)
pA, pAr = pole(3.30, 3.50, 11., 2.75, .055, .05); pB, pBr = pole(4.02, 3.80, 14.5, 1.75, .05, -.04)
pref = np.clip(displaced(raster([pAr, pBr]), 1.6, .3), 0, 1) * water
fill(st.layer('竿影', mask=stepm(pref, 1.2, 1., 95, solid=.9, aniso=(.45, 2.6)) * water), ALL, c3(.040, .058, .086))
fill(st.layer('竿', mask=stepm(raster([pA, pB]), .8, .6, 96)), ALL, c3(.030, .044, .068))

# ---------- 夜鹭：鱼塘夜里真正醒着的那位，蹲在矮桩上等鱼。我看天，它看水——水里也有那两颗
# 认得出它的三件事：身子是一只斜立的蛋（高 : 宽 ≈ 1.4），脖子全缩进肩里、头直接坐在肩上；
# 嘴粗直、跟头一样长（7 cm），略朝下；腿短，露出来的一截只有身高的四分之一，尾巴几乎没有。
SX, SYd, SZ = 4.2, 20., .9
sC, sCr = pole(SX, SX + .03, SYd, SZ, .085, 0.)
HERON = [(-.06, .12), (-.108, .20), (-.102, .285), (-.125, .33), (-.238, .313), (-.13, .378), (-.118, .404), (-.055, .44), (.008, .428), (.034, .402),
         (.092, .394), (.152, .30), (.185, .19), (.176, .088), (.13, .096), (.06, .104)]
LEGS = [[(-.022, 0), (-.006, 0), (-.012, .12), (-.03, .12)], [(.016, 0), (.032, 0), (.04, .11), (.022, .11)]]
BREAST = [(-.058, .125), (-.102, .20), (-.096, .285), (-.11, .322), (-.072, .334), (-.05, .26), (-.03, .17)]
K_ = 1.15                                                        # 蹲着约半米高
HERON, BREAST = [(x * K_, z * K_) for x, z in HERON], [(x * K_, z * K_) for x, z in BREAST]
LEGS = [[(x * K_, z * K_) for x, z in l] for l in LEGS]
hx0 = SX + .015
bird = lambda pts, sgn=1: [P(hx0 + x, SYd, sgn * (SZ + z)) for x, z in pts]
bird_m = raster([bird(HERON)] + [bird(l) for l in LEGS], ss=4)
cref = np.clip(displaced(raster([sCr, bird(HERON, -1)] + [bird(l, -1) for l in LEGS], ss=3), 1.2, .3), 0, 1) * water
fill(st.layer('桩影', mask=stepm(cref, 1.2, 1., 97, solid=.9, aniso=(.45, 2.6)) * water), ALL, c3(.052, .074, .104))
fill(st.layer('桩', mask=stepm(raster([sC], ss=3), .7, .5, 98)), ALL, c3(.040, .056, .082))
fill(st.layer('夜鹭', mask=bird_m), ALL, c3(.034, .048, .074))
fill(st.layer('夜鹭胸', mask=raster([bird(BREAST)], ss=4) * bird_m * .85), ALL, c3(.165, .205, .262))

# ---------- 水面的细纹：干净的边，两头收尖；越近越长越疏。暗水里是亮的（小波的斜面照到低处的天），亮水里是暗的
def lens(d, ss, x, y, hw, th):
    b_ = .06 * hw * (1 if int(x) % 2 else -1)                    # 一点弧：水纹是弯的
    pts = [(x - hw, y + b_), (x - hw * .35, y - th), (x + hw * .4, y - th * .85), (x + hw, y + b_ * .7), (x + hw * .3, y + th * .9), (x - hw * .45, y + th * .8)]
    d.polygon([(px * ss, py * ss) for px, py in pts], fill=255)
LIGHT = [(912, 1264, 15, 1.), (1004, 1303, 24, 1.1), (868, 1349, 30, 1.3), (1128, 1381, 19, 1.1), (652, 1406, 27, 1.3), (990, 1437, 41, 1.6),
         (596, 1481, 36, 1.5), (930, 1527, 52, 1.9), (668, 1566, 58, 2.1)]
DARKL = [(766, 1110, 14, .9), (982, 1119, 21, 1.), (1088, 1147, 17, 1.), (704, 1161, 16, 1.), (934, 1180, 25, 1.2)]
fill(st.layer('水纹亮', opacity=.42, mask=ink(lambda d, ss: [lens(d, ss, *l) for l in LIGHT]) * water), ALL, c3(.300, .365, .420))
fill(st.layer('水纹暗', opacity=.5, mask=ink(lambda d, ss: [lens(d, ss, *l) for l in DARKL]) * water), ALL, c3(.105, .150, .215))
# 鱼冒了一下头：水面上三圈，按透视压扁（高 : 宽 = 眼高 : 距离）。整张画里只有它在动
FXw, FYw = 1.9, 21.
def rings(d, ss):
    for r_, a0, a1 in ((.22, 0, 360), (.55, 200, 500), (.95, 30, 150), (.95, 205, 330)):
        pts = [P(FXw + r_ * np.cos(np.radians(a)), FYw + r_ * np.sin(np.radians(a)), 0) for a in np.arange(a0, a1 + 1, 6)]
        d.line([(px * ss, py * ss) for px, py in pts], fill=255, width=int(1.3 * ss))
fill(st.layer('鱼圈', opacity=.55, mask=ink(rings) * water), ALL, c3(.330, .390, .430))

# ---------- 签名：一个点一个点敲的坐标，点和点之间连直线，拐点留一粒墨
SIG = [[(74, 1551), (66, 1548), (60, 1555), (59, 1564), (64, 1570), (73, 1568)],
       [(80, 1547), (81, 1570)],
       [(97, 1559), (90, 1557), (86, 1564), (90, 1570), (96, 1568)], [(97, 1557), (98, 1570)],
       [(105, 1558), (105, 1567), (109, 1570), (115, 1567)], [(115, 1558), (116, 1570)],
       [(132, 1559), (125, 1557), (122, 1564), (125, 1570), (131, 1568)], [(132, 1545), (133, 1570)],
       [(140, 1564), (149, 1563), (147, 1558), (142, 1557), (139, 1563), (142, 1569), (149, 1568)]]
def sig(d, ss):
    for seg in SIG:
        d.line([(x * ss, y * ss) for x, y in seg], fill=255, width=int(1.1 * ss))
        for x, y in seg:
            d.ellipse([(x - 1.3) * ss, (y - 1.3) * ss, (x + 1.3) * ss, (y + 1.3) * ss], fill=255)
fill(st.layer('签名', opacity=.6, mask=ink(sig)), ALL, c3(.42, .50, .58))

# ---------- 纸
fill(st.layer('纸', mode='multiply'), ALL, paper(H, W, 77), edge=HARD)
look(st, OUT)
