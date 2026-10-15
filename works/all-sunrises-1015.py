# 《所有的日出》 · 10.15
# 从哪儿看：站在月面上靠近边缘的地方，眼睛离地一米多，平视，长焦（横向约 14°），朝着低悬在一道远山脊上方的地球。
# 这张在说什么：月全食那一刻，从月亮上看是地球把太阳整个挡住。地球是一个黑盘子，只剩一圈红——那一刻地球上所有的日出和日落加在一起。
#     月面没有别的光，就被这一圈照成铜色。1967 年勘测者 3 号在月面上拍到过这一圈。
# 光：唯一的光是那一圈，离地平线两度，大小两度，红橙为主；晨昏线上晴的几段亮成金，云挡住的几段暗下去；外沿一丝青（臭氧层）。
#     光很弱，从画面前方来，全是逆光：东西朝我们的面都在暗里，只有朝着地球的顶和棱接住一线铜。影子朝我们拖，很长，从地球正下方那点散开。
#     环外一层极淡的珍珠色，是日冕从地球边上露出来的一点。没有天光，影子近黑、偏冷。
# 规矩：月亮上没有空气，远处不退色、不变软——远山和近处一样黑、一样锐。深度只靠叠压、大小、纹理的疏密、影子的长短。
# 明暗：整张落在最暗那段。天是最黑的平阶（近黑的蓝紫），地球的盘子比天还黑一阶。地面：地球正下方那一片尘土向前散射，泛一条暗铜（地面最亮），
#     往两边、往近处暗下去；左下近处一大块环形山壁最暗。最亮是环，第二是远山脊那一线铜，第三是近处山壁的脊线。
# 视线：先落在那一圈红（全画唯一鲜的色、唯一的光），顺着它往下落到远山脊的铜线、地面那片暗铜，
#     再被左边那道斜下来的脊线领回来，最后停在黑盘子上那几粒城里的灯——那一面正是地球的夜里。
# 构图：地球在右侧、刚离开山脊，几乎贴着；左下一块近处的山壁从框外斜着压进来；天占大半，空着，只有星。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get('KBRUSH_ENGINE_DIR', './kbrush-cloud')  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(15)
from shapes import Shape
from blocks import lasso, line, soft, look, smooth, spray, paper
from layers import Stack, fill, HARD
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'rough.png' if ROUGH else 'final.png')
H, W = 800, 1200
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
def nz(sig, seed):
    r = np.random.default_rng(seed)
    n = gaussian_filter(r.standard_normal((H, W)), sig); return n / n.std()
if ROUGH:
    spray = lambda region, sp, seed, solid=.96, **_: np.asarray(getattr(region, 'mask', region), float) * solid

# ---------- 几何
EX, EY, R = 812., 378., 80.                 # 地球
er = np.hypot(xx - EX, yy - EY)
ang = np.degrees(np.arctan2(yy - EY, xx - EX)) % 360       # 0 = 右，90 = 下
YH = 556.                                   # 平原的地平线
# 远山脊：从左边平原的地平线起，往右抬成一道山，峰在地球右下，右边又有一个小肩
RIDGE = [(-20, 558), (180, 553), (360, 556), (470, 548), (560, 534), (640, 516), (700, 506), (742, 500), (770, 494),
         (812, 476), (850, 470), (884, 462), (912, 470), (950, 488), (1000, 498), (1040, 492), (1086, 506), (1130, 520), (1180, 532), (1230, 540)]
rp = np.array(smooth(RIDGE, 3, closed=False))
ridge_y = np.interp(xx[0], rp[:, 0], rp[:, 1])
ridge_y = ridge_y + 2.2 * gaussian_filter(np.random.default_rng(3).standard_normal(W), 2)     # 碎石把轮廓咬得毛一点
sky = yy < ridge_y[None, :]
ground = ~sky
# 近处的环形山壁：左下，从框外斜着压进来，脊线是上边
# 近处一块岩脊：从左下框外压进来，棱角分明，几级台阶往右下掉，被下框切掉
SLOPE = [(-30, 520), (22, 512), (58, 518), (84, 508), (120, 522), (150, 530), (176, 548), (212, 552), (236, 572), (262, 578), (284, 600), (318, 612),
         (338, 640), (372, 652), (390, 684), (420, 702), (436, 738), (462, 760), (478, 800), (486, 830), (-30, 830)]
slope = lasso(H, W, SLOPE, jag=2.2, scale=4, seed=21)
top_y = np.where(slope.mask.any(0), slope.mask.argmax(0), H).astype(float)     # 岩脊每一列的顶
crest_pts = [(float(x), top_y[x]) for x in range(0, W, 3) if top_y[x] < H - 2]

# 光从地球来，在地平线以上两度：影子从地球正下方那一点往我们这边散开
P = np.array([EX, YH])

# ---------- 颜色（台阶）
SKY = (.043, .040, .070)
DISC = (.016, .015, .024)
G0 = (.060, .043, .045)       # 平原，近处
G1 = (.088, .056, .050)       # 暗铜一阶
G2 = (.150, .082, .058)       # 暗铜二阶，地球正下方
G3 = (.250, .128, .074)       # 贴着地平线最亮的那条
RIDGE_C = (.050, .036, .040)  # 远山，背光
SLOPE_C = (.030, .022, .026)  # 近处山壁，最暗
COPPER = (.78, .38, .17)
COPPER_HI = (.95, .58, .28)

st = Stack(K.Paper(H, W, seed=3), ground=SKY)

# 1 地面：先铺平原
fill(st.layer('平原', mask=spray(ground, 1.2, 11)), ALL, G0)
# 地球正下方那一片尘土往前散射：越贴地平线、越正对地球越亮，往两边、往近处暗
dxn = (xx - EX) / (150 + .55 * (yy - YH).clip(0))
dyn = (yy - YH).clip(0)
sheen = np.exp(-dxn ** 2) * np.exp(-dyn / 62) + .035 * nz(22, 31) + .015 * nz(3, 32)
plain = ground & (yy > YH - 2)
fill(st.layer('暗铜1', mask=spray(plain & (sheen > .30), 9, 12)), ALL, G1)
fill(st.layer('暗铜2', mask=spray(plain & (sheen > .58), 6, 13, holes=.015)), ALL, G2)
fill(st.layer('暗铜3', mask=spray(plain & (sheen > .80) & (yy < YH + 26), 3, 14)), ALL, G3)

# 2 远山：背光，整块暗；朝上、朝地球那几块坡接住一点
foot = YH + 3 + 2.5 * gaussian_filter(np.random.default_rng(4).standard_normal(W), 6) * 2 + 5 * np.exp(-((xx[0] - 880) / 110) ** 2) + 3 * np.exp(-((xx[0] - 1060) / 60) ** 2)
ridge_mass = ground & (yy < foot[None, :]) & (yy >= ridge_y[None, :])
fill(st.layer('远山', mask=spray(ridge_mass, 1.0, 15)), ALL, RIDGE_C)
# 山上几道朝左上的坡面（沟和坡的走向顺着山势往下斜），只亮一线顶
rr = np.random.default_rng(16)
gully = np.zeros((H, W))
for k in range(26):
    x0 = rr.uniform(560, 1180); y0 = np.interp(x0, rp[:, 0], rp[:, 1]) + rr.uniform(3, 10)
    L = rr.uniform(14, 46); a = np.radians(rr.uniform(52, 78) * (1 if x0 > 884 else -1) + 90)
    pts = [(x0 + np.cos(a) * L * t, y0 + abs(np.sin(a)) * L * t) for t in (0, .5, 1)]
    gully = np.maximum(gully, line(H, W, pts, width=lambda t: 1.4 * (1 - .7 * t), seed=40 + k).mask * rr.uniform(.35, .8))
gl = gully * ridge_mass * np.exp(-np.abs(xx - EX) / 300)
fill(st.layer('山坡亮', opacity=.28, mask=gl), ALL, (.30, .16, .10))

# 3 平原上的坑：只有沿的顶接住光，一段一段断开；远沿（上半）那条最常见，近沿淡；坑里是影。越远越小越扁
craters = []
rc = np.random.default_rng(17)
for k in range(60):
    y = YH + 5 + (rc.random() ** 1.6) * 236
    x = rc.uniform(470, 1230)
    s_ = (y - YH)
    w = s_ * rc.uniform(.22, .75) + 2.5
    hh = w * (.075 + s_ / 2400)
    if slope.mask[int(min(y + hh, H - 1)), int(min(max(x - w, 0), W - 1))] or slope.mask[int(min(y, H - 1)), int(min(max(x, 0), W - 1))]: continue
    if x + w > 1060 and y > 730: continue
    craters.append((x, y, w, hh))
    if len(craters) >= 20: break
cr_rim = np.zeros((H, W)); cr_in = np.zeros((H, W), bool)
broken = gaussian_filter(np.random.default_rng(171).standard_normal((H, W)), 2.2); broken /= broken.std()
for x, y, w, hh in craters:
    d = ((xx - x) / w) ** 2 + ((yy - y) / hh) ** 2
    t = np.arctan2((yy - y) / hh, (xx - x) / w)
    cr_in |= d < .82
    th_ = (.07 + .05 * (w / 120)) * (1 + .7 * np.clip(broken, -1, 1.5))
    far = (np.abs(d - .92) < th_) & (yy < y) & (np.abs(np.cos(t)) < .93)
    near = (np.abs(d - .95) < th_ * .8) & (yy > y) & (np.abs(np.cos(t)) < .7)
    lit_w = np.exp(-((x - EX) / (360 + 1.5 * (y - YH))) ** 2)
    amp = (.35 + .65 * lit_w) * rc.uniform(.6, 1.0)
    cr_rim = np.maximum(cr_rim, far * amp * (broken > -.1) + near * amp * .35 * (broken > .4))
cr_in &= plain
fill(st.layer('坑里', mode='multiply', mask=spray(cr_in, .8, 18, solid=.6) * (yy > YH + 3)), ALL, (.66, .64, .74))
fill(st.layer('坑沿', mask=soft(Shape(cr_rim > .05), feather=.5) * cr_rim * plain * (yy > YH + 3)), ALL, (.60, .29, .14))
# 正对地球那片的碎光：尘土和小石子的顶，一粒粒，越近越疏越大
rg = np.random.default_rng(172)
pr = np.clip(sheen, 0, 1) ** 2 * .02 * np.exp(-(yy - YH).clip(0) / 22)
glit = plain & (rg.random((H, W)) < pr) & (yy > YH + 2)
glit = gaussian_filter(glit.astype(float), .45) > .12
fill(st.layer('碎光', mask=glit.astype(float) * .8), ALL, (.72, .36, .17))

# 4 石头：棱角的剪影，坐在地上；只有朝地球那侧的顶棱一线铜。影子很长，从 P 往我们这边散开，宽度跟着石头收，越远越淡
rocks = [(612, 600, 8, 5), (905, 614, 12, 7), (1012, 586, 6, 4), (728, 668, 17, 10), (1094, 702, 24, 14), (842, 748, 13, 8), (690, 576, 5, 3)]
rock_m = np.zeros((H, W), bool); rock_top = np.zeros((H, W)); rshadow = np.zeros((H, W))
rr2 = np.random.default_rng(61)
for k, (x, y, rw, rh) in enumerate(rocks):
    n_ = 6
    pts = []
    for j in range(n_):
        a_ = np.pi + np.pi * j / (n_ - 1)                      # 上半个轮廓：左 → 顶 → 右，棱角
        pts.append((x + rw * np.cos(a_) * rr2.uniform(.75, 1.1), y + rh * np.sin(a_) * rr2.uniform(.6, 1.15)))
    pts += [(x + rw * .9, y + rh * .12), (x - rw * .8, y + rh * .15)]
    b_ = lasso(H, W, pts, jag=.5, scale=3, seed=60 + k)
    rock_m |= b_.mask
    top = b_.mask & (edt(b_.mask) < 1.5) & (yy < y - rh * .3) & (((xx - x) * np.sign(EX - x)) > -rw * .3)
    rock_top = np.maximum(rock_top, top * np.exp(-((x - EX) / 520) ** 2))
    v = np.array([x - P[0], y - P[1]]); v /= np.linalg.norm(v); nrm = np.array([-v[1], v[0]])
    along = (xx - x) * v[0] + (yy - y) * v[1]; across = (xx - x) * nrm[0] + (yy - y) * nrm[1]
    Ls = rh * 30 * (y - YH) / 120 + 40                      # 两度高的光：影子是石头高的近三十倍，越近越长
    f = np.clip(along / Ls, 0, 1)
    persp = 1 + 1.6 * f * (Ls / max(y - YH, 1)) * .5              # 往我们这边，近大远小：影子越拖越宽
    pen = 1 + 2.8 * f                                               # 光有两度大：离石头越远，半影越宽、越虚
    wid = rw * .78 * persp * (1 - .35 * f)
    sh = (along > 0) & (along < Ls) & (np.abs(across) < wid + pen) & ~b_.mask
    core = np.clip((wid + pen - np.abs(across)) / (2 * pen), 0, 1)
    rshadow = np.maximum(rshadow, sh * core * (1 - f) ** .9)
sg = np.random.default_rng(191).random((H, W))
lit_g = np.clip((sheen - .2) / .5, 0, 1)
fill(st.layer('石影', mode='multiply', mask=((sg < rshadow * 1.1) | (rshadow > .85)).astype(float) * plain * (.25 + .75 * lit_g)), ALL, (.50, .47, .56))
fill(st.layer('石头', mask=spray(rock_m, .7, 20)), ALL, (.034, .026, .030))
fill(st.layer('石顶', mask=soft(Shape(rock_top > .05), feather=.5) * rock_top), ALL, COPPER)

# 4b 一台三条腿的着陆器：远远站在地球正下方那片暗铜上，剪影；朝地球那侧的棱接住一线铜。一九六七年就是这样一台机器在月面上拍到了这一圈
LX, LY, LS = 748., 584., 1.55
def P_(dx, dy): return (LX + dx * LS, LY + dy * LS)
def seg(p0, p1, w_):
    return line(H, W, [p0, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2), p1], width=w_ * LS, seed=7).mask
lander = np.zeros((H, W), bool)
feet = [P_(-11, 0), P_(10, .4), P_(1.5, 1.6)]
for f_ in feet:
    lander |= seg(f_, P_(np.sign(f_[0] - LX) * 3.2, -8.5), .8)
    lander |= ((xx - f_[0]) / (2.2 * LS)) ** 2 + ((yy - f_[1]) / (.9 * LS)) ** 2 < 1
lander |= lasso(H, W, [P_(-4.6, -12.5), P_(4.4, -12.8), P_(4.8, -7.6), P_(-4.4, -7.2)], jag=0).mask             # 机身
lander |= seg(P_(.4, -12), P_(1.4, -25), .6)                                                                       # 桅杆
panel = lasso(H, W, [P_(-6.5, -23.5), P_(6.0, -27.6), P_(7.2, -25.4), P_(-5.2, -21.4)], jag=0).mask             # 太阳能板，斜着
lander |= panel
lander |= ((xx - P_(5.2, -17)[0]) / (3.0 * LS)) ** 2 + ((yy - P_(5.2, -17)[1]) / (1.4 * LS)) ** 2 < 1          # 平板天线
upper_edge = lander & ~np.roll(lander, 1, 0)                                                                        # 每一块的上沿
lan_top = upper_edge & (yy < LY - 6 * LS)
lan_top = gaussian_filter(lan_top.astype(float), .3) > .2
fill(st.layer('着陆器', mask=spray(lander, .5, 82, solid=1.)), ALL, (.030, .024, .028))
fill(st.layer('着陆器顶', mask=lan_top.astype(float) * .9), ALL, COPPER)

# 5 近处的山壁：最暗，脊线接住一线铜；脊上几块石头剪影
fill(st.layer('山壁', mask=spray(slope, 1.4, 21)), ALL, SLOPE_C)
cp = np.array(crest_pts)
crest_band = slope.mask & (yy - top_y[None, :] < 2.2)
cdist = np.hypot(xx - EX, yy - YH)
cbright = np.clip(1.15 - cdist / 1100, .25, 1) * (.75 + .25 * nz(9, 23).clip(-1, 1))
cmask = crest_band
fill(st.layer('脊线', mask=spray(cmask, .9, 24, solid=.9) * cbright), ALL, COPPER)
# 山壁上沿：朝上的那截坡掠着接一点光，一阶极暗的铜，往下很快没了
sb = slope.mask & (inner_ := edt(slope.mask)) < 0 if False else None
inn = edt(slope.mask)
band = slope.mask & (inn < 26 + 10 * nz(8, 36)) & (inn > 2.5)
fill(st.layer('坡上沿', mask=spray(band, 6, 37) * slope.mask * np.clip(cbright, .3, 1) * .8), ALL, (.068, .042, .040))
# 脊上的碎石：几簇，大小不一，不规则；剪影，朝地球那边的顶一点亮
rb = np.random.default_rng(25)
cbits = np.zeros((H, W), bool); ctop = np.zeros((H, W))
for c_t in (.13, .41, .47, .70):
    for k in range(rb.integers(1, 4)):
        t = c_t + rb.normal(0, .015); i = int(np.clip(t, 0, 1) * (len(cp) - 1)); x, y = cp[i]
        s = (2.5 + 11 * t) * rb.uniform(.5, 1.4)
        pts = [(x + s * np.cos(a) * rb.uniform(.7, 1.3) * 1.4, y - s * .35 + s * np.sin(a) * rb.uniform(.7, 1.2)) for a in np.linspace(0, 2 * np.pi, 7, endpoint=False)]
        m = lasso(H, W, pts, jag=.4, scale=3, seed=int(rb.integers(1e6))).mask
        cbits |= m
        e = m & (edt(m) < 1.6)
        ctop = np.maximum(ctop, e * (yy < y - s * .6) * (xx > x - s * .2))
fill(st.layer('脊石', mask=spray(cbits, .6, 26)), ALL, SLOPE_C)
fill(st.layer('脊石顶', mask=soft(Shape(ctop > 0), feather=.5) * .75), ALL, COPPER)
# 山壁身上：一点点从脊上漏下来的颗粒，越往下越没有
inner = edt(slope.mask)
grit = slope.mask & (inner < 60) & (np.random.default_rng(27).random((H, W)) < .02 * np.exp(-inner / 18))
fill(st.layer('坡上颗粒', opacity=.5, mask=grit.astype(float)), ALL, (.26, .13, .08))

# 6 远山脊线：一线铜，正对地球那段最亮，两边淡下去；山尖上几粒石头的亮
ridge_line = np.zeros((H, W), bool)
yi = np.round(ridge_y).astype(int)
for x in range(W):
    ridge_line[max(yi[x], 0):max(yi[x], 0) + 2, x] = True
rb_ = np.exp(-((xx - EX) / 260) ** 2) * .85 + .15
rb_ = rb_ * (.8 + .2 * nz(3, 28).clip(-1, 1))
fill(st.layer('山脊线', mask=spray(ridge_line & ~slope.mask, .7, 29, solid=.92) * rb_), ALL, COPPER)
fill(st.layer('山脊最亮', mask=spray(ridge_line & ~slope.mask & (np.abs(xx - EX) < 120), .6, 30, solid=.8) * np.exp(-((xx - EX) / 90) ** 2)), ALL, COPPER_HI)

# 7 天：星。疏密不匀，几颗亮的带一点颜色；离环近的被它的光吃掉
rs = np.random.default_rng(33)
star = np.zeros((H, W)); star_c = np.zeros((H, W, 3))
clump = gaussian_filter(rs.standard_normal((H, W)), 60); clump = (clump - clump.min()) / (clump.max() - clump.min())
for k in range(420):
    x, y = rs.uniform(0, W), rs.uniform(0, H)
    if not sky[int(y), int(x)] or rs.random() > clump[int(y), int(x)] ** 1.6 * 1.5: continue
    if np.hypot(x - EX, y - EY) < R + 70: continue
    m = rs.random() ** 3.2
    r_ = .6 + 1.4 * m
    d = np.hypot(xx[max(int(y) - 4, 0):int(y) + 5, max(int(x) - 4, 0):int(x) + 5] - x, yy[max(int(y) - 4, 0):int(y) + 5, max(int(x) - 4, 0):int(x) + 5] - y)
    a = np.clip(1 - d / r_, 0, 1) * (.18 + .82 * m)
    sub = star[max(int(y) - 4, 0):int(y) + 5, max(int(x) - 4, 0):int(x) + 5]
    np.maximum(sub, a, out=sub)
    c = rs.choice([(.80, .84, 1.0), (.95, .93, .88), (1.0, .82, .62), (.86, .88, .98)])
    star_c[max(int(y) - 4, 0):int(y) + 5, max(int(x) - 4, 0):int(x) + 5][a > 0] = c
fill(st.layer('星', mask=star * sky), ALL, star_c)

# 8 日冕：地球边上露出来的一点，珍珠色，极淡，一缕缕往外
th = np.radians(ang)
streak = .55 + .45 * np.cos(4 * th + .6) * np.cos(7 * th + 2.0)
cor = np.clip(1 - (er - R) / (70 * streak + 30), 0, 1) ** 2.2 * (er > R)
fill(st.layer('日冕', mode='screen', opacity=.07, mask=cor), ALL, (.75, .76, .82))

# 9 地球：比天还黑的一个盘子；夜里那一面的几粒灯
disc = er < R
fill(st.layer('地球', mask=spray(disc, .4, 34, solid=1.)), ALL, DISC)
lights = np.zeros((H, W))
LP = [(-.46, .30, .9), (-.40, .36, .6), (-.33, .32, .7), (-.30, .41, .5), (-.22, .38, .4), (-.52, .18, .5), (.18, -.22, .45), (.25, -.14, .35), (.05, .50, .3), (-.62, .05, .3)]
for ux, uy, b in LP:
    x, y = EX + ux * R, EY + uy * R
    lights = np.maximum(lights, np.clip(1 - np.hypot(xx - x, yy - y) / 1.4, 0, 1) * b)
fill(st.layer('城里的灯', mask=lights * disc * .7), ALL, (.88, .62, .33))

# 10 那一圈：内沿是地球的边，锐；红在最里，往外是橙，晴的几段亮成金；外沿一丝青；再往外一层红晕，颗粒一路稀到底
bright = np.full((H, W), .42)
for a0, a1, b in [(18, 52, .92), (84, 108, .10), (132, 160, .70), (192, 216, .14), (246, 284, 1.0), (316, 340, .22)]:
    mid, half = (a0 + a1) / 2, (a1 - a0) / 2
    da = np.abs((ang - mid + 180) % 360 - 180)
    bright = np.where(da < half, bright + (b - .42) * np.clip((half - da) / 9, 0, 1), bright)
bright = bright * (.85 + .15 * np.cos(np.radians(ang) * 23 + 1.3) * np.cos(np.radians(ang) * 9))
ring_red = (er >= R) & (er < R + 1.1 + 3.4 * bright)
ring_or = (er >= R + .6) & (er < R + 1.2 + 3.0 * bright) & (bright > .5)
ring_au = (er >= R + .9) & (er < R + 2.6) & (bright > .8)
ring_blue = (er >= R + 1.1 + 3.4 * bright) & (er < R + 2.1 + 3.4 * bright) & (bright > .6)
gl_r = np.clip(1 - (er - R) / (6 + 18 * bright), 0, 1) ** 2.0 * (er > R) * (bright ** 1.3)
fill(st.layer('红晕', mode='screen', opacity=.9, mask=gl_r), ALL, (.66, .16, .06))
glow_dots = (er > R + 4) & (er < R + 34) & (np.random.default_rng(35).random((H, W)) < .18 * np.exp(-(er - R - 4) / 8) * bright)
fill(st.layer('红晕颗粒', opacity=.7, mask=glow_dots.astype(float)), ALL, (.70, .26, .10))
fill(st.layer('环红', mask=soft(Shape(ring_red), feather=.6) * np.clip(.25 + .9 * bright, 0, 1)), ALL, (.80, .26, .12))
fill(st.layer('环橙', mask=soft(Shape(ring_or), feather=.7) * np.clip((bright - .5) * 2.2, 0, 1)), ALL, (.98, .55, .22))
fill(st.layer('环金', mask=soft(Shape(ring_au), feather=.6) * np.clip((bright - .8) * 5, 0, 1)), ALL, (1.0, .86, .55))
fill(st.layer('环青', mask=soft(Shape(ring_blue), feather=.7) * np.clip((bright - .6) * 2.5, 0, 1) * .85), ALL, (.42, .74, .82))
# 内沿最锐：地球的边压回去一线黑，环就从盘子边上直接起
fill(st.layer('盘边', mask=((er < R + .2) & (er > R - 1.2)).astype(float)), ALL, DISC)

# 11 签名：右下，一个点一个点敲，点间连直线，拐点留一粒墨
SIG = [[(1093, 768), (1088, 764), (1083, 765), (1081, 771), (1083, 777), (1089, 778), (1093, 775)],
       [(1098, 760), (1098, 778)],
       [(1110, 769), (1105, 768), (1102, 772), (1104, 777), (1109, 777), (1111, 772), (1111, 778)],
       [(1117, 768), (1117, 775), (1120, 778), (1125, 776), (1126, 768), (1126, 778)],
       [(1140, 762), (1140, 778), (1135, 777), (1132, 772), (1134, 768), (1140, 769)],
       [(1146, 772), (1154, 772), (1153, 768), (1149, 767), (1145, 770), (1146, 776), (1151, 778), (1155, 776)]]
sig = np.zeros((H, W), bool)
for s in SIG:
    sig |= line(H, W, s, width=1.3, seed=90).mask
    for p in s[1:-1]:
        sig |= np.hypot(xx - p[0], yy - p[1]) < 1.2
fill(st.layer('签名', mask=sig.astype(float) * .55), ALL, (.55, .30, .18))

# 12 纸
fill(st.layer('纸', mode='multiply'), ALL, paper(H, W), edge=HARD)
if os.environ.get('RECORD'):
    from record import record; record(st, 'process.mp4', width=720, slow=('环', '红晕', '着陆器', '城里的灯'))
look(st, OUT)
print('done', OUT)
