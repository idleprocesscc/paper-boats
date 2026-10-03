# 雨天的正午 · 白鹭（10.3）
# 从哪儿看：平视，站在山坡小路上，眼睛比稻田高一点，往对面山谷看。地平线藏在山后面，大约在 y=330。
# 这张在说什么：雨把远处一层层收走，只把最近的、和一只白鹭，留给你。
# 光：正午，阴雨，天光从头顶来，大、散、冷灰；没有投影。湿的东西朝天的面映着天，亮边干净（湿叶、水面）；其余都是糙的。
# 明暗：天最亮；白鹭比天还亮，是全画最亮最脆的一处。往近处一层层变深：最远的山几乎是天色，最近的茶丛最深，最深那阶换蓝黑。
# 层次：远山四层，每层往天色挪一阶，阶越来越少，颗粒越来越细越淡；山脚都泡在雾里（透明叠色：一条浅带压过去，叠出第三个颜色）。
#       中景稻田平的水面映天，远处的田映山（暗一阶），近处的映高处的天（亮）。田埂手勾，越远越细越挤。近处左下一丛湿茶树压住画框。
# 视线：先落在白鹭（白、锐、在灰里），顺着田埂往远处的村子，再被雾带进山里。
# 质感（10.3 从木内、Shyama 那几张原图里看出来的，挑三样）：雨是斜的布纹，纸纹的纤维也顺着雨走；雾是透明叠色；茶叶是斑驳加对折的两阶。
# 雨：细长的亮线，只在暗的东西前面看得见（茶丛、近山），亮的地方看不见雨。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get("KBRUSH_ENGINE_DIR", "./kbrush-cloud")  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(3)
from shapes import Shape
from blocks import lasso, line, soft, look
from layers import Stack, fill, HARD, SOFT, LOST
from scipy.ndimage import gaussian_filter, rotate, distance_transform_edt as edt
from PIL import Image as _I, ImageDraw as _D

H, W = 800, 1200
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))

def open_smooth(pts, n=3):
    p = np.array(pts, float)
    for _ in range(n):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [.75 * a + .25 * b, .25 * a + .75 * b]
        q.append(p[-1]); p = np.array(q)
    return [tuple(v) for v in p]
def curve_y(pts, n=3):
    t = np.array(open_smooth(pts, n)); o = np.argsort(t[:, 0])
    return np.interp(np.arange(W), t[o, 0], t[o, 1])
def nz(sig, seed):
    n = gaussian_filter(np.random.default_rng(seed).standard_normal((H, W)), sig); return n / n.std()
def spray(region, sp, seed, solid=.96, dens=.7, cl=.5):
    """一阶：区域里实实地盖上，边外喷一圈颗粒，越远越稀；sp 越小越窄，cl 越小颗粒越细（远处细、近处粗）"""
    r_ = np.random.default_rng(seed); dout = edt(~region)
    dots = (r_.random((H, W)) < dens * np.exp(-dout / sp)) & (gaussian_filter(r_.random((H, W)), cl) > .47 + .02 * (cl - .5))
    return np.where(region, solid, dots * .95)
def poly(pts):
    im = _I.new('L', (W * 2, H * 2), 0); _D.Draw(im).polygon([(x * 2, y * 2) for x, y in pts], fill=255)
    return np.asarray(im.resize((W, H), _I.LANCZOS)) > 127

st = Stack(K.Paper(H, W, seed=3), ground='#cdd1cc')
SKY = np.array([.80, .82, .80])
fill(st.layer('天'), ALL, tuple(SKY), edge=HARD)
sky_top = yy < 46 + 10 * nz(140, 99)                            # 天顶的雨云重一阶：阴雨天越往头顶越暗。边要宽、要平，v3 起伏一大就成了天上又一道山
fill(st.layer('云', mask=spray(sky_top, 26, 98, dens=.55)), ALL, (.74, .77, .75), edge=HARD)

# ---------- 1 山：从远到近，每层往天挪一阶；山脚的雾是一条浅带压过去（透明叠色）
RIDGES = [  # 顶线（手勾）, 颜色, 顶边喷多宽, 雾离山顶多远
    ([(-20, 208), (70, 198), (150, 203), (230, 186), (300, 190), (360, 176), (430, 183), (520, 200), (610, 196), (700, 170), (760, 158), (820, 163), (900, 182), (980, 196), (1060, 190), (1140, 197), (1220, 193)], (.73, .76, .75), 1.2, 30),
    ([(-20, 258), (60, 246), (130, 250), (200, 236), (260, 232), (330, 240), (420, 252), (500, 246), (560, 232), (640, 240), (720, 262), (800, 256), (880, 236), (940, 222), (1000, 226), (1080, 244), (1160, 240), (1220, 246)], (.65, .70, .69), 1.8, 36),
    ([(-20, 300), (80, 292), (170, 286), (240, 296), (320, 310), (400, 298), (470, 288), (530, 296), (600, 314), (680, 306), (760, 318), (850, 312), (930, 296), (1000, 286), (1060, 290), (1140, 302), (1220, 298)], (.55, .62, .60), 2.6, 44),
    ([(-20, 396), (100, 394), (200, 398), (300, 402), (380, 396), (450, 384), (520, 368), (580, 350), (640, 338), (700, 326), (760, 318), (820, 322), (880, 330), (950, 344), (1010, 352), (1080, 348), (1150, 342), (1220, 346)], (.40, .49, .46), 3.5, None),
]
for i, (top, col, sp, fog) in enumerate(RIDGES):
    ty = curve_y(top)
    bump = nz(2.5, 100 + i) * (1.0 + 1.6 * (i == 3))            # 近的那道山顶是一排树冠，毛一点
    reg = yy > ty[None, :] + bump
    fill(st.layer(f'山{i}', mask=spray(reg, sp, 110 + i, cl=.38 + .08 * i)), ALL, col, edge=HARD)
    if i == 3:   # 近山上的林子：大团的深浅（斑驳），不是细颗粒
        blot = reg & (nz(4, 130) > 1.0) & (yy > ty[None, :] + 6)            # v1 大团成了迷彩；改小、改近
        fill(st.layer('林', mask=spray(blot, 2, 131)), ALL, (.36, .45, .42), edge=HARD)
    if fog is not None:
        fl = ty[None, :] + fog + 14 * nz(30, 140 + i)
        band = yy > fl
        fill(st.layer(f'雾{i}', opacity=.55, mask=spray(band, 6 + 2 * i, 150 + i, dens=.5)), ALL, tuple(SKY + .02), edge=HARD)

# ---------- 2 谷底：灌满水的稻田。平视的地面越远越挤，按距离的倒数挤（v5 以前田埂间距均匀变大，整片田像朝我竖起来了——一个朋友："山有远，田没有远"）
HZ = 330.                                                        # 视平线
VP = np.array([640., HZ])
FLOOR = curve_y([(-20, 404), (300, 406), (600, 410), (900, 418), (1220, 424)])
floor = yy > FLOOR[None, :]
def dt(y): return np.clip((y - HZ) / (H - HZ), 0, 1)             # 0 天边 … 1 脚下：粗细、颗粒、颜色都跟它走
XC = np.arange(W, dtype=float)
LY = []
for n in range(1, 9):                                            # 每块田差不多一样深：第 n 道埂在 HZ + 74/(1 - .105n)
    off = 74 / (1 - .105 * n) * (1 + .04 * (XC - 600) / 600) * (1 + .012 * np.sin(XC / (70 + 17 * n) + 1.7 * n))
    LY.append(HZ + off)
BANDS = [FLOOR] + LY + [np.full(W, H + 50.)]
band_of = np.zeros((H, W), int)
for k in range(len(BANDS) - 1):
    band_of[floor & (yy > BANDS[k][None, :])] = k
# 纵向的埂：地上一列列往视平线收；脚下那一排上隔 330px 一列，远处自然挤；每一排田随机断开几列，田就错开了
XB = (VP[0] + (xx - VP[0]) * (H - HZ) / np.maximum(yy - HZ, 1))   # 每个点投回脚下那一排的 x
COLW = 330.; col_of = np.floor((XB + 1600) / COLW).astype(int)
rc = np.random.default_rng(171); NCOL = int(5000 / COLW) + 2
chosen = rc.random((len(BANDS), NCOL)) < .55
chosen[band_of[683, 850], col_of[683, 1010]] = True              # 白鹭那块田在右边收住，不然一大块绿压满右下角，跟左下的茶丛两头一样重
cell = np.zeros((H, W), int)
for k in range(len(BANDS) - 1):
    eff = np.cumsum(chosen[k]); m = band_of == k
    cell[m] = k * 1000 + eff[np.clip(col_of[m], 0, NCOL - 1)]
WATER = [(.60, .66, .64), (.63, .68, .66), (.66, .71, .69), (.69, .74, .72), (.71, .76, .74), (.73, .77, .76), (.75, .79, .78), (.77, .80, .79), (.78, .81, .80), (.79, .82, .81)]
DELTA = np.array([0, 0, .02, -.022, .012])
rd = np.random.default_rng(172); dmap = {c: DELTA[rd.integers(len(DELTA))] for c in np.unique(cell[floor])}
wcol = np.zeros((H, W, 3))
for k in range(len(BANDS) - 1):
    wcol[band_of == k] = WATER[k]
dl = np.vectorize(lambda c: dmap.get(c, 0.))(cell)
wcol = np.clip(wcol + dl[..., None] * (dt(yy)[..., None] ** .5), 0, 1)       # 远处的田几乎一个色，近处才分得出块
fill(st.layer('田水'), Shape(floor), wcol, edge=HARD)
# 埂：越远越细、越淡（往雾里退），越近越粗、越绿
wv = .4 + 7.6 * dt(yy) ** 1.2
lv = np.zeros((H, W), bool)
for k, ly in enumerate(LY):
    lv |= np.abs(yy - ly[None, :]) < wv / 2
colx = VP[0] + ((np.floor((XB + 1600) / COLW)) * COLW - 1600 - VP[0]) * (yy - HZ) / (H - HZ)   # 这一点左边那一列的 x
dxc = np.abs(xx - colx); dxc2 = np.abs(xx - (colx + COLW * (yy - HZ) / (H - HZ)))
for k in range(len(BANDS) - 1):
    m = band_of == k
    ci = np.clip(col_of, 0, NCOL - 1)
    on = chosen[k][ci] & (dxc < wv * .45) | chosen[k][np.clip(ci + 1, 0, NCOL - 1)] & (dxc2 < wv * .45)
    lv |= m & on
lv &= floor
fadec = (1 - dt(yy)) ** 1.4
LVC = np.array([.37, .47, .38])[None, None, :] * (1 - fadec[..., None]) + np.array([.62, .68, .66])[None, None, :] * fadec[..., None]
fill(st.layer('田埂', mask=spray(lv, 1.2, 161, cl=.45)), ALL, LVC, edge=HARD)
fill(st.layer('埂亮', mask=spray(lv & np.roll(~lv, 2, 0) & (dt(yy) > .25), .8, 162, solid=.7)), ALL, (.52, .62, .52), edge=HARD)   # 近处埂顶朝天那一线亮一阶

# 田间小路：几十条埂一样粗就没主次、眼睛没路走（nerolette："抽一两条路宽一点会不会更有呼吸感"）。
# 一条从脚下斜穿到山脚的村子（地上的直路投到画上还是直的，宽度跟着远近），一道横埂加宽成路跟它接上；近处路上两个映着天的水洼
# 路顺着田埂走：先沿一列往远处收，到那道横埂拐弯，沿埂往左，再斜着上去进村（v9 一根直线斜穿过去，又是尺子拉的）
def ly_at(k, x): return float(LY[k][int(np.clip(x, 0, W - 1))])
def col_x(xb, y): return VP[0] + (xb - VP[0]) * (y - HZ) / (H - HZ)
PTS = [(col_x(380, y), y) for y in np.linspace(810, ly_at(5, 420), 14)]          # 近处那段别切在画面正中；就压在那条纵埂上（XB=380），v12 跟埂隔着一条缝（nerolette）
PTS += [(x, ly_at(5, x)) for x in np.linspace(PTS[-1][0] - 8, 330, 10)]
PTS += [(330 - 40 * u - 50 * u * u, ly_at(5, 330) + (414 - ly_at(5, 330)) * u) for u in np.linspace(.1, 1, 10)]
cp = np.array(open_smooth(PTS, 2)); cp, _ = (lambda q: (q, None))(cp)
d_ = np.gradient(cp, axis=0); nrm = np.c_[d_[:, 1], -d_[:, 0]]; nrm /= np.linalg.norm(nrm, axis=1)[:, None] + 1e-9
rw = np.random.default_rng(167)
jit = gaussian_filter(rw.standard_normal(len(cp)), 3) * 6
cp = cp + nrm * (jit * dt(cp[:, 1]))[:, None]                     # 手抖：近处抖得开，远处几乎不抖
hw = (.8 + 7.5 * dt(cp[:, 1]) ** 1.1) * (1 + .18 * gaussian_filter(rw.standard_normal(len(cp)), 4) / .2)
path = poly([tuple(v) for v in cp + nrm * hw[:, None]] + [tuple(v) for v in (cp - nrm * hw[:, None])[::-1]])
P0, P1 = cp[0], cp[-1]
wide = np.abs(yy - LY[5][None, :]) < (wv * 1.9) / 2                 # 那道横埂整条加宽成路
road = (path | wide) & floor
RDC = np.array([.58, .59, .53])[None, None, :] * (1 - fadec[..., None]) + np.array([.66, .70, .68])[None, None, :] * fadec[..., None]
fill(st.layer('路', mask=spray(road, 1.4, 167, cl=.6)), ALL, RDC, edge=HARD)
fill(st.layer('路沿', mask=spray(road & ~np.roll(road, -3, 0) & (dt(yy) > .3), .8, 168, solid=.8)), ALL, (.45, .53, .44), edge=HARD)   # 路边的草沿暗一阶
pud = np.zeros((H, W), bool)
for (u, rx, ry) in ((.06, 13, 3.0), (.2, 8, 2.0)):
    c = cp[int(u * len(cp))]
    pud |= np.hypot((xx - c[0]) / rx, (yy - c[1]) / ry) + .25 * nz(2, 169 + int(u * 100)) < 1
fill(st.layer('水洼'), Shape(pud & path), (.79, .82, .81), edge=HARD)                 # 水洼映天：光滑，边干净

# 白鹭那块田插了秧：暗一阶的灰绿，一行行秧苗朝视平线收，白的鹭才跳得出来
EGX, WL = 850, 683                                              # v6 鹭站在埂上、身后是浅水，白跳不出来；挪进秧田里，按远近放大
planted = cell == cell[WL, EGX]
SC = (683 - HZ) / (621 - HZ)
def E(x, y): return (850 + (x - 815) * SC, 683 + (y - 621) * SC)
fill(st.layer('秧田', mask=spray(planted, 1.5, 163)), ALL, (.50, .58, .52), edge=HARD)
im = _I.new('L', (W * 2, H * 2), 0); dr = _D.Draw(im); rs = np.random.default_rng(164)
ys_, xs_ = np.nonzero(planted); y_lo, y_hi = ys_.min(), ys_.max()
for xb in np.arange(-1600, 3200, 18):                            # 秧行：脚下隔 18px，往远处收；一簇三根叶子张开（v6 一粒粒圆点像点阵屏）
    y = float(y_lo)
    while y < y_hi:
        x = VP[0] + (xb - VP[0]) * (y - HZ) / (H - HZ) + rs.normal(0, .6)
        if 0 <= x < W and planted[int(y), int(x)]:
            h_ = (2 + 10 * dt(y)) * rs.uniform(.75, 1.15)
            for k in (-1, 0, 1):
                dr.line([(x * 2, y * 2), ((x + k * .38 * h_ + rs.normal(0, .5)) * 2, (y - h_ * (1 - .15 * abs(k))) * 2)], fill=255, width=2)
        y += 2.5 + 6 * dt(y)
tufts = np.asarray(im.resize((W, H), _I.LANCZOS)) > 90
fill(st.layer('秧'), Shape(tufts & planted), (.36, .47, .38), edge=HARD)
fill(st.layer('秧尖'), Shape(tufts & planted & np.roll(~tufts, 2, 0)), (.60, .68, .56), edge=HARD)
band5 = planted
# 远处靠村子那边还有一小块插了秧的：白鹭那块的回声，把眼睛往村子领
planted2 = cell == cell[442, 330]
fill(st.layer('秧田远', mask=spray(planted2, 1, 166, cl=.4)), ALL, (.55, .62, .57), edge=HARD)

# ---------- 3 村子：山脚一小撮，树团后面压着几间屋，雾从底下漫过去
fill(st.layer('村树', mask=spray(lasso(H, W, open_smooth([(80, 408), (96, 392), (132, 384), (168, 390), (200, 380), (246, 384), (290, 392), (330, 398), (352, 410)], 2) + [(352, 416), (80, 416)], jag=1.6, scale=3, seed=21).mask, 1.6, 170)), ALL, (.33, .42, .38), edge=HARD)
WALL, ROOF = st.layer('墙'), st.layer('瓦')
for (x0, x1, y0, h) in ((126, 158, 397, 8), (172, 212, 401, 9), (228, 250, 398, 7), (266, 302, 402, 8)):
    fill(WALL, Shape((xx > x0 + 3) & (xx < x1 - 3) & (yy > y0) & (yy < y0 + h)), (.75, .74, .70), edge=HARD)
    fill(ROOF, lasso(H, W, [(x0, y0 + 1), (x0 + 4, y0 - 6), (x1 - 4, y0 - 6), (x1, y0 + 1)], jag=.4, scale=2, seed=x0), (.29, .32, .35), edge=HARD)
fl = FLOOR[None, :] + 4 + 6 * nz(25, 180)
fill(st.layer('谷雾', opacity=.5, mask=spray((yy > fl - 2) & (yy < fl + 24), 5, 181, dens=.5)), ALL, tuple(SKY + .02), edge=HARD)   # 只漫过村子的脚，屋顶留着

# ---------- 4 白鹭：中景偏右，站在田水里。全画最亮最锐的一处；肚子和脖子背光那侧冷一阶；倒影被雨打碎
body = lasso(H, W, open_smooth([E(797, 596), E(806, 589), E(818, 588), E(829, 591), E(838, 597), E(847, 604), E(836, 606), E(822, 608), E(808, 606), E(799, 602)], 2), jag=.3, scale=2, seed=31).mask
neck = line(H, W, [E(804, 594), E(798, 585), E(800, 576), E(806, 570), E(803, 563)], width=lambda t: (5.5 - 2.5 * t) * SC).mask
_hx, _hy = E(801.5, 561.5); head = (np.hypot((xx - _hx) / (4.6 * SC), (yy - _hy) / (3.6 * SC)) < 1)
egret = body | neck | head
fill(st.layer('鹭'), Shape(egret), (.97, .97, .95), edge=HARD)
_b = E(803, 599)[1]; _nx, _n1 = E(803, 566); _n2 = E(803, 590)[1]
fill(st.layer('鹭冷'), Shape(egret & ((yy > _b) | ((xx > _nx) & (yy > _n1) & (yy < _n2)))), (.84, .87, .88), edge=HARD)
fill(st.layer('喙'), line(H, W, [E(798, 561), E(791, 563), E(784, 565.5)], width=lambda t: (2.0 - 1.2 * t) * SC), (.86, .70, .26), edge=HARD)
_ex, _ey = E(800, 560.5); fill(st.layer('眼'), Shape(np.hypot(xx - _ex, yy - _ey) < 1.0), (.10, .10, .10), edge=HARD)
legs = line(H, W, [E(814, 607), E(813, 621)], width=.9).mask | line(H, W, [E(821, 607), E(824, 621)], width=.9).mask
fill(st.layer('腿'), Shape(legs), (.20, .21, .20), edge=HARD)
refl = np.zeros((H, W), bool)
src = legs                                                       # 秧田里只看得见腿的倒影；v13 整只鹭的倒影在秧苗缝里成了一摞浅色砖头
for y in range(WL, min(H, WL + 64)):
    sy = 2 * WL - y
    if 0 <= sy < H: refl[y] = src[sy]
ripple = nz((0.4, 6), 190) > -.35                               # 横着被雨点打断
fill(st.layer('鹭影', opacity=.35, mask=refl & ripple & band5), ALL, (.30, .33, .31), edge=HARD)
# 雨点在水上的圈：压扁的小椭圆，近处大远处小，白鹭周围多几个
rr = np.random.default_rng(195); rings = np.zeros((H, W), bool)
for _ in range(40):
    y0 = rr.uniform(470, 790); x0 = rr.uniform(300, 1190)
    if rr.random() < .15: x0, y0 = rr.normal(812, 60), rr.normal(632, 18)
    s = 1.5 + 9 * (y0 - 450) / 350
    d = np.hypot((xx - x0) / s, (yy - y0) / (s * .32))
    rings |= (np.abs(d - 1) < .9 / s * 1.4) & (yy > y0 - s) & ~lv
fill(st.layer('雨圈', opacity=.45), Shape(rings & floor & ~planted), (.88, .91, .90), edge=HARD)

# ---------- 5 近处：左下一丛湿茶树。里面是大团的深浅（斑驳）；边上一片片叶子，每片对折成两阶：朝天的半边映着天，另一半暗
bush = lasso(H, W, open_smooth([(-20, 478), (30, 462), (82, 470), (124, 494), (166, 520), (208, 556), (244, 604), (270, 670), (286, 750), (294, 820), (-20, 820)], 3), jag=5, scale=12, seed=41).mask
fill(st.layer('茶', mask=spray(bush, 5, 200, cl=1.1)), ALL, (.19, .28, .24), edge=HARD)
mott = bush & (nz(14, 201) > .25)
fill(st.layer('茶斑', mask=spray(mott, 4, 202)), ALL, (.13, .21, .19), edge=HARD)
deep = bush & (.4 * nz(20, 203) + (yy - 620) / 110 + (70 - xx) / 80 > 1.5)      # 只贴着左下框角，v1 在丛里成了两个黑洞
fill(st.layer('茶深', mask=spray(deep, 4, 204)), ALL, (.07, .10, .14), edge=HARD)        # 最深那阶换蓝黑
rl = np.random.default_rng(210)
bd = edt(bush)
cand = np.argwhere(bush & (bd < 130))
LA, LB, LS = st.layer('叶暗'), st.layer('叶亮'), st.layer('叶光')
leaf_dark = np.zeros((H, W), bool); leaf_lit = np.zeros((H, W), bool); sheen = np.zeros((H, W), bool)
# v5 叶子全朝上朝外、都挤在轮廓上，成了一排剑龙背板（nerolette）。茶叶是互生的，往各个方向斜着叠，湿了往下垂；轮廓上多是侧着、垂着的，边是碎的
n_ok = 0
while n_ok < 120:
    cy, cx = cand[rl.integers(len(cand))]
    ang = np.deg2rad(rl.uniform(15, 165) if rl.random() < .55 else rl.uniform(-180, 180))
    ax = np.array([np.cos(ang), np.sin(ang)])
    if bd[cy, cx] < 18 and ax[1] < -.45 and rl.random() < .75: continue   # 轮廓上朝天竖着的少要
    n_ok += 1
    L_ = rl.uniform(16, 42) * (1 + .3 * (cy - 480) / 320)
    edge_on = rl.random() < .22
    wd = L_ * (rl.uniform(.10, .14) if edge_on else rl.uniform(.28, .38))
    nr = np.array([-ax[1], ax[0]])
    t = np.linspace(0, 1, 14)
    mid = np.array([cx, cy])[None, :] + ax[None, :] * (t[:, None] * L_)
    mid += nr[None, :] * (np.sin(np.pi * t) * rl.normal(0, .08) * L_)[:, None]      # 叶子带一点弯
    half = (wd / 2 * np.sin(np.pi * t) ** .8)[:, None]
    side_a = [tuple(v) for v in mid + nr * half]; side_b = [tuple(v) for v in mid - nr * half]
    spine = [tuple(v) for v in mid]
    if edge_on:
        leaf_dark |= poly(side_a + side_b[::-1]); continue
    up_is_a = (nr[1] < 0)
    A = poly(spine + side_a[::-1]); B = poly(spine + side_b[::-1])
    lit, dk = (A, B) if up_is_a else (B, A)
    if ax[1] > .6: lit, dk = dk & False, dk | lit                 # 垂下来的叶子看到的是背面，一个色
    leaf_lit |= lit; leaf_dark |= dk
    if rl.random() < .3 and ax[1] < .6:                           # 湿叶上一道天光：干净的边，两头收尖
        e = side_a if up_is_a else side_b
        sheen |= line(H, W, e[3:11], width=lambda tt: 1.7 * np.sin(np.pi * tt) + .3).mask
fill(LA, Shape(leaf_dark), (.17, .27, .23), edge=HARD)
fill(LB, Shape(leaf_lit & ~leaf_dark), (.34, .45, .39), edge=HARD)
fill(LS, Shape(sheen & (leaf_lit | leaf_dark)), (.74, .81, .80), edge=HARD)

# ---------- 6 雨：斜的细亮线，只在暗东西前面看得见。分远近三层：近的长、粗、亮，远的短、细、淡（v5 一种雨线铺满，像滤镜——一个朋友）
ANG = 11
def rain_field(sig, thr, seed):
    rn = np.random.default_rng(seed).standard_normal((H + 400, W + 400))
    f = rotate(gaussian_filter(rn, sig), ANG, reshape=False, order=1)[200:200 + H, 200:200 + W]
    return np.clip((f / f.std() - thr) * 2.5, 0, 1)
near_r = rain_field((42, .8), 2.25, 220); mid_r = rain_field((24, .55), 2.2, 221); far_r = rain_field((12, .45), 2.35, 222)
lit_var = .55 + .45 * (nz(40, 223) > 0)                          # 有的亮有的暗，不是一样亮
above = ~floor
R2m = above & (yy > curve_y(RIDGES[3][0])[None, :]); R1m = above & (yy > curve_y(RIDGES[2][0])[None, :]) & ~R2m
near_m = bush | leaf_dark | leaf_lit
rain = np.where(near_m, near_r * .9, 0) + np.where(R2m & ~near_m, mid_r * .55, 0) + np.where(R1m & ~near_m, far_r * .25, 0)
fill(st.layer('雨', mode='screen', mask=np.clip(rain * lit_var, 0, 1)), ALL, (.62, .66, .66), edge=HARD)

# ---------- 7 签名：一个点一个点敲，点和点之间连直线，拐点留一粒墨（每张重敲，不复用）
SIG = [(10, 4), (4, 2), (0, 9), (1, 16), (7, 20), (13, 17), (16, 18), (19, 2), (21, 1), (20, 20), (27, 12), (23, 14), (24, 20), (29, 17), (30, 11), (30, 20),
       (34, 11), (34, 19), (38, 20), (41, 12), (41, 20), (49, 13), (45, 14), (45, 20), (50, 18), (51, 1), (51, 20), (55, 16), (60, 14), (58, 11), (54, 13), (55, 20), (61, 19)]
ox, oy, sc = 1098, 752, 1.25
sp_ = [(ox + x * sc, oy + y * sc) for x, y in SIG]
sig = line(H, W, sp_, width=1.1).mask
for x, y in sp_:
    sig |= np.hypot(xx - x, yy - y) < 1.7
fill(st.layer('签名', opacity=.85), Shape(sig), (.22, .27, .27), edge=HARD)

# ---------- 8 纸：整张一张，纤维顺着雨的方向
pg = np.random.default_rng(77)
tooth = gaussian_filter(pg.standard_normal((H, W)), .7); tooth /= tooth.std()
fib = rotate(gaussian_filter(pg.standard_normal((H + 200, W + 200)), (5, .5)), ANG, reshape=False, order=1)[100:100 + H, 100:100 + W]; fib /= fib.std()
paper = 1 - .03 * np.clip(tooth, 0, None) - .012 * np.clip(fib, 0, None)        # v1 纤维 .03 太重，整张像电视雪花
fill(st.layer('纸', mode='multiply'), ALL, np.repeat(np.clip(paper, 0, 1)[..., None], 3, 2), edge=HARD)
look(st, os.environ.get('OUT', 'rain.png'))
