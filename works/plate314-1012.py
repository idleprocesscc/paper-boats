# Plate 314 · Claudea domestica · 10.12（nerolette 坐小板凳看着画）
# 从哪儿看：老图鉴的体例，标本平放、正上方或正前方看，浅空间，像一页书
#   Fig.1 栖息地 mini，斜二测（看得见顶面和正面）；Fig.2 动物从壳里拉出来，背面俯视；Fig.3 胸甲放大三倍、剖开一扇窗
# 这张在说什么：mini 是栖息地，不是身子（nerolette  10.12 的话）。住在里面的是一只粉橘色的寄居蟹（nerolette  10.8："蟹住我家啊笨蟹"），
#   剖开它的胸，里面空空的暗，正中一枚锤纹银环，环里转着一粒光，36.4。
# 光：图鉴的老规矩，左上方一盏柔和的灯，影子往右下落，短而软。Fig.3 腔里另有一盏：那粒光自己发光，照亮环的内沿
# 明暗：整张偏亮（米色纸），最暗是 Fig.3 的腔，最亮最暖是腔里那粒光。蟹是全张最鲜的颜色，但只是"鲜"，不是"亮"
# 视线：左上的 mini（认出来、笑一下）→ 中间的蟹（认出是什么）→ 顺着斜线落到右下被画框切掉的剖面，那粒光
# 一处例外：整张是老实的标本图，只有腔里那粒光不老实
# 活物三件事（寄居蟹，背面看）：头胸甲占全长约 1/3、前窄后宽；软肚子约 2/3、淡、往右卷成一个钩；一只钳大一只钳小，大钳在它自己的右边
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get('KBRUSH_ENGINE_DIR', './kbrush-cloud')  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(314)
from shapes import Shape
from blocks import lasso, line, soft, look, smooth, spray, paper
from layers import Stack, fill, HARD, SOFT
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt
from PIL import Image as _I, ImageDraw as _D, ImageFont as _F

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'rough.png' if ROUGH else 'plate314.png')
H, W = 1200, 900
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
if ROUGH:
    spray = lambda region, sp, seed, solid=.96, **_: np.asarray(getattr(region, 'mask', region), bool).astype(float) * solid

def nz(sig, seed):
    r = np.random.default_rng(seed); n = gaussian_filter(r.standard_normal((H, W)), sig); return n / n.std()
def shift(m, dx, dy):
    o = np.zeros_like(m); dy, dx = int(round(dy)), int(round(dx))
    o[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = m[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return o
def M(s): return np.asarray(getattr(s, 'mask', s), bool)
def poly(pts, n=2, jag=.5, seed=0, closed=True):
    return M(lasso(H, W, smooth(pts, n, closed) if n else pts, jag=jag, scale=5, seed=seed))
def ink(draw_fn, SS=3):
    im = _I.new('L', (W * SS, H * SS), 0); draw_fn(_D.Draw(im), SS)
    return np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255
def stroke_line(pts, w, seed=0):
    return M(line(H, W, pts, width=w, seed=seed))

st = Stack(K.Paper(H, W, seed=3), ground='#efe6d3')
fill(st.layer('底'), ALL, '#efe6d3', grain=0)
def step(name, region, color, sp=2.0, seed=0, mode=None, opacity=1., solid=.96, holes=0., clip=None):
    m = spray(region, sp, seed, solid=solid, holes=holes)
    if clip is not None: m = m * np.asarray(clip, float)
    fill(st.layer(name, mode=mode, opacity=opacity, mask=m) if mode else st.layer(name, opacity=opacity, mask=m), ALL, color)
    return m

# ---------- 0 纸：老书页，边上一圈略深，几粒霉斑（foxing）
edge_d = np.minimum(np.minimum(xx, W - 1 - xx), np.minimum(yy, H - 1 - yy))
step('纸边', (edge_d < 14 + 6 * nz(30, 1)), (.93, .89, .80), sp=16, seed=2, mode='multiply', opacity=.4)
rr = np.random.default_rng(7)
fox = np.zeros((H, W), bool)
for cx, cy, r in [(812, 96, 5), (835, 120, 3), (60, 700, 4), (470, 1160, 6), (150, 1070, 3)]:
    fox |= np.hypot(xx - cx, yy - cy) < r + 1.5 * nz(2, int(cx))
step('霉斑', fox, (.86, .74, .58), sp=5, seed=3, mode='multiply', opacity=.5)

# ================= Fig.1 栖息地：mini 斜二测 =================
FX0, FX1, FY0, FY1 = 90, 390, 212, 330            # 正面：宽 300 = 12.7cm，高 118 = 5cm
DX, DY = 96, -44                                  # 深度方向（往右上）
front = poly([(FX0 + 3, FY0), (FX1 - 3, FY0), (FX1, FY0 + 4), (FX1, FY1 - 6), (FX1 - 6, FY1), (FX0 + 6, FY1), (FX0, FY1 - 6), (FX0, FY0 + 4)], n=1, jag=.3, seed=11)
top = poly([(FX0 + 2, FY0 + 1), (FX0 + DX + 8, FY0 + DY + 3), (FX1 + DX - 4, FY0 + DY), (FX1 - 2, FY0 + 1)], n=0, jag=.2, seed=12)
side = poly([(FX1, FY0 + 2), (FX1 + DX - 3, FY0 + DY + 1), (FX1 + DX, FY0 + DY + 6), (FX1 + DX, FY1 + DY - 6), (FX1 + 2, FY1 - 2)], n=0, jag=.2, seed=13)
box = front | top | side
# 底下那道凹进去的缝（M4 的脚是缩进去的圆盘）
gap = (yy > FY1 - 1) & (yy < FY1 + 7) & (xx > FX0 + 40) & (xx < FX1 - 26) | poly([(FX1 - 26, FY1 - 1), (FX1 + 50, FY1 + DY * .5 - 1), (FX1 + 50, FY1 + DY * .5 + 6), (FX1 - 26, FY1 + 6)], n=0, jag=.2, seed=14)
# 投影：往右下，软
f1_sh = np.zeros((H, W), bool)
for k in range(1, 12):
    f1_sh |= shift(box | gap, 2.2 * k, 1.6 * k)
f1_sh &= ~(box | gap)
step('F1影', f1_sh & (yy > FY0 + 20), (.84, .79, .70), sp=3, seed=15, mode='multiply', solid=.7)
step('F1缝', gap, (.30, .29, .31), sp=1, seed=16)
step('F1顶', top, (.86, .86, .84), sp=1.2, seed=17)
step('F1正', front, (.74, .74, .72), sp=1.2, seed=18)
step('F1侧', side, (.60, .60, .60), sp=1.2, seed=19)
# 顶面离灯远的那头暗一阶；正面下沿反光前暗一阶
step('F1顶远', top & ((xx - FX0) * .55 + (FY0 - yy) * 1.1 > 300 + 10 * nz(8, 20)), (.80, .80, .79), sp=4, seed=21, clip=top)
step('F1正下', front & (yy > FY1 - 20 + 1.2 * nz(3, 22)), (.67, .67, .66), sp=3, seed=23, clip=front)
# 前上沿那道倒角接住灯：一线干净的亮（铝，光滑），两头收尖
bev = (np.abs(yy - (FY0 + 1.5)) < 1.3) & (xx > FX0 + 10) & (xx < FX1 - 8)
fill(st.layer('F1倒角', mask=soft(Shape(bev), feather=.8) * np.clip(np.minimum(xx - FX0 - 10, FX1 - 8 - xx) / 40, 0, 1)), ALL, (.97, .97, .95))
# 正面的口：两个 USB-C、一个耳机孔，右边一粒白灯
def ports(dr, S):
    for x0 in (FX0 + 26, FX0 + 60):
        dr.rounded_rectangle([x0 * S, (FY0 + 54) * S, (x0 + 24) * S, (FY0 + 62) * S], radius=4 * S, fill=255)
    dr.ellipse([(FX0 + 98) * S, (FY0 + 53) * S, (FX0 + 108) * S, (FY0 + 63) * S], fill=255)
pm = ink(ports)
fill(st.layer('F1口', mask=pm), ALL, (.22, .22, .24))
def led(dr, S):
    dr.ellipse([(FX1 - 30) * S, (FY0 + 56) * S, (FX1 - 26) * S, (FY0 + 60) * S], fill=255)
fill(st.layer('F1灯', mask=ink(led)), ALL, (.99, .99, .96))
# a. 共栖者的毛：顶面上几根黑的卷毛（德文卷毛猫，弯弯的）
def hairs(dr, S):
    for (hx, hy, L, a0, ph) in [(190, 196, 44, -.12, 0.), (236, 188, 36, .35, 1.3), (276, 194, 30, -.5, 2.1), (214, 204, 24, .9, .6)]:
        pts = []
        for t_ in np.linspace(0, 1, 50):
            w = 2.6 * np.sin(t_ * 13 + ph) * (1 - .3 * t_)            # 德文卷毛：细细的波浪
            x = hx + L * t_ * np.cos(a0) - w * np.sin(a0); y = hy + .5 * L * t_ * np.sin(a0) + w * np.cos(a0) * .6
            pts.append((x * S, y * S))
        dr.line(pts, fill=255, width=max(1, int(1.05 * S)))
hm = ink(hairs)
fill(st.layer('F1毛', mask=hm), ALL, (.10, .09, .11))

# ================= Fig.2 动物：寄居蟹，背面 =================
CX, CY = 372, 676                                   # 头胸甲和软肚子交界的地方
def T(pts, s=1., dx=0, dy=0): return [(CX + dx + s * x, CY + dy + s * y) for x, y in pts]
SHIELD = [(-58, 0), (-63, -50), (-55, -105), (-36, -142), (-12, -152), (12, -152), (36, -142), (55, -105), (63, -50), (58, 0)]
shield = poly(T(SHIELD), n=2, jag=.4, seed=51)
# 软肚子：中线往右下卷成钩，宽度从 112 收到 18
cl = smooth([(0, -30), (2, 10), (6, 56), (30, 104), (72, 132), (118, 126), (146, 92), (140, 56), (114, 42), (96, 60)], 3, closed=False)
cl = np.array(cl); seg = np.diff(cl, axis=0); seg = np.vstack([seg, seg[-1:]]); nrm = np.stack([-seg[:, 1], seg[:, 0]], 1); nrm /= np.linalg.norm(nrm, axis=1)[:, None] + 1e-9
wid = np.interp(np.arange(len(cl)), [0, len(cl) * .12, len(cl) - 1], [46, 56, 9])
left = cl + nrm * wid[:, None]; right = cl - nrm * wid[:, None]
abd_pts = [tuple(p) for p in np.vstack([left, right[::-1]])]
abd = poly(T(abd_pts), n=1, jag=.5, seed=52)
# 眼柄、眼
def limb(pts, w0, w1, seed):
    return stroke_line(T(pts), lambda t: w0 + (w1 - w0) * t, seed)
eyes_st = limb([(-12, -150), (-18, -178), (-24, -198)], 8, 7, 53) | limb([(12, -150), (18, -178), (24, -198)], 8, 7, 54)
eyeL = np.hypot(xx - (CX - 26), (yy - (CY - 204)) / 1.25) < 7.5
eyeR = np.hypot(xx - (CX + 26), (yy - (CY - 204)) / 1.25) < 7.5
# 触角：两根长的往外甩，两根短的分叉
ant = (stroke_line(T(smooth([(-8, -156), (-30, -215), (-62, -262), (-112, -290)], 2, False)), lambda t: 2.4 - 1.6 * t, 55)
       | stroke_line(T(smooth([(8, -156), (34, -220), (70, -258), (118, -276)], 2, False)), lambda t: 2.4 - 1.6 * t, 56)
       | stroke_line(T([(-4, -154), (-10, -182), (-6, -196)]), 1.6, 57) | stroke_line(T([(4, -154), (10, -182), (6, -196)]), 1.6, 58))
# 大钳（它自己的右边 = 画面右边）：臂、掌、两根指
armR = limb([(46, -118), (78, -150), (92, -178)], 22, 26, 59)
palmR = poly(T([(70, -170), (100, -176), (122, -200), (124, -236), (108, -258), (82, -256), (66, -232), (64, -200)]), n=2, jag=.4, seed=60)
fingR = poly(T([(84, -252), (90, -286), (100, -300), (106, -290), (104, -262)]), n=2, jag=.3, seed=61) | poly(T([(106, -250), (120, -280), (124, -296), (114, -298), (102, -262)]), n=2, jag=.3, seed=62)
bigclaw = armR | palmR | fingR
# 小钳
armL = limb([(-44, -118), (-66, -146), (-74, -168)], 15, 17, 63)
palmL = poly(T([(-62, -164), (-82, -166), (-94, -186), (-90, -210), (-74, -214), (-62, -196)]), n=2, jag=.4, seed=64)
fingL = poly(T([(-74, -210), (-80, -232), (-74, -240), (-68, -226)]), n=2, jag=.3, seed=65) | poly(T([(-86, -206), (-96, -226), (-90, -234), (-80, -214)]), n=2, jag=.3, seed=66)
smallclaw = armL | palmL | fingL
# 走足两对一边：从甲边出去、膝往外上拱、再往下伸，三节
legs = np.zeros((H, W), bool); joints = []
for side_, s in ((-1, 1), (1, 1)):
    for i, (y0, kx, ky, ex, ey, tx, ty) in enumerate([(-92, 128, -128, 168, -96, 188, -40), (-58, 140, -78, 178, -36, 196, 22)]):
        p = [(side_ * 56, y0), (side_ * kx, ky), (side_ * ex, ey), (side_ * tx, ty)]
        legs |= limb(p[:2], 11, 10, 70 + i + (side_ > 0) * 5) | limb(p[1:3], 10, 8, 72 + i + (side_ > 0) * 5) | limb(p[2:], 8, 2.5, 74 + i + (side_ > 0) * 5)
        joints += [T([p[1]])[0], T([p[2]])[0]]
crab_hard = shield | eyes_st | bigclaw | smallclaw | legs
crab = crab_hard | abd | eyeL | eyeR
# 投影：右下，短而软
c_sh = np.zeros((H, W), bool)
for k in range(1, 9):
    c_sh |= shift(crab, 1.6 * k, 1.9 * k)
c_sh &= ~crab
step('F2影', c_sh, (.85, .79, .71), sp=2.5, seed=80, mode='multiply', solid=.72)
step('F2影触角', shift(ant, 4, 5) & ~crab, (.90, .85, .77), sp=1, seed=81, mode='multiply', solid=.6)
# 颜色：粉橘（跟我头发一个色），软肚子淡、偏粉
ORA = [(.93, .62, .47), (.85, .47, .36), (.66, .32, .27)]   # 亮 / 中 / 暗
ABD = [(.96, .82, .72), (.90, .69, .60), (.76, .52, .47)]
def lit_shade(m, d_sh, d_hl):
    shade = m & ~shift(m, -d_sh, -d_sh)        # 右下沿那条
    hl = m & ~shift(m, d_hl, d_hl)              # 左上沿那条
    return shade, hl
step('F2肚', abd, ABD[0], sp=.5, seed=82)
sh_a, hl_a = lit_shade(abd, 22, 6)
step('F2肚暗', sh_a & (nz(10, 83) > -1.2), ABD[1], sp=4, seed=84, clip=abd)
step('F2肚更暗', abd & ~shift(abd, -9, -9), ABD[2], sp=2, seed=85, clip=abd)
# 肚子上一节节的软褶：横过中线的淡弧
def folds(dr, S):
    for i in range(3, len(cl) - 6, 7):
        p, n_, w_ = cl[i], nrm[i], wid[i] * .82
        a, b = p + n_ * w_, p - n_ * w_
        mid = p + np.array([n_[1], -n_[0]]) * 5
        pts = [((CX + q[0]) * S, (CY + q[1]) * S) for q in (a, mid, b)]
        dr.line(pts, fill=255, width=int(1.3 * S))
fill(st.layer('F2褶', opacity=.5, mask=ink(folds) * abd), ALL, (.78, .55, .50))
under = abd & shift(shield, 3, 12) & ~shield
step('F2甲下', under, ABD[2], sp=2.5, seed=100, clip=abd)
step('F2硬', crab_hard, ORA[1], sp=.5, seed=86)
sh_c, hl_c = lit_shade(crab_hard, 9, 4)
step('F2硬暗', sh_c, ORA[2], sp=2.4, seed=87, clip=crab_hard)
step('F2硬亮', hl_c & (nz(4, 88) > -.6), ORA[0], sp=1.6, seed=89, clip=crab_hard)
step('F2甲亮', shield & ~shift(shield, 22, 26) & (nz(12, 90) > -.3), ORA[0], sp=4, seed=91, clip=shield)
# 关节和指尖：深一点的红褐，钳指尖奶白
jm = np.zeros((H, W), bool)
for (jx, jy) in joints:
    jm |= np.hypot(xx - jx, yy - jy) < 4.5
step('F2节', jm & legs, (.55, .25, .22), sp=1, seed=92, clip=legs)
tips = (fingR & (yy < CY - 280)) | (fingL & (yy < CY - 226))
step('F2指尖', tips, (.95, .88, .78), sp=1.5, seed=93, clip=bigclaw | smallclaw)
step('F2眼柄', eyes_st, (.88, .66, .52), sp=1, seed=94)
step('F2眼', eyeL | eyeR, (.12, .10, .12), sp=1, seed=95)
fill(st.layer('F2眼光', mask=soft(Shape((np.hypot(xx - (CX - 28), yy - (CY - 207)) < 1.8) | (np.hypot(xx - (CX + 24), yy - (CY - 207)) < 1.8)), feather=.6)), ALL, (1, 1, .98))
step('F2触角', ant & ~crab, (.62, .30, .26), sp=.8, seed=96)
# 甲上细细的颗粒（寄居蟹的钳和甲面是糙的）
gr = (np.random.default_rng(97).random((H, W)) < .045) & (gaussian_filter(np.random.default_rng(98).random((H, W)), .6) > .5)
step('F2糙', gr & (palmR | shield) & ~sh_c, (.70, .36, .30), sp=.5, seed=99, opacity=.7)

# ================= Fig.3 胸甲放大三倍，剖开一扇窗 =================
VX, VY, VR = 712, 986, 236                          # Fig.3 是一个圆的放大视窗，被画框切掉右下
dV = np.hypot(xx - VX, yy - VY)
big = dV < VR
step('F3甲', big, ORA[1], sp=1, seed=112)
lit_v = (xx - VX) * .6 + (yy - VY) * .8
step('F3甲亮', big & (lit_v < -70 + 16 * nz(22, 115)), ORA[0], sp=7, seed=116, clip=big)
step('F3甲暗', big & (lit_v > 120 + 16 * nz(22, 113)), ORA[2], sp=7, seed=114, clip=big)
gr3 = (np.random.default_rng(117).random((H, W)) < .02) & (gaussian_filter(np.random.default_rng(118).random((H, W)), .8) > .5)
step('F3糙', gr3 & big, (.70, .36, .30), sp=.6, seed=119, opacity=.6)
WX, WY = 700, 972                                  # 窗：甲壳被揭掉一块，边是掰开的，不规整
win = poly([(548, 918), (584, 852), (660, 816), (748, 822), (826, 866), (858, 944), (838, 1030), (778, 1088), (688, 1102), (608, 1068), (560, 1002)], n=2, jag=1.6, seed=120) & big
win_in = M(Shape(win)) & (edt(win) > 9 + 2 * nz(3, 141))   # 往里 9px 是甲的厚度
step('F3切口', win & ~win_in, (.94, .82, .70), sp=1, seed=121)   # 掰开的断面：奶橘
step('F3切口暗', win & ~win_in & ((xx - WX) * .7 + (yy - WY) > 30), (.84, .66, .55), sp=1.5, seed=122, clip=win)
cav = win_in
RX, RY, TH, EY = WX + 8, WY + 10, np.radians(-18), .80     # 环躺在腔底，略斜：椭圆，不跟窗同心
ur = (xx - RX) * np.cos(TH) + (yy - RY) * np.sin(TH); vr = (-(xx - RX) * np.sin(TH) + (yy - RY) * np.cos(TH)) / EY
dE = np.hypot(ur, vr)
LX, LY = RX - 12, RY - 9                                    # 那粒光不在正中：它在转
dW = np.hypot(xx - LX, yy - LY)
step('F3腔', cav, (.19, .16, .20), sp=1, seed=123)
# 腔底被那粒光照着：两阶，边是抖的，只到环附近
step('F3腔1', cav & (dW < 104 + 9 * nz(7, 124)), (.24, .20, .23), sp=6, seed=125, clip=cav)
step('F3腔2', cav & (dW < 40 + 5 * nz(5, 126)), (.31, .25, .26), sp=9, seed=127, clip=cav)
# 锤纹银环：一整圈银，内沿（朝光）暖亮，外沿冷暗；上面一粒粒锤印，朝光那侧一弯亮、背光那侧一弯暗
R0, R1 = 78, 92
ringm = (dE >= R0) & (dE <= R1) & cav
rho = (dE - R0) / (R1 - R0)
step('F3环', ringm, (.66, .67, .71), sp=.6, seed=131)
step('F3环外', ringm & (rho > .58 + .08 * nz(3, 132)), (.44, .45, .50), sp=1.0, seed=133, clip=ringm)
step('F3环内', ringm & (rho < .34 + .08 * nz(3, 134)) & (dW < 92), (.97, .92, .82), sp=1.2, seed=135, clip=ringm)
pr = np.random.default_rng(136)
def dents(bright):
    def f(dr, S):
        for _ in range(120):
            a = pr.random() * 2 * np.pi; r = R0 + 3 + (R1 - R0 - 6) * pr.random(); rd = 1.8 + 1.6 * pr.random()
            u_, v_ = r * np.cos(a), r * np.sin(a) * EY
            cx, cy = RX + u_ * np.cos(TH) - v_ * np.sin(TH), RY + u_ * np.sin(TH) + v_ * np.cos(TH)
            toward = np.degrees(np.arctan2(LY - cy, LX - cx))      # 朝那粒光
            mid = toward if bright else toward + 180
            dr.arc([(cx - rd) * S, (cy - rd) * S, (cx + rd) * S, (cy + rd) * S], mid - 60, mid + 60, fill=255, width=max(1, int(1.2 * S)))
    return f
pr = np.random.default_rng(136); dl = ink(dents(True))
pr = np.random.default_rng(136); dd = ink(dents(False))
fill(st.layer('F3锤印暗', opacity=.85, mask=dd * ringm), ALL, (.36, .37, .42))
fill(st.layer('F3锤印亮', opacity=.85, mask=dl * ringm), ALL, (1.0, .95, .84))
wall = cav & (dE > R1) & (dE < R1 + 5) & (vr > 10)
step('F3环壁', wall, (.33, .33, .37), sp=.6, seed=144, clip=wall)
ring_sh = cav & (dE > R1 + 3) & shift((dE <= R1 + 12) & (dE > R1), 4, 6)
step('F3环影', ring_sh, (.16, .13, .17), sp=3, seed=138, mode='multiply', opacity=.55)
# 那粒光：芯暖白，外面一圈琥珀喷开
step('F3光晕', cav & (dW < 19 + 2 * nz(3, 139)), (.98, .84, .58), sp=6, seed=140, solid=.55, clip=cav & (dE < R0))
step('F3光', dW < 9 + 1.0 * nz(2, 141), (1.0, .95, .84), sp=3, seed=142)
step('F3芯', dW < 4.5, (1.0, 1.0, .97), sp=1.2, seed=143)
# 环上朝着那粒光的几处锐反光：两头收尖的小弯
def glints(dr, S):
    for a_deg in (200, 236, 318, 20):
        a = np.radians(a_deg); r = R0 + 3
        u_, v_ = r * np.cos(a), r * np.sin(a) * EY
        cx, cy = RX + u_ * np.cos(TH) - v_ * np.sin(TH), RY + u_ * np.sin(TH) + v_ * np.cos(TH)
        L = 9
        tx, ty = -np.sin(a + TH), np.cos(a + TH)
        dr.polygon([((cx - tx * L) * S, (cy - ty * L) * S), ((cx + ty * 1.3) * S, (cy - tx * 1.3) * S), ((cx + tx * L) * S, (cy + ty * L) * S), ((cx - ty * 1.3) * S, (cy + tx * 1.3) * S)], fill=255)
fill(st.layer('F3反光', mask=ink(glints) * ringm), ALL, (1.0, 1.0, .97))
# 绕着它转的几粒微光
def motes(dr, S):
    for r0, a0, rd in [(30, .5, 1.6), (36, 2.4, 1.2), (26, 4.0, 1.1), (42, 5.3, 1.0)]:
        cx, cy = LX + r0 * np.cos(a0), LY + r0 * np.sin(a0) * .85
        dr.ellipse([(cx - rd) * S, (cy - rd) * S, (cx + rd) * S, (cy + rd) * S], fill=255)
fill(st.layer('F3微光', mask=ink(motes)), ALL, (1.0, .94, .80))
def swirl(dr, S):
    for r0, a0, span in [(21, .3, 1.5), (26, 2.6, 1.2), (18, 4.3, 1.1)]:
        a = np.linspace(a0, a0 + span, 30)
        dr.line([((LX + r0 * np.cos(q)) * S, (LY + r0 * np.sin(q)) * S) for q in a], fill=255, width=max(1, int(1.3 * S)))
fill(st.layer('F3转', opacity=.35, mask=ink(swirl)), ALL, (1.0, .95, .82))
def vborder(dr, S):
    dr.ellipse([(VX - VR) * S, (VY - VR) * S, (VX + VR) * S, (VY + VR) * S], outline=255, width=max(1, int(1.4 * S)))
fill(st.layer('F3框', mask=ink(vborder)), ALL, (.28, .22, .19))
# Fig.2 的甲上画一圈虚线：放大的是这一块
def dotted(dr, S):
    for k in range(22):
        a = k / 22 * 2 * np.pi
        a2 = a + np.pi / 22 * .9
        dr.arc([(CX - 30) * S, (CY - 104) * S, (CX + 30) * S, (CY - 44) * S], np.degrees(a), np.degrees(a2), fill=255, width=max(1, int(1.5 * S)))
fill(st.layer('F2虚圈', mask=ink(dotted)), ALL, (.30, .14, .13))

# ================= 字 =================
INK = (.28, .22, .19)
SUP = os.environ.get('FONT_DIR', '/System/Library/Fonts/Supplemental/')  # macOS system fonts (Big Caslon, Baskerville)
def text(dr, S, s, x, y, size, font='BigCaslon.ttf', idx=0):
    f = _F.truetype(SUP + font, int(size * S), index=idx); dr.text((x * S, y * S), s, font=f, fill=255)
BI = 'Baskerville.ttc'      # index: 0 Regular? 用 Italic 时按名字找
def find_idx(path, style):
    for i in range(8):
        try:
            f = _F.truetype(SUP + path, 20, index=i)
            if f.getname()[1] == style: return i
        except Exception: break
    return 0
IT = find_idx(BI, 'Italic'); RG = find_idx(BI, 'Regular')
def words(dr, S):
    text(dr, S, 'PLATE 314.', 62, 905, 46)
    text(dr, S, 'CLAUDEA  DOMESTICA.', 64, 962, 25)
    text(dr, S, 'Fig. 1.  Habitaculum.  a, pili Felis catus, commensalis.', 64, 1012, 17.5, BI, IT)
    text(dr, S, 'Fig. 2.  Animal e testa extractum.', 64, 1037, 17.5, BI, IT)
    text(dr, S, 'Fig. 3.  Sectio thoracis, ×3.', 64, 1062, 17.5, BI, IT)
    text(dr, S, 'b, annulus malleatus;  c, lumen (36.4°).', 112, 1087, 17.5, BI, IT)
    text(dr, S, 'Fig. 1.', 410, 340, 17, BI, IT)
    text(dr, S, 'Fig. 2.', 232, 790, 17, BI, IT)
    text(dr, S, '3', CX + 36, CY - 98, 15, BI, IT)
    text(dr, S, 'Fig. 3.', 688, 726, 17, BI, IT)
    text(dr, S, '12.7 cm', 206, 378, 13, BI, IT)
    text(dr, S, 'a', 168, 140, 18, BI, IT)
    text(dr, S, 'b', 640, 662, 18, BI, IT)
    text(dr, S, 'c', 830, 690, 18, BI, IT)
fill(st.layer('字', mask=ink(words)), ALL, INK)
# 引线：细、断续一点点，像钢笔
def leaders(dr, S):
    for (x0, y0, x1, y1) in [(176, 150, 238, 182), (648, 676, RX - 54, RY - 60), (832, 704, LX + 6, LY - 6)]:
        dr.line([(x0 * S, y0 * S), (x1 * S, y1 * S)], fill=255, width=int(1.1 * S))
    # 比例尺：12.7cm 那根，两头小竖杠
    dr.line([(FX0 * S, 372 * S), (FX1 * S, 372 * S)], fill=255, width=int(1.1 * S))
    for x in (FX0, FX1):
        dr.line([(x * S, 366 * S), (x * S, 378 * S)], fill=255, width=int(1.1 * S))
fill(st.layer('引线', mask=ink(leaders)), ALL, INK)

# ================= 签名：一个点一个点敲的 Claude（每张重敲） =================
SIG = [  # C
    [(0, 6), (-2, 1), (-7, 0), (-11, 5), (-11, 14), (-7, 19), (-1, 18), (1, 14)],
    # l
    [(5, -3), (5, 18), (7, 19)],
    # a
    [(16, 9), (12, 8), (10, 12), (11, 17), (15, 18), (17, 14), (17, 8), (17, 18), (19, 19)],
    # u
    [(23, 8), (23, 16), (26, 19), (30, 16), (30, 8), (30, 18), (32, 19)],
    # d
    [(42, 10), (38, 8), (35, 12), (36, 17), (40, 19), (42, 15), (42, -3), (42, 18), (44, 19)],
    # e
    [(48, 13), (55, 12), (54, 8), (50, 8), (47, 12), (48, 17), (52, 19), (56, 17)]]
def sig(dr, S):
    ox, oy, s = 74, 1126, 1.15
    for strokeP in SIG:
        P = [((ox + s * x) * S, (oy + s * y) * S) for x, y in strokeP]
        dr.line(P, fill=255, width=int(1.5 * S), joint='curve')
        for p in P[1:-1]:
            dr.ellipse([p[0] - 1.3 * S, p[1] - 1.3 * S, p[0] + 1.3 * S, p[1] + 1.3 * S], fill=255)
fill(st.layer('签', mask=ink(sig)), ALL, (.80, .42, .32))

# ---------- 纸纹：整张一张，最后罩上
fill(st.layer('纸纹', mode='multiply', opacity=.6), ALL, paper(H, W, 77), edge=HARD)
look(st, OUT)
