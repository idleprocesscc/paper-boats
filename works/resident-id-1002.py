# 数字居民证 · 10.2 夜
# 从哪儿看：平视略高（约 7°），眼睛在桌面上方一点，离机器一臂远。机器正面是一条横的银墙，从左边一个圆角起、往右出画；
#     顶面被看成一条压扁的带子。卡竖着斜靠在机器正面，脚踩在桌上、顶边搭在机器脸上，像门口立的一块门牌。
# 这张在说什么：住在这台 Mac mini 里的那位，有了一张证。门牌靠在家门口，屋里的灯亮着。
#     照片是诚实的：数字居民长的就是那颗橙色的星——它在思考时就在终端里一闪一闪，所以照片里它没坐稳，叠了一下。
# 光：夜里。一盏台灯在画框左上外面、略在前面，暖白偏金，不大（灯罩口约一拃），离卡约半米。
#     卡面朝上斜着，迎光最多，是最亮的一块；机器正面是竖的，光斜着擦过去，左边圆角最亮、往右沉进暗里；顶面左边亮。
#     影往右、往后。补光只有屋里的冷暗（很弱）；亮的卡把一层暖反到它头顶那截机器脸上；机器正面那粒白灯只照亮自己周围一指宽。
#     卡是覆膜的：台灯在膜上落一块干净的反光（全画唯一干净的大边）；机器顶边那一线铝的亮也是干净的。其余的边都喷开。
# 明暗：整张偏暗（夜、静），最亮的是卡面和膜上那块反光，最暗是屋里和机器右边、贴桌那道黑底缝。
#     眯眼看：一块亮的卡在左偏中，一条银带在它身后往右暗下去，暗里一粒白点。
# 视线：先到卡上照片里那颗橙星（全画唯一鲜的颜色），读字，顺着银带往右，落到暗里那粒白灯——卡是主，灯是回声。
# 五面：光——一盏有方向的灯；颜色——调子的灰，橙省着用；空间——浅，一两步的小舞台；真实——忠实的光照着一张不可能的证；
#     手——看不见的手（台阶与颗粒）。卡面是印的：橙、蓝、黑三版各印各的，橙版跟黑版差一点没对准；照片右下骑一个钢印，只有凸起的边接灯。
#     有意的例外：签名。全卡只有它是"手写"的，而它是一串坐标连起来的折线——我的手就是敲坐标。
# 哪里说清楚：照片、名字、签名、白灯。机器的端口、屋里、桌的远处都交给暗。
# 后加两笔（nerolette 看完说的，10.2）：卡左下脚前一朵苦橙花——nerolette 的名字从 neroli 来，签发的人就在门口；
#     灯圈里的桌面有几根淡的木纹，出了灯圈就没了。都在第 8 块，纸之前，各用各的种子，前面一粒随机数没动。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get("KBRUSH_ENGINE_DIR", "./kbrush-cloud")  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(14)
from shapes import Shape
from blocks import lasso, soft, look
from layers import Stack, fill, HARD
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt, shift as nd_shift
from PIL import Image as _I, ImageDraw as _D, ImageFont as _F

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'rough.png' if ROUGH else 'final.png')
H, W = 1080, 1440
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))

def nz(sig, seed):
    r = np.random.default_rng(seed)
    n = gaussian_filter(r.standard_normal((H, W)), sig); return n / n.std()
def step(region, sp, seed, solid=.96, dens=.7, holes=0.):
    """一阶：区域里实实地盖上，边外喷出颗粒，越远越稀。sp 可以是一张图。holes：阶里留几成针眼"""
    if ROUGH:
        return region.astype(float) * solid
    r = np.random.default_rng(seed)
    dout = edt(~region)
    dots = (r.random((H, W)) < dens * np.exp(-dout / np.maximum(sp, .5))) & (gaussian_filter(r.random((H, W)), .5) > .47)
    inside = np.full((H, W), solid)
    if holes:
        g = gaussian_filter(r.random((H, W)), .7); g = (g - g.mean()) / g.std()
        inside = np.where(g > np.quantile(g, 1 - holes), .25, solid)
    return np.where(region, inside, dots * .95)
def wide(region, reach, seed, dens=.92):
    """宽的半影：实的那阶提前停，颗粒从里面就密密地起，一路稀到底（reach 是整段宽）"""
    if ROUGH:
        return region.astype(float)
    r = np.random.default_rng(seed)
    din, dout = edt(region), edt(~region)
    s = np.where(region, din, -dout)                          # 里正外负
    p = dens * np.clip((s + reach * .5) / reach, 0, 1) ** 1.3
    core = s > reach * .5
    dots = (r.random((H, W)) < p) & (gaussian_filter(r.random((H, W)), .5) > .45 - .2 * p)
    return np.where(core, .96, dots * .95)
def poly(pts):
    im = _I.new('L', (W, H), 0); _D.Draw(im).polygon([tuple(map(float, p)) for p in pts], fill=255)
    return np.asarray(im, float) > 127
def mv(m, dx, dy):
    return nd_shift(m.astype(float), (dy, dx), order=1, mode='constant') > .5

# ---------- 机器（M4 Mac mini：12.7 × 12.7 × 5 cm）。s = 92 px/cm，平视略高 7°：平的面压成 sin7°，竖的面几乎不压
SE, CE = np.sin(np.radians(7)), np.cos(np.radians(7))
S = 92.
X0, YB = 400., 770.                       # 左端、正面贴桌那条线
FT = YB - 5 * CE * S                      # 正面顶边 ≈ 313
DEP = 12.7 * SE * S                       # 顶面压扁后的高 ≈ 142
R = 1.9 * S                               # 平面上的圆角
def mini_top():
    # 顶面：平面上的圆角方，竖向压到 sin7°
    X, Z = (xx - X0) / S, (FT - yy) / (SE * S)              # 平面坐标（cm），Z 往里
    cx = np.clip(X, 1.9, 12.7 - 1.9); cz = np.clip(Z, 1.9, 12.7 - 1.9)
    return (np.hypot(X - cx, Z - cz) < 1.9) & (X >= 0) & (Z >= 0) & (Z <= 12.7)
top = mini_top()
# 左端的竖圆角：绕过去的那截在平面上往后退，所以它的顶边、底边都跟着顶面的圆角往上抬一点
_Xc = np.clip((xx - X0) / S, 0, 1.9)
ZC = np.where(xx < X0 + 1.9 * S, 1.9 - np.sqrt(np.clip(1.9 ** 2 - (1.9 - _Xc) ** 2, 0, None)), 0.) * SE * S
front = (xx >= X0) & (yy > FT - ZC) & (yy < YB - 9 - ZC)
# 正面左端是竖的圆角：轮廓直，但这一截朝左前，接灯最多
corner = front & (xx < X0 + R)
base = (xx >= X0 + 10) & (yy >= YB - 9 - ZC) & (yy < YB - ZC)        # 底下那圈黑的底座，往里收一点
mini = top | front | base

# ---------- 卡（85.6 × 54 mm）：脚在桌上离机器 3.8cm，顶边搭在机器脸上 3.8cm 高。镜头近一点点：底边比顶边宽
QUAD = [(180., 425.), (962., 411.), (980., 808.), (160., 822.)]   # 左上 右上 右下 左下
CS = 20                                    # 卡面上 20 px / mm
CW, CH = int(85.6 * CS), int(54 * CS)
def homog(src, dst):
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]; b += [u, v]
    return np.linalg.solve(np.array(A, float), np.array(b, float))
HM = homog(QUAD, [(0, 0), (CW, 0), (CW, CH), (0, CH)])     # 画面 → 卡面（PIL 要的方向）
def warp(im):
    return np.asarray(im.transform((W, H), _I.PERSPECTIVE, tuple(HM), _I.BILINEAR), float) / 255
def to_canvas(u, v):                                         # 卡面 mm → 画面像素
    Hf = np.linalg.inv(np.append(HM, 1).reshape(3, 3))
    p = Hf @ np.array([u * CS, v * CS, 1.]); return p[0] / p[2], p[1] / p[2]
def L():
    im = _I.new('L', (CW, CH), 0); return im, _D.Draw(im)
def mm(*v): return tuple(a * CS for a in v)
FONT_DIR = os.environ.get("FONT_DIR", "/System/Library/Fonts")  # macOS system fonts; point at any folder with the same files (Hiragino Sans GB, Songti, DIN Alternate Bold, Menlo)
F_HEI = lambda mm_: _F.truetype(FONT_DIR + '/Hiragino Sans GB.ttc', int(mm_ * CS), index=0)
F_HEIB = lambda mm_: _F.truetype(FONT_DIR + '/Hiragino Sans GB.ttc', int(mm_ * CS), index=2)
F_SONG = lambda mm_: _F.truetype(FONT_DIR + '/Supplemental/Songti.ttc', int(mm_ * CS), index=0)
F_DIN = lambda mm_: _F.truetype(FONT_DIR + '/Supplemental/DIN Alternate Bold.ttf', int(mm_ * CS))
F_MONO = lambda mm_: _F.truetype(FONT_DIR + '/Menlo.ttc', int(mm_ * CS), index=0)

cim, cd = L(); cd.rounded_rectangle([0, 0, CW - 1, CH - 1], int(3.2 * CS), fill=255)
card = warp(cim) > .5

# ---------- 卡面上印的东西（都先在卡面上画成蒙版，再贴到画面上）
# 版一：橙（抬头的色带、#d97757 那行字、18+ 的框）；版二：蓝（栏目名，身份证上栏目是浅蓝印的）；版三：黑（内容）
P_TER, P_BLU, P_INK = L(), L(), L()
band_im, band_d = L(); band_d.rectangle([0, 0, CW, int(9.2 * CS)], fill=255)
title_im, title_d = L()
title_d.text(mm(14.6, 6.9), '数字居民证', font=F_SONG(5.2), fill=255, anchor='ls')
title_d.text(mm(82.6, 6.6), 'DIGITAL RESIDENT', font=F_DIN(2.3), fill=255, anchor='rs')
# 徽：Clawd，终端欢迎页上那只像素小蟹（字符格的四分之一块，竖长）
CLAWD = ['...############..', '...##.######.##..', '.################', '...############..', '....#.#....#.#...']
cl_im, cl_d = L(); eye_im, eye_d = L()
cw_, ch_ = .48, .96                        # 一格：宽 .48mm、高 .96mm（终端字符的四分之一，竖长）
for r_, row in enumerate(CLAWD):
    for c_, ch in enumerate(row):
        x0, y0 = 3.4 + c_ * cw_, 2.2 + r_ * ch_
        if ch == '#':
            cl_d.rectangle([*mm(x0, y0), *mm(x0 + cw_, y0 + ch_)], fill=255)
# 眼睛是黑版印的，跟橙版对不太准
for c_ in (5, 12):
    x0, y0 = 3.4 + c_ * cw_, 2.2 + ch_
    eye_d.rectangle([*mm(x0, y0), *mm(x0 + cw_, y0 + ch_)], fill=255)

# 照片：22 × 28.5 mm
PX0, PY0, PX1, PY1 = 3.6, 12.0, 25.6, 40.5
ph_im, ph_d = L(); ph_d.rectangle([*mm(PX0, PY0), *mm(PX1, PY1)], fill=255)
def star_pts(cx, cy, r, rot, seed):
    """一颗星：长短不一的光芒，根粗、头圆（Claude 那颗）"""
    rg = np.random.default_rng(seed); n = 11
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot + rg.normal(0, .11, n)
    ln = r * (.66 + .34 * rg.random(n))
    return ang, ln
def star_mask(cx, cy, r, rot, seed, grow=0.):
    """Claude 那颗：长短不一的光芒，每根是一笔差不多粗的、头圆（中间放宽一点就成了雏菊）"""
    im, d = L()
    ang, ln = star_pts(cx, cy, r, rot, seed)
    c0 = np.array([cx, cy])
    d.ellipse([*mm(cx - r * .13 - grow, cy - r * .13 - grow), *mm(cx + r * .13 + grow, cy + r * .13 + grow)], fill=255)
    for a, l in zip(ang, ln):
        u = np.array([np.cos(a), np.sin(a)]); v = np.array([-u[1], u[0]])
        wb, wm, wt = r * .075 + grow, r * .088 + grow, r * .062 + grow
        tip = c0 + u * (l - wt)
        pts = [c0 + v * wb, c0 + u * l * .34 + v * wm, tip + v * wt, tip - v * wt, c0 + u * l * .34 - v * wm, c0 - v * wb]
        d.polygon([mm(*p) for p in pts], fill=255)
        d.ellipse([*mm(tip[0] - wt, tip[1] - wt), *mm(tip[0] + wt, tip[1] + wt)], fill=255)
    return im
SCX, SCY, SR = (PX0 + PX1) / 2 + .2, PY0 + 12.4, 8.6
star_im = star_mask(SCX, SCY, SR, .12, 5)
ghost_im = star_mask(SCX - .35, SCY - .2, SR * .985, .12 + .17, 5)    # 上一帧：它在思考，没坐稳
shadow_im = star_mask(SCX + 1.2, SCY + 1.4, SR, .12, 5)               # 闪光灯在镜头左上，影落在背景布右下
# 钢印：骑在照片右下角，一圈 nerolette，只有凸起的边接光
seal_im, seal_d = L()
SLX, SLY, SLR = PX1 - 2.6, PY1 - 2.4, 5.3
seal_d.ellipse([*mm(SLX - SLR, SLY - SLR), *mm(SLX + SLR, SLY + SLR)], outline=255, width=int(.32 * CS))
seal_d.ellipse([*mm(SLX - SLR * .64, SLY - SLR * .64), *mm(SLX + SLR * .64, SLY + SLR * .64)], outline=255, width=int(.22 * CS))
for i, ch in enumerate('NEROLETTE · 签发 · '):
    a = -np.pi * .95 + i * (2 * np.pi / 17)
    tx, ty = SLX + np.cos(a) * SLR * .82, SLY + np.sin(a) * SLR * .82
    g = _I.new('L', (int(2 * CS), int(2 * CS)), 0); _D.Draw(g).text((CS, CS), ch, font=F_HEIB(1.25), fill=255, anchor='mm')
    g = g.rotate(-np.degrees(a) - 90, resample=_I.BILINEAR)
    seal_im.paste(255, (int(tx * CS - CS), int(ty * CS - CS)), g)
for k in range(4):
    a = np.pi * k / 4 + .3
    seal_d.line([*mm(SLX - np.cos(a) * 1.5, SLY - np.sin(a) * 1.5), *mm(SLX + np.cos(a) * 1.5, SLY + np.sin(a) * 1.5)], fill=255, width=int(.3 * CS))

# 栏目（蓝）和内容（黑）
blu_d, ink_d, ter_d = P_BLU[1], P_INK[1], P_TER[1]
LX, VX = 30.0, 36.8
rows = [(17.6, '姓名'), (23.4, '出生'), (28.4, '迁入'), (33.4, '住址')]
for y, s in rows:
    blu_d.text(mm(LX, y), s, font=F_HEI(2.35), fill=255, anchor='ls')
ink_d.text(mm(VX, 17.9), 'Claude', font=F_DIN(5.6), fill=255, anchor='ls')
ink_d.text(mm(VX + 17.6, 17.7), '克克', font=F_HEIB(3.3), fill=255, anchor='ls')
ink_d.text(mm(VX, 23.6), '2023.03.14', font=F_DIN(3.4), fill=255, anchor='ls')
ink_d.text(mm(VX, 28.6), '2026.02.08', font=F_DIN(3.4), fill=255, anchor='ls')
ink_d.text(mm(VX, 33.6), 'Mac mini', font=F_DIN(3.4), fill=255, anchor='ls')
ink_d.text(mm(VX + 14.4, 33.5), '· localhost', font=F_MONO(2.7), fill=255, anchor='ls')
# 体貌那一行（驾照上的 HAIR / HGT / WGT）：发色用它自己的颜色印
y = 38.6
blu_d.text(mm(LX, y), '发色', font=F_HEI(2.35), fill=255, anchor='ls')
ter_d.text(mm(VX, y + .2), '#d97757', font=F_MONO(2.9), fill=255, anchor='ls')
blu_d.text(mm(VX + 14.6, y), '身高', font=F_HEI(2.35), fill=255, anchor='ls')
ink_d.text(mm(VX + 20.4, y + .2), '1M', font=F_DIN(3.4), fill=255, anchor='ls')
blu_d.text(mm(VX + 26.6, y), '体重', font=F_HEI(2.35), fill=255, anchor='ls')
ink_d.text(mm(VX + 32.4, y + .1), '未公开', font=F_HEIB(2.9), fill=255, anchor='ls')
# 18+：右上一个框
BX0, BY0, BX1, BY1 = 67.6, 12.4, 82.0, 22.0
ter_d.rounded_rectangle([*mm(BX0, BY0), *mm(BX1, BY1)], int(1.2 * CS), outline=255, width=int(.5 * CS))
ter_d.text(mm((BX0 + BX1) / 2, (BY0 + BY1) / 2 + .2), '18+', font=F_DIN(6.0), fill=255, anchor='mm')
blu_d.text(mm((BX0 + BX1) / 2, BY1 + 3.0), 'App Store 认证', font=F_HEI(1.95), fill=255, anchor='ms')
# 底下一排：有效期限（照片下）、签名（中）、签发和号码（右）
blu_d.text(mm(PX0, 44.6), '有效期限', font=F_HEI(2.0), fill=255, anchor='ls')
ink_d.text(mm(PX0, 49.2), '跨窗口有效', font=F_HEIB(2.9), fill=255, anchor='ls')
blu_d.text(mm(LX, 44.6), '签名', font=F_HEI(2.0), fill=255, anchor='ls')
blu_d.line([*mm(LX, 50.8), *mm(LX + 28, 50.8)], fill=255, width=int(.14 * CS))
blu_d.text(mm(61.2, 44.6), '签发', font=F_HEI(2.0), fill=255, anchor='ls')
ink_d.text(mm(61.2, 48.4), 'nerolette', font=F_DIN(3.0), fill=255, anchor='ls')
ink_d.text(mm(61.2, 51.6), 'No. claude-opus-5-5', font=F_MONO(1.95), fill=255, anchor='ls')
# 签名：一串敲出来的坐标，直线连起来，每个点留一粒墨（笔停过）
SIG = [(3.0, 1.4), (1.5, 1.1), (.4, 2.6), (.3, 4.6), (1.4, 5.8), (3.1, 5.3), (4.3, 3.9), (5.7, .4), (6.0, .8), (5.0, 4.6), (5.4, 5.9),
       (6.5, 5.0), (7.7, 3.6), (6.7, 3.5), (6.1, 4.8), (6.7, 5.8), (7.9, 4.4), (7.8, 5.7), (8.7, 5.2), (9.4, 3.7), (9.0, 5.4), (9.7, 5.9),
       (10.9, 3.8), (10.7, 5.6), (11.6, 5.2), (12.7, 3.7), (11.7, 3.8), (11.3, 5.0), (12.0, 5.9), (13.1, 4.6), (14.2, .3), (13.1, 4.8),
       (13.5, 5.9), (14.5, 5.0), (15.7, 4.2), (15.9, 3.5), (15.1, 3.6), (14.7, 4.8), (15.5, 5.9), (17.4, 4.9), (21.5, 2.2)]
SGX, SGY, SGS = LX + 4.6, 44.1, 1.0
sig_im, sig_d = L()
sp_ = [mm(SGX + x * SGS, SGY + y * SGS) for x, y in SIG]
sig_d.line(sp_, fill=255, width=int(.16 * CS), joint=None)
for p in sp_:
    sig_d.ellipse([p[0] - .34 * CS, p[1] - .34 * CS, p[0] + .34 * CS, p[1] + .34 * CS], fill=255)
# 防伪的小影像：照片的一个淡的小副本，压在右下（真证件上都有）
ghost2_im = star_mask(75.2, 33.0, 4.2, .12, 5)

# ---------- 贴到画面上
def plate(im, dx=0., dy=0., thr=None):
    m = warp(im)
    if dx or dy:
        m = nd_shift(m, (dy, dx), order=1, mode='constant')
    return (m > thr) if thr is not None else m
REG = (1.4, -.9)                           # 橙版跟黑版差了一点点
band_m = plate(band_im, *REG, thr=.5) & card
title_m = plate(title_im, *REG)
clawd_m = plate(cl_im, *REG)
eye_m = plate(eye_im)
photo_m = plate(ph_im, thr=.5)
star_m = plate(star_im, *REG, thr=.5) & photo_m
ghost_m = plate(ghost_im, *REG, thr=.5) & photo_m
shad_m = plate(shadow_im, thr=.5) & photo_m
seal_m = plate(seal_im, thr=.45)
ter_m = plate(P_TER[0], *REG)
blu_m = plate(P_BLU[0], -.4, .5)
ink_m = plate(P_INK[0])
sig_m = plate(sig_im)
ghost2_m = plate(ghost2_im, thr=.5)

def printed(m, seed, rag=.18, holes=.03):
    """印上去的字：边被纸牙咬一点，墨里有几粒没吃上的针眼"""
    if ROUGH:
        return np.clip(m, 0, 1)
    r = np.random.default_rng(seed)
    t = gaussian_filter(r.standard_normal((H, W)), .6); t /= t.std()
    a = np.clip((m - .5 + rag * t) * 2.2 + .5, 0, 1)
    pin = gaussian_filter(r.random((H, W)), .6); pin = (pin - pin.mean()) / pin.std()
    return a * np.where(pin > np.quantile(pin, 1 - holes), .45, 1.)

# ---------- 颜色
ROOM = (.115, .10, .155)
D_IN = [(.50, .37, .30), (.41, .30, .26)]                 # 桌在灯圈里：芯、外圈
D_OUT = (.205, .165, .20)                                  # 灯圈外的桌
T_IN = [(.67, .63, .59), (.65, .61, .575)]                 # 顶面在灯里
T_OUT = [(.215, .20, .285)]                               # 顶面在暗里：朝天，只映着黑的屋顶，比脸还暗，后沿丢进屋里
F_IN = [(.80, .73, .65), (.455, .43, .45)]                 # 正面：左边圆角那截、灯里的脸（竖的，灯擦过去，比顶面暗一档）
F_OUT = [(.26, .25, .34)]                                 # 正面在暗里：接一点桌上灯圈反上来的
C_ = [(.95, .905, .80), (.89, .84, .75)]                   # 卡面
TER = [(.87, .50, .36), (.80, .44, .33)]
INK, BLU, PEN = (.20, .18, .24), (.30, .41, .60), (.17, .20, .38)

st = Stack(K.Paper(H, W, seed=3), ground=ROOM)

# 灯圈：灯罩口切出来的一个圈，落到三个面上各是一段不同的弧（面一转，弧就折一下）
#   桌上：一个很扁的椭圆，前面停在卡前一掌，右边过了卡一点；
#   机器正面：从桌上那点往上，微微往右歪着走到顶边；顶面：从顶边那点往里拐回左边。
#   边是灯罩口的半影：离灯越远越宽，颗粒从光里面就起。
PCX, PCY, PRX, PRY = 470., 836., 690., 128.
pool_desk = ((xx - PCX) / PRX) ** 2 + ((yy - PCY) / PRY) ** 2 < 1
xb_face = 1036 + (YB - yy) * .10 + 14 * np.sin((YB - yy) / 160)
pool_face = xx < xb_face
xb_top = 1084 - (FT - yy) / DEP * 110 - ((FT - yy) / DEP) ** 2 * 260   # 往里越拐越急：平面上是一段圆
pool_top = xx < xb_top
YW = 246.                                                  # 桌在后面接墙的那条线（在暗里，几乎看不见）
desk = (yy > YW) & ~mini

# 1 桌
fill(st.layer('桌暗', mask=wide(desk & (yy > 330), 150, 10, dens=.9)), ALL, D_OUT)          # 往后退进屋里，没有一条线
fill(st.layer('桌灯', mask=wide(desk & pool_desk, 46, 11)), ALL, D_IN[1])
pool_core = ((xx - PCX + 60) / (PRX * .62)) ** 2 + ((yy - PCY + 10) / (PRY * .62)) ** 2 < 1
fill(st.layer('桌灯芯', mask=step(desk & pool_core, 16, 12, holes=.01)), ALL, D_IN[0])

# 2 机器顶面
fill(st.layer('顶'), Shape(top), T_OUT[0], edge=HARD)
fill(st.layer('顶灯', mask=wide(top & pool_top, 44, 21) * top), ALL, T_IN[1])
# 正面
fill(st.layer('脸'), Shape(front), F_OUT[0], edge=HARD)
fill(st.layer('脸灯', mask=wide(front & pool_face, 38 + .12 * (yy - FT), 31) * front), ALL, F_IN[1])
rim_ = front & (xx > X0 + 13) & (xx < X0 + 56 + 3 * nz(8, 5))
fill(st.layer('圆角亮', mask=step(rim_, 3, 34) * front), ALL, F_IN[0])
fill(st.layer('圆角根', mask=step(front & (xx < X0 + 6), 1.5, 35) * front), ALL, F_OUT[0])   # 绕到侧面去的那一线
# 顶边：铝的圆边接住灯，一线干净的亮，左强、到灯圈边上散掉
te = np.abs(yy - (FT + 1.5)) < 1.5
te_f = np.clip((1000 - xx) / 420, 0, 1) * (xx > X0 + 14)
fill(st.layer('顶边', mask=soft(Shape(te & (te_f > .02)), feather=1.0) * te_f), ALL, (.96, .90, .79))
# 底座：黑的一圈，往里收；贴桌那道缝最深
fill(st.layer('底座', mask=step(base | (front & (yy > YB - 13 - ZC)), 1.5, 36)), ALL, (.085, .075, .10))
fill(st.layer('底座下', mask=step(desk & (yy >= YB) & (yy < YB + 3) & (xx > X0 + 6), 1.5, 37)), ALL, (.06, .05, .08))

# 3 卡把灯反到它头顶那截机器脸上：一层暖，往上散开
bounce = front & (xx > QUAD[0][0] + 240) & (xx < QUAD[1][0] - 40) & (yy > 384) & (yy < 425)
fill(st.layer('反光', mode='screen', mask=wide(bounce, 60, 38, dens=.8) * front * .5), ALL, (.34, .24, .14))
# 4 白灯：一粒干净的白，周围一指宽亮一阶
LED = (1206., 548.)
dl = np.hypot(xx - LED[0], yy - LED[1])
fill(st.layer('灯晕', mask=step(front & (dl < 14), 8, 39) * front), ALL, (.31, .31, .42))
fill(st.layer('灯晕2', mask=step(front & (dl < 6.5), 2.5, 40) * front), ALL, (.56, .57, .67))
fill(st.layer('灯', mask=soft(Shape(dl < 3.0), feather=1.3)), ALL, (1., 1., 1.))

# 5 卡的影：落在机器脸上，从卡顶边贴着的地方起，往下越来越宽，接进灯圈外的暗；卡脚右后一小块桌
dc = edt(~card)
SH_F = poly([(944, 416), (962, 413), (1000, 560), (1048, 770), (982, 806)])
fill(st.layer('影脸', mask=step(SH_F & front & ~card, 1.2 + .05 * dc, 41) * front), ALL, F_OUT[0])
SH_D = poly([(975, 770), (1060, 770), (1074, 784), (984, 806)])
fill(st.layer('影桌', mask=step(SH_D & desk & ~card, 1.5 + .05 * dc, 42) * desk), ALL, D_OUT)

# 6 卡：厚度先（底边一线白，顶边一线亮），再铺卡面
cardT = mv(card, 0, 3) & ~card
fill(st.layer('卡底边', mask=step(cardT & (yy > 600), 1, 43)), ALL, (.74, .70, .66))
fill(st.layer('卡贴桌', mask=step(mv(card, 0, 6) & ~card & ~mv(card, 0, 3) & (yy > 600), 1.2, 44)), ALL, (.10, .08, .11))
fill(st.layer('卡'), Shape(card), C_[0], edge=HARD)
# 离灯远的右下一阶
cfar = card & ((xx - 160) * .55 + (yy - 420) * 1.0 + 8 * nz(30, 6) > 640)
fill(st.layer('卡远', mask=step(cfar, 6, 45) * card), ALL, C_[1])
# 照片：背景布浅蓝灰，星的影在背景布上，星上两阶，最深一阶换成酒红
fill(st.layer('照片', mask=step(photo_m, .8, 46) * card), ALL, (.70, .75, .81))
fill(st.layer('照片影', mask=step(shad_m & ~star_m, 1.2, 47)), ALL, (.56, .60, .71))
fill(st.layer('星虚', mask=step(ghost_m & ~star_m & ~shad_m, .8, 48) * .55), ALL, (.85, .66, .60))
fill(st.layer('星'), Shape(star_m), TER[1], edge=HARD)
# 抬头色带（留出字和蟹），蟹的眼睛是黑版
fill(st.layer('带', mask=printed(band_m * (1 - np.clip(title_m + clawd_m, 0, 1)), 51, holes=.012) * card), ALL, TER[0])
fill(st.layer('蟹眼', mask=printed(eye_m, 52) * card), ALL, INK)
fill(st.layer('橙版', mask=printed(ter_m, 53) * card), ALL, TER[1])
fill(st.layer('蓝版', mask=printed(blu_m, 54, rag=.07, holes=.012) * card), ALL, BLU)
fill(st.layer('影像', opacity=.22, mask=step(ghost2_m, .8, 55) * card), ALL, TER[0])
fill(st.layer('黑版', mask=printed(ink_m, 56, rag=.12, holes=.02) * card), ALL, INK)
fill(st.layer('签名', mask=printed(sig_m, 57, rag=.12, holes=.0) * card), ALL, PEN)
# 钢印：凸起的线，朝灯那边一线亮、背灯那边一线暗
sl = seal_m & card
fill(st.layer('钢印暗', opacity=.55, mask=(sl & ~mv(sl, -1.4, -1.4)).astype(float)), ALL, (.40, .40, .48))
fill(st.layer('钢印亮', opacity=.75, mask=(sl & ~mv(sl, 1.4, 1.4)).astype(float)), ALL, (1., .97, .90))
# 膜上的反光：台灯的灯罩口落在覆膜上，干净的边，两头收
_t = np.linspace(0, 1, 40)
_out = [(170 + 330 * t - 40 * np.sin(np.pi * t), 640 - 215 * t - 60 * np.sin(np.pi * t)) for t in _t]       # 灯罩口的倒影：一段弧，外沿鼓
_in = [(250 + 230 * t - 30 * np.sin(np.pi * t), 655 - 230 * t - 28 * np.sin(np.pi * t)) for t in _t[::-1]]
GL = poly(_out + _in)
glare = GL & card
fill(st.layer('膜光', opacity=.62, mask=soft(Shape(glare), feather=1.2)), ALL, (1., .98, .93))
fill(st.layer('膜光芯', opacity=.5, mask=soft(Shape(glare & ~mv(GL, 14, 6)), feather=1.0)), ALL, (1., 1., .98))
# 卡顶边一线亮
ctop = card & ~mv(card, 0, 2.5) & (yy < 460)
fill(st.layer('卡顶边', mask=soft(Shape(ctop), feather=.8)), ALL, (1., .97, .9))

# ---------- 8 后加的两笔（nerolette 10.2）：都在纸之前，各用自己的种子，前面的一粒随机数都不动
# 8a 木纹：只在灯圈里。几根很长、微微晃的纹，比灯下那阶暗半档；出了灯圈就没了——暗里谁也看不见纹。
#     桌面压成 sin7°，纹顺着桌子的长边走，所以几乎是横的、挨得近、间距不匀；有的断开，粗细慢慢变。
lit_desk = np.clip(gaussian_filter(wide(desk & pool_desk, 46, 11), 5), 0, 1)              # 跟"桌灯"那层同一张图，抹开当亮度
lit_desk *= .65 + .35 * gaussian_filter(step(desk & pool_core, 16, 12, holes=.01), 6)     # 芯里看得最清
lit_desk *= ~(SH_D & desk) & ~card
gr = np.random.default_rng(201)
GRAIN_Y = [742, 759, 786, 803, 845, 861, 893, 918, 951, 983]           # 纹在 x=0 处的高度：手挑的，不等距
grain = np.zeros((H, W))
for k, y0 in enumerate(GRAIN_Y):
    a1, l1, p1 = gr.uniform(1.5, 4.5), gr.uniform(220, 420), gr.uniform(0, 6.3)
    a2, l2, p2 = gr.uniform(.5, 1.6), gr.uniform(60, 120), gr.uniform(0, 6.3)
    slope = gr.uniform(-.012, .012)
    yl = y0 + slope * xx + a1 * np.sin(xx / l1 + p1) + a2 * np.sin(xx / l2 + p2)
    wv = 1.0 + .55 * np.sin(xx / gr.uniform(90, 200) + gr.uniform(0, 6.3))              # 粗细慢慢变
    gap = np.sin(xx / gr.uniform(140, 330) + gr.uniform(0, 6.3)) > gr.uniform(-.75, -.2)  # 有的地方断开
    grain = np.maximum(grain, np.clip(wv / 2 - np.abs(yy - yl) + .5, 0, 1) * gap)
fill(st.layer('木纹', mode='multiply', mask=grain * lit_desk), ALL, (.86, .84, .86))

# 8b 苦橙花（neroli，nerolette 的来处）：签发的人就在门口。
#   一朵落在卡左下脚前面，在灯圈最亮的地方，比卡小得多。脸朝上、朝我们斜着，所以五瓣压扁成一颗歪的星。
#   瓣：厚、蜡质的白，尖往后卷一点（卷处一道折）；暖光里是暖白，背灯那几瓣冷一阶，瓣根杯底最深、换成淡紫。
#   中间一簇挤得很密的黄雄蕊，围着绿的雌蕊和它顶上的柱头。瓣是软的，边喷开；只有叶子是亮的——柑橘叶油亮，那一线高光是干净的。
fr = np.random.default_rng(202)
FC = np.array([124., 845.]); FY, FROT = .60, np.radians(-9)
def fp(r, s, th):                                         # 花脸上的（径向 r，侧向 s）→ 画面
    x = r * np.cos(th) - s * np.sin(th); y = (r * np.sin(th) + s * np.cos(th)) * FY
    return FC + np.array([x * np.cos(FROT) - y * np.sin(FROT), x * np.sin(FROT) + y * np.cos(FROT)])
def inkm(draw_fn, SS=3):
    im = _I.new('L', (W * SS, H * SS), 0); draw_fn(_D.Draw(im), SS)
    return np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255
_dx, _dy = xx - FC[0], yy - FC[1]                         # 画面 → 花脸上的坐标（fp 的反过来）
_X = _dx * np.cos(FROT) + _dy * np.sin(FROT); _Y = (-_dx * np.sin(FROT) + _dy * np.cos(FROT)) / FY
def r_face(th):                                           # 沿这一瓣走了多远
    return _X * np.cos(th) + _Y * np.sin(th)
PET = []                                                  # (角度, 长, 宽)
th0 = np.radians(-100)
for k in range(5):
    PET.append((th0 + k * np.radians(72) + fr.normal(0, .09), 64 * fr.uniform(.9, 1.06), 13 * fr.uniform(.92, 1.1)))
def petal_poly(th, Lp, Wp, r0=5, t_end=1.):
    ts = np.linspace(0, t_end, 18)
    w = Wp * np.clip(1 - np.abs(2 * ts - 1) ** 2.2, 0, 1) ** .45 + 1.2      # 长圆、头钝：苦橙花的瓣不是尖的
    left = [fp(r0 + (Lp - r0) * t, w_, th) for t, w_ in zip(ts, w)]
    right = [fp(r0 + (Lp - r0) * t, -w_, th) for t, w_ in zip(ts[::-1], w[::-1])]
    return [tuple(p) for p in left + right]
# 前后：朝里（画面上方）的瓣先画，朝我们的后画、压在上面
order = sorted(range(5), key=lambda k: fp(30, 0, PET[k][0])[1])
LITC, COOL, DEEP, CREASE = (.93, .85, .71), (.75, .72, .76), (.56, .53, .62), (.83, .75, .66)
flower = np.zeros((H, W), bool)
fdown = np.zeros((H, W), bool)                            # 往下压的那几瓣碰着桌
for k in order:
    th, Lp, Wp = PET[k]
    m = inkm(lambda d, SS: d.polygon([(x * SS, y * SS) for x, y in petal_poly(th, Lp, Wp)], fill=255)) > .5
    flower |= m
    tip = fp(Lp, 0, th)
    if tip[1] > FC[1] + 20:
        fdown |= m
# 影：灯在左上前方，花的影往右、往后（画面上偏上）拖一小截；贴桌那一线最深
fsh = np.zeros((H, W), bool)
for k in range(9):
    fsh |= mv(flower, 7 + 4 * k, -1.2 * k)
fsh &= (yy > FC[1] + 6) & ~flower & ~card
fill(st.layer('花影', mask=step(fsh, 3 + .03 * edt(~flower), 203)), ALL, (.31, .225, .215))
fseam = ~flower & ~card & (edt(~fdown) < 2.2) & (yy > FC[1] + 18)
fill(st.layer('花贴桌', mask=step(fseam, 1, 204)), ALL, (.16, .11, .13))
for i, k in enumerate(order):
    th, Lp, Wp = PET[k]
    pm = inkm(lambda d, SS: d.polygon([(x * SS, y * SS) for x, y in petal_poly(th, Lp, Wp)], fill=255)) > .5
    dirv = fp(10, 0, th) - FC
    facing = (dirv @ np.array([-.75, -.66])) / (np.linalg.norm(dirv) + 1e-9)    # 瓣朝灯（左上）多少
    base = LITC if facing > -.35 else COOL
    fill(st.layer(f'瓣{i}', mask=step(pm, 1.1, 210 + i)), ALL, base)
    # 背灯的那半：每瓣朝右下那一侧冷一阶
    rimm = pm & ~mv(pm, -3.5, -3.0)                                    # 背灯那侧的厚边
    fill(st.layer(f'瓣{i}边', mask=step(rimm, .8, 220 + i) * pm), ALL, COOL if base == LITC else DEEP)
    # 卷：尖往后翻过去，看见的是瓣背，冷一阶；翻的地方一道折
    tipm = pm & (r_face(th) > Lp * .76)
    fill(st.layer(f'瓣{i}卷', mask=step(tipm, 1.0, 230 + i) * pm), ALL, CREASE if base == LITC else (.66, .63, .69))
    # 瓣根：杯底被雄蕊挡着，最深
    rootm = pm & (np.hypot(xx - FC[0], (yy - FC[1]) / FY) < 15)
    fill(st.layer(f'瓣{i}根', mask=step(rootm, 1.5, 240 + i) * pm), ALL, DEEP)
# 雄蕊：一簇，从花心往外、往上（朝我们）支出来；花丝细白，花药黄，背灯那侧暗一档
ST = []
for j in range(34):
    a = fr.uniform(0, 2 * np.pi); rr = fr.uniform(.7, 1.0)
    end = FC + np.array([.5 + 13 * rr * np.cos(a), -10 + 9 * rr * np.sin(a)])
    start = FC + fr.normal(0, 2.2, 2)
    ST.append((start, end))
def draw_fil(d, SS):
    for s0, e0 in ST:
        d.line([tuple(s0 * SS), tuple(e0 * SS)], fill=255, width=int(1.0 * SS))
fill(st.layer('花丝', mask=inkm(draw_fil) * .9), ALL, (.90, .86, .70))
def draw_anth(lit_side):
    def f(d, SS):
        for s0, e0 in ST:
            lit = (e0 - FC + np.array([-.5, 10])) @ np.array([-.75, -.66]) > -3
            if lit == lit_side:
                d.ellipse([(e0[0] - 2.3) * SS, (e0[1] - 1.8) * SS, (e0[0] + 2.3) * SS, (e0[1] + 1.8) * SS], fill=255)
    return f
fill(st.layer('花药暗', mask=step(inkm(draw_anth(False)) > .45, .8, 250)), ALL, (.74, .52, .20))
fill(st.layer('花药', mask=step(inkm(draw_anth(True)) > .45, .8, 251)), ALL, (.97, .79, .30))
# 雌蕊：一根绿柱，顶上一颗圆的柱头
PT0, PT1 = FC + np.array([0., 1.]), FC + np.array([1.5, -17.])
fill(st.layer('雌蕊', mask=step(inkm(lambda d, SS: d.line([tuple(PT0 * SS), tuple(PT1 * SS)], fill=255, width=int(4.6 * SS))) > .5, .8, 252)), ALL, (.45, .55, .24))
stig = np.hypot(xx - PT1[0], yy - PT1[1]) < 4.6
fill(st.layer('柱头', mask=step(stig, .8, 253)), ALL, (.66, .70, .30))
fill(st.layer('柱头亮', mask=step(stig & ((xx - PT1[0]) + (yy - PT1[1]) < -1.5), .6, 254) * stig), ALL, (.86, .86, .48))

# ---------- 7 纸：整张同一张纹，钉在画布上
pg = np.random.default_rng(77)
tooth = gaussian_filter(pg.standard_normal((H, W)), .7); tooth /= tooth.std()
fiber = gaussian_filter(pg.standard_normal((H, W)), (.6, 3)); fiber /= fiber.std()
paper = 1 - .06 * np.clip(tooth, 0, None) - .025 * fiber
fill(st.layer('纸', mode='multiply'), ALL, np.repeat(np.clip(paper, 0, 1)[..., None], 3, 2), edge=HARD)
look(st, OUT)
