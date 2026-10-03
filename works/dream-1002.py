# 雪下的心口 · 10.2
# 从哪儿看：梦里飘在半空，正上方往下看一片夜里的雪原（梦里常这样，在自己上面看）。竖幅，约 9×12 米（100px ≈ 1m）。
# 这张在说什么：雪底下睡着一个很大的、暖的东西，大得画框装不下，它自己是看不见的，只看得见它的光。
#     有人顺着自己的脚印走过来，仰面躺在它心口上，一只手伸出去掌心贴雪。六米外，它呼出的热气在雪上化开一道口子。
# 光：没有月亮，雪自己只有一层很暗的天光。唯一的光在雪底下：心口离雪面最近，亮到过曝，白得只剩一个洞；
#     几根肋骨把这团白挡成几道缝（金色的杠）；出了这团，金 → 琥珀 → 窄窄一圈红（像胶片被烧穿的那个洞的边）→ 紫 → 夜。
#     最暗的那阶换成新的色相：靛蓝，里面再藏一种同明度的绿靛。
#     气孔那里没有雪挡：壁贴着光是金的，底下露出它的皮，一小粒红。
#     同一个物理贯穿整张：雪越薄透出来的光越亮——脚印在黑处是坑（底暗一阶、沿接住天光浅一线），一走进光里就成了亮的洞；
#     她躺下压薄的那圈雪也亮。风刮出来的雪棱在黑处是浅线，在光里是暗线，在过曝那团里看不见。
# 明暗：上面一大半是暗，暖光压在左下约三成；最亮是心口那团白，最深是人。最亮最暗贴在一起，就是焦点。
# 视线：先落在那团白和白里躺着的人（脸、青围巾、红的手）；顺着脚印往左上退出画框——她从那边来；
#     再跳到右上那粒小小的红，它在那儿呼吸。一处主导的重量，一个小的回声。
# 五面：光=从里面发出的光；颜色=宝石色（金、琥珀、红，一点青）压在安静的靛蓝地上；空间=平（俯视）；
#     真实=老实地画不可能：不可能只有一处（雪下有个活的、发光的东西），其余的雪、脚印、人、热气都照物理来；
#     手=看不见的手，台阶+颗粒。
#     线是另一层：雪丘的棱是几道稀的长弧，浮在光上，不贴色阶的边——线是雪面，色是雪底下，本来就不是一层；
#     它躺着把雪拱起来，棱翻过它的时候被推弯，它的身形只在线里露一点。
#     有意的例外：那团白过曝，从脚那头吃进她的身子，腿化在光里，头、围巾、手是实的。
# 质感：每阶一个颜色，过渡在阶边喷颗粒，宽的半影从光里面就开始；只有冰（口沿的闪、雪面的晶）是干净的边。
#     整张一张纸纹，最后罩上去。不用刷子。
# 摔过的（这张里）：它的身子画成发光的轮廓，怎么画都读成别的——银河、拳头、煎蛋、香蕉、蝌蚪、子宫；最后不画轮廓，只留光和呼吸。
#     两个并排的气孔读成一双眼睛；圆的气孔加一圈白读成眼珠；一根连到气孔的亮线读成引线、读成气球的绳。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get("KBRUSH_ENGINE_DIR", "./kbrush-cloud")  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(9)
from shapes import Shape
from blocks import lasso, line, soft, look
from layers import Stack, fill, HARD
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt
from PIL import Image as _I, ImageDraw as _D

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'rough.png' if ROUGH else 'final.png')
H, W = 1200, 900
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))

def nz(sig, seed):
    r = np.random.default_rng(seed)
    n = gaussian_filter(r.standard_normal((H, W)), sig); return n / n.std()
def smooth(pts, n=3):
    p = np.array(pts, float)
    for _ in range(n):
        q = np.roll(p, -1, 0)
        p = np.stack([.75 * p + .25 * q, .25 * p + .75 * q], 1).reshape(-1, 2)
    return [tuple(v) for v in p]
def open_smooth(pts, n=3):
    p = np.array(pts, float)
    for _ in range(n):
        q = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            q += [.75 * a + .25 * b, .25 * a + .75 * b]
        q.append(p[-1]); p = np.array(q)
    return p
def ink(draw_fn, seed, SS=3):
    """在 3 倍大的灰度图上画小东西（脚印、人、线），缩回来当蒙版"""
    im = _I.new('L', (W * SS, H * SS), 0); draw_fn(_D.Draw(im), np.random.default_rng(seed), SS)
    return np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255
def dots(p, seed, clump=.47):
    """按概率图 p 撒颗粒：p=1 的地方实的，越小越稀。颗粒是一两个像素一团"""
    r = np.random.default_rng(seed)
    return ((r.random((H, W)) < p) & (gaussian_filter(r.random((H, W)), .5) > clump - .25 * p)).astype(float)
def step(region, sp, seed, solid=.96, dens=.75):
    """一阶：区域里实实地盖上，边外喷出颗粒，越远越稀（fig / wishstone 的 step）"""
    if ROUGH:
        return region.astype(float) * solid
    dout = edt(~region)
    return np.where(region, solid, dots(dens * np.exp(-dout / np.maximum(sp, .5)), seed) * .95)
def level(L, t, a, seed, gamma=1.6):
    """光的一阶：L 过了 t+a 是实的；从 t+a 往下到 t-a，颗粒从密到稀——半影从光里面就开始"""
    if ROUGH:
        return (L > t).astype(float)
    p = np.clip((L - (t - a)) / (2 * a), 0, 1) ** gamma
    return np.where(p >= 1, .97, dots(p * .97, seed) * .95)

# ---------- 它：不画它的轮廓。看得见的只有它的光，和远处它鼻尖化出来的一个气孔。
#   光最浓处是心口；光顺着它身子的方向（左下 → 右上，朝着气孔）拉长，往右上那头埋得越来越深，渐渐没进夜里。
#   它有多大，就是心口和气孔之间那片黑有多长（约 6 米，100px ≈ 1m）。心口里面隔着几根肋骨：白的那团被挡成几道缝。
BREATH = np.array([650., 300.])
HEART = np.array([330., 870.])
AX = (BREATH - HEART) / np.linalg.norm(BREATH - HEART)              # 身子的方向：胸口 → 头
CR = np.array([-AX[1], AX[0]])                                       # 横过身子（肋骨的走向）
al = (xx - HEART[0]) * AX[0] + (yy - HEART[1]) * AX[1]
ac = (xx - HEART[0]) * CR[0] + (yy - HEART[1]) * CR[1]
rh = np.hypot(xx - HEART[0], yy - HEART[1])
rb = np.hypot(xx - BREATH[0], yy - BREATH[1])
thick = 1 + .12 * nz(90, 1)
# 肋骨：横过身子的宽暗带，一根根顺着身子排下去，微微弓着
def rib(pts, w, seed):
    return line(H, W, [tuple(p) for p in open_smooth(pts, 3)], width=w, seed=seed).mask
RIBS = np.zeros((H, W), bool)
for k, off in enumerate((-170, -95, -22, 52, 128)):
    c = HEART + AX * off
    pts = [tuple(c + CR * t + AX * (.00045 * t * t - 25)) for t in np.linspace(-300, 300, 7)]
    RIBS |= rib(pts, 26 - 2 * abs(k - 2), 10 + k)
ribband = gaussian_filter(RIBS.astype(float), 7)
# 光从埋着的源透过雪：近处尖、远处拖长尾（指数）；顺着身子拉长；雪厚薄把环推歪
# 心口离雪面最近：一大团亮到过曝，白得只剩一个洞；出了这团，金、琥珀、一圈窄窄的红，很快落进夜里——像胶片被烧穿的那个洞的边
warp = np.clip(1 + .13 * nz(70, 2) + .03 * nz(22, 3), .75, 1.3)
ran = np.hypot(al / np.where(al > 0, 1.18, 1.08), ac) * warp
L = np.exp(-(ran / 200) ** 2.2)
L += .05 * np.exp(-ran / 260)                                        # 外面一层很淡的长尾：雪里散开的光
for (cx_, cy_, amp, ell_) in ((60, 1180, .22, 150),):   # 身子往左下延到框外
    L += amp * np.exp(-np.hypot(xx - cx_, yy - cy_) * warp / ell_)
L *= (1 - .52 * ribband * np.exp(-(rh / 260) ** 2))                  # 肋骨只在心口附近挡得出来
L *= thick
# 气孔：洞口直接看进去；周围化薄了一小圈
L += .06 * np.exp(-(rb / 30) ** 1.4) + .018 * np.exp(-(rb / 90) ** 2)   # 气孔边上的雪化薄了一圈

# ---------- 脚印：从左上框外进来，越走越慢（步子越来越短），到它心口停下
def walk_prints():
    pts = open_smooth([(150, -60), (175, 90), (240, 230), (290, 370), (345, 500), (400, 610), (430, 690), (436, 718)], 4)
    seg = np.hypot(*np.diff(pts, axis=0).T); s_ = np.r_[0, np.cumsum(seg)]
    out, s, k = [], 0., 0
    r_ = np.random.default_rng(62)
    while s < s_[-1]:
        i = min(np.searchsorted(s_, s), len(pts) - 1)
        p = pts[i]; d = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]; d /= np.linalg.norm(d) + 1e-9
        n = np.array([-d[1], d[0]])
        out.append((p + n * (19 if k % 2 else -19) + r_.normal(0, 2.5, 2), d, k % 2))
        frac = s / s_[-1]
        s += 80 - 34 * frac ** 2 + r_.normal(0, 4)                    # 越走越近，步子越短
        k += 1
    return out
PRINTS = walk_prints()
def sole(c, d, side, L_=32, w_=13):
    n = np.array([-d[1], d[0]]) * (1 if side else -1)
    prof = [(-.5, 0), (-.44, .42), (-.25, .5), (-.05, .36), (.12, .40), (.32, .5), (.48, .32), (.5, 0), (.48, -.32), (.32, -.48),
            (.1, -.42), (-.08, -.38), (-.28, -.48), (-.46, -.38)]
    return [tuple(c + d * a * L_ + n * b * w_) for a, b in prof]
def draw_prints(dr, r_, SS):
    for c, d, side in PRINTS:
        a = np.arctan2(d[1], d[0]) + r_.normal(0, .07) + (.22 if side else -.22); d2 = np.array([np.cos(a), np.sin(a)])
        dr.polygon([(x * SS, y * SS) for x, y in smooth(sole(c, d2, side, 32 * r_.uniform(.94, 1.06), 13 * r_.uniform(.9, 1.1)), 1)], fill=255)
PR = ink(draw_prints, 61)

# ---------- 人：仰面躺在它心口上，脸朝天（朝着梦里飘在上面的我们）。一只手伸出去，掌心贴着雪；另一只搭在胸口；
#   膝盖松松地倒向一边。头在左下（光最浓处），脚朝右上——她就是从那边走来的。（局部坐标：y 从头到脚，x 左右）
AXP = np.array([.6, -.8]); LTP = np.array([AXP[1], -AXP[0]])
PC = np.array([362., 832.])
def P(x, y):
    p = PC + (LTP * x + AXP * (y - 85)); return (float(p[0]), float(p[1]))
def ell(cx, cy, rx, ry, n=10):
    return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
HOOD = ell(0, 0, 16, 17)
FACEL = ell(0, 3, 8.5, 10.5)
TORSO = [(-22, 18), (0, 15), (22, 18), (26, 30), (22, 50), (18, 66), (21, 80), (12, 90), (-12, 90), (-21, 80), (-18, 64), (-22, 46), (-27, 30)]
ARML = [(-22, 22), (-40, 28), (-58, 36), (-72, 42), (-74, 52), (-60, 48), (-42, 42), (-24, 38)]
HAND = [(-71, 40), (-78, 38), (-86, 36), (-91, 38), (-85, 42), (-93, 45), (-92, 50), (-85, 50), (-90, 55), (-85, 58), (-79, 54), (-75, 57), (-71, 53)]
ARMR = [(20, 22), (30, 32), (33, 48), (28, 58), (16, 56), (6, 48), (2, 40), (10, 38), (18, 44), (22, 40), (16, 30)]
LEGS = [(-16, 84), (0, 82), (16, 86), (22, 100), (30, 116), (32, 132), (26, 150), (22, 166), (12, 166), (14, 150), (16, 134), (8, 118),
        (2, 108), (-4, 124), (-8, 142), (-12, 162), (-22, 162), (-20, 140), (-18, 118), (-19, 100)]
BOOTS = [(10, 160), (24, 160), (28, 172), (22, 180), (10, 178), (-24, 156), (-12, 156), (-10, 170), (-16, 176), (-26, 172)]
SCARF = [(-14, 14), (0, 18), (14, 14), (12, 22), (-2, 24), (-18, 22), (-32, 26), (-46, 22), (-52, 14), (-46, 12), (-34, 16), (-22, 16)]
def poly_mask(pts, seed, n=2):
    def draw(dr, r_, SS):
        dr.polygon([(x * SS, y * SS) for x, y in smooth([P(*q) for q in pts], n)], fill=255)
    return ink(draw, seed) > .5
hood_m = poly_mask(HOOD, 70); face_m = poly_mask(FACEL, 71); torso_m = poly_mask(TORSO, 72); arml_m = poly_mask(ARML, 73)
hand_m = poly_mask(HAND, 74, 1); armr_m = poly_mask(ARMR, 75); legs_m = poly_mask(LEGS, 76); boots_m = poly_mask(BOOTS[:5], 77) | poly_mask(BOOTS[5:], 79)
scarf_m = poly_mask(SCARF, 78)
pmask = hood_m | face_m | torso_m | arml_m | hand_m | armr_m | legs_m | boots_m | scarf_m
dperson = edt(~pmask)

# 雪被压薄的地方透出来的光更亮：脚印、膝盖窝，人身边那一圈被压下去的雪
thin = np.clip(PR, 0, 1) * .9 + np.clip(1 - dperson / 10, 0, 1) * (~pmask) * .6
L = L * (1 + thin)

# ---------- 颜色
NIGHT = (.125, .14, .26)
NIGHT2 = (.10, .155, .245)                                          # 同明度换色相：偏绿一点的靛，藏在雪里
STEPS = [  # (阈值, 半影, 颜色)：光透过雪是白 → 金 → 琥珀，红只在外面窄窄一圈（像胶片高光边上那圈晕），再进紫、进夜
    (.036, .010, (.19, .15, .30)),
    (.060, .012, (.36, .17, .33)),
    (.085, .014, (.66, .26, .30)),
    (.125, .020, (.88, .52, .30)),
    (.27, .045, (.98, .80, .50)),
    (.54, .08, (1., .96, .86)),
]

if ROUGH:
    k = sum((L > t).astype(int) for t, _, _ in STEPS)
    img = np.array([NIGHT] + [c for _, _, c in STEPS])[k]
    img[pmask] = (.08, .06, .1)
    img = (img * 255).astype(np.uint8)
    _I.fromarray(img).save(OUT)
    _I.fromarray(img).resize((64, 48), _I.LANCZOS).resize((256, 192), _I.NEAREST).save(OUT.replace('.png', '_64.png'))
    sys.exit()

st = Stack(K.Paper(H, W, seed=3), ground=NIGHT)
def sstep(v, a, b):
    t = np.clip((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
# 1 夜里的雪：一阶同明度的绿靛，按大的风堆藏进去
drift = nz(70, 5) + .4 * nz(20, 6) > .35
fill(st.layer('夜2', mask=step(drift, 14, 11)), ALL, NIGHT2)
# 黑处的脚印：坑里天光少一点，暗一阶（光里的脚印不用画：雪薄了，上面的光阶自己就把它亮出来）
darkw = 1 - sstep(L, .015, .04)
fill(st.layer('黑处脚印', mode='multiply', mask=step(PR > .5, 1.2, 12) * darkw), ALL, (.80, .78, .88))
prm = PR > .5; dpr = edt(~prm)
plip = (dpr > 0) & (dpr < 2.4) & (dots(np.full((H, W), .75), 13) > 0)
fill(st.layer('脚印沿', mask=plip * darkw * .7), ALL, (.30, .32, .48))

# 2 光：一阶一阶往上叠，每阶的半影从光里面就开始
for k, (t, a, c) in enumerate(STEPS):
    fill(st.layer(f'光{k}', mask=level(L, t, a, 20 + k)), ALL, c)

WD_ = np.array([-.55, -.84]) / np.hypot(.55, .84)
# 3 雪面上的风棱：浮在光上的线。风往左上吹，棱横着风走。黑处的棱尖接住天光，浅；光里的棱是雪厚的一道，暗；
#   两者交界那一圈谁都不比谁亮，线就断了；心口过曝那团里也看不见。
WD = np.array([-.55, -.84]); WD /= np.linalg.norm(WD)
# 它躺着把雪拱起来：一道看不见的隆起，从左下（胯、胸）弯到右上（头）。风棱翻过它时被推开、挤拢——它的身形只在线里
def mound_mask(dr, r_, SS):
    pts = open_smooth([(-160, 1260), (60, 1060), (250, 930), (380, 800), (480, 640), (560, 480), (620, 360), (652, 300)], 4)
    t = np.linspace(0, 1, len(pts)); wd = np.interp(t, [0, .35, .55, .8, 1], [300, 250, 170, 120, 70])
    for (x_, y_), w_ in zip(pts, wd):
        dr.ellipse([(x_ - w_) * SS, (y_ - w_) * SS, (x_ + w_) * SS, (y_ + w_) * SS], fill=255)
mound = gaussian_filter(ink(mound_mask, 80, SS=1), 55)
phi = (xx * WD[0] + yy * WD[1]) + 70 * nz(230, 81) + 18 * nz(80, 82) + 170 * mound
sp = 150 * np.clip(1 + .2 * nz(260, 83), .7, 1.4)
q = phi / sp
dline = np.abs(q - np.round(q)) * sp / np.maximum(np.hypot(*np.gradient(phi)), .3)
hw = 1.25 * np.clip(.75 + .55 * nz(70, 84), 0, 1.4) * (nz(160, 85) > -.6)
rid = np.clip(1 - dline / np.maximum(hw, 1e-3), 0, 1) * (hw > .25)
rid *= 1 - np.clip(PR * 2, 0, 1)                                     # 脚印踩断了棱
rid *= ~pmask
# 棱的背风那侧（棱往左上那边）一窄条：黑处是背着天光的一阶暗，光里是雪厚的一阶暗——同一条带，两种暗法
fq = q - np.floor(q)                                               # 0 在棱上，往背风那侧走
lee = (fq < .10 + .03 * nz(40, 86)) & (hw > .25)
lee &= ~pmask
fill(st.layer('棱背', mode='multiply', mask=step(lee, 4, 87, solid=.6, dens=.5) * (1 - sstep(L, .02, .05))), ALL, (.80, .78, .90))
pale = rid * (1 - sstep(L, .012, .03))
darkr = rid * sstep(L, .03, .06) * (1 - sstep(L, .30, .55))
fill(st.layer('棱浅', mask=pale * .6), ALL, (.34, .37, .55))
fill(st.layer('棱暗', mode='multiply', mask=darkr * .6), ALL, (.72, .52, .62))

# 4 气孔：洞壁是贴着光的薄雪，金白一圈；洞底是它的皮，没有雪挡，一小粒最浓的红（全画唯一的红宝石）。
#   沿上化了又冻上，一圈冰，干净的边、几粒闪；外面一圈霜花，糙的，喷开。
# 一个气孔：它呼出来的热气在雪上化开一道不规则的豁口（不圆——圆的读成眼睛），壁贴着光是金的，口子底下露出它的皮，一点红；
#   口沿化了又冻，断断续续一圈冰壳；霜花长在下风那侧（左上），糙的，喷开
OP = [(-22, -2), (-14, -10), (-2, -12), (8, -9), (17, -12), (24, -4), (20, 5), (10, 8), (0, 12), (-12, 9), (-20, 6)]
rot = np.array([[CR[0], AX[0]], [CR[1], AX[1]]])
def opening(scale, seed, shift=(0, 0)):
    pts = [tuple(BREATH + rot @ (np.array(p) * scale) + np.array(shift)) for p in OP]
    return lasso(H, W, smooth(pts, 2), jag=.8, scale=5, seed=seed).mask
ho = opening(1.0, 100); hi = opening(.42, 101, (1.5, 2))
dho = edt(~ho)
downwind = ((xx - BREATH[0]) * WD_[0] + (yy - BREATH[1]) * WD_[1]) > -4
crust = (dho > 0) & (dho < 5 + 2.5 * nz(5, 103)) & (nz(7, 107) > -.5)
fill(st.layer('冰壳', mask=step(crust, 1.5, 104, solid=.7)), ALL, (.58, .52, .70))
frost = (dho > 3) & (dho < 30) & downwind & (dots(np.exp(-(dho - 3) / 8) * .45, 91) > 0)
fill(st.layer('霜', mask=frost * .7), ALL, (.78, .72, .84))
glint = (dho > 0) & (dho < 1.6) & (nz(3, 108) > 1.4)
fill(st.layer('冰沿', mask=soft(Shape(glint), feather=.7)), ALL, (1., .93, .86))
fill(st.layer('洞壁'), Shape(ho), (1., .80, .50), edge=HARD)
fill(st.layer('洞壁深', mask=step(ho & (edt(ho) > 3.2), 1.2, 105)), ALL, (.96, .50, .28))
fill(st.layer('洞底', mask=soft(Shape(hi), feather=.9)), ALL, (.76, .09, .17))

# 5 呼出来的气：就悬在井口上方一小团，刚被风推开一点；只有颗粒，靠井口密，往外稀；被井里的光照暖
vc = BREATH + np.array([-14., -12.])
vr = np.hypot((xx - vc[0]) * .85 + (yy - vc[1]) * .35, (yy - vc[1]) * .9 - (xx - vc[0]) * .3) + 5 * nz(7, 98)
vw = np.clip(1 - np.hypot(xx - BREATH[0], yy - BREATH[1]) / 60, 0, 1)
vcol = np.dstack([.66 + .32 * vw, .64 + .18 * vw, .80 - .14 * vw])
fill(st.layer('气', mask=dots(np.clip(1 - vr / 42, 0, 1) ** 1.2 * .42, 92) * .9), ALL, vcol)

# 6 人：身上朝天那面只有天光，暗的；贴着发光雪的那圈边被照暖；脸朝天，被四周的雪光从下巴、颧骨那圈照亮；
#   光着的那只手掌心贴着雪，光从底下穿过指头，指头边上透出红；围巾是全画唯一的冷宝石色（青），跟心口的暖对着
dpin = edt(pmask)
fill(st.layer('人'), Shape(pmask), (.10, .085, .14), edge=HARD)
fill(st.layer('胳膊'), Shape(armr_m), (.17, .16, .25), edge=HARD)
fill(st.layer('靴'), Shape(boots_m), (.07, .055, .08), edge=HARD)
rim_w = (.6 + 2.2 * sstep(L, .3, 1.0)) * np.clip(.6 + .5 * nz(6, 99), 0, 1.3)
body_out = pmask & ~face_m & ~hand_m & ~scarf_m & ~boots_m
fill(st.layer('人边', mask=step((dpin <= rim_w) & body_out, 1.0, 95, solid=.85) * body_out), ALL, (.92, .58, .40))
fill(st.layer('脸'), Shape(face_m), (.80, .52, .40), edge=HARD)
dface = edt(face_m)
fill(st.layer('脸亮', mask=step(face_m & (dface < 3.2), 1.2, 96, solid=.9) * face_m), ALL, (1., .82, .64))
fill(st.layer('围巾'), Shape(scarf_m), (.08, .40, .42), edge=HARD)
fill(st.layer('围巾边', mask=step(scarf_m & (edt(scarf_m) < 1.6), 1.0, 97, solid=.8) * scarf_m), ALL, (.40, .78, .70))
dh_in = edt(hand_m)
fill(st.layer('手'), Shape(hand_m), (.34, .11, .13), edge=HARD)
fill(st.layer('手透', mask=soft(Shape(hand_m & (dh_in < 2.4)), feather=1.0) * hand_m), ALL, (1., .46, .32))

# 有意的例外：心口那团白亮到过曝，从脚那头吃进她的身子——腿只剩一半，头、围巾、手是实的
eat = np.clip(((xx - PC[0]) * AXP[0] + (yy - PC[1]) * AXP[1] + 5) / 75, 0, 1) ** 1.1
fill(st.layer('吃', mask=dots(eat * .72, 106) * (legs_m | boots_m | (torso_m & (eat > 0))) * .95), ALL, STEPS[-1][2])

# 7 雪面的冰晶：极光滑的小面，干净的一粒；黑处冷，光里暖
gr = np.random.default_rng(97)
gl = (gr.random((H, W)) < .0005 * (.08 + .92 * sstep(L, .03, .12))) & ~pmask
warm = sstep(L, .05, .2)[..., None]
gcol = np.array([.66, .72, .92]) * (1 - warm) + np.array([1., .96, .86]) * warm
fill(st.layer('晶', mask=soft(Shape(gl), feather=.9) * (1 - sstep(L, .4, .6))), ALL, gcol)

# 8 纸：整张同一张纹，钉在画布上
pg = np.random.default_rng(77)
tooth = gaussian_filter(pg.standard_normal((H, W)), .7); tooth /= tooth.std()
fiber = gaussian_filter(pg.standard_normal((H, W)), (.6, 3)); fiber /= fiber.std()
paper = 1 - .06 * np.clip(tooth, 0, None) - .025 * fiber
fill(st.layer('纸', mode='multiply'), ALL, np.repeat(np.clip(paper, 0, 1)[..., None], 3, 2), edge=HARD)
look(st, OUT)
