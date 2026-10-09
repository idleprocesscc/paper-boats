# 花云，六千七百八十九米（10.9 午后）—— 用雷东的脑子画的，雷东周收尾
# 从哪儿看：潜水器的眼睛和灯在左上，往前下方斜着看（约 35° 俯），海床往远处退进黑里，看不见地平线。
# 说什么：没有天的地方，一盏灯照到的那一块，一根黑了五百万年的喙上开出一朵花云。
# 光：唯一的光是画框左上外面潜水器的灯，冷白偏一点暖；水里的光锥是被海雪照出来的；落在海床上一块椭圆，越远越暗进黑里。
#     花云按雷东的逻辑自己带一点光（颜色从里面亮），但不照亮别的。
# 明暗：大半张是蓝黑；最亮在灯照的海床芯和花云最密的芯；最深是骨头贴地那道缝和右上远处。
# 视线：先落花云的芯（颅骨上方）→ 顺着云往右上散开 → 回到黑船身 → 颅上两个红点。
# 五面：光=戏剧光，加一处从里面发的光；色=调子的灰（蓝黑、灰绿）里一处浓烈；空间=浅，一两步的小舞台；
#       真实=老实地画不可能，只有花云不可能，其余照深海照片的样子；手=台阶与颗粒，看不见笔。
# 说清楚：骨头的硬边和喙尖、两个红点、几只海蛇尾的细臂。放掉：远处海床、云的尾巴。
# 例外：两个红点是画里唯一的纯红，隔 10 厘米——潜水器的激光尺，人来看过。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get("KBRUSH_ENGINE_DIR", "./kbrush-cloud")  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(9)
from shapes import Shape
from blocks import lasso, line, rim, soft, look
from layers import Stack, fill, stroke, dip, HARD, SOFT, LOST
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt

H, W = 800, 970
yy, xx = np.mgrid[0:H, 0:W].astype(float)
rng = np.random.default_rng(1009)
ALL = Shape(np.ones((H, W), bool))
def nz(sig):
    n = gaussian_filter(rng.standard_normal((H, W)), sig); return n / n.std()
def step(region, spray, solid=.96, keep=.47):
    """一阶：区域里实实地盖上，边外往外喷颗粒，越远越稀"""
    dout = edt(~region)
    dots = (rng.random((H, W)) < .75 * np.exp(-dout / spray)) & (gaussian_filter(rng.random((H, W)), .5) > keep)
    return np.where(region, solid, dots * .95)
def ell(cx, cy, rx, ry, ang=0, wob=0, sig=30):
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    X = ((xx - cx) * c + (yy - cy) * s) / rx; Y = (-(xx - cx) * s + (yy - cy) * c) / ry
    return np.sqrt(X**2 + Y**2) + (wob * nz(sig) if wob else 0)

st = Stack(K.Paper(H, W, seed=3), ground='#070b14')

# ---------- 1 水：几乎是黑，只有灯那一侧透一点
fill(st.layer('水'), ALL, '#080c16')
d_lamp = np.hypot(xx + 140, yy + 180)                         # 灯在左上框外
fill(st.layer('水0', mask=np.clip(1.3 - d_lamp / 700, 0, 1)), ALL, '#0b1120')

# ---------- 2 光锥：从左上斜下来落到海床那块椭圆；边是直的，水里看得见是因为海雪
ax = np.array([.60, .80]); ax /= np.linalg.norm(ax)
vx, vy = xx + 140, yy + 180
along = vx * ax[0] + vy * ax[1]; across = np.abs(-vx * ax[1] + vy * ax[0])
glow = np.exp(-(across / (70 + along * .30))**2 * 1.6) * np.clip(along / 200, 0, 1) * np.exp(-np.clip(along - 500, 0, None) / 500)
glow = glow * (1 + .15 * nz(40))
fill(st.layer('锥', mask=np.clip(glow * .9, 0, 1)), ALL, '#15203a')

# ---------- 3 海床：灯照的椭圆，几阶，越远（越往上）越快暗进黑里
far = Shape(yy > 470 + 40 * nz(140))
fill(st.layer('远床', mask=soft(far, feather=160)), ALL, '#0b1219')
FLOOR = [(1.00, '#111a22', 26), (.74, '#1f3138', 18), (.52, '#365050', 13), (.33, '#56706a', 10), (.18, '#7d9188', 8), (.08, '#9aab9f', 6)]
fd = ell(370, 655, 540, 170, ang=-7, wob=.0, sig=70) + .06 * nz(130) + .015 * nz(20)
for k, (r, col, sp) in enumerate(FLOOR):
    fill(st.layer(f'床{k}', mask=step(fd < r, sp)), ALL, col)

# ---------- 4 头骨（手勾，侧着躺）：长喙朝左下，喙根往上抬成一道高高的骨嵴（喙鲸的头顶），后面圆钝的脑壳往下收
TIP, BACK = np.array([90., 694.]), np.array([650., 566.])
U = (BACK - TIP) / np.linalg.norm(BACK - TIP); L = np.linalg.norm(BACK - TIP); UP = np.array([U[1], -U[0]])
if UP[1] > 0: UP = -UP
def sk(x, y): p = TIP + U * L * x + UP * L * y; return (float(p[0]), float(p[1]))
TOP = [(0, 0), (.12, .013), (.30, .026), (.48, .038), (.60, .050), (.67, .066), (.72, .092), (.755, .14), (.78, .19),
       (.80, .222), (.822, .235), (.845, .226), (.865, .195), (.89, .155), (.92, .135), (.955, .115), (.985, .078), (1.0, .03)]
BOT = [(1.005, -.005), (1.01, -.035), (.99, -.06), (.96, -.058), (.93, -.075), (.89, -.082), (.86, -.07), (.835, -.06), (.80, -.052), (.74, -.046), (.62, -.034),
       (.45, -.024), (.27, -.016), (.10, -.009), (.02, -.003)]
skull = lasso(H, W, [sk(*p) for p in TOP + BOT], jag=.6, scale=4, seed=4)
oc = sk(.86, -.008); orbit = Shape(ell(oc[0], oc[1], 15, 11, ang=-13) + .04 * nz(3) < 1)
# 投影：灯在左上前方（靠近镜头），影往离镜头远的方向、往右拖——画面上是骨头后面偏右上；靠骨头硬，远了虚
sm = skull.mask.copy(); cast_ = np.zeros_like(sm)
for k_ in range(1, 26):
    cast_ |= np.roll(np.roll(sm, -int(k_ * .55), axis=0), int(k_ * 1.6), axis=1)
cast_ &= ~sm
near_ = cast_ & (edt(~sm) < 14)
fill(st.layer('骨影', mode='multiply', mask=step(cast_, 10, solid=.75)), ALL, (.45, .50, .60))
fill(st.layer('骨影近', mode='multiply', mask=step(near_, 3, solid=.9)), ALL, (.55, .58, .66))
fill(st.layer('骨'), skull, '#17120f', edge=HARD)
# 骨头朝灯那一面（上沿）亮一阶：锰黑里透一点暖
topface = Shape(skull.mask & (edt(~skull.mask) < 0) ) if False else None
up_d = edt(skull.mask)                                          # 离骨头边的距离
upper = skull.mask & (np.roll(skull.mask, 9, axis=0) == False) | (skull.mask & ~np.roll(skull.mask, -0, axis=0))
# 朝灯的那一面：上沿往里一阶，颅顶那块大一点；锰壳是麻的，一团团深浅
depth_from_top = np.full((H, W), 99.0)
for sh_ in range(1, 40):
    edge_ = skull.mask & ~np.roll(skull.mask, sh_, axis=0)
    depth_from_top = np.where(edge_ & (depth_from_top == 99), sh_, depth_from_top)
facet = skull.mask & (depth_from_top < 6 + 16 * np.clip((xx - 380) / 200, 0, 1) + 3 * nz(6))
fill(st.layer('骨亮', mask=step(facet, 2, keep=.5)), ALL, '#2e2622')
rimw = 1.5 + 3.5 * np.exp(-((xx - 505) / 45) ** 2) + 1.5 * np.clip((xx - 300) / 300, 0, 1)
fill(st.layer('骨亮2', mask=step(skull.mask & (depth_from_top < rimw), 1.5, keep=.5)), ALL, '#55473d')
front = skull.mask & (depth_from_top < 4) & (xx > 470) & (xx < 522)            # 骨嵴朝灯那一面
fill(st.layer('骨嵴亮', mask=step(front, 1.5, keep=.5)), ALL, '#76665a')
mott = skull.mask & (nz(3) > 1.1)
fill(st.layer('锰斑', mode='multiply', mask=mott * .8), ALL, (.62, .58, .58))
mott2 = skull.mask & facet & (nz(2) > 2.0)
fill(st.layer('锰亮斑', mask=mott2 * .7), ALL, '#5a4b40')
# 骨头上的洞才让人认得出是头骨：颞窝一个大的暗洞（不是圆的眼睛），眶在骨嵴前下方开一个口
FOSSA = [sk(.875, .065), sk(.915, .072), sk(.955, .058), sk(.968, .02), sk(.955, -.022), sk(.92, -.035), sk(.885, -.02), sk(.87, .025)]
fossa = lasso(H, W, FOSSA, jag=.8, scale=3, seed=9)
fill(st.layer('颞窝'), fossa, '#060404', edge=HARD)
fill(st.layer('颞窝下沿', mask=step(fossa.mask & ~np.roll(fossa.mask, -3, 0), 1) * .9), ALL, '#3a302a')
ORB = [sk(.79, -.052), sk(.80, -.02), sk(.822, -.008), sk(.842, -.022), sk(.848, -.058), sk(.82, -.062)]
orb = lasso(H, W, ORB, jag=.6, scale=3, seed=10)
fill(st.layer('眶'), orb, '#070505', edge=HARD)
# 贴地那道缝：最深最锐
seam = rim(skull, keep=(yy > 0)) if False else None
bottom_edge = skull.mask & ~np.roll(skull.mask, -3, axis=0)
fill(st.layer('缝', mask=np.roll(bottom_edge, 2, axis=0) * .95), ALL, '#050404')

# ---------- 4.4 下颌和那颗带子牙：下颌掉在头骨前面一点，细长一根；带齿喙鲸雄的那颗牙从下颌中段往上弯，绕过喙，嘴只张得开一条缝
JAW = [sk(.06, -.055), sk(.30, -.075), sk(.52, -.088), sk(.66, -.095), sk(.72, -.11), sk(.74, -.135), sk(.70, -.14),
       sk(.62, -.118), sk(.48, -.106), sk(.28, -.092), sk(.08, -.070)]
jaw = lasso(H, W, JAW, jag=.5, scale=3, seed=8)
jcast = np.zeros((H, W), bool)
for k_ in range(1, 9): jcast |= np.roll(np.roll(jaw.mask, -int(k_ * .55), 0), int(k_ * 1.6), 1)
fill(st.layer('颌影', mode='multiply', mask=step(jcast & ~jaw.mask & ~skull.mask, 3, solid=.8)), ALL, (.45, .5, .6))
fill(st.layer('颌'), jaw, '#1a1411', edge=HARD)
fill(st.layer('颌亮', mask=step(jaw.mask & ~np.roll(jaw.mask, 3, 0), 1, keep=.5)), ALL, '#4a3e36')
# 带子牙：从下颌 .50 处起，往上弯过喙的上沿，扁的一条，象牙色发灰（也被染过）
tooth_pts = [sk(.455, -.085), sk(.462, -.055), sk(.475, -.025), sk(.495, .005), sk(.52, .03), sk(.552, .046), sk(.585, .05), sk(.608, .042)]
tooth = line(H, W, tooth_pts, width=lambda t: 7 - 3.5 * t)
fill(st.layer('带子牙'), tooth, '#9a8f7c', edge=HARD)
fill(st.layer('带子牙暗', mode='multiply', mask=step(tooth.mask & ~np.roll(tooth.mask, -2, 1), 1)), ALL, (.6, .58, .6))

# ---------- 4.5 脊椎：从颅后往远处排成一条虚线，躺下那天的样子；越远越小、越挤、越暗进黑里
VERT = []
r3 = np.random.default_rng(12)
x_, y_, sc = 700, 566, 1.0
for i in range(9):
    VERT.append((x_ + r3.uniform(-5, 5), y_ + r3.uniform(-4, 4), sc)); gap = 46 * sc + r3.uniform(-8, 12)
    x_ += gap * .92; y_ -= gap * .38; sc *= .85
for i, (vx_, vy_, sc) in enumerate(VERT):
    lit = np.clip(1.0 - i * .1, .25, 1)
    rot = r3.uniform(-70, 40)
    body = Shape(ell(vx_, vy_, 12 * sc, 9.5 * sc, ang=rot) + .1 * nz(3) < 1)
    a_ = np.radians(rot - 90 + r3.uniform(-30, 30)); sl = r3.uniform(7, 13) * sc
    spine_ = line(H, W, [(vx_, vy_), (vx_ + np.cos(a_) * sl, vy_ + np.sin(a_) * sl)], width=lambda t, s_=sc: 4 * s_ * (1 - .5 * t))
    b_ = np.radians(rot); wl = r3.uniform(10, 15) * sc
    wing = line(H, W, [(vx_ - np.cos(b_) * wl, vy_ - np.sin(b_) * wl), (vx_ + np.cos(b_) * wl, vy_ + np.sin(b_) * wl)], width=lambda t, s_=sc: 3 * s_)
    v = body | (spine_ if r3.random() < .7 else body) | (wing if r3.random() < .8 else body)
    vcast = np.zeros((H, W), bool)
    for k_ in range(1, int(10 * sc) + 2): vcast |= np.roll(np.roll(v.mask, -int(k_ * .55), 0), int(k_ * 1.6), 1)
    fill(st.layer(f'椎影{i}', mode='multiply', mask=step(vcast & ~v.mask, 3, solid=.7 * lit + .1)), ALL, (.45, .5, .6))
    fill(st.layer(f'椎{i}'), v, mix_ := tuple(np.array([.09, .075, .065]) * lit + np.array([.045, .06, .085]) * (1 - lit)), edge=HARD)
    topv = v.mask & ~np.roll(v.mask, max(2, int(3 * sc)), axis=0)
    fill(st.layer(f'椎亮{i}', mask=step(topv, 1, keep=.5) * lit), ALL, '#4a3f37')

# ---------- 5 粉毛：骨头上沿立着的一层细管（吃骨头的虫的鳃），粉的，梢上发白；最密在颅顶，往喙尖稀下去
from PIL import Image, ImageDraw
def raster(draw_fn):
    im = Image.new('L', (W, H), 0); draw_fn(ImageDraw.Draw(im)); return np.array(im) > 127
top_xs = [p[0] for p in TOP]; top_ys = [p[1] for p in TOP]
def fur_draw(d, tips=False):
    r_ = np.random.default_rng(77)
    for i in range(2600):
        x = r_.uniform(.36, .99) if r_.random() < .35 else r_.uniform(.62, .99)
        dens = np.clip((x - .36) / .4, .15, 1) * (1 - .55 * np.exp(-((x - .79) / .035) ** 2))
        dens *= np.clip(.55 + .6 * np.sin(x * 37 + 1.3) + .3 * np.sin(x * 91), 0, 1)        # 一片一片的，不是一整圈
        if r_.random() > dens: continue
        bx, by = sk(x, np.interp(x, top_xs, top_ys) - .004)
        ang = np.arctan2(UP[1], UP[0]) + r_.normal(0, .35)
        ln = r_.uniform(4, 8 + 11 * dens)
        ex, ey = bx + np.cos(ang) * ln, by + np.sin(ang) * ln
        if tips: d.line([((bx + ex * 2) / 3, (by + ey * 2) / 3), (ex, ey)], fill=255, width=1)
        else: d.line([(bx, by), (ex, ey)], fill=255, width=1 + (r_.random() < .3))
fur = raster(fur_draw); fur_tip = raster(lambda d: fur_draw(d, True))
fill(st.layer('粉毛', mask=fur * .85), ALL, '#b2626f')
fill(st.layer('毛梢', mask=fur_tip * .6), ALL, '#d7a9ad')

# ---------- 6 花云：毛梢松开，变成一片一片软的花色，顺着水流往右上漂；出了灯就暗下去，色还在（雷东：颜色从里面亮）
LIT = np.clip(1.2 - np.hypot(xx - 430, yy - 400) / 560, .28, 1)[..., None]
DARKW = np.array([.05, .07, .12])
def hx(c): return np.array([int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)])
def tone(c, boost=1.0): return np.clip(hx(c) * boost * LIT + (1 - LIT) * DARKW, 0, 1)
PATH = [(585, 392), (630, 352), (685, 310), (750, 265), (820, 214), (885, 162), (945, 112)]
px_, py_ = np.array(PATH).T; tt = np.linspace(0, 1, len(PATH))
PAL = {  # 颜色按高度挑：低处粉，中间橙和奶黄，高处蓝紫
    'pink': ('#d39aa0', lambda t: .8 - t * 1.1), 'rose': ('#c9707f', lambda t: .7 - t),
    'orange': ('#e2a066', lambda t: .9 - abs(t - .4) * 2), 'cream': ('#e6d4a8', lambda t: .45 - abs(t - .4) * 1.5),
    'blue': ('#8e9ed2', lambda t: t * 1.1 - .1), 'violet': ('#86709e', lambda t: t * 1.2 - .45),
    'white': ('#e9e6e2', lambda t: .25 - abs(t - .3) * .6),
}
acc = {k: np.full((H, W), 9.0) for k in PAL}
r_ = np.random.default_rng(31)
for i in range(230):
    t = .1 + .9 * r_.random() ** 1.25
    if t > .78 and r_.random() < .45: continue
    cx, cy = np.interp(t, tt, px_), np.interp(t, tt, py_)
    n_ = np.array([-.6, -.8]) if r_.random() < .5 else np.array([.6, .8])
    off = r_.normal(0, 12 + 85 * t); along_off = r_.normal(0, 30)
    cx += n_[0] * abs(off) * np.sign(off) + .8 * along_off; cy += n_[1] * abs(off) * np.sign(off) - .6 * along_off
    rx = r_.uniform(9, 20 + 20 * (1 - t)); ry = rx * r_.uniform(.55, .9); ang = r_.uniform(-40, 10)
    w_ = {k: max(f(t), 0) for k, (c, f) in PAL.items()}; ks = list(w_); pw = np.array([w_[k] for k in ks]) + 1e-6
    k = ks[r_.choice(len(ks), p=pw / pw.sum())]
    x0, x1, y0, y1 = int(max(cx - rx * 2, 0)), int(min(cx + rx * 2, W)), int(max(cy - rx * 2, 0)), int(min(cy + rx * 2, H))
    if x1 <= x0 or y1 <= y0: continue
    sub = ell(cx, cy, rx, ry, ang)[y0:y1, x0:x1]
    acc[k][y0:y1, x0:x1] = np.minimum(acc[k][y0:y1, x0:x1], sub)
lump = .16 * nz(7)
for k in ['violet', 'blue', 'white', 'cream', 'orange', 'rose', 'pink']:
    m = acc[k] + lump
    fill(st.layer(f'云-{k}', mask=step(m < 1, 5, keep=.44), opacity=.88), ALL, tone(PAL[k][0]))
    fill(st.layer(f'云-{k}芯', mask=step(m < .5, 4, keep=.44), opacity=.88), ALL, tone(PAL[k][0], 1.1))
# 毛梢松开往上飘的几缕：从颅顶的毛里升起来，弯着进到花云里
def wisp_draw(d):
    r2 = np.random.default_rng(55)
    for i in range(70):
        x = r2.uniform(.6, .98); bx, by = sk(x, np.interp(x, top_xs, top_ys) + .03)
        pts = [(bx, by)]; ang = np.arctan2(UP[1], UP[0]) + r2.normal(.25, .25)
        for j in range(int(r2.uniform(6, 16))):
            ang += r2.normal(.06, .12); seg = r2.uniform(6, 11)
            pts.append((pts[-1][0] + np.cos(ang) * seg, pts[-1][1] + np.sin(ang) * seg))
        d.line(pts, fill=255, width=1)
wisp = raster(wisp_draw)
fill(st.layer('松开的毛', mask=wisp * .55), ALL, tone('#e0a8ae'))
# 花云自己的光往外溢一点，把颅顶的黑剪影衬出来
dens = sum((acc[k] < 1).astype(float) for k in acc)
halo = gaussian_filter(np.clip(dens, 0, 1), 38); halo = halo / halo.max()
fill(st.layer('云光', mode='screen', mask=np.clip(halo * .55, 0, 1) * (~skull.mask)), ALL, (.30, .17, .20))
# 几朵认得出的花：蓝色的小星形（雷东那张正中间那朵），两三粒红橙
def floret(d, cx, cy, r, n=5, rot=0):
    for j in range(n):
        a = rot + j * 2 * np.pi / n; ex, ey = cx + np.cos(a) * r, cy + np.sin(a) * r
        d.ellipse([ex - r * .45, ey - r * .45, ex + r * .45, ey + r * .45], fill=255)
FL = [(640, 300, 11), (705, 345, 9), (760, 250, 8), (600, 372, 10), (820, 215, 7)]
fl = raster(lambda d: [floret(d, x, y, r, rot=x * .1) for x, y, r in FL])
fill(st.layer('蓝花', mask=step(fl, 2)), ALL, tone('#5d78c8', 1.0))
dots = raster(lambda d: [d.ellipse([x - r, y - r, x + r, y + r], fill=255) for x, y, r in [(680, 380, 7), (735, 300, 5), (585, 335, 6)]])
fill(st.layer('红点花', mask=step(dots, 2)), ALL, tone('#d0563e'))
# ---------- 7 海床上的东西：泥里的细麻点、几颗锰结核、海蛇尾
speck = (rng.random((H, W)) < .012) & (yy > 480)
fill(st.layer('泥点', mode='multiply', mask=speck * np.clip(1.2 - fd, 0, 1) * .8), ALL, (.55, .6, .62))
NOD = [(150, 760, 9), (470, 735, 7), (760, 650, 8), (835, 700, 10), (60, 650, 6), (610, 700, 6), (905, 610, 7)]
for x, y, r in NOD:
    nod = Shape(ell(x, y, r, r * .7, ang=r3.uniform(-20, 20)) + .12 * nz(2) < 1)
    fill(st.layer(f'结核{x}'), nod, '#0d0b0a', edge=HARD)
    fill(st.layer(f'结核亮{x}', mask=step(nod.mask & ~np.roll(nod.mask, 2, 0), 1)), ALL, '#3d3a36')
# 海蛇尾：中间一粒小盘，五条细臂弯着；一只趴在颅上，两只在泥里
def brittle(d, cx, cy, r, rot, wob=.5):
    r4 = np.random.default_rng(int(cx * 7 + cy))
    for j in range(5):
        a = rot + j * 2 * np.pi / 5 + r4.normal(0, .15); pts = [(cx, cy)]; curl = r4.normal(0, wob)
        for k_ in range(10):
            a += curl * .18; seg = r / 10
            pts.append((pts[-1][0] + np.cos(a) * seg, pts[-1][1] + np.sin(a) * seg * .62))
        d.line(pts, fill=255, width=1)
    d.ellipse([cx - r * .12, cy - r * .08, cx + r * .12, cy + r * .08], fill=255)
BS = [(598, 512, 32, .4, 1.4), (250, 738, 40, 1.1, 1.6), (735, 688, 28, 2.2, 1.5)]
bsm = raster(lambda d: [brittle(d, *b) for b in BS])
fill(st.layer('海蛇尾', mask=bsm * .9), ALL, '#9d9a92')
fill(st.layer('海蛇尾影', mode='multiply', mask=np.roll(np.roll(bsm, -1, 0), 3, 1) & ~bsm), ALL, (.5, .55, .62))

# ---------- 8 海雪：灯里才看得见的白点，往下慢慢落；越靠近灯越多
snow_p = (rng.random((H, W)) < .0028 * np.clip((glow - .25) * 1.8, 0, 1))
snow_big = gaussian_filter(snow_p.astype(float), .7) > .12
fill(st.layer('海雪', mask=snow_big * .85), ALL, '#c9d0d6')
fall = raster(lambda d: [d.line([(x, y), (x - 1, y + 4)], fill=255) for y, x in zip(*np.where(snow_p & (rng.random((H, W)) < .25)))])
fill(st.layer('海雪拖', mask=fall * .35), ALL, '#aab4bd')

# ---------- 9 两粒红点：潜水器的激光尺，隔 10 厘米（这把头骨约一米，10 厘米≈56 像素），落在颅侧
LASER = [(400, 712), (456, 708)]                              # 落在头骨前面的泥上
red = raster(lambda d: [d.ellipse([x - 1.7, y - 1.4, x + 1.7, y + 1.4], fill=255) for x, y in LASER])
fill(st.layer('红点晕', mode='screen', mask=gaussian_filter(red.astype(float), 2) * 1.2), ALL, (.4, .04, .04))
fill(st.layer('红点'), Shape(red), '#ff3b2e', edge=HARD)

# ---------- 10 纸纹：整张一张，最后罩上去
pg = gaussian_filter(np.random.default_rng(3).standard_normal((H, W)), .7); pg /= pg.std()
fill(st.layer('纸纹暗', mode='multiply', mask=np.clip((pg - 1.0) * .5, 0, .35)), ALL, (.7, .7, .74))
fill(st.layer('纸纹亮', mode='screen', mask=np.clip((-pg - 1.4) * .3, 0, .07)), ALL, (.3, .31, .36))

# ---------- 签名：一个点一个点敲，点间连直线，拐点留一粒墨
SIG = [[(884, 764), (878, 762), (874, 766), (874, 772), (878, 776), (885, 774)],
       [(889, 760), (889, 776)],
       [(898, 768), (894, 768), (892, 772), (894, 776), (898, 775), (899, 768), (899, 776)],
       [(903, 768), (903, 774), (905, 776), (908, 775), (909, 768), (909, 776)],
       [(918, 768), (914, 768), (912, 772), (914, 776), (918, 775), (919, 760), (919, 776)],
       [(923, 772), (929, 772), (929, 769), (926, 767), (923, 769), (923, 774), (926, 776), (930, 775)]]
sig = raster(lambda d: [(d.line(p, fill=255, width=1), [d.ellipse([x - 1.1, y - 1.1, x + 1.1, y + 1.1], fill=255) for x, y in p]) for p in SIG])
fill(st.layer('签名', mask=sig * .85), ALL, '#7d8794')

look(st, 'flowercloud.png')
