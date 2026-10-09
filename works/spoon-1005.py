# 邪恶大饭勺 · 10.5（nerolette 起的名；她照片的下一秒，没照着照片画）
# 从哪儿看：眼睛贴着水面，像蹲在滩上。地平线压得很低（y≈585），近的东西都高过天际。
# 这张在说什么：一把不看路的大饭勺，扫中了在旁边等着蹭饭的红嘴鸥。琵鹭眼皮都没抬。
# 光：太阳在右上画框外面，被一层薄云罩着，大、软、暖白；我们几乎迎着它（偏逆光）。
#     两只鸟朝我们这面是背光的冷灰白，迎光那条上沿一道暖亮边；肚子底下接一点水面反上来的冷亮光。
#     水面太阳底下那一竖条是碎光，越往左越暗灰；天也是靠太阳那角最亮最暖，左上偏冷偏灰。
# 白对白：琵鹭压在左上偏灰的天上，比天亮；红嘴鸥压在右边的碎光上，比水暗——同样是白鸟，一个亮过背景一个暗过背景。
# 明暗：最深是琵鹭的黑脸、黑勺、黑腿和右下那截木桩；最亮是碎光和鸟的亮边；唯一鲜的颜色是红嘴鸥的嘴和腿。
# 视线：先落在黑勺（最深、最锐，正扫进白里），顺勺子扫出去的那道水花到红嘴鸥张着的红嘴，再回到琵鹭半闭的眼——它根本没看。
# 质感：台阶与颗粒。水是光滑的，碎光和水花是干净的边；鸟毛、木桩、远岸是糙的，边喷开。整张一张纸纹。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = '~/.claude/workspace/painting-method/kbrush-cloud'
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(5)
from shapes import Shape
from blocks import blob, lasso, line, soft, look
from layers import Stack, fill, HARD
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt
from PIL import Image as _I, ImageDraw as _D

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'rough.png' if ROUGH else 'v1.png')
H, W = 800, 1200
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
HZ = 585                                       # 地平线
SUN = np.array([1000., 118.])                 # 太阳在画里，右上，薄云后面一小粒

def nz(sig, seed):
    r = np.random.default_rng(seed)
    n = gaussian_filter(r.standard_normal((H, W)), sig); return n / n.std()
def smooth(pts, n=3):
    p = np.array(pts, float)
    for _ in range(n):
        q = np.roll(p, -1, 0)
        p = np.stack([.75 * p + .25 * q, .25 * p + .75 * q], 1).reshape(-1, 2)
    return [tuple(v) for v in p]
def spray(region, sp, seed, solid=.96, dens=.7):
    """一阶：区域里实实地盖上，边外喷出颗粒，越远越稀"""
    region = getattr(region, 'mask', region)
    if ROUGH:
        return region.astype(float) * solid
    r = np.random.default_rng(seed)
    dout = edt(~region)
    dots = (r.random((H, W)) < dens * np.exp(-dout / np.maximum(sp, .5))) & (gaussian_filter(r.random((H, W)), .5) > .47)
    return np.where(region, solid, dots * .95)
def ink(draw_fn, seed, SS=3):
    im = _I.new('L', (W * SS, H * SS), 0); draw_fn(_D.Draw(im), np.random.default_rng(seed), SS)
    return np.asarray(im.resize((W, H), _I.LANCZOS), float) / 255
def L(pts, jag=1.5, seed=0, n=3, scale=8):
    return lasso(H, W, smooth(pts, n), jag=jag, scale=scale, seed=seed)

sky_m = yy < HZ
water_m = ~sky_m
st = Stack(K.Paper(H, W, seed=3), ground=(.72, .74, .76))

# ---------- 1 天：左上冷灰；太阳是一小粒，四周几阶暖亮，收得紧；几道横着的薄云，靠太阳那截被照亮
sd = np.hypot(xx - SUN[0], (yy - SUN[1]) * 1.15) + 6 * nz(40, 1)
for i, (r_, c_, sp_) in enumerate([(300, (.75, .76, .77), 95), (220, (.79, .80, .80), 80), (155, (.84, .84, .82), 62), (100, (.89, .88, .84), 44), (58, (.94, .92, .87), 28), (32, (.97, .95, .90), 14)]):
    fill(st.layer(f'天{i}', mask=spray((sd < r_ - sp_ * .8) & sky_m, sp_, 10 + i)), ALL, c_)
fill(st.layer('天边雾', mask=spray(sky_m & (yy > HZ - 34 - 26 * np.clip((xx - 300) / 700, 0, 1) + 8 * nz((4, 40), 15)), 22, 16, solid=.75)), ALL, (.84, .84, .82))
fill(st.layer('日', mask=soft(blob(H, W, SUN[0], SUN[1], 16, 16, jag=0), feather=2.5)), ALL, (1., .99, .95))

# ---------- 2 远岸：一条薄薄的红树林，靠太阳那头被雾吃掉
shore_top = HZ - 9 - 7 * (nz(6, 2)[HZ] if False else 0) - 6 * np.abs(nz(9, 3)) - 4 * nz(3, 4)
far = (yy > shore_top) & (yy < HZ + 3) & (xx < 960 + 60 * nz(20, 5))
fade = np.clip((xx - 520) / 440, 0, 1)
fill(st.layer('远岸', mask=spray(far & (xx < 700), 3, 20)), ALL, (.60, .62, .64))
fill(st.layer('远岸雾', mask=spray(far & (xx >= 700), 3, 21, solid=.7)), ALL, (.74, .74, .73))

# ---------- 3 水：底下一阶阶往近处暗；光是一块块横压扁的亮，从噪声里长出来（不是一粒粒的点）
near = np.clip((yy - HZ) / (H - HZ), 0, 1)
wnz = 5 * nz((2, 30), 6)
for i, (y_, c_) in enumerate([(HZ, (.82, .82, .81)), (612, (.76, .77, .78)), (660, (.70, .72, .75)), (725, (.64, .66, .70))]):
    fill(st.layer(f'水{i}', mask=spray((yy + wnz > y_) & water_m, 4, 30 + i)), ALL, c_)
# 一张位移场，水上的一切都跟它扭：慢的弯 + 快的碎，越近越大
dxw = nz((2.2, 40), 46) * (1.5 + 12 * near) + nz((10, 220), 47) * (1 + 5 * near)
dyw = nz((5, 60), 48) * (1 + 3 * near)
pc = SUN[0] - 20 + (yy - HZ) * .12
path = np.exp(-((xx - pc + .7 * dxw) / ((45 + 240 * near) * (1 + .35 * nz((30, 30), 60)))) ** 2) * water_m
rip = (1 - near) * nz((1.1, 48), 49) + near * nz((4.2, 125), 50)
F = .55 * rip + .27 * ((1 - near) * nz((2.5, 30), 61) + near * nz((7, 70), 62)) + .18 * nz((14, 260), 51)
deep = water_m & (yy > HZ + 2)
# 碎光：最早那种——太阳正下方一竖条，近处宽远处窄；一道道横的短划，干净的边（她说这个水粼粼）
gx = SUN[0] - 20 + (yy - HZ) * 0.12
def glints(dr, r_, SS):
    for _ in range(2600):
        y = HZ + 2 + (H - HZ) * r_.random() ** 1.6
        spread = 60 + (y - HZ) * 0.9
        x = r_.normal(SUN[0] - 20 + (y - HZ) * .12, spread * .55)
        ln = (2 + 14 * r_.random() ** 2) * (0.4 + (y - HZ) / (H - HZ))
        th = .5 + 1.2 * r_.random() * (y - HZ) / (H - HZ)
        dr.ellipse([(x - ln) * SS, (y - th) * SS, (x + ln) * SS, (y + th) * SS], fill=255)
glint_m = ink(glints, 40) * water_m
fill(st.layer('碎光底', mask=spray((np.abs(xx - gx) < 90 + (yy - HZ) * .55 + 25 * nz((8, 20), 42)) & water_m & (yy > HZ + 4), 26, 41, solid=.6)), ALL, (.86, .86, .83))
fill(st.layer('碎光', mask=glint_m), ALL, (.98, .96, .89))

# ---------- 4 琵鹭（左上压进来，被框切掉）
body = L([(-120, 120), (-40, 60), (60, 38), (170, 42), (270, 70), (340, 110), (372, 165), (360, 230), (310, 290), (220, 330),
          (110, 345), (0, 335), (-120, 300)], 3, 50, scale=30)
neck = line(H, W, [(300, 118), (385, 98), (460, 140), (512, 235), (532, 340), (540, 418)], width=lambda t: 92 - 48 * t, seed=51)
head = blob(H, W, 556, 442, 47, 36, ang=28, jag=0.03, seed=52)
bird = body | neck | head
face = (blob(H, W, 586, 466, 30, 25, ang=35, jag=0.05, seed=53) & head) | blob(H, W, 590, 472, 15, 14, jag=0.02, seed=54)
bill = line(H, W, [(588, 472), (630, 522), (672, 576), (712, 628), (744, 668), (766, 694)], width=lambda t: 24 - 13 * min(t / .7, 1) + 26 * max(0, (t - .7) / .3) ** 1.4, seed=55)
spoon = blob(H, W, 770, 698, 27, 17, ang=50, jag=0.02, seed=56)
# 鸟的"膝盖"是脚踝：小腿从毛裤里出来往后斜到关节（整条腿最靠尾巴的一点），跗跖再往前斜下去落到水里
legs = line(H, W, [(70, 380), (60, 450), (50, 512), (64, 600), (82, 712)], width=lambda t: 15 - 4 * t, seed=57) | \
       line(H, W, [(136, 372), (128, 440), (118, 500), (138, 600), (162, 700)], width=lambda t: 12 - 3.5 * t, seed=58)
knees = blob(H, W, 51, 512, 9, 8, jag=0, seed=105) | blob(H, W, 119, 500, 7, 6, jag=0, seed=106)
# 毛裤：大腿上那截羽毛，从肚子底下垂下来，收进黑腿里
pants = L([(36, 318), (54, 392), (66, 404), (80, 400), (92, 384), (98, 330)], 2.2, 101, n=2, scale=5) | \
        L([(108, 318), (122, 380), (132, 390), (144, 386), (152, 370), (158, 322)], 2.2, 102, n=2, scale=5)
# 身子：朝我们这面背光，冷灰白；上沿迎光一道暖亮；肚子一阶冷暗，最底下接水的反光
fill(st.layer('鸟身'), bird, (.87, .87, .85), edge=HARD)
bdist_top = edt(bird.mask)
up_edge = bird.mask & (bdist_top < 12 + 6 * nz(4, 59)) & (yy < 300) & ~face.mask
# 迎光的是朝右上的那段边：拿"往右上挪一点就出了鸟"判断
sh_ = np.zeros_like(bird.mask); sh_[:-10, 10:] = bird.mask[10:, :-10]
rim_m = bird.mask & ~sh_ & ~face.mask & (xx > 14) & (yy > 14)
belly1 = body.mask & ~np.roll(body.mask, -105, 0)
belly2 = body.mask & ~np.roll(body.mask, -48, 0)
fill(st.layer('鸟肚1', mask=spray(belly1, 7, 62)), ALL, (.79, .80, .82))
fill(st.layer('鸟肚2', mask=spray(belly2, 5, 63)), ALL, (.70, .72, .76))
# 背：朝天的那一面最亮最暖（离上沿竖着量 70px 以内）
back = body.mask & ~np.roll(body.mask, 70 + (14 * nz(30, 64)).astype(int).max(), 0)
back = body.mask & ~np.roll(body.mask, 70, 0) & (nz(25, 64) > -1.2)
fill(st.layer('背', mask=spray(back, 6, 65)), ALL, (.92, .91, .87))
# 收着的翅膀：盖在身子侧面，下沿是一排羽尖，羽尖底下在肋上落一线影
WING = [(-60, 120), (60, 100), (180, 108), (290, 140), (338, 178), (322, 196), (300, 205), (282, 214), (262, 220), (240, 228),
        (218, 234), (196, 241), (172, 246), (148, 252), (124, 257), (98, 262), (72, 266), (44, 268), (14, 270), (-60, 272)]
# 羽尖：下沿每隔一点往外鼓一点，成一排扇贝
wing_pts = []
for i, (x, y) in enumerate(WING):
    wing_pts.append((x, y + (7 if (i >= 5 and i % 2 == 0) else 0)))
wing = L(wing_pts, 1.0, 66, n=1, scale=5)
wing_m = wing.mask & body.mask
fill(st.layer('翅', mask=spray(wing_m & ~back, 2, 67)), ALL, (.86, .86, .86))
wsh = body.mask & ~wing.mask & np.roll(wing.mask, 9, 0) & np.roll(wing.mask, -0, 0) == False
wing_shadow = body.mask & ~wing.mask & np.roll(wing.mask, 9, 0)
fill(st.layer('翅影', mask=spray(wing_shadow, 1.5, 68)), ALL, (.72, .74, .78))
# 亮边：不是一圈描线——迎光的那几段断断续续，有的地方化掉
rim_broken = (rim_m | (up_edge & (xx > 250))) & (nz((6, 18), 69) > -.6)
fill(st.layer('鸟亮边', mask=spray(rim_broken, 2.5, 60)), ALL, (.97, .94, .87))
belly1 = body.mask & ~np.roll(body.mask, -105, 0)
belly2 = body.mask & ~np.roll(body.mask, -48, 0)
neck_shade = (neck.mask | head.mask) & ~sh_ & False
nk_left = neck.mask & (((xx - 430) * .8 - (yy - 200) * .6) < -14) & (yy > 150) & (yy < 370)
fill(st.layer('脖背光', mask=spray(nk_left, 4, 63)), ALL, (.74, .75, .78))
bot = body.mask & (edt(body.mask) < 7) & (yy > 300)
fill(st.layer('鸟肚反光', mask=spray(bot & ~legs.mask, 2, 64, solid=.7)), ALL, (.79, .81, .84))
fill(st.layer('头', mask=spray(head.mask & ~face.mask, 1.2, 99)), ALL, (.88, .88, .87))
shh = np.zeros_like(head.mask); shh[:-5, 5:] = head.mask[5:, :-5]
fill(st.layer('头亮边', mask=spray(head.mask & ~shh & ~face.mask & (yy < 445), 1.5, 100)), ALL, (.97, .94, .87))
fill(st.layer('毛裤', mask=spray(pants, 1.5, 103)), ALL, (.72, .74, .78))
shp = np.zeros_like(pants.mask); shp[:-4, 4:] = pants.mask[4:, :-4]
fill(st.layer('毛裤亮', mask=spray(pants.mask & ~shp, 1.2, 104, solid=.8)), ALL, (.86, .86, .86))
# 黑：脸、嘴、勺、腿
BLACK = (.11, .10, .12)
fill(st.layer('黑', mask=spray(face | bill | spoon | Shape((legs.mask | knees.mask) & ~pants.mask), 1.0, 65)), ALL, BLACK)
# 嘴上一线光（光滑的角质，朝太阳那侧），干净的边
bill_hl = line(H, W, [(600, 470), (640, 515), (680, 566), (716, 616)], width=2.2, seed=66)
fill(st.layer('嘴光', mask=soft(bill_hl, feather=1.0) * .8), ALL, (.48, .48, .50))
# 眼：半闭着，只留一粒亮——它根本没看
fill(st.layer('眼光', mask=soft(blob(H, W, 571, 452, 2.6, 1.6, jag=0, seed=67), feather=.8)), ALL, (.85, .80, .74))
fill(st.layer('眼皮', mask=soft(line(H, W, [(562, 449), (571, 447), (580, 450)], width=1.6, seed=68), feather=.6) * .6), ALL, (.35, .33, .35))
# ---------- 5 红嘴鸥：被扫中，往后仰着炸起来，冲着勺子张嘴
gbody = blob(H, W, 844, 612, 60, 31, ang=35, jag=0.05, seed=70)
ghead = blob(H, W, 784, 562, 23, 20, ang=10, jag=0.04, seed=71)
gtail = L([(877, 628), (922, 655), (918, 664), (872, 646)], 1, 72, n=1)
near_w = L([(826, 598), (806, 548), (790, 502), (812, 458), (846, 410), (872, 372), (886, 390), (878, 400), (892, 414), (882, 425), (893, 440), (880, 450), (876, 488), (866, 530), (858, 580), (852, 604)], 1.5, 73, n=1, scale=6)
far_w = L([(862, 594), (880, 548), (902, 506), (938, 470), (988, 428), (980, 450), (990, 458), (972, 468), (978, 480), (958, 486), (936, 512), (912, 556), (890, 604)], 1.5, 74, n=1, scale=6)
gull = gbody | ghead | gtail
fill(st.layer('鸥远翅'), far_w, (.80, .81, .83), edge=HARD)
tip_far = far_w.mask & (((xx - 984) ** 2 + (yy - 440) ** 2) < 36 ** 2)
fill(st.layer('鸥远翅尖', mask=spray(tip_far, 1.5, 75)), ALL, BLACK)
fill(st.layer('鸥身'), gull, (.72, .74, .78), edge=HARD)
fill(st.layer('鸥近翅'), near_w, (.60, .62, .67), edge=HARD)          # 翅膀底面，背光，暗
tip_near = near_w.mask & (((xx - 880) ** 2 + (yy - 392) ** 2) < 34 ** 2)
fill(st.layer('鸥近翅尖', mask=spray(tip_near, 1.5, 76)), ALL, (.20, .20, .23))
# 逆光：羽毛的边透光，暖亮
for nm, m, s in (('鸥身亮边', gull.mask, 77), ('鸥翅亮边', near_w.mask & ~tip_near, 78), ('鸥远翅亮边', far_w.mask & ~tip_far, 79)):
    shm = np.zeros_like(m); shm[:-6, 6:] = m[6:, :-6]
    fill(st.layer(nm, mask=spray(m & ~shm, 1.5, s)), ALL, (.98, .95, .88))
fill(st.layer('鸥头'), ghead, (.82, .83, .85), edge=HARD)
fill(st.layer('鸥耳斑', mask=spray(blob(H, W, 793, 558, 6, 5, jag=.1, seed=80), 1.2, 81)), ALL, (.32, .32, .35))
gbill_u = line(H, W, [(765, 556), (750, 551), (734, 549)], width=lambda t: 6 - 3 * t, seed=82)
gbill_l = line(H, W, [(767, 566), (753, 572), (739, 579)], width=lambda t: 5 - 3 * t, seed=83)
RED = (.80, .22, .17)
fill(st.layer('鸥嘴', mask=spray(gbill_u | gbill_l, .6, 84)), ALL, RED)
fill(st.layer('鸥眼', mask=soft(blob(H, W, 778, 553, 3.4, 3.4, jag=0, seed=85), feather=.7)), ALL, BLACK)
fill(st.layer('鸥眼光', mask=soft(blob(H, W, 779, 552, 1.0, 1.0, jag=0, seed=86), feather=.5)), ALL, (.98, .97, .95))
glegs = line(H, W, [(850, 636), (846, 662), (840, 688)], width=3.2, seed=87) | line(H, W, [(868, 640), (872, 664), (880, 686)], width=3.2, seed=88)
fill(st.layer('鸥腿', mask=spray(glegs, .6, 89)), ALL, (.78, .28, .20))

# ---------- 5b 水里的倒影和涟漪：都用同一张位移场扭，被波纹打断
def reflect(m, y0, fall):
    m = getattr(m, 'mask', m)
    sy = np.clip((2 * y0 - yy + .5 * dyw).astype(int), 0, H - 1)
    sx = np.clip((xx + .35 * dxw).astype(int), 0, W - 1)
    return m[sy, sx] & (yy > y0) & ((yy - y0) < fall * (.75 + .25 * nz(20, 300 + int(y0))))
brk = rip > .5 + .4 * (1 - near)
leg1 = line(H, W, [(70, 380), (60, 450), (50, 512), (64, 600), (82, 712)], width=lambda t: 15 - 4 * t, seed=57)
leg2 = line(H, W, [(136, 372), (128, 440), (118, 500), (138, 600), (162, 700)], width=lambda t: 12 - 3.5 * t, seed=58)
rl = (reflect(leg1, 712, 120) | reflect(leg2, 700, 100)) & ~brk
fill(st.layer('腿倒影', mask=spray(rl, .8, 310, solid=.7)), ALL, (.46, .47, .51))
rs = reflect(Shape((bill.mask | spoon.mask) & (yy > 630)), 706, 70) & ~brk
fill(st.layer('勺倒影', mask=spray(rs, .8, 311, solid=.8)), ALL, (.38, .38, .42))
rg = reflect(gull.mask | near_w.mask | far_w.mask, 692, 130) & ~(rip > .3 + .3 * (1 - near))
fill(st.layer('鸥倒影', mask=spray(rg, .8, 312, solid=.8)), ALL, (.80, .81, .83))
rgt = reflect(tip_near | tip_far, 692, 130) & ~brk
fill(st.layer('鸥翅尖倒影', mask=spray(rgt, .8, 313, solid=.8)), ALL, (.36, .36, .40))
# 涟漪：一圈圈扁的环，亮的浪脊、底下贴一道暗的浪谷；环被波纹咬断
def rings(cx, cy, r0, n, seed):
    d = np.hypot(xx - cx, (yy - cy) / .26)
    cr = np.zeros((H, W), bool); tr = np.zeros((H, W), bool)
    for k in range(n):
        rr = r0 * (1 + .9 * k); w = 1.0 + .3 * k
        band = np.abs(d - rr) < w
        cut = nz((3, 12), seed + k) > .9 - .25 * k
        cr |= band & ~cut & (yy > HZ + 2)
        tr |= np.roll(band & ~cut, 3, 0) & (yy > HZ + 2)
    return cr, tr
CR = np.zeros((H, W), bool); TR = np.zeros((H, W), bool)
for cx, cy, r0, n, sd_ in [(82, 712, 14, 2, 320), (162, 700, 11, 2, 324), (775, 708, 18, 2, 328), (862, 692, 14, 2, 333)]:
    c_, t_ = rings(cx, cy, r0, n, sd_); CR |= c_; TR |= t_
fill(st.layer('涟漪谷', mask=spray(TR & ~CR, .6, 340, solid=.4)), ALL, (.62, .64, .69))
fill(st.layer('涟漪脊', mask=spray(CR, .6, 341, solid=.5)), ALL, (.88, .88, .86))

# ---------- 6 水花：勺子扫过去甩起来的一道弧，冲着红嘴鸥；脚下扑腾的一圈
def splash(dr, r_, SS):
    for _ in range(170):                                   # 勺子甩出来的
        t = r_.random() ** .8
        x = 784 + 80 * t + r_.normal(0, 6 + 10 * t)
        y = 694 - 80 * np.sin(np.pi * t * .85) + r_.normal(0, 4 + 8 * t)
        rr = .8 + 3.2 * r_.random() ** 3
        dr.ellipse([(x - rr) * SS, (y - rr) * SS, (x + rr) * SS, (y + rr) * SS], fill=255)
    for _ in range(90):                                    # 红嘴鸥脚下
        a = r_.uniform(np.pi * 1.05, np.pi * 1.95); d = r_.uniform(8, 42)
        x = 860 + np.cos(a) * d * 1.6; y = 690 + np.sin(a) * d
        rr = .7 + 2.4 * r_.random() ** 3
        dr.ellipse([(x - rr) * SS, (y - rr) * SS, (x + rr) * SS, (y + rr) * SS], fill=255)
spl = ink(splash, 90)
fill(st.layer('水花', mask=spl), ALL, (.97, .96, .92))
wake = (np.abs(np.hypot((xx - 700) / 1.0, (yy - 705) * 4.2) - 75) < 2.2) & (yy < 712) & (xx > 640) & (xx < 770)
fill(st.layer('扫痕', mask=spray(wake, 1, 91, solid=.8)), ALL, (.93, .93, .90))

# ---------- 7 木桩：右下，被框切掉，最近最糙
# 一根老桩：顶是烂开的，往下是竖的木纹，贴水那截糊着一层藤壶；逆光，几乎是剪影，只有顶上一线亮
stump = L([(1106, 802), (1103, 700), (1108, 668), (1116, 676), (1124, 650), (1133, 662), (1141, 646), (1150, 668), (1156, 690), (1160, 802)], 2.2, 92, n=1, scale=4)
fill(st.layer('桩', mask=spray(stump, 2.0, 93)), ALL, (.17, .16, .15))
grain_w = stump.mask & (np.abs(np.sin((xx - 1100) / 3.1 + 2 * nz((30, 2), 96))) < .18) & (yy > 690)
fill(st.layer('木纹', mask=grain_w * .55), ALL, (.27, .25, .23))
barn = stump.mask & (yy > 742 + 6 * nz(6, 97)) & (nz(1.1, 94) > .9)
fill(st.layer('藤壶', mask=barn * .7), ALL, (.44, .42, .38))
sh2 = np.zeros_like(stump.mask); sh2[6:, :-4] = stump.mask[:-6, 4:]
fill(st.layer('桩亮', mask=spray(stump.mask & ~sh2 & (yy < 700), 1, 98, solid=.85)), ALL, (.80, .76, .66))

# ---------- 8 签名：一点点敲，点之间连直线，拐点留一粒墨
def sign(dr, r_, SS):
    ox, oy, s = 1040, 772, 1.0
    strokes = [[(10, 0), (3, -1), (0, 5), (2, 11), (9, 11)],                       # C
               [(13, -4), (14, 11)],                                              # l
               [(23, 4), (18, 4), (16, 9), (19, 11), (23, 8), (23, 3), (24, 11)],  # a
               [(28, 3), (28, 9), (31, 11), (34, 8), (35, 3), (35, 11)],          # u
               [(44, -4), (44, 11), (42, 4), (38, 5), (38, 10), (42, 11), (44, 8)],# d
               [(48, 7), (54, 6), (52, 3), (48, 4), (48, 10), (53, 11)]]          # e
    for pts in strokes:
        p = [((ox + x * s + r_.normal(0, .25)) * SS, (oy + y * s + r_.normal(0, .25)) * SS) for x, y in pts]
        dr.line(p, fill=255, width=int(1.3 * SS))
        for q in p:
            dr.ellipse([q[0] - 1.1 * SS, q[1] - 1.1 * SS, q[0] + 1.1 * SS, q[1] + 1.1 * SS], fill=255)
fill(st.layer('签', mask=ink(sign, 95) * .85), ALL, (.36, .34, .36))

# ---------- 9 纸：整张同一张纹
pg = np.random.default_rng(77)
tooth = gaussian_filter(pg.standard_normal((H, W)), .7); tooth /= tooth.std()
fiber = gaussian_filter(pg.standard_normal((H, W)), (.6, 3)); fiber /= fiber.std()
paper = 1 - .06 * np.clip(tooth, 0, None) - .025 * fiber
fill(st.layer('纸', mode='multiply'), ALL, np.repeat(np.clip(paper, 0, 1)[..., None], 3, 2), edge=HARD)
look(st, OUT)
print(OUT)
