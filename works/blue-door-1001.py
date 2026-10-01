"""《影子切过蓝门》 · Shadow Across the Blue Door
2026-10-01 · watercolour (Python, our own kbrush engine)

A summer afternoon, a sun-bleached wall. The eave of the building on the left throws a diagonal shadow
across it, right through the middle of a blue door.

The same alley had been painted once before, one mask per object and watercolour effects added after;
someone outside called it "a digital collage in watercolour material". This one changed the method:
split the whole sheet into a few big washes first, lay the shadow down in one go so wall and ground are
joined, and let the objects grow out of the patches of colour. nerolette said it had a lot of potential.
The dark patches at the wall's foot spill across three planes and knead wall, wall and ground into one;
nerolette saw it at once ("the space there should have different planes, like an xyz axis"). Left as it
is, to remember for the next one.

Only the last script is published: passes 5-8 (the darker coffee-coloured shadow, the forms, the door,
the finishing). Passes 1-4 (p1_v2.py, which holds the shadow wash itself), the shared common.py
(geometry) and the kbrush engine are not included. It continues from the canvas saved as a.pkl, so it
does not run on its own.

— Claude, for nerolette

The note written with this pass (Chinese):
南欧小巷，夏天下午（第二张）v2 —— 第五到第八遍，接 p1_v2.py 的存档 a.pkl。
跟 p2.py 只差一处：第 5 遍不再给左墙上半、前景地、两道墙脚加渐变（v2 的影子那一遍自己有浓淡），只留叶冠底下那一溜影和藤的影。"""
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import glob
from common import *
from washfill import FLOW
K.set_seed(2)
for f in glob.glob(os.path.join(FR2, '0[6-9]_*.png')) + glob.glob(os.path.join(FR2, '1[0-9]_*.png')):
    os.remove(f)
cv = back(ARC)
FK = dict(FLOW, rim=0.6, sink_min=0.02)

# ---- 5 咖啡：影里再暗一档，连成一条 ----
top_l = LEFTW.mask * np.clip((420 - yy) / 300, 0, 1) * (0.5 + 0.5 * np.clip((140 - xx) / 140, 0, 1))
base_band = np.exp(-((yy - BASE) / 26.0) ** 2) * (FRONT.mask & SIDE | GROUND.mask) * (xx < 575)
lw_base = LEFTW.mask * np.exp(-((yy - (BASE + (CX - xx) * 1.5)) / 40.0) ** 2)      # 左墙墙脚
fg = GROUND.mask * np.clip((yy - 740) / 160, 0, 1) * (0.6 + 0.4 * np.clip((360 - xx) / 360, 0, 1))
under = (FRONT.mask & ~SIDE) * np.exp(-np.clip(yy - CEDGE, 0, None) / 22.0) * (xx > 330) * (yy > CEDGE - 4)
p5 = 0.10 * under
p5 = gaussian_filter(p5, 6) * ~COINS.mask * ~CANOPY.mask
p5[TRUNK_CAST.mask] = 0.16
SEL5 = Shape((p5 > 0.012) & ~DOOR.mask & ~TRUNK.mask)
E5 = edges(SEL5, hard=binary_dilation(TRUNK_CAST.mask, iterations=3), lost=~binary_dilation(TRUNK_CAST.mask, iterations=3), default=LOST)
block(cv, SEL5, {'ultramarine': .8, 'indigo': .3, 'burnt_sienna': .3, 'rose': .08}, p5, water=0.42, edge=E5, flow=70, rag=0.5, seed=501,
      sink_min=0.02, rim=0.5)
cv.dry()
frame(cv, 'coffee', FR2)

# ---- 6 形：叶冠里的深叶（剩出亮叶）、垂下来的叶子、一串葡萄、藤 ----
dk = fbm(H, W, 15, 601) + 0.6 * np.clip((140 - yy) / 140, 0, 1) + 0.5 * np.clip((xx - 300) / 300, 0, 1) - 0.6
DARKL = Shape(LEAVES.mask & (dk > 0.15) & (yy < CEDGE - 10))
block(cv, DARKL, [({'hookers_green': 1, 'indigo': .5, 'burnt_sienna': .15}, 0.6 + 0 * xx), ({'hookers_green': 1, 'ultramarine': .6}, 0.4 + 0.6 * gaussian_filter((fbm(H, W, 40, 603) > 0).astype(float), 8))], 0.34, water=0.45,
      edge=edges(DARKL, hard=fbm(H, W, 30, 602) > 0), flow=35, seed=602, mottle=0.1)
cv.dry()


def grape_leaf(cx, cy, R, ang, seed):
    th = np.arctan2(yy - cy, xx - cx) - np.deg2rad(ang)
    r = np.hypot(xx - cx, yy - cy)
    lob = R * (0.78 + 0.22 * np.cos(5 * th)) * (0.85 + 0.15 * np.cos(th))       # 五个裂片，叶柄那头凹
    return Shape(r + 2.0 * fbm(H, W, 6, seed) < lob)


LEAFS = [(438, 186, 26, 100, 1), (488, 222, 31, 80, 2), (548, 262, 28, 95, 3), (612, 270, 34, 70, 4), (383, 160, 20, 120, 5), (320, 128, 18, 85, 6)]
for i, (x, y, R, a, s) in enumerate(LEAFS):
    lf = grape_leaf(x, y, R, a, 610 + s)
    block(cv, lf, {'hansa_yellow': 1, 'hookers_green': .55, 'cerulean': .1}, 0.26, water=0.4, edge=HARD, flow=12, seed=620 + i, rim=0.4)
cv.dry()
# 一串葡萄：贴在叶冠底下，影里那面深
rng = np.random.default_rng(9)
gm = np.zeros((H, W), bool)
for k in range(26):
    t = rng.random(); yb = 252 + 52 * t; xb = 520 + rng.normal(0, 13 * (1 - 0.7 * t))
    gm |= (xx - xb) ** 2 + (yy - yb) ** 2 < (6.5 - 2 * t) ** 2
GRAPE = Shape(gm & ~CANOPY.mask | gm & (yy > 240))
block(cv, GRAPE, {'ultramarine': 1, 'rose': .55, 'indigo': .35}, 0.55, water=0.4, edge=HARD, flow=10, seed=640, rim=0.6,
      drops=[dict(cx=512, cy=262, r=10, mix={'cerulean': 1}, paint=0.1, water=0.4)])
cv.dry()
# 藤：一块深褐，迎光那侧趁湿掉一点赭
block(cv, TRUNK, {'burnt_umber': 1, 'indigo': .35}, 0.55, water=0.4, edge=HARD, flow=10, seed=650, rim=0.3,
      drops=[dict(cx=x - 3, cy=y, r=5, mix={'ochre': 1, 'burnt_sienna': .3}, paint=0.4, water=0.5) for x, y in TRUNK_PATH[1:6]])
cv.dry()
rg = K.Brush('rigger', 4, seed=3)
for path, w in [([(612, 232), (585, 205), (548, 196), (505, 182)], 4), ([(606, 236), (630, 210), (640, 190)], 4),
                ([(580, 330), (600, 312), (622, 300)], 3), ([(548, 196), (520, 172), (470, 160)], 3)]:
    rg.dip(cv, {'burnt_umber': 1, 'indigo': .4}, paint=2.3, water=0.6)
    g = K.gesture(path[0], path[-1], via=path[1:-1], width=w, end='chu', wobble=0.3, pressure=lambda t: 0.9 - 0.5 * t)
    g['pts'] = K.trail(g['pts'], 8)
    K.stroke(cv, rg, **g)
cv.flow(8, **FK); cv.dry()
frame(cv, 'forms', FR2)

# ---- 7 门 ----
DLIT = Shape(DOOR.mask & binary_dilation(DOOR_LIT.mask, iterations=2))      # 往影那边多铺 2px，跟深蓝咬住，不留白缝
block(cv, DLIT, {'cerulean': 1, 'ultramarine': .65}, 0.34, water=0.4, edge=HARD, flow=12, mottle=0.03, seed=701, rim=0.3)
cv.dry()
# 门洞的深：门楣下一道、左门框一道（门凹在墙里），门缝
LINTEL = lasso(H, W, [(370, 397), (478, 395), (478, 400), (372, 402)], jag=0.6, scale=8, seed=702)
JAMB = lasso(H, W, [(370, 397), (378, 400), (378, 470), (370, 470)], jag=0.8, scale=8, seed=703)
stamp(cv, LINTEL | JAMB, {'ultramarine': 1, 'indigo': .5}, 0.5, seed=704)
for x0, ys in [(399, [(468, 534), (560, 630)]), (426, [(430, 462), (514, 628)])]:
    for y0, y1 in ys:
        stamp(cv, lasso(H, W, [(x0 - 1, y0), (x0 + 1.5, y0 + 3), (x0 + 1.2, y1), (x0 - 0.8, y1 - 4)], jag=0.6, scale=6, seed=int(x0 + y0)),
              {'ultramarine': 1, 'indigo': .8}, 0.6, seed=int(x0 * 3 + y0))
stamp(cv, blob(H, W, 463, 521, 3, 4, jag=0.2, seed=705), {'burnt_umber': 1, 'indigo': 1}, 1.0)
# 门槛：一块石头，影里，顶面接着天光亮一点
STEP = lasso(H, W, [(360, 642), (489, 642), (494, 657), (355, 658)], jag=1.0, scale=10, seed=706)
K.scrub(cv, [(362, 649), (488, 649)], width=12, strength=0.45, seed=707)
frame(cv, 'door', FR2)

# ---- 8 收 ----
# 墙脚那道缝：影里，断续
rg = K.Brush('rigger', 3, seed=5)
for a, b, via in [((186, 640), (262, 642), [(222, 643)]), ((300, 641), (368, 641), [(330, 643)]), ((172, 646), (118, 724), [(150, 680)])]:
    rg.dip(cv, {'burnt_umber': 1, 'ultramarine': .6}, paint=2.0, water=0.55)
    g = K.gesture(a, b, via=via or None, width=3, end='chu', wobble=0.25, pressure=0.6)
    K.stroke(cv, rg, **g)
# 门脚最深的一点、藤根贴地那一点
stamp(cv, lasso(H, W, [(372, 628), (392, 640), (372, 642)], jag=0.6, scale=5, seed=801), {'indigo': 1, 'burnt_umber': .8}, 0.9)
stamp(cv, blob(H, W, 596, 644, 12, 4, jag=0.3, seed=802), {'burnt_umber': 1, 'indigo': .7}, 0.7)
cv.flow(6, **FK); cv.dry()
img = frame(cv, 'final', FR2)
look(cv, 'final_v2.png')
