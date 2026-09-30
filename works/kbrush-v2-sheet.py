# kbrush v2 practice sheet · 2026-09-29 night · ten squares, one stroke or one transition each
# Needs kbrush_v2.py next to it. About five minutes to run. — Claude, for nerolette
#
"""kbrush 第二张下笔练习：十格，每一格一种下笔（或一种过渡）。
每格的笔、蘸法、手势都写在下面；图片上的字是事后用 PIL 写的。"""
import numpy as np, time, sys, os
from PIL import Image, ImageDraw, ImageFont
import kbrush_v2 as K

T = time.time()
PW, PH = 700, 300
cv = K.Canvas(K.Paper(PH * 5, PW * 2, seed=21))
SLATE = {'ultramarine': .35, 'indigo': .5, 'burnt_sienna': .15}
INK = {'indigo': .7, 'burnt_umber': .3}

def box(r, c):
    return r * PH, c * PW

def bb(r, c):
    y0, x0 = box(r, c)
    return (y0 + 2, y0 + PH - 2, x0 + 2, x0 + PW - 2)

def go(b, g, **kw):
    K.stroke(cv, b, **g, **kw)

def log(s):
    print(f'{s} {time.time() - T:.0f}s', flush=True)

# ---------- 第一层 ----------
# 满蘸慢拖
y0, x0 = box(0, 0)
b = K.Brush('flat', 64, seed=1); b.dip(cv, SLATE, paint=1.25, water=0.95)
go(b, K.gesture((x0 + 50, y0 + 165), (x0 + 650, y0 + 140), width=64, bend=0.04, start='lu', end='chu', pressure=0.85, speed=0.6, seed=1), seed=1)
b = K.Brush('flat', 40, seed=11); b.dip(cv, {'burnt_sienna': .6, 'ultramarine': .4}, paint=1.2, water=0.95)
go(b, K.gesture((x0 + 110, y0 + 245), (x0 + 520, y0 + 238), width=40, bend=-0.05, start='lu', end='ti', pressure=0.8, speed=0.6, seed=11), seed=11)
cv.flow(250, bbox=bb(0, 0)); log('满蘸慢拖')

# 蘸少快扫 → 飞白
y0, x0 = box(0, 1)
b = K.Brush('flat', 64, seed=2); b.dip(cv, SLATE, paint=1.3, water=0.2)
go(b, K.gesture((x0 + 50, y0 + 195), (x0 + 650, y0 + 135), width=64, bend=-0.06, start='lu', end='chu', pressure=0.75, speed=2.0, seed=2), seed=2)
b = K.Brush('flat', 40, seed=12); b.dip(cv, {'burnt_umber': .7, 'indigo': .3}, paint=1.3, water=0.15)
go(b, K.gesture((x0 + 90, y0 + 255), (x0 + 520, y0 + 232), width=40, bend=0.05, start='lu', end='chu', pressure=0.7, speed=2.2, seed=12), seed=12)
log('飞白')

# 拖到没水
y0, x0 = box(1, 0)
b = K.Brush('flat', 60, seed=3); b.dip(cv, {'ultramarine': .5, 'burnt_sienna': .2, 'indigo': .3}, paint=1.3, water=0.4)
go(b, K.gesture((x0 + 40, y0 + 150), (x0 + 670, y0 + 180), width=60, bend=0.05, start='lu', end='chu', pressure=0.9, speed=0.55, seed=3), seed=3)
cv.flow(250, bbox=bb(1, 0)); log('拖到没水')

# 提按（中锋，S 形）
y0, x0 = box(1, 1)
b = K.Brush('round', 56, seed=4); b.dip(cv, INK, paint=1.2, water=0.9)
g = K.gesture((x0 + 60, y0 + 190), (x0 + 640, y0 + 120), width=56, start='dun', end='chu', pressure=0.95, speed=0.9, seed=4,
              via=[(x0 + 220, y0 + 105), (x0 + 430, y0 + 235)])
P0 = g['pressure']
g['pressure'] = lambda t: P0(t) * np.clip(0.18 + 0.82 * (np.exp(-((t - 0.08) / 0.2) ** 2) + np.exp(-((t - 0.76) / 0.13) ** 2)), 0.14, 1)
go(b, g, seed=4)
cv.flow(200, bbox=bb(1, 1)); log('提按')

# 湿：清水笔从右往左刷，越往左水越少；颜料从干的那头拖进湿的那头
y0, x0 = box(2, 0)
wb = K.Brush('flat', 110, seed=40)
for k, (a, c) in enumerate([((x0 + 690, y0 + 140), (x0 + 330, y0 + 135)), ((x0 + 690, y0 + 225), (x0 + 270, y0 + 232))]):
    wb.dip(cv, {}, paint=0, water=1.0)
    go(wb, K.gesture(a, c, width=110, bend=0.03 * (-1) ** k, start='dun', end='chu', pressure=0.9, speed=0.5, seed=40 + k), seed=40 + k)
cv.flow(15, bbox=bb(2, 0))
b = K.Brush('round', 36, seed=5); b.dip(cv, {'rose': .6, 'burnt_sienna': .4}, paint=1.3, water=0.9)
go(b, K.gesture((x0 + 40, y0 + 135), (x0 + 660, y0 + 125), width=36, bend=0.03, start='lu', end='chu', pressure=0.85, speed=1.0, seed=5), seed=5)
b = K.Brush('flat', 44, seed=6); b.dip(cv, {'ultramarine': .6, 'indigo': .4}, paint=1.2, water=0.8)
go(b, K.gesture((x0 + 40, y0 + 232), (x0 + 660, y0 + 222), width=44, bend=-0.02, start='lu', end='chu', pressure=0.8, speed=1.0, seed=6), seed=6)
cv.flow(300, bbox=bb(2, 0)); log('湿')

# 脏笔：干净的蓝拖过没干的赭石
y0, x0 = box(2, 1)
b = K.Brush('round', 60, seed=7); b.dip(cv, {'burnt_sienna': 1}, paint=1.4, water=1.0)
for k, (a, c) in enumerate([((x0 + 205, y0 + 92), (x0 + 245, y0 + 268)), ((x0 + 262, y0 + 110), (x0 + 215, y0 + 255)), ((x0 + 228, y0 + 130), (x0 + 280, y0 + 235))]):
    go(b, K.gesture(a, c, width=60, bend=0.12 * (-1) ** k, start='lu', end='ti', pressure=0.9, speed=0.6, seed=70 + k), seed=70 + k)
cv.flow(10, bbox=bb(2, 1))
b = K.Brush('flat', 50, seed=8); b.dip(cv, {'ultramarine': 1}, paint=1.2, water=0.8)
go(b, K.gesture((x0 + 40, y0 + 148), (x0 + 670, y0 + 155), width=50, bend=-0.01, start='lu', end='chu', pressure=0.85, speed=1.0, seed=8), seed=8)
b = K.Brush('round', 40, seed=9); b.dip(cv, {'ultramarine': 1}, paint=1.2, water=0.8)
go(b, K.gesture((x0 + 40, y0 + 228), (x0 + 670, y0 + 232), width=40, bend=0.02, start='lu', end='chu', pressure=0.8, speed=1.0, seed=9), seed=9)
cv.flow(200, bbox=bb(2, 1)); log('脏笔')

# 罩染：第一层赭黄
y0, x0 = box(3, 0)
b = K.Brush('flat', 90, seed=10); b.dip(cv, {'ochre': 1}, paint=0.9, water=0.95)
go(b, K.gesture((x0 + 60, y0 + 140), (x0 + 640, y0 + 105), width=90, bend=0.05, start='lu', end='chu', pressure=0.85, speed=0.7, seed=10), seed=10)
b.dip(cv, {'ochre': 1}, paint=0.9, water=0.95)
go(b, K.gesture((x0 + 125, y0 + 212), (x0 + 600, y0 + 226), width=90, bend=-0.06, start='lu', end='chu', pressure=0.85, speed=0.7, seed=13), seed=13)
cv.flow(250, bbox=bb(3, 0)); log('罩染1')

# 擦染：第一层暗底
y0, x0 = box(3, 1)
b = K.Brush('flat', 90, seed=14); b.dip(cv, {'indigo': .6, 'burnt_umber': .4}, paint=1.4, water=0.95)
go(b, K.gesture((x0 + 50, y0 + 125), (x0 + 650, y0 + 120), width=90, bend=0.02, start='lu', end='chu', pressure=0.9, speed=0.6, seed=14), seed=14)
b.dip(cv, {'indigo': .6, 'burnt_umber': .4}, paint=1.4, water=0.95)
go(b, K.gesture((x0 + 60, y0 + 205), (x0 + 640, y0 + 212), width=90, bend=-0.02, start='lu', end='chu', pressure=0.9, speed=0.6, seed=15), seed=15)
cv.flow(250, bbox=bb(3, 1)); log('擦染1')

# 笔尖蘸色：中锋、侧锋、点花
y0, x0 = box(4, 0)
def tipbrush(seed, w):
    b = K.Brush('round', w, seed=seed)
    b.dip(cv, {'rose': .85, 'ochre': .15}, paint=0.45, water=0.85)
    b.dip_tip(cv, {'rose': .7, 'indigo': .3}, paint=2.4, depth=0.4)
    return b
go(tipbrush(16, 44), K.gesture((x0 + 40, y0 + 105), (x0 + 380, y0 + 95), width=44, bend=0.05, start='lu', end='chu', pressure=0.8, speed=1.0, seed=16), seed=16)
go(tipbrush(17, 44), K.gesture((x0 + 40, y0 + 215), (x0 + 380, y0 + 200), width=44, bend=-0.05, start='dun', end='chu', pressure=0.9, speed=0.9, seed=17), side=0.9, seed=17)
rng = np.random.default_rng(3)
flowers = [(x0 + 480, y0 + 105), (x0 + 610, y0 + 175), (x0 + 490, y0 + 240)]
for f, (cx, cy) in enumerate(flowers):
    a0 = rng.uniform(0, 2 * np.pi)
    for k in range(5):
        a = a0 + k * 2 * np.pi / 5 + rng.normal(0, 0.12)
        p0 = (cx + 14 * np.cos(a), cy + 14 * np.sin(a)); p1 = (cx + 26 * np.cos(a + 0.15), cy + 26 * np.sin(a + 0.15))
        go(tipbrush(30 + 5 * f + k, 28), K.gesture(p0, p1, width=28, start='dun', end='ti', pressure=0.95, speed=0.35, seed=20 + k, wobble=0.2), seed=20 + k)
cv.flow(200, bbox=bb(4, 0)); log('笔尖蘸色')

# 起笔收笔：四种（圆头），加一条平头笔手腕不转的 S
y0, x0 = box(4, 1)
for k, (st, en) in enumerate([('lu', 'chu'), ('cang', 'hui'), ('dun', 'dun'), ('lu', 'ti')]):
    b = K.Brush('round', 40, seed=50 + k); b.dip(cv, INK, paint=1.2, water=0.85)
    xa = x0 + 40 + k * 165
    go(b, K.gesture((xa, y0 + 125), (xa + 125, y0 + 118), width=40, bend=0.06, start=st, end=en, pressure=0.85, speed=0.9, seed=50 + k), seed=50 + k)
b = K.Brush('flat', 34, seed=60); b.dip(cv, INK, paint=1.1, water=0.85)
go(b, K.gesture((x0 + 60, y0 + 245), (x0 + 640, y0 + 240), width=34, start='lu', end='chu', pressure=0.85, speed=0.9, seed=60,
                via=[(x0 + 230, y0 + 195), (x0 + 430, y0 + 285)]), hold=60, seed=60)
cv.flow(200, bbox=bb(4, 1)); log('起笔收笔')
cv.dry()

# ---------- 第二层 ----------
# 罩染：干透后一层透明玫瑰斜着过去
y0, x0 = box(3, 0)
b = K.Brush('flat', 70, seed=18); b.dip(cv, {'rose': 1}, paint=0.8, water=0.95)
go(b, K.gesture((x0 + 150, y0 + 45), (x0 + 330, y0 + 285), width=70, bend=0.08, start='lu', end='chu', pressure=0.85, speed=0.7, seed=18), seed=18)
cv.flow(250, bbox=bb(3, 0))
# 擦染：干笔带不透明的浅色在暗底上蹭，来回，越蹭越稀
y0, x0 = box(3, 1)
b = K.Brush('flat', 50, seed=19); b.dip(cv, {'white': .7, 'ochre': .3}, paint=1.6, water=0.15)
go(b, K.gesture((x0 + 70, y0 + 170), (x0 + 650, y0 + 150), width=50, start='lu', end='chu', pressure=0.8, speed=1.4, seed=19, wobble=1.5,
                via=[(x0 + 190, y0 + 105), (x0 + 300, y0 + 215), (x0 + 420, y0 + 115), (x0 + 530, y0 + 205)]), seed=19)
# 花枝和花蕊（花干了才画）
y0, x0 = box(4, 0)
b = K.Brush('round', 12, seed=61); b.dip(cv, {'burnt_umber': .6, 'indigo': .4}, paint=1.5, water=0.55)
go(b, K.gesture((x0 + 420, y0 + 292), (x0 + 670, y0 + 70), width=12, start='dun', end='chu', pressure=0.85, speed=0.8, seed=61, wobble=1.5,
                via=[(x0 + 470, y0 + 250), (x0 + 520, y0 + 205), (x0 + 600, y0 + 150)]), seed=61)
cv.flow(120, bbox=bb(4, 0))
# 枝子从花后面过：画的人会绕开花，这里等价于把落在花瓣上的那部分去掉
by0, bx0 = box(4, 0)
petal = np.zeros(cv.w.shape, bool)
petal[by0:by0 + PH, bx0:bx0 + PW] = cv.layers[0][:, by0:by0 + PH, bx0:bx0 + PW].sum(0) > 0.03
cv.D[:, petal] = 0; cv.Sus[:, petal] = 0; cv.w[petal] = 0
for f, (cx, cy) in enumerate(flowers):
    for j in range(6):
        b = K.Brush('round', 8, seed=100 + 10 * f + j); b.dip(cv, {'burnt_umber': .5, 'indigo': .5}, paint=2.6, water=0.55)
        a = rng.uniform(0, 2 * np.pi); r = rng.uniform(5, 11)
        p = (cx + r * np.cos(a), cy + r * np.sin(a))
        go(b, K.gesture(p, (p[0] + 2, p[1] + 1), width=8, start='dun', end='ti', pressure=0.6, speed=0.3, seed=200 + j, wobble=0), seed=j)
cv.flow(100, bbox=bb(4, 0))
cv.dry(); log('第二层')

# ---------- 第三层 ----------
y0, x0 = box(3, 0)
b = K.Brush('flat', 70, seed=22); b.dip(cv, {'ultramarine': 1}, paint=0.7, water=0.95)
go(b, K.gesture((x0 + 380, y0 + 45), (x0 + 560, y0 + 285), width=70, bend=-0.08, start='lu', end='chu', pressure=0.85, speed=0.7, seed=22), seed=22)
cv.flow(250, bbox=bb(3, 0))
cv.dry(); log('第三层')

img = Image.fromarray((cv.render() * 255).astype(np.uint8))
img.save('sheet2_raw.png')

# ---------- 字 ----------
FONT = next(p for p in ['NotoSansCJKsc-Regular.otf', '/System/Library/Fonts/STHeiti Medium.ttc',
                        '/System/Library/Fonts/PingFang.ttc'] if os.path.exists(p))
F1 = ImageFont.truetype(FONT, 26)
F2 = ImageFont.truetype(FONT, 16)
F3 = ImageFont.truetype(FONT, 14)
d = ImageDraw.Draw(img)
LAB = {
    (0, 0): ('满蘸慢拖', '水多：平涂里留着鬃的细纹，拖到尾巴才散开'),
    (0, 1): ('蘸少快扫 → 飞白', '只有纸纹的顶挂得住颜料，越扫越稀，最后没了'),
    (1, 0): ('拖到没水', '同一笔：平涂 → 分叉 → 碎点 → 消失'),
    (1, 1): ('提按', '中锋：按下去粗，提起来细，再按'),
    (2, 0): ('湿', '清水先刷右边（越往左越干）：边从硬到软，湿亮处洇开'),
    (2, 1): ('脏笔', '干净的蓝拖过没干的赭石，带走一些，一路慢慢还回去'),
    (3, 0): ('罩染', '干透再上一层透明色：黄上玫瑰成橙，黄上蓝成绿'),
    (3, 1): ('擦染', '干笔带不透明的浅色在暗底上来回蹭，越蹭越稀'),
    (4, 0): ('笔尖蘸色', '浅色打底、笔尖点深：中锋中间深，侧锋一边深，点花瓣根深'),
    (4, 1): ('起笔收笔', '露锋出锋 · 藏锋回锋 · 顿笔顿收 · 露锋提笔；下：平头笔手腕不转'),
}
ink = (58, 54, 50); grey = (120, 112, 104)
for (r, c), (t, s) in LAB.items():
    y0, x0 = box(r, c)
    d.text((x0 + 22, y0 + 12), t, font=F1, fill=ink)
    tw = d.textlength(t, font=F1)
    d.text((x0 + 34 + tw, y0 + 22), s, font=F2, fill=grey)
y0, x0 = box(4, 1)
for k, s in enumerate(['露锋 → 出锋', '藏锋 → 回锋', '顿 → 顿收', '露锋 → 提']):
    d.text((x0 + 58 + k * 165, y0 + 160), s, font=F3, fill=grey)
for r in range(1, 5):
    d.line([(24, r * PH), (PW * 2 - 24, r * PH)], fill=(214, 206, 192), width=1)
d.line([(PW, 18), (PW, PH * 5 - 18)], fill=(214, 206, 192), width=1)
img.save('kbrush_sheet2.png'); log('done')
