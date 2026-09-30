# kbrush v1 practice sheet · 2026-09-29 · seven squares, one way of putting the brush down each
# Needs kbrush_v1.py next to it. — Claude, for nerolette
#
"""同一支笔，七种下笔。每一行只改了蘸多少、压多重、走多快、纸湿不湿、下面是什么——笔是同一个。"""
import numpy as np, sys, time
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from kbrush_v1 import Paper, Canvas, Brush, stroke
OUT = sys.argv[1] if len(sys.argv) > 1 else 'kbrush_sheet.png'
H, W = 1330, 1400
cv = Canvas(Paper(H, W, seed=3))
SLATE = {'ultramarine': .35, 'indigo': .5, 'burnt_sienna': .15}
line = lambda y, x0=120, x1=1280: [(x0, y), (x1, y)]
def B(seed):
    return Brush(70, 90, seed=seed)   # 每行都是同一型号的笔，只是新拿一支
bump = lambda t, c, s: np.exp(-((t - c) / s) ** 2)

# 1 满蘸 · 慢拖
b = B(1); b.dip(cv, SLATE, paint=1.2, water=1.0)
stroke(cv, b, line(110), pressure=0.85, speed=0.7, seed=1)
# 2 蘸少 · 快扫
b = B(2); b.dip(cv, SLATE, paint=0.6, water=0.25)
stroke(cv, b, line(250), pressure=0.7, speed=2.2, seed=2)
# 3 提按：起笔按、中间提、收笔一顿
b = B(3); b.dip(cv, SLATE, paint=1.0, water=0.8)
pr = lambda t: np.where(t < 0.6, np.clip(1 - 3.2 * t, 0.1, 1), 0.1 + 0.9 * bump(t, 0.8, 0.08))
stroke(cv, b, line(390), pressure=pr, speed=lambda t: 1.0 - 0.6 * bump(t, 0.8, 0.06), seed=3)
# 4 同一笔，右半边纸是湿的
m = np.zeros((H, W)); m[470:610, 640:1300] = 1
cv.wet(gaussian_filter(m, 4) > 0.5, 0.7)
b = B(4); b.dip(cv, SLATE, paint=1.0, water=0.8)
stroke(cv, b, line(530), pressure=0.8, speed=1.0, seed=4)
# 5 脏笔：左边一段没干的赭红，蓝笔从它身上穿过去，把赭带走一路
b = B(5); b.dip(cv, {'burnt_sienna': 1}, paint=1.3, water=1.0)
stroke(cv, b, [(120, 670), (470, 670)], pressure=0.9, speed=0.8, seed=5)
cv.flow(15)
b = B(7); b.dip(cv, SLATE, paint=0.7, water=0.5)
stroke(cv, b, [(300, 670), (1280, 670)], pressure=0.85, speed=1.0, pickup=0.35, seed=7)
cv.flow(160)
cv.dry()
# 6 罩染 / 7 擦染 的底色
b = Brush(110, 120, seed=8)
for k, y in enumerate(range(900, 1120, 40)):
    if k % 2 == 0: b.dip(cv, {'ochre': .8, 'burnt_sienna': .2}, paint=1.0, water=1.0)   # 刷两道回去蘸一次
    stroke(cv, b, [(120, y), (640, y)] if k % 2 == 0 else [(640, y), (120, y)], pressure=0.9, speed=1, seed=20 + k)
b = Brush(110, 120, seed=9)
for k, y in enumerate(range(900, 1120, 40)):
    if k % 2 == 0: b.dip(cv, {'indigo': .7, 'burnt_umber': .3}, paint=1.6, water=1.0)
    stroke(cv, b, [(760, y), (1280, y)] if k % 2 == 0 else [(1280, y), (760, y)], pressure=0.9, speed=1, seed=30 + k)
cv.flow(160)
cv.dry()
# 6 罩染：深的透明色（凡戴克棕一样的 umber+indigo，很稀）压在浅上
b = B(10); b.dip(cv, {'burnt_umber': .6, 'indigo': .4}, paint=0.5, water=0.9)
stroke(cv, b, [(150, 960), (610, 960)], pressure=0.9, speed=0.9, seed=10)
b = B(11); b.dip(cv, {'burnt_umber': .6, 'indigo': .4}, paint=0.5, water=0.9)
stroke(cv, b, [(150, 1060), (610, 1060)], pressure=0.9, speed=0.9, seed=11)
# 7 擦染：浅的不透明色（白 + 一点赭）几乎干着蹭在深上
b = B(12); b.dip(cv, {'white': .85, 'ochre': .15}, paint=0.7, water=0.2)
stroke(cv, b, [(790, 960), (1250, 960)], pressure=0.6, speed=2.2, seed=12)
b = B(13); b.dip(cv, {'white': .85, 'ochre': .15}, paint=0.7, water=0.2)
stroke(cv, b, [(790, 1060), (1250, 1060)], pressure=0.6, speed=2.2, seed=13)
cv.flow(100)
cv.dry()
im = Image.fromarray((cv.render() * 255).astype(np.uint8)); d = ImageDraw.Draw(im)
try: f = ImageFont.truetype('/System/Library/Fonts/STHeiti Medium.ttc', 26)
except Exception: f = None
if f:
    ink = (90, 85, 80)
    for y, s in [(110, '满蘸 · 慢拖（颜料够走完一整行，最后几寸笔干了自己开叉）'), (250, '蘸少 · 快扫 → 飞白（干笔只碰得到纸纹的顶）'),
                 (390, '提按：起笔按、中间提、收笔一顿'), (530, '同一笔，右半边纸是湿的'),
                 (670, '脏笔：蓝笔穿过没干的赭，把它带走一小段路')]:
        d.text((120, y - 72), s, fill=ink, font=f)
    d.text((120, 800), '罩染：深的透明色压在浅上', fill=ink, font=f)
    d.text((760, 800), '擦染：浅的不透明色蹭在深上', fill=ink, font=f)
im.save(OUT)
