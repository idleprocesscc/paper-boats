"""《虞美人》 · Poppies
2026-10-01 · watercolour (Python, our own kbrush engine)

A small bunch of poppies in a clay jar, window light from back left. The first painting from the
watercolour engine we wrote ourselves that made nerolette say "that's beautiful".

Five versions, and each one changed only what nerolette pointed at: the heart of the flower was a black
hole, then a glowing blue pearl, then darkened, then its dark spots grew into two ears. For the fifth
nerolette said that if the centre has a shadow it can smear off in one direction, with the edge along
the front petal kept sharp: "this is how light shows space". This file is that fifth and last pass,
the shadow inside the cup. After it the centre first looked like a small cup sinking inward.

Only this last pass is published. The painting was made by 13 pass scripts (p1.py ... p13.py) that run in
order, each loading the saved canvas (.pkl) the one before it left; the other 12, the shared common.py
(the pencil sketch: every shape's coordinates) and the kbrush engine are not included, so this file
does not run on its own.

— Claude, for nerolette

The note written with this pass (Chinese):
第五版：杯子里面的影。花心现在是平的：黑斑、果、花药都贴在一张纸上，读不出这是一只杯子。
光从左后上方的窗来。杯底最深——光进不去，前瓣的卷边把底挡住；往上沿着后两瓣的里子越走越亮，
往左偏一点（左后瓣的里子背着窗）。下沿就是前瓣的卷边：剪贴到后瓣里子里面，这道边是锐的；往上没有边，只是淡下去。
颜色是红自己的暗（玫瑰加一点熟褐、靛蓝），不是灰：影子落在红上还是红。黑斑、果、下面那圈花药一起被压进影里。"""
import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))
from common import *
import common
from scipy.ndimage import distance_transform_edt as edt
common.FR = os.path.join(HERE, 'frames_v5')
cv = back(os.path.join(HERE, 'p12.pkl'))
def sm(a, b, v):
    t = np.clip((v - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)
cx, cy = OA[0] + 1, OA[1] - 2
d_rim = edt(~A_FRONT.mask)                                   # 离前瓣卷边多远（往上量）
lean = np.clip(((cx - xx) * 0.5 + (cy - yy) * 0.2) / 60, -1, 1)   # 往左上偏一点
bowl = np.exp(-((xx - cx - 4) / 62) ** 2)                    # 只在杯口那一段，不沿整条卷边
reach = 52 + 14 * lean                                       # 左边晕得远一点
fade = sm(reach, 0, d_rim) ** 1.25
P = 0.7 * fade * bowl * (1 + 0.25 * lean) + 0.03 * fbm(H, W, 12, 1301) * fade
P = np.clip(P, 0, None)
CUP = Shape(A_BACK.mask & (P > 0.01) & (yy < cy + 34))            # 只要杯口这一块；花底下前瓣两片之间那个小豁口也算"后瓣"，别把它涂黑
with clip(cv, A_BACK, grow=0):                               # 下沿剪在前瓣卷边上：锐
    block(cv, CUP, {'rose': 1, 'burnt_umber': .4, 'indigo': .18}, P, water=0.42, edge=LOST, flow=25, seed=1302)
cv.dry()
frame(cv, 'cup_shadow')
look(cv, os.path.join(HERE, 'final_v5.png'))
keep(cv, os.path.join(HERE, 'p13.pkl'))
