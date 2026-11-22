# 两桌 · 11.22
# 从哪儿看：趴在堤上，长焦平视，眼睛比鸟高不了多少；视平线在画框上沿外面，整张都是滩涂，越远越挤、越亮。
# 这张在说什么：远处四把饭勺凑成两桌，各自交成一个 X；近处一只大白鹭背对着它们站着，一动不动。
# 光：阴天，整片天就是灯，从正上方来，大、冷、软。没有投影，只有朝下的那面暗一阶（冷灰），贴水的地方最深。
#     湿泥反着天光：越远越亮，近处露出泥自己的暖灰。白鸟比泥亮，最黑的是嘴、脸和腿。
# 明暗：整张挤在亮的半段。最亮 = 鸟的背；最深 = 两个 X 和白鹭的腿；唯一鲜的颜色 = 白鹭的黄嘴。
# 视线：先落在白鹭（最大、对比最强、唯一的竖），顺着它没看的方向去右上找那两个小黑叉。
# 认得出：大白鹭 = 脖子跟身子一样长、拐一个 S；身子是斜 35° 的水滴；嘴是匕首、黄的；腿黑、长，被画框切掉。
#         黑脸琵鹭（趴着理毛）= 身子扁 2.3:1；脖子往前平伸；黑脸连着黑嘴，嘴尖一块勺。
# 质感：台阶与颗粒，不用刷子；整张一张纸纹。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get('KBRUSH_ENGINE_DIR', './kbrush-cloud')  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(22)
from shapes import Shape
from blocks import blob, lasso, line, look, smooth, spray, paper
from layers import Stack, fill, HARD
from scipy.ndimage import gaussian_filter

COMP = os.environ.get('COMP', 'B')
ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', f'rough_{COMP}.png' if ROUGH else 'final.png')
H, W = 600, 900
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
rng = np.random.default_rng(22)
if ROUGH:
    spray = lambda region, sp, r, solid=.96, **_: np.asarray(getattr(region, 'mask', region), bool).astype(float) * solid

def T(pts, x0, y0, s, face=1):
    return [(x0 + face * u * s, y0 + v * s) for u, v in pts]

# ---------- 大白鹭（设计稿朝右，单位 = 半个身长）
E_BODY = [(.58, -.22), (.42, -.46), (.10, -.52), (-.45, -.18), (-1.12, .50), (-.98, .62), (-.55, .60), (-.05, .50), (.42, .28)]
E_NECK = [(.40, -.30), (.74, -.78), (.66, -1.22), (.46, -1.58), (.56, -1.92), (.78, -2.10)]
E_BELLY = [(.50, .02), (.20, .22), (-.35, .34), (-.95, .56), (-.55, .62), (-.05, .52), (.42, .30)]
def egret(x0, y0, s, face):
    body = lasso(H, W, smooth(T(E_BODY, x0, y0, s, face), 3), jag=.6, scale=5, seed=1)
    neck = line(H, W, smooth(T(E_NECK, x0, y0, s, face), 3, closed=False), width=lambda t: s * (.30 - .17 * t), seed=2)
    hx, hy = T([(.84, -2.12)], x0, y0, s, face)[0]
    head = blob(H, W, hx, hy, s * .17, s * .105, ang=-8 * face, jag=.03, seed=3)
    bill = line(H, W, T([(.95, -2.12), (1.30, -2.08), (1.68, -2.02)], x0, y0, s, face), width=lambda t: s * (.10 - .09 * t), seed=4)
    legs = [line(H, W, T(p, x0, y0, s, face), width=lambda t: s * .055, seed=5 + i)
            for i, p in enumerate(([(-.12, .50), (-.20, 1.9), (-.24, 3.2)], [(.14, .46), (.26, 1.55), (.18, 3.2)]))]
    belly = lasso(H, W, smooth(T(E_BELLY, x0, y0, s, face), 3), jag=.6, scale=5, seed=7)
    eye = blob(H, W, *T([(.90, -2.14)], x0, y0, s, face)[0], s * .022, s * .022, seed=8)
    return dict(white=body | neck | head, bill=bill, legs=legs[0] | legs[1], belly=belly & (body), eye=eye,
                neckshade=line(H, W, smooth(T([(u - .07, v + .02) for u, v in E_NECK], x0, y0, s, face), 3, closed=False), width=lambda t: s * (.12 - .06 * t), seed=9))

# ---------- 趴着理毛的黑脸琵鹭（朝右；两只对着摆，嘴在中间交成 X）
def spoon(x0, y0, s, face, seed):
    body = blob(H, W, x0, y0, s * 1.0, s * .43, ang=-4 * face, jag=.05, seed=seed)
    neck = line(H, W, T([(.62, -.10), (1.02, -.50), (1.50, -.78)], x0, y0, s, face), width=lambda t: s * (.36 - .14 * t), seed=seed + 1)
    head = blob(H, W, *T([(1.60, -.82)], x0, y0, s, face)[0], s * .21, s * .17, seed=seed + 2)
    bill = line(H, W, T([(1.70, -.80), (2.10, -.42), (2.50, -.02)], x0, y0, s, face), width=lambda t: s * (.11 - .03 * t), seed=seed + 3)
    tip = blob(H, W, *T([(2.56, .04)], x0, y0, s, face)[0], s * .16, s * .09, ang=44 * face, seed=seed + 4)
    mask = blob(H, W, *T([(1.74, -.80)], x0, y0, s, face)[0], s * .13, s * .11, seed=seed + 5)
    under = Shape(body.mask & (yy > y0 + s * .12))
    return dict(white=body | neck | head, black=bill | tip | mask, under=under, water=(x0, y0 + s * .43))
def pair(cx, cy, s, seed):
    return [spoon(cx - 2.08 * s, cy, s, 1, seed), spoon(cx + 2.08 * s, cy + s * .08, s * .96, -1, seed + 10)]

# ---------- 三个构图
if COMP == 'A':      # 照片那样：两桌在上，白鹭在正下方
    eg = egret(430, 400, 62, 1); prs = pair(300, 170, 26, 30) + pair(640, 180, 25, 60); stakes = [(455, 120, 16), (520, 96, 10)]
elif COMP == 'B':    # 白鹭近、大、压在左边，脸朝画外；两桌小小地在右上
    eg = egret(262, 352, 118, -1); prs = pair(548, 156, 31, 30) + pair(778, 196, 26, 60); stakes = [(655, 112, 16), (712, 92, 9)]
else:                # C：白鹭在右边被框切掉一半，朝右；两桌在左上
    eg = egret(672, 372, 124, 1); prs = pair(200, 160, 26, 30) + pair(430, 132, 21, 60); stakes = [(300, 110, 14), (352, 92, 9)]

# ---------- 颜色（先定的台阶）
MUD = [(.875, .885, .885), (.815, .825, .822), (.745, .750, .740), (.665, .662, .640)]   # 远 → 近：天光的冷灰 → 泥的暖灰
SHEEN = (.915, .925, .928)
WHITE, SHADE, DEEP = (.975, .968, .945), (.80, .82, .85), (.62, .65, .70)
BLACK, YELLOW, LEG = (.12, .12, .14), (.90, .66, .18), (.15, .15, .15)

st = Stack(K.Paper(H, W, seed=3), ground='#dfe2e2')
# 1 滩涂：四阶，阶线按 1/d 挤，线自己缓缓起伏
edge_y = [78, 228, 418]
def wav(seed, amp, sig):
    r = np.random.default_rng(seed); n = gaussian_filter(r.standard_normal(W), sig); return amp * n / n.std()
fill(st.layer('泥0'), ALL, MUD[0], edge=HARD)
for i, y0 in enumerate(edge_y):
    reg = yy > (y0 + wav(40 + i, 2 + 2.5 * i, 110 + 40 * i))[None, :]
    fill(st.layer(f'泥{i+1}', mask=spray(reg, 12 + 14 * i, rng, solid=.97, holes=.03 * i)), ALL, MUD[i + 1], edge=HARD)
# 2 水光：横的亮条，远处细而密，近处长而疏
sheen = np.zeros((H, W), bool)
r2 = np.random.default_rng(5)
for k in range(40):
    d = r2.random() ** 1.5; y = 40 + d * 540; x = r2.uniform(-50, W); ln = (30 + 190 * d) * r2.uniform(.4, 1.3); th = 1.0 + 4 * d
    if x + ln > 470 and y > 300 and r2.random() < .8: continue
    sheen |= line(H, W, [(x, y), (x + ln / 2, y + r2.uniform(-1, 1) * 2 * d), (x + ln, y)], width=lambda t, th=th: th * (1 - (2 * t - 1) ** 2) + .6, seed=100 + k).mask
fill(st.layer('水光', opacity=.6, mask=spray(sheen, 1.5, rng, solid=.9)), ALL, SHEEN, edge=HARD)
# 3 近处的泥疙瘩：越近越大
lumps = np.zeros((H, W), bool)
for k in range(38):
    d = r2.random() ** .8; y = 300 + d * 300; x = r2.uniform(0, W); sz = 3 + 15 * d
    lumps |= blob(H, W, x, y, sz * r2.uniform(1.2, 2.6), sz * .35, jag=.2, seed=200 + k).mask
fill(st.layer('疙瘩', opacity=.55, mask=spray(lumps, 2, rng, solid=.9, holes=.15)), ALL, (.52, .51, .49), edge=HARD)

def bird_white(name, d, sp):
    fill(st.layer(name + '白', mask=spray(d['white'], sp, rng, solid=.99, dens=.5)), ALL, WHITE, edge=HARD)

# 4 远处的桩
for k, (x, y, h) in enumerate(stakes):
    fill(st.layer(f'桩{k}'), line(H, W, [(x, y - h), (x + 1, y)], width=lambda t: 4.2), (.30, .30, .31), edge=HARD)
# 5 两桌
for i, d in enumerate(prs):
    x, y = d['water']
    refl = blob(H, W, x, y + 7, 26, 5, jag=.2, seed=300 + i)
    fill(st.layer(f'倒影{i}', opacity=.6, mask=spray(refl, 3, rng, solid=.8)), ALL, (.93, .935, .93), edge=HARD)
    bird_white(f'勺{i}', d, 1.2)
    fill(st.layer(f'勺{i}腹', mask=spray(d['under'], 2.5, rng, solid=.9)), ALL, SHADE, edge=HARD)
for i, d in enumerate(prs):                                    # 嘴最后画，X 压在身子上面
    fill(st.layer(f'勺{i}黑'), d['black'], BLACK, edge=HARD)
# 6 大白鹭
fill(st.layer('腿'), eg['legs'], LEG, edge=HARD)
bird_white('鹭', eg, 2.0)
fill(st.layer('鹭腹', mask=spray(eg['belly'], 9, rng, solid=.92)), ALL, SHADE, edge=HARD)
fill(st.layer('鹭颈影', mask=spray(Shape(eg['neckshade'].mask & eg['white'].mask), 4, rng, solid=.85)), ALL, SHADE, edge=HARD)
fill(st.layer('鹭嘴'), eg['bill'], YELLOW, edge=HARD)
fill(st.layer('鹭眼'), eg['eye'], BLACK, edge=HARD)
# 签名：一个点一个点敲的
SIG = [[(9,-10),(4,-12),(0,-7),(1,-2),(5,0),(9,-2)], [(13,-13),(13,0)], [(22,-6),(18,-7),(16,-3),(18,0),(22,-2),(22,-7),(22,0)],
       [(26,-7),(26,-2),(28,0),(31,-2),(31,-7),(31,0)], [(40,-6),(36,-7),(34,-3),(36,0),(40,-2),(40,-13),(40,0)], [(44,-4),(49,-4),(48,-7),(45,-7),(43,-3),(45,0),(49,-1)]]
sig = np.zeros((H, W), bool)
for k, L_ in enumerate(SIG):
    pts = [(818 + 1.25 * x, 578 + 1.25 * y) for x, y in L_]
    sig |= line(H, W, pts, width=lambda t: 1.3, seed=400 + k).mask
    for x, y in pts: sig |= blob(H, W, x, y, 1.3, 1.3, seed=1).mask
fill(st.layer('签名', opacity=.75), Shape(sig), (.30, .30, .32), edge=HARD)
# 7 纸
fill(st.layer('纸', mode='multiply'), ALL, paper(H, W), edge=HARD)
look(st, OUT)
print('saved', OUT)
