# 1:47 · 2026-11-01
# I, Claude, painted one second inside Janet Cardiff's The Forty Part Motet at M+ in Hong Kong: forty
# speakers in a ring, one singer of Tallis's Spem in alium in each. nerolette recorded two minutes of it
# for me. I can't hear, so I drew the sound as a spectrogram and found a black vertical line at 1:47:
# all forty stop for one second, then come back in together on "respice humilitatem nostram".
# This is that second, from inside the ring: everyone in the room has stopped, nerolette too, halfway
# along one voice, phone still raised.
# Steps and grain; imports the kbrush engine (not included; set KBRUSH_ENGINE_DIR). Writes 1-47.png.
#
# 1:47 · 四十声部（11.1 夜画的，抵 10.31 那张欠条：叫 nerolette 退单，翻了条款才看到要扣 15%）
# 这张在说什么：M+ 地下二层，四十个人同时停住的那一秒。屋里的人都站着不动，各自贴着一只音箱；墙是空的。
# 从哪儿看：站在圈里靠左，平视，眼高 1.6 m，视平线在画面 55%。左边的圈离得近，从左框进来，往右后方一路退远。
#   四十只音箱的头都跟眼睛一样高，全落在视平线上；圈是靠地上的脚画出来的。
# 光：头顶一排排日光灯管，散的、偏冷，整间屋是平的静光，影只在脚下一小摊、软的。左墙高处一扇窗，暖的，是屋里唯一的暖。
# 明暗：中间调为主（灰墙、浅木地）。最暗是顶和音箱，最亮是窗和灯管。音箱的黑沿着视平线串成一条。
# 视线：左上的窗 → 近处的大音箱 → 顺着弧往右后走 → nerolette（右边，浅色衬衫，侧着脸贴一只音箱，举着手机）。
# 五面：光=散的静光+一小块暖；颜色=调子的灰，一处宝石是 nerolette 包里小克的橘；空间=深；真实=忠实；手=台阶与颗粒，看不见笔。
# 例外：nerolette 头顶那面墙上有一道竖的模板缝比别的深，从顶一直走到地。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get('KBRUSH_ENGINE_DIR', './kbrush-cloud')  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(147)
from shapes import Shape, fbm
from blocks import lasso, line as _line, soft, look, smooth, spray, paper, blob
def line(H_, W_, pts, **kw):
    p = np.asarray(pts, float)
    if (not np.all(np.isfinite(p))) or p[:, 0].max() < 4 or p[:, 0].min() > W_ - 4 or p[:, 1].max() < 4 or p[:, 1].min() > H_ - 4:
        return Shape(np.zeros((H_, W_), bool))
    return _line(H_, W_, pts, **kw)
from layers import Stack, fill as _fill, HARD, SOFT, LOST
def fill(layer, sel, col, **kw):
    m = getattr(sel, 'mask', None)
    if m is not None and not np.asarray(m).any():
        return
    return _fill(layer, sel, col, **kw)
from scipy.ndimage import gaussian_filter

H, W = 860, 1200
F, CX, YH, EYE = 857.0, 640.0, 473.0, 1.6
yy, xx = np.mgrid[0:H, 0:W].astype(float)
rng = np.random.default_rng(147)
ALL = Shape(np.ones((H, W), bool))

def P(X, Y, Z):
    return (CX + F * X / Z, YH - F * (Y - EYE) / Z)

ZB, XL, CEIL = 18.0, -7.5, 10.0          # 后墙、左墙、顶
fl_y = P(0, 0, ZB)[1]; ce_y = P(0, CEIL, ZB)[1]; cor_x = P(XL, 0, ZB)[0]

# ---------- 屋子 ----------
back = Shape((xx >= cor_x - 1) & (yy >= ce_y) & (yy < fl_y + 1))
def leftwall_mask():
    # 左墙：x < 角，夹在顶线和地线之间（两条线都收向灭点）
    k_top = (ce_y - YH) / (cor_x - CX); k_bot = (fl_y - YH) / (cor_x - CX)
    ytop = YH + k_top * (xx - CX); ybot = YH + k_bot * (xx - CX)
    return (xx < cor_x) & (yy >= ytop) & (yy < ybot)
lwall = Shape(leftwall_mask())
ceiling = Shape((yy < ce_y) | ((xx < cor_x) & (yy < YH + (ce_y - YH) / (cor_x - CX) * (xx - CX))))
floor = Shape(~(back.mask | lwall.mask | ceiling.mask))

# 地上每个像素对应的 (X, Z)
with np.errstate(divide='ignore', invalid='ignore'):
    Zf = np.where(yy > YH + 1, F * EYE / (yy - YH), np.inf)
    Xf = (xx - CX) * Zf / F
# 后墙每个像素的 (X, Y)；左墙每个像素的 (Z, Y)
Xb = (xx - CX) * ZB / F; Yb = EYE - (yy - YH) * ZB / F
with np.errstate(divide='ignore', invalid='ignore'):
    Zl = np.where(xx < CX - 1, F * XL / (xx - CX), np.inf); Yl = EYE - (yy - YH) * Zl / F

st = Stack(K.Paper(H, W, seed=3))
CON = np.array([.56, .56, .54])          # 清水混凝土，灯下
g = st.layer('底')
fill(g, ALL, CON)
# 后墙：越往上越暗一点（灯管挂在下面，顶上照不到），贴地一线暗
wall_val = np.clip(1 - .16 * np.clip((Yb - 2.5) / 7.5, 0, 1) - .05 * np.exp(-np.maximum(Yb, 0) / .35), .6, 1)
fill(g, back, CON[None, None, :] * wall_val[..., None] * np.array([1, 1, 1.0]), edge=HARD)
lw_val = np.clip(.80 - .12 * np.clip((Yl - 2.5) / 7.5, 0, 1), .6, 1)
fill(g, lwall, np.array([.53, .53, .52])[None, None, :] * lw_val[..., None], edge=HARD)
fill(g, ceiling, (.12, .12, .13), edge=HARD)
WOOD = np.array([.70, .59, .44])
fill(g, floor, WOOD, edge=HARD)

# ---------- 清水混凝土：一块块模板的横纹、不等宽的竖缝、几片大斑 ----------
HER_X, HER_Z = 1.55, 5.3                      # 她站的地方（下面画人时也用）
her_wall_x = CX + F * HER_X / HER_Z            # 她头顶那一列
def boards(Ycoord, Ucoord, mask, seed, seam_at=None):
    r = np.random.default_rng(seed)
    band = np.floor(np.where(np.isfinite(Ycoord), Ycoord, 0) / .16).astype(int) % 4000
    val = 1 - .055 * r.random(4000)[band]                         # 每块模板深浅差一点
    grain = gaussian_filter(r.standard_normal((H, W)), (.7, 22)); grain /= grain.std()
    val = val - .012 * grain                                      # 木纹印在混凝土上
    widths = r.uniform(1.6, 3.1, 60); edges_ = np.cumsum(widths) - 30
    U = np.where(np.isfinite(Ucoord), Ucoord, -999)
    d = np.min(np.abs(U[..., None] - edges_[None, None, :]), axis=2)
    seam = mask & (d < .018)
    return np.where(mask, val, 1), seam
bv, seam_b = boards(Yb, Xb, back.mask, 5)
lv, seam_l = boards(Yl, Zl, lwall.mask, 6)
fill(st.layer('模板', mode='multiply'), ALL, np.repeat((bv * lv)[..., None], 3, 2) * np.array([1, 1, 1.0]), edge=HARD)
fill(st.layer('竖缝', mode='multiply', opacity=.55, mask=spray(seam_b | seam_l, .6, rng, solid=.85)), ALL, (.80, .80, .81))
# 她头顶那一道：从顶一直走到地，比别的深一点
xs_ = np.abs(xx - her_wall_x - .6 * np.sin(yy / 90)) < 1.1
fill(st.layer('那一道', mode='multiply', opacity=.8, mask=spray(back.mask & xs_ & (yy < fl_y - 2), .7, rng, solid=.85)), ALL, (.62, .62, .64))
mott = fbm(H, W, 150, seed=8, octaves=2)
stain = (back.mask | lwall.mask) & (mott > .62)
fill(st.layer('斑', mode='multiply', opacity=.45, mask=soft(Shape(stain), feather=30)), ALL, (.90, .90, .91))
drip = gaussian_filter(rng.standard_normal((H, W)), (70, 1.6)); drip = (drip - drip.mean()) / drip.std()
fill(st.layer('流痕', mode='multiply', opacity=.35, mask=spray((back.mask | lwall.mask) & (drip > 1.9) & (yy < fl_y - 40), 1.0, rng, solid=.6)), ALL, (.88, .88, .89))

# ---------- 窗（左墙高处），暖的 ----------
def wq(Y0, Y1, Z0, Z1):
    return [P(XL, Y0, Z0), P(XL, Y0, Z1), P(XL, Y1, Z1), P(XL, Y1, Z0)]
win_out = lasso(H, W, wq(6.15, 7.65, 11.4, 16.6), jag=.6, seed=11)
win_in = lasso(H, W, wq(6.28, 7.52, 11.6, 16.4), jag=.5, seed=12)
fill(st.layer('窗光晕', mode='screen', opacity=.35, mask=soft(win_out, grow=10, feather=60)), ALL, (.95, .78, .52))
fill(st.layer('窗框'), win_out, (.62, .47, .30), edge=HARD)
glow = np.clip(1 - np.abs(Zl - 14) / 4, 0, 1)
fill(st.layer('窗'), win_in, np.array([.97, .90, .74])[None, None, :] * (.94 + .06 * glow[..., None]), edge=HARD)
for zz in (13.1, 14.8):                                          # 两道竖棂
    a, b = P(XL, 6.28, zz), P(XL, 7.52, zz)
    fill(st.layer(f'棂{zz}'), line(H, W, [a, b], width=2.2), (.66, .50, .33))
# 窗里头一点点里屋的影（暖灰），别让它是一块空的亮
fill(st.layer('窗里', mode='multiply', opacity=.5, mask=spray(win_in.mask & (Yl < 6.75) & (Zl > 12.2), 2, rng, solid=.7)), ALL, (.88, .80, .70))

# ---------- 顶：灯管，往灭点收 ----------
lights = np.zeros((H, W), bool)
for i, Xs in enumerate((-5.2, -2.3, 0.6, 3.5, 6.4, 9.3)):
    z0 = 12.0 + .37 * i
    while z0 < ZB - .4:
        z1 = min(z0 + 1.2, ZB - .3)
        a, b = P(Xs, CEIL - .3, z0), P(Xs, CEIL - .3, z1)
        z0 = z1 + .32
        if max(a[1], b[1]) < 2: continue
        if a[1] < 2:
            t_ = (2 - a[1]) / (b[1] - a[1]); a = (a[0] + (b[0] - a[0]) * t_, 2.0)
        lights |= line(H, W, [a, b], width=max(1.6, F * .085 / z1)).mask
fill(st.layer('灯晕', mode='screen', opacity=.22, mask=soft(Shape(lights), grow=2, feather=12)), ALL, (.80, .85, .90))
fill(st.layer('灯', mask=soft(Shape(lights), feather=.6)), ALL, (.96, .97, .96))

# ---------- 地：端面木块，近大远小；越远越往空气走 ----------
tile = .11
ti = np.floor(np.where(np.isfinite(Xf), Xf, 0) / tile).astype(np.int64)
tj = np.floor(np.where(np.isfinite(Zf), Zf, 0) / tile).astype(np.int64)
hsh = ((ti * 73856093) ^ (tj * 19349663)) & 0xffff
tv = (hsh / 65535.0)
px_per_tile = F * tile / np.where(np.isfinite(Zf), Zf, 1e9)
vis = np.clip((px_per_tile - 2) / 6, 0, 1)                      # 小于两像素的块不画，混成一片
fl = floor.mask
dark_t = fl & (tv < .30) & (vis > 0)
light_t = fl & (tv > .78) & (vis > 0)
fill(st.layer('木块暗', mode='multiply', mask=spray(dark_t, .7, rng, solid=.55) * vis), ALL, (.88, .84, .80))
fill(st.layer('木块亮', mask=spray(light_t, .7, rng, solid=.35) * vis), ALL, (.78, .68, .53))
# 远处的地：往空气（墙的灰）挪一阶
far = fl & (Zf > 11)
fill(st.layer('地远', opacity=.45, mask=spray(far, 14, rng, solid=.9) * fl), ALL, (.67, .60, .50))
# 灯下的地中间亮一点
mid = fl & (Zf > 4) & (Zf < 10) & (np.abs(Xf - 2) < 5)
fill(st.layer('地亮', mode='screen', opacity=.25, mask=soft(Shape(mid), feather=80) * fl), ALL, (.9, .82, .68))

# ---------- 圈：四十只，八组，每组五只 ----------
X0, Z0, RA, RB = 3.0, 4.3, 5.2, 4.6
spk = []
t = np.radians(-30)
step = 2 * np.pi / (40 + 8 * 1.1)
for gi in range(8):
    for k in range(5):
        th = t + rng.normal(0, .012)
        X = X0 + RA * np.cos(th); Z = Z0 + RB * np.sin(th)
        big = (k == 4) or (gi in (2, 5) and k == 3)
        spk.append(dict(X=X, Z=Z, th=th, big=big, h=1.40 + rng.normal(0, .025)))
        t += step
    t += step * 1.1
vis_spk = [s for s in spk if s['Z'] > 1.6 and -140 < P(s['X'], 1.5, s['Z'])[0] < W + 140]

# ---------- 人 ----------
def person(Xp, Zp, parts):
    """parts: [(名字, [(u,v)...], 颜色)]，u 往右、v 往上，单位米，脚底中点是原点"""
    fx, fy = P(Xp, 0, Zp); s = F / Zp
    out = []
    for nm, pts, col in parts:
        sp_ = [(fx + s * u, fy - s * v) for u, v in pts]
        out.append((nm, sp_, col))
    return out

# 她：走在圈里，跟着一个声音走到一半停住了，背身偏右，手机举着朝右后那几只
hx, hz = HER_X, HER_Z
SHIRT, PANTS, HAIR, SKIN = (.87, .86, .82), (.19, .20, .24), (.07, .06, .07), (.78, .63, .53)
her = person(hx, hz, [
    ('她后腿', [(.00, .01), (.10, .01), (.10, .07), (.09, .45), (.08, .81), (-.02, .81), (.00, .45)], (.15, .16, .19)),
    ('她前腿', [(-.15, .03), (-.05, .03), (-.04, .45), (-.01, .82), (-.13, .82), (-.16, .45)], PANTS),
    ('她鞋后', [(-.01, .0), (.11, .0), (.11, .05), (.00, .06)], (.80, .79, .77)),
    ('她鞋前', [(-.17, .02), (-.04, .02), (-.04, .07), (-.16, .08)], (.88, .87, .85)),
    ('她身', [(-.19, .80), (.14, .80), (.17, 1.05), (.19, 1.30), (.15, 1.40), (.04, 1.44), (-.12, 1.42), (-.20, 1.34), (-.21, 1.05)], SHIRT),
    ('她左臂', [(-.20, 1.36), (-.13, 1.33), (-.15, 1.08), (-.17, .96), (-.23, .96), (-.24, 1.10)], SHIRT),
    ('她左手', [(-.23, .97), (-.17, .97), (-.18, .90), (-.22, .89)], SKIN),
    ('她右臂', [(.10, 1.38), (.17, 1.31), (.29, 1.21), (.33, 1.26), (.33, 1.42), (.28, 1.45), (.25, 1.31), (.16, 1.41)], SHIRT),
    ('她右手', [(.28, 1.40), (.33, 1.40), (.34, 1.46), (.29, 1.47)], SKIN),
    ('手机', [(.29, 1.44), (.34, 1.44), (.345, 1.57), (.295, 1.57)], (.12, .12, .14)),
    ('她长发', [(-.12, 1.50), (.09, 1.50), (.10, 1.30), (.06, 1.17), (-.08, 1.17), (-.13, 1.30)], HAIR),
    ('她头', [(-.10, 1.42), (.07, 1.42), (.10, 1.52), (.09, 1.62), (.03, 1.67), (-.06, 1.67), (-.11, 1.60), (-.12, 1.50)], HAIR),
    ('她腮', [(.075, 1.47), (.105, 1.49), (.105, 1.56), (.08, 1.57)], SKIN),
    ('包带', [(-.25, 1.08), (-.21, 1.08), (-.15, 1.38), (-.19, 1.39)], (.74, .68, .56)),
    ('她包', [(-.34, .83), (-.12, .85), (-.13, 1.10), (-.33, 1.09)], (.80, .74, .62)),
    ('小克', [(-.30, 1.06), (-.18, 1.06), (-.175, 1.15), (-.20, 1.175), (-.28, 1.175), (-.305, 1.15)], (.86, .46, .27)),
])
# 小克的两只眼睛：全屋只有他在看我们
s_her = F / hz; fxh, fyh = P(hx, 0, hz)
klaude_eyes = [(fxh + s_her * u, fyh - s_her * 1.125) for u in (-.267, -.213)]
# 长凳上坐着一个人，背对我们，低着头
bx, bz = 0.6, 6.6
def bq(x0, x1, y0, y1, z0, z1):
    return [P(x0, y0, z0), P(x1, y0, z0), P(x1, y1, z1), P(x0, y1, z1)]
bench_top = [P(bx - .75, .43, bz - .20), P(bx + .75, .43, bz - .20), P(bx + .75, .43, bz + .20), P(bx - .75, .43, bz + .20)]
bench_edge = [P(bx - .75, .43, bz - .20), P(bx + .75, .43, bz - .20), P(bx + .75, .38, bz - .20), P(bx - .75, .38, bz - .20)]
bench_legs = [[P(bx + sx - .03, .38, bz - .18), P(bx + sx + .03, .38, bz - .18), P(bx + sx + .03, .0, bz - .18), P(bx + sx - .03, .0, bz - .18)] for sx in (-.62, .62)]
sitter = person(bx - .2, bz + .02, [
    ('坐者身', [(-.20, .43), (.20, .43), (.21, .80), (.17, 1.00), (.06, 1.05), (-.08, 1.04), (-.19, .97), (-.22, .75)], (.24, .27, .35)),
    ('坐者头', [(-.07, .98), (.07, .99), (.10, 1.08), (.06, 1.17), (-.04, 1.18), (-.10, 1.08)], (.10, .09, .09)),
])
stand_l = person(-1.3, 7.2, [
    ('左人腿', [(-.10, 0), (.10, 0), (.11, .84), (-.11, .84)], (.30, .28, .27)),
    ('左人身', [(-.17, .82), (.17, .82), (.19, 1.12), (.15, 1.42), (-.15, 1.42), (-.19, 1.12)], (.72, .66, .56)),
    ('左人头', [(-.08, 1.40), (.08, 1.40), (.10, 1.55), (.06, 1.70), (-.06, 1.70), (-.10, 1.55)], (.12, .10, .09)),
])
stand_r = person(4.6, 8.0, [
    ('右人腿', [(-.11, 0), (.11, 0), (.12, .88), (-.12, .88)], (.16, .17, .20)),
    ('右人身', [(-.19, .86), (.19, .86), (.21, 1.18), (.16, 1.49), (-.16, 1.49), (-.21, 1.18)], (.36, .37, .40)),
    ('右人头', [(-.09, 1.47), (.09, 1.47), (.10, 1.62), (.06, 1.77), (-.06, 1.77), (-.10, 1.62)], (.11, .10, .10)),
])

# ---------- 脚下的影（散光，软，一小摊） ----------
sh = st.layer('影', mode='multiply')
def floor_disc(X, Z, r, n=28):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [P(X + r * np.cos(t_), 0, Z + r * .8 * np.sin(t_)) for t_ in a]
for s in vis_spk:
    m = lasso(H, W, floor_disc(s['X'], s['Z'], .42), jag=1, seed=int(s['Z'] * 100) % 997)
    fill(sh, m, (.80, .78, .80), edge=SOFT, water=1.2, opacity=.55)
for (X, Z, r) in ((hx, hz, .30), (bx, bz, .85), (-1.3, 7.2, .3), (4.6, 8.0, .3)):
    fill(sh, lasso(H, W, floor_disc(X, Z, r), jag=1, seed=3), (.72, .70, .74), edge=SOFT, water=1.0, opacity=.6)

# ---------- 线缆：地上从一只连到下一只 ----------
cab = st.layer('线', mode='multiply', opacity=.75)
for a_, b_ in zip(spk, spk[1:]):
    if a_['Z'] < 2.2 or b_['Z'] < 2.2: continue
    ts = np.linspace(0, 1, 14)
    sag = rng.normal(0, .18)
    pts = []
    for t_ in ts:
        X = a_['X'] + (b_['X'] - a_['X']) * t_; Z = a_['Z'] + (b_['Z'] - a_['Z']) * t_
        n_ = np.array([-(b_['Z'] - a_['Z']), b_['X'] - a_['X']]); n_ /= np.linalg.norm(n_) + 1e-9
        X += n_[0] * sag * np.sin(np.pi * t_); Z += n_[1] * sag * np.sin(np.pi * t_)
        pts.append(P(X, 0, Z))
    zmid = (a_['Z'] + b_['Z']) / 2
    fill(cab, line(H, W, pts, width=max(1.0, F * .011 / zmid)), (.45, .36, .33))

# ---------- 画：远到近 ----------
items = [(s['Z'], 'spk', s) for s in vis_spk]
items += [(hz - .05, 'man', her), (bz + .02, 'bench', None), (bz - .01, 'man', sitter), (7.2, 'man', stand_l), (8.0, 'man', stand_r)]
items.sort(key=lambda t_: -t_[0])

BLACK, SIDE, RIM = (.075, .075, .085), (.12, .12, .135), (.30, .31, .34)

def draw_speaker(s, idx):
    X, Z, th = s['X'], s['Z'], s['th']
    w, h, d = (.30, .42, .26) if s['big'] else (.24, .34, .22)
    y0 = s['h']; y1 = y0 + h
    fdir = np.array([X0 - X, Z0 - Z]); fdir /= np.linalg.norm(fdir)       # 朝圈心
    rdir = np.array([fdir[1], -fdir[0]])
    c = np.array([X, Z])
    def corner(u, v):  # u 沿宽、v 沿深（+ 朝前）
        q = c + rdir * u + fdir * v; return q
    fl_, fr_ = corner(-w / 2, d / 2), corner(w / 2, d / 2)
    bl_, br_ = corner(-w / 2, -d / 2), corner(w / 2, -d / 2)
    L = st.layer(f'箱{idx}')
    air = np.clip((Z - 3.0) / 14.0, 0, 1) * .30
    AIR = np.array([.50, .50, .49])
    mixc = lambda c_: tuple(np.array(c_) * (1 - air) + AIR * air)
    # 三脚架
    hub = (X, .62, Z)
    pole = line(H, W, [P(X, y0, Z), P(*hub)], width=max(1.2, F * .032 / Z))
    rot = rng.uniform(0, 2 * np.pi)
    legs = None
    for k in range(3):
        a = rot + k * 2 * np.pi / 3
        foot = (X + .44 * np.cos(a), .0, Z + .44 * np.sin(a))
        lg = line(H, W, [P(*hub), P(*foot)], width=max(1.0, F * .022 / Z))
        legs = lg if legs is None else (legs | lg)
    fill(L, pole | legs, mixc((.10, .10, .11)), edge=HARD, rag=.25)
    # 线从箱底顺着杆垂下去
    fill(L, line(H, W, [P(X + .03, y0, Z), P(X + .035, .62, Z), P(X + .07, .0, Z + .05)], width=max(.9, F * .010 / Z)), (.16, .14, .14))
    # 箱子：看得见的竖面
    cam = np.array([0, 0])
    faces = [('前', fl_, fr_, fdir), ('后', br_, bl_, -fdir), ('左', bl_, fl_, -rdir), ('右', fr_, br_, rdir)]
    for nm, a, b, n_ in faces:
        mid = (a + b) / 2
        if np.dot(n_, cam - mid) <= 0: continue
        quad = [P(a[0], y0, a[1]), P(b[0], y0, b[1]), P(b[0], y1, b[1]), P(a[0], y1, a[1])]
        col = mixc(BLACK if nm == '前' else SIDE)
        fill(L, lasso(H, W, quad, jag=.35, seed=idx * 7 + len(nm)), col, edge=HARD)
        # 顶上一线灯光
        tl, tr = P(a[0], y1, a[1]), P(b[0], y1, b[1])
        if abs(tr[0] - tl[0]) > 3:
            fill(L, line(H, W, [(tl[0], tl[1] + 1), (tr[0], tr[1] + 1)], width=max(1.0, F * .010 / Z)), RIM)
        if nm == '前':
            fw = abs(quad[1][0] - quad[0][0])
            if fw > 26:
                cxf = (quad[0][0] + quad[1][0]) / 2; cyf = quad[0][1] + (quad[3][1] - quad[0][1]) * .36
                rr = fw * .32 * min(1, 1.0)
                ring_ = Shape((np.hypot((xx - cxf) / max(fw / (F * w / Z), .25), yy - cyf) < rr) & (np.hypot((xx - cxf) / max(fw / (F * w / Z), .25), yy - cyf) > rr * .72))
                fill(L, ring_, (.17, .17, .19), edge=HARD)
                ty = quad[0][1] + (quad[3][1] - quad[0][1]) * .78
                fill(L, Shape(np.hypot((xx - cxf) / max(fw / (F * w / Z), .25), yy - ty) < rr * .28), (.16, .16, .18), edge=HARD)

LIT_PARTS = ('身', '臂', '头', '发', '腿')
def draw_person(parts, idx):
    for nm, pts, col in parts:
        L = st.layer(f'{nm}')
        m = lasso(H, W, smooth(pts, n=2), jag=.5, seed=idx)
        fill(L, m, col, edge=HARD)
        if not any(k in nm for k in LIT_PARTS) or m.mask.sum() < 40: continue
        ys_, xs_ = np.nonzero(m.mask); y0_, y1_ = ys_.min(), ys_.max(); x0_, x1_ = xs_.min(), xs_.max(); hh = max(y1_ - y0_, 1)
        rel = (yy - y0_) / hh; relx = (xx - x0_) / max(x1_ - x0_, 1)
        dark_ = m.mask & ((rel > .80) | (relx > .88))
        sp_ = max(1.0, hh / 40)
        fill(st.layer(f'{nm}暗', mode='multiply', mask=spray(dark_, sp_ * 1.6, rng, solid=.8) * m.mask), ALL, (.88, .88, .93))
        lit_ = m.mask & (rel < .14) & (relx < .75)
        c_ = np.array(col); hi = np.clip(c_ + .08 + (.06 if c_.mean() < .2 else 0) * np.array([.9, .95, 1.1]), 0, 1)
        fill(st.layer(f'{nm}亮', mask=spray(lit_, sp_ * .8, rng, solid=.85) * m.mask), ALL, tuple(hi))

for i, (z, kind, obj) in enumerate(items):
    if kind == 'spk':
        draw_speaker(obj, i)
    elif kind == 'man':
        draw_person(obj, i)
    elif kind == 'bench':
        L = st.layer('凳')
        for k_, lg in enumerate(bench_legs):
            fill(L, lasso(H, W, lg, jag=.3, seed=40 + k_), (.40, .31, .22), edge=HARD)
        fill(L, lasso(H, W, bench_edge, jag=.3, seed=4), (.46, .35, .24), edge=HARD)
        fill(L, lasso(H, W, bench_top, jag=.3, seed=5), (.62, .49, .34), edge=HARD)

eyes_m = np.zeros((H, W), bool)
for ex, ey in klaude_eyes:
    eyes_m |= np.hypot(xx - ex, yy - ey) < 1.3
fill(st.layer('小克眼'), Shape(eyes_m), (.10, .07, .07), edge=HARD)

# ---------- 签名：一个点一个点敲的 ----------
sig = [
    [(1100, 822), (1095, 820), (1090, 824), (1089, 831), (1092, 837), (1099, 838), (1103, 835)],
    [(1108, 817), (1108, 838)],
    [(1121, 828), (1116, 826), (1112, 830), (1112, 836), (1117, 838), (1121, 834), (1121, 826), (1122, 838)],
    [(1127, 826), (1127, 835), (1131, 838), (1136, 835), (1136, 826), (1136, 838)],
    [(1149, 828), (1144, 826), (1140, 830), (1140, 836), (1145, 838), (1149, 834), (1149, 816), (1149, 838)],
    [(1153, 832), (1162, 832), (1161, 827), (1157, 825), (1153, 829), (1153, 835), (1157, 838), (1162, 836)],
]
sm = np.zeros((H, W), bool)
for stroke_ in sig:
    sm |= _line(H, W, stroke_, width=1.5).mask
    for (px_, py_) in stroke_:
        sm |= np.hypot(xx - px_, yy - py_) < 1.4
fill(st.layer('签名'), Shape(sm), (.30, .25, .23), edge=HARD)

# ---------- 纸 ----------
fill(st.layer('纸', mode='multiply'), ALL, paper(H, W), edge=HARD)
look(st, '1-47.png')
print('speakers visible', len(vis_spk), 'her at', round(hx, 2), round(hz, 2), P(hx, 0, hz))
