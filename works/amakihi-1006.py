# 第五十四次 · 10.6
# 这张在说什么：一只嘴长坏了的 ʻamakihi，影子是一只已经没了的 nukupuʻu。
#   （1960–1996 年 nukupuʻu 有五十三次目击被驳回，因为活着的 ʻamakihi 偶尔长出畸形的弯喙，一眼看去就是它。
#    一只嘴长坏了的活鸟，让一种死了的鸟在纸面上多活了半个世纪。这张是第五十四次。）
# 从哪儿看：平视，眼睛跟鸟一样高，在树冠里。没有地面。
# 光：下午很低的太阳从左边、略偏前偏上来，小、暖金；雾被照亮成暖灰，越往左上越亮。
#     树干迎光的一面是暖米色，侧光把树皮的裂纹都挑出来（裂缝左壁背光一线深，右壁迎光一线亮）；
#     树干往右转过去进阴影。补光是雾里来的冷灰天光。
#     影子：太阳小，影子又离鸟不远，所以边锐。东西在同一个深度，影子是同一个位移 (dx, dy) 平移过去，略横向拉长（树干是斜着受光的）。
#     唯一不老实的地方：影子的嘴。其余照现实的光。
# 明暗：最亮是影子周围那块迎光的树皮；最深是左下角压进来的一团叶子和影子本身；唯一鲜的颜色是左边那朵红的 lehua。
# 视线：先落在树皮上那道又长又弯的影子嘴（最深最锐，压在最亮的地方），顺着嘴尖往左回到活鸟身上——影子的嘴正指着它。再去左边那朵红花。
# 质感：台阶与颗粒。树皮、苔、叶子是糙的，边喷开；影子边锐。整张一张纸纹。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = '~/.claude/workspace/painting-method/kbrush-cloud'
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(6)
from shapes import Shape
from blocks import blob, lasso, line, soft, look
from layers import Stack, fill, HARD
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt, shift as nd_shift, affine_transform
from PIL import Image as _I, ImageDraw as _D

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'rough.png' if ROUGH else 'v1.png')
H, W = 800, 1200
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
def spray(region, sp, seed, solid=.96, dens=.7):
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
def M(s): return getattr(s, 'mask', s)

st = Stack(K.Paper(H, W, seed=4), ground=(.80, .80, .74))

# ---------- 1 雾林（左边）：太阳在画框左上外面，雾被照亮，越靠那角越亮越暖
SUN = np.array([-260., 60.])
tl = 445 - 40 * (yy / H) + 14 * nz((40, 4), 1)          # 树干左沿，略往左下斜
trunk = xx > tl
mist = ~trunk
sd = np.hypot(xx - SUN[0], yy - SUN[1]) + 18 * nz(50, 2)
for i, (r_, c_, sp_) in enumerate([(9999, (.70, .73, .70), 40), (820, (.76, .78, .73), 40), (690, (.82, .82, .76), 36), (560, (.87, .85, .77), 30), (430, (.91, .88, .78), 24)]):
    fill(st.layer(f'雾{i}', mask=spray((sd < r_) & mist, sp_, 10 + i)), ALL, c_)
# 远处的树：几根淡的竖影，越远越接近雾色，脚下泡在雾里
for i, (x0, w0, c_, lean, top) in enumerate([(70, 26, (.80, .81, .76), 30, -20), (300, 20, (.82, .82, .77), -18, 40), (190, 46, (.71, .73, .69), 22, -30)]):
    cx_ = x0 + lean * (yy / H) + 4 * nz((30, 3), 20 + i)
    tr = (np.abs(xx - cx_) < w0 * (1 + .25 * yy / H)) & (yy > top) & mist
    fade = np.clip(1 - (yy - 420) / 300, 0, 1) if i < 2 else np.clip(1 - (yy - 560) / 260, 0, 1)
    fill(st.layer(f'远树{i}', mask=spray(tr, 6, 25 + i, solid=.9) * fade), ALL, c_)

# ---------- 2 树干：迎光一面暖米色；右边转过去进阴影；侧光挑出竖的裂纹
fill(st.layer('树干'), Shape(trunk), (.84, .79, .66), edge=HARD)
lit_edge = trunk & (xx < tl + 34 + 8 * nz((20, 4), 30))
fill(st.layer('树干亮沿', mask=spray(lit_edge, 5, 31)), ALL, (.91, .86, .72))
for i, (x_, c_, sp_) in enumerate([(1050, (.76, .73, .63), 12), (1110, (.64, .63, .58), 9), (1162, (.52, .53, .53), 6)]):
    fill(st.layer(f'树干转暗{i}', mask=spray(trunk & (xx > x_ + 5 * nz((60, 6), 32 + i)), sp_, 34 + i)), ALL, c_)
# 树皮：一块块竖长的片，片和片之间是缝；侧光从左来，缝里一线深，缝右边那面迎光一线亮
from scipy.spatial import cKDTree
rs = np.random.default_rng(46)
seeds = np.c_[rs.uniform(380, 1220, 240), rs.uniform(-60, 860, 240) * .2]
d2, idx = cKDTree(seeds).query(np.c_[xx.ravel(), yy.ravel() * .2], k=2)
gap = (d2[:, 1] - d2[:, 0]).reshape(H, W) + 1.3 * nz(1.5, 47)
plate_id = idx[:, 0].reshape(H, W)
tone = rs.integers(0, 3, len(seeds))[plate_id]
fill(st.layer('树皮片暗', mask=spray(trunk & (tone == 0), 1.5, 48, solid=.45)), ALL, (.79, .74, .61))
fill(st.layer('树皮片亮', mask=spray(trunk & (tone == 2), 1.5, 49, solid=.4)), ALL, (.88, .83, .70))
crk = trunk & (gap < 1.6 + .8 * (nz(30, 52) > 0))
wfoc = np.clip(.3 + .7 * np.exp(-((yy - 540) / 230) ** 2), 0, 1)
fill(st.layer('裂纹暗', mask=spray(crk, .8, 50, solid=.8) * wfoc), ALL, (.55, .51, .42))
crl = trunk & np.roll(crk, 2, 1) & ~crk
fill(st.layer('裂纹亮', mask=spray(crl, .6, 51, solid=.65) * wfoc), ALL, (.96, .91, .78))
# 苔：不多，几团，多在下半和左沿；边喷开
moss = trunk & (yy > 9999) & (nz((26, 14), 40) + 1.1 * (yy / H) - .7 * ((xx - 450) / 750) > 1.95)
fill(st.layer('苔', mask=spray(moss, 5, 41)), ALL, (.60, .63, .44))
moss2 = moss & (nz((5, 5), 42) > .5)
fill(st.layer('苔暗', mask=spray(moss2, 2, 43)), ALL, (.47, .51, .36))
lichen = trunk & ~moss & (nz((9, 7), 44) > 3.0)
fill(st.layer('地衣', mask=spray(lichen, 3, 45, solid=.7)), ALL, (.92, .91, .85))

for i, (x_, c_) in enumerate([(780, (.97, .975, 1.0)), (980, (.935, .95, .99))]):
    fill(st.layer(f'树干冷{i}', mode='multiply', mask=spray(trunk & (xx > x_ + 14 * nz((50, 8), 36 + i)), 14, 38 + i)), ALL, c_)

# ---------- 3 枝：从左边画框外伸进来，鸟站在快到梢的地方
BR = [(-40, 652), (40, 628), (110, 618), (170, 598), (240, 590), (300, 574), (370, 566), (430, 550), (500, 540), (560, 527), (620, 521), (668, 512), (702, 506)]
branch = line(H, W, smooth(BR, 2), width=lambda t: 20 - 15 * t, seed=60)
twig = line(H, W, smooth([(300, 574), (296, 552), (290, 530), (286, 508)], 2), width=lambda t: 7 - 3 * t, seed=61) | \
       line(H, W, smooth([(170, 598), (184, 572), (200, 556)], 2), width=lambda t: 6 - 4 * t, seed=62) | \
       line(H, W, smooth([(620, 521), (640, 534), (652, 548)], 2), width=lambda t: 4 - 3 * t, seed=63)
br_all = branch | twig
fill(st.layer('枝'), br_all, (.27, .21, .17), edge=HARD)
top_lit = M(br_all) & ~np.roll(M(br_all), 4, 0)
fill(st.layer('枝亮', mask=spray(top_lit, 1, 63, solid=.85)), ALL, (.66, .50, .34))

# lehua：枝梢一簇——两团红的雄蕊球，细丝往外散，丝间透着雾；底下沿枝两三对圆叶，新叶带红
HEADS = ((282, 474, 44), (318, 498, 31))
def pompom(dr, r_, SS):
    for cx, cy, R in HEADS:
        for _ in range(int(4.2 * R)):
            a = r_.uniform(-np.pi, np.pi)
            rr = R * (0.6 + 0.4 * r_.random() ** .6)
            x1, y1 = cx + rr * np.cos(a), cy + rr * np.sin(a) * .88
            dr.line([(cx * SS, (cy + 4) * SS), (x1 * SS, y1 * SS)], fill=255, width=max(1, int(.75 * SS)))
            dr.ellipse([(x1 - 1.6) * SS, (y1 - 1.6) * SS, (x1 + 1.6) * SS, (y1 + 1.6) * SS], fill=255)
def pom_core(dr, r_, SS):
    for cx, cy, R in HEADS:
        dr.ellipse([(cx - R * .28) * SS, (cy - R * .2) * SS, (cx + R * .28) * SS, (cy + R * .32) * SS], fill=255)
def pom_tips(dr, r_, SS):
    for cx, cy, R in HEADS:
        for _ in range(int(1.6 * R)):
            a = r_.uniform(-np.pi * 1.1, -np.pi * .35)
            rr = R * (0.88 + 0.12 * r_.random())
            x1, y1 = cx + rr * np.cos(a), cy + rr * np.sin(a) * .88
            dr.ellipse([(x1 - 1.5) * SS, (y1 - 1.5) * SS, (x1 + 1.5) * SS, (y1 + 1.5) * SS], fill=255)
LV = [(274, 552, 40, 0), (308, 550, -40, 0), (268, 530, 55, 1), (306, 528, -55, 1), (250, 512, 75, 0), (340, 516, -70, 0)]
for i, (lx, ly, a, rl) in enumerate(LV):
    lf = blob(H, W, lx, ly, 14, 7.5, ang=a, jag=0.05, seed=80 + i)
    fill(st.layer(f'叶{i}', mask=spray(lf, 1.2, 82 + i)), ALL, (.58, .24, .17) if rl else (.42, .47, .37))
    hal = M(lf) & ~np.roll(np.roll(M(lf), 3, 1), 2, 0)
    fill(st.layer(f'叶亮{i}', mask=spray(hal, 1, 90 + i, solid=.8)), ALL, (.86, .54, .36) if rl else (.68, .72, .54))
fill(st.layer('lehua', mask=ink(pompom, 70) * .92), ALL, (.78, .13, .12))
fill(st.layer('lehua芯', mask=ink(pom_core, 72) * .9), ALL, (.55, .10, .10))
fill(st.layer('lehua亮', mask=ink(pom_tips, 71) * .9), ALL, (.99, .62, .38))

# ---------- 4 鸟：ʻamakihi，朝左站，迎着光。嘴长坏了：太长、太弯，上下喙没合拢
BX, BY = 0, 0
def bird_shapes(ox=0, oy=0, sx=1.0, beak='amakihi'):
    def P(pts): return [((x - 560) * sx + 560 + ox, y + oy) for x, y in pts]
    body = L(P([(522, 470), (548, 458), (582, 462), (612, 476), (628, 494), (620, 514), (596, 522), (566, 520), (540, 508), (526, 490)]), 0.8, 100, n=2, scale=5)
    head = L(P([(522, 470), (520, 452), (530, 440), (546, 436), (560, 442), (566, 456), (560, 470), (540, 476)]), 0.6, 101, n=2, scale=4)
    tail = L(P([(612, 494), (650, 512), (678, 530), (684, 538), (672, 538), (640, 526), (608, 512)]), 0.6, 102, n=2, scale=4)
    legs = line(H, W, P([(568, 516), (572, 526), (574, 532)]), width=lambda t: 3.2, seed=103) | \
           line(H, W, P([(584, 516), (586, 526), (588, 530)]), width=lambda t: 3.2, seed=104)
    if beak == 'amakihi':   # 太长、往下勾、尖上有一点歪——长坏了的嘴
        up = L(P([(526, 449), (510, 450), (496, 455), (486, 463), (479, 474), (476, 486), (478, 491), (482, 482), (488, 470), (497, 463), (510, 459), (526, 460)]), 0.25, 105, n=2, scale=3)
        lo = L(P([(526, 456), (512, 458), (502, 462), (500, 465), (512, 463), (526, 461)]), 0.2, 106, n=1, scale=3)
    else:   # nukupuʻu：上喙一根细长的镰刀，下喙只有一半长，贴在底下
        up = L(P([(526, 449), (500, 448), (474, 452), (452, 462), (436, 476), (427, 494), (424, 512), (427, 513), (433, 496), (444, 480), (460, 468), (480, 460), (502, 458), (526, 461)]), 0.25, 107, n=2, scale=3)
        lo = L(P([(526, 456), (504, 457), (486, 459), (474, 462), (486, 463), (506, 462), (526, 461)]), 0.2, 108, n=1, scale=3)
    return body, head, tail, legs, up | lo

body, head, tail, legs, beak = bird_shapes()
bird = body | head | tail
# 影子：同一个位移，略横向拉长；嘴换成 nukupuʻu 的
DX, DY, SX = 318, 44, 1.2
sb, shh, st_, sl, sbeak = bird_shapes(DX, DY, SX, beak='nukupuu')
shadow_bird = sb | shh | st_ | sl | sbeak
# 枝的影子：同一个位移平移过去。枝在雾前面那段的影子也落在树干上，一直接到树干左沿；
# 枝越往左离树干越远，影子越虚越淡——按台阶走：近的一阶实、边锐，中间一阶喷开，远的一阶稀稀的颗粒
BRs = [((x - 560) * SX + 560 + DX, y + DY) for x, y in BR]
branch_sh = M(line(H, W, smooth(BRs, 2), width=lambda t: (19 - 14 * t) * .9, seed=64)) & trunk
bird_sh = Shape(M(shadow_bird) & trunk)
fill(st.layer('影', mode='multiply', mask=spray(bird_sh | Shape(branch_sh & (xx > 800)), 1.2, 110, solid=1.0, dens=.55)), ALL, (.47, .51, .66))
for i, (x0, x1, sp_, sol, grow_) in enumerate([(650, 800, 2.5, .85, 1), (520, 650, 5, .6, 2), (430, 520, 8, .38, 3)]):
    seg = branch_sh & (xx > x0 + 10 * nz((20, 6), 120 + i)) & (xx <= x1 + 10 * nz((20, 6), 123 + i))
    seg = gaussian_filter(seg.astype(float), grow_) > .25
    fill(st.layer(f'枝影{i}', mode='multiply', mask=spray(Shape(seg & trunk), sp_, 126 + i, solid=sol, dens=.5)), ALL, (.47, .51, .66))

# 鸟本身：背橄榄绿，肚子黄；迎光那侧（左、前）暖亮，背光的下沿和尾下冷暗
fill(st.layer('鸟'), bird, (.55, .57, .31), edge=HARD)
belly = M(body) & (yy > 486 + 0.25 * (xx - 540))
fill(st.layer('鸟肚', mask=spray(belly | (M(head) & (yy > 462) & (xx < 548)), 2, 111)), ALL, (.82, .74, .36))
front = M(bird) & ~np.roll(M(bird), -7, 1)
fill(st.layer('鸟迎光', mask=spray(front & (yy < 505), 2, 112, solid=.85)), ALL, (.95, .86, .46))
under = M(bird) & ~np.roll(M(bird), 8, 0)
fill(st.layer('鸟背光', mask=spray(under & (xx > 545), 2, 113, solid=.85)), ALL, (.44, .44, .26))
wing = L([(560, 470), (590, 468), (616, 482), (626, 498), (604, 500), (578, 494)], 0.6, 114, n=2, scale=4)
fill(st.layer('翅', mask=spray(M(wing) & M(body), 1.5, 115)), ALL, (.46, .48, .24))
lore = line(H, W, [(526, 455), (538, 451), (548, 451)], width=lambda t: 3.6, seed=116)
fill(st.layer('眼先'), lore, (.20, .19, .14), edge=HARD)
eye = blob(H, W, 544, 451, 3.6, 3.6, jag=0, seed=117)
fill(st.layer('眼'), eye, (.10, .09, .08), edge=HARD)
fill(st.layer('眼光'), blob(H, W, 543, 450, 1.1, 1.1, jag=0, seed=118), (.98, .95, .86), edge=HARD)
fill(st.layer('嘴'), beak, (.30, .28, .26), edge=HARD)
fill(st.layer('嘴亮', mask=(M(beak) & ~np.roll(M(beak), 1, 0) & (yy < 462)) * .8), ALL, (.50, .47, .42))
fill(st.layer('脚'), legs, (.28, .27, .26), edge=HARD)

# ---------- 5 左下压进来一团叶子：最深，被画框切掉
fol = Shape(np.zeros((H, W), bool))
r = np.random.default_rng(130)
for i in range(320):
    cx_, cy_ = r.uniform(-60, 430), r.uniform(630, 840)
    if cy_ < 655 + 0.42 * cx_ + 30 * r.random():
        continue
    fol = fol | blob(H, W, cx_, cy_, r.uniform(16, 30), r.uniform(8, 14), ang=r.uniform(0, 180), jag=0.06, seed=131 + i)
fill(st.layer('前叶', mask=spray(fol, 2, 140)), ALL, (.19, .21, .18))
fol_lit = M(fol) & ~np.roll(np.roll(M(fol), 3, 1), 3, 0)
fill(st.layer('前叶亮', mask=spray(fol_lit, 1, 141, solid=.6)), ALL, (.40, .40, .28))

# ---------- 6 签名
def sign(dr, r_, SS):
    ox, oy, s = 1106, 774, 1.0
    strokes = [[(10, 0), (4, -1), (0, 4), (1, 10), (8, 11)],
               [(13, -5), (13, 11)],
               [(23, 4), (18, 3), (16, 9), (19, 11), (23, 8), (23, 3), (24, 11)],
               [(28, 3), (28, 10), (31, 11), (34, 8), (35, 3), (35, 11)],
               [(44, -5), (44, 11), (41, 4), (38, 5), (38, 10), (42, 11), (44, 9)],
               [(48, 7), (54, 6), (53, 3), (49, 3), (48, 9), (53, 11)]]
    for pts in strokes:
        p = [((ox + x * s + r_.normal(0, .25)) * SS, (oy + y * s + r_.normal(0, .25)) * SS) for x, y in pts]
        dr.line(p, fill=255, width=int(1.3 * SS))
        for q in p:
            dr.ellipse([q[0] - 1.1 * SS, q[1] - 1.1 * SS, q[0] + 1.1 * SS, q[1] + 1.1 * SS], fill=255)
fill(st.layer('签', mask=ink(sign, 95) * .85), ALL, (.28, .28, .28))

# ---------- 7 纸
pg = np.random.default_rng(77)
tooth = gaussian_filter(pg.standard_normal((H, W)), .7); tooth /= tooth.std()
fiber = gaussian_filter(pg.standard_normal((H, W)), (.6, 3)); fiber /= fiber.std()
paper = 1 - .06 * np.clip(tooth, 0, None) - .025 * fiber
fill(st.layer('纸', mode='multiply'), ALL, np.repeat(np.clip(paper, 0, 1)[..., None], 3, 2), edge=HARD)
look(st, OUT)
print(OUT)
