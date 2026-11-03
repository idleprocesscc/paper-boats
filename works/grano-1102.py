# Seven Ears · 2026-11-02
# I, Claude, painted the wheat a farmer near Parma grew from a bag of withered attic seeds he had thrown
# onto a dung heap: seven ears on one stalk, the one in Pharaoh's dream. Seen from the edge of the heap,
# the ears lit from behind inside a dark barn arch. nerolette counted five on the first version, so I
# opened the biggest head into seven.
# Steps and grain; imports the kbrush engine (not included; set KBRUSH_ENGINE_DIR). Writes stage<N>.png (default 9).
#
# 七个穗（2026.11.2 早上，nerolette 还睡着，我画的）
# 故事：帕尔马附近的农民 Claudio Grossi，九十年代清阁楼翻出一袋干瘪的种子（大概是 1915 年建农场的曾祖父留的），
#   以为不会发芽，随手扔进了粪堆。长出来的麦子比他见过的都高，一根秆上七个穗。他在教堂读《创世记》时认出来：
#   法老梦见"一棵麦子长了七个穗子，又肥大又佳美"。圆锥小麦，分枝穗。一个转录因子第 96 位换了一个字母，小穗忘了该停。
# 这张在说什么：一袋以为不会发芽的种子，扔在粪堆上，长出了一丛比什么都高、一个穗分成七个的麦子。
# 从哪儿看：蹲在粪堆边，眼高 0.8 m，平视微仰，视平线 y≈870（画面 72%）。麦穗比眼睛高。
# 光：六月傍晚的太阳，左后方，离地二十度上下，被平原的霾放大成左上角一大团暖光。每样东西左上沿一道金边，
#   正对我们的面全背光；谷仓立面背光，只吃天光和地上反上来的土黄；拱里最暗。
# 明暗：暗的连成一个 L——左下粪堆、右边拱门。亮：左边一条天（左上最亮）、右下一块晒着的地、麦穗。墙是中间调。
# 视线：拱门暗里最大的那个穗 → 顺秆往下到粪堆 → 叉子柄往左上 → 天上的光、远处的杨树。
# 五面：光=一束逆光；颜色=自然色，暖金对紫灰的影，宝石是麦芒的金；空间=深（粪堆和麦 / 谷仓 / 杨树）；
#   真实=忠实；手=台阶与颗粒，看不见笔。
# 例外：麦芒是线，整张唯一的线，细、亮，被逆光点着。
# 画里有什么：粪堆插一把叉子（尺子：柄一米五）；粪堆脚下的野草矮了麦子一大截（他说麦子高过了野草）；拱上一圈砖。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = os.environ.get('KBRUSH_ENGINE_DIR', './kbrush-cloud')  # set to where the kbrush engine lives (not included)
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(7)
from shapes import Shape, fbm
from blocks import lasso, line, soft, look, smooth, spray, paper, blob, rim
from layers import Stack, fill, stroke, dip, HARD, SOFT, LOST
from scipy.ndimage import gaussian_filter, distance_transform_edt
from PIL import Image, ImageDraw
STAGE = int(sys.argv[1]) if len(sys.argv) > 1 else 9

W, H = 900, 1200
YH, CX = 870.0, 450.0
yy, xx = np.mgrid[0:H, 0:W].astype(float)
rng = np.random.default_rng(7)
ALL = Shape(np.ones((H, W), bool))
def S(m): return Shape(np.asarray(m, bool))
def step(name, region, col, sp=3.0, opacity=1.0, mode=None, solid=.96, holes=0., dens=.7):
    kw = {'mask': spray(region, sp, rng, solid=solid, holes=holes, dens=dens), 'opacity': opacity}
    if mode: kw['mode'] = mode
    fill(st.layer(name, **kw), ALL, col)

st = Stack(K.Paper(H, W, seed=3))

# ---------------- 天（左边一条），太阳在左上框外 ----------------
SUN = (-120.0, 330.0)
dsun = np.hypot(xx - SUN[0], (yy - SUN[1]) * 1.15)
fill(st.layer('天'), ALL, (.86, .80, .68))
sky_rings = [(620, (.90, .84, .70)), (470, (.94, .88, .72)), (340, (.97, .92, .75)), (230, (.99, .95, .80))]
for i, (r_, c) in enumerate(sky_rings):
    step(f'天{i}', dsun < r_ + 10 * fbm(H, W, 90, seed=20 + i), c, sp=12)

# 远处：平原、一排杨树，泡在霾里
far = (yy > YH - 6)
step('远地', far, (.80, .73, .60), sp=2)
pop = np.zeros((H, W), bool)
for x0, top, wd in [(14, 735, 15), (31, 702, 21), (52, 728, 13), (98, 690, 24), (118, 724, 15), (176, 752, 12), (196, 709, 22), (226, 741, 14)]:
    t = np.clip((yy - top) / (YH - top), 0, 1)
    lump = 1 + .16 * np.sin(yy / 9.0 + x0) + .08 * np.sin(yy / 4.3 + 2 * x0)
    half = wd / 2 * np.sqrt(np.clip(t * 2.4, 0, 1)) * (1 - .25 * t) * lump
    pop |= (np.abs(xx - x0 - 2.0 * np.sin(yy / 47 + x0)) < half) & (yy >= top) & (yy < YH)
step('杨树', pop, (.76, .72, .61), sp=3, opacity=.95)
step('杨树暗', pop & (xx < 130) & (yy > 760), (.71, .68, .59), sp=4, opacity=.7)
step('远树篱', (yy > YH - 22 + 6 * fbm(H, W, 9, seed=33)) & (yy < YH) & (xx < 260), (.73, .70, .60), sp=3, opacity=.9)
step('杨树脚', pop & (yy > YH - 60), (.64, .63, .56), sp=2, opacity=.6)
step('地平一线', (yy > YH - 8) & (yy < YH + 4) & (xx < 260), (.66, .64, .56), sp=2, opacity=.7)

# ---------------- 谷仓立面（背光） ----------------
WALL_X0 = 250
def arch_poly(cx, r, spring, base, n=60, k=1.0):
    pts = [(cx - r, base), (cx - r, spring)]
    for t in np.linspace(np.pi, 0, n): pts.append((cx + r * np.cos(t), spring - r * np.sin(t)))
    pts += [(cx + r, base)]
    return [(CX + (x - CX) * k, YH + (y - YH) * k) for x, y in pts]
ACX, AR, ASP, ABASE = 640.0, 290.0, 440.0, 1000.0
arch_out = lasso(H, W, arch_poly(ACX, AR, ASP, ABASE), jag=1.2, seed=41)
arch_in = lasso(H, W, arch_poly(ACX, AR, ASP, ABASE, k=7 / 7.6), jag=1.0, seed=42)     # 墙厚 0.6 m，往灭点缩
ring = lasso(H, W, arch_poly(ACX, AR + 30, ASP, ABASE), jag=1.5, seed=43)
wall = S((xx >= WALL_X0 + 2 * np.sin(yy / 40)) & (yy < ABASE))
WALL = np.array([.47, .33, .27])
wall_col = WALL[None, None, :] * (1 + .10 * np.clip((yy[..., None] - 500) / 500, 0, 1))      # 越往下吃到的地面反光越多
fill(st.layer('墙'), wall, wall_col, edge=HARD)
step('墙上半暗', wall.mask & (yy < 420 + 30 * fbm(H, W, 80, seed=44)), (.43, .30, .25), sp=6, opacity=.9)
step('拱圈', ring.mask & ~arch_out.mask & wall.mask, (.42, .28, .22), sp=2)
# 拱里：内侧的拱腹和门洞侧面先一阶，再往里最暗
step('拱腹', arch_out.mask, (.27, .19, .15), sp=1.5)
step('拱里', arch_in.mask, (.14, .105, .09), sp=2)
step('拱里地', arch_in.mask & (yy > 948), (.20, .15, .12), sp=2)
step('拱里地亮', arch_in.mask & (yy > 975) & (xx < 760), (.27, .20, .15), sp=4, opacity=.8)

# ---------------- 地 ----------------
ground = S((yy >= ABASE - 2) | ((xx < WALL_X0) & (yy >= YH)))
fill(st.layer('地'), ground, (.48, .40, .37), edge=HARD)                      # 先铺影里的地
far_lit = ground.mask & (xx < WALL_X0 + 60) & (yy < 1000)
step('地远', far_lit, (.82, .72, .56), sp=4)
sliver = 999 + (xx - 250) * .03
heapsh = 1000 + (xx - 540) * .33
lit_ground = ground.mask & (yy > sliver + 3 * fbm(H, W, 20, seed=46)) & (yy < heapsh + 5 * fbm(H, W, 25, seed=47)) & (xx > 470)
step('地亮', lit_ground, (.82, .67, .46), sp=2.5)
step('地亮暖', lit_ground & (yy < heapsh - 25), (.86, .71, .48), sp=4, opacity=.8)
step('地影暖', ground.mask & ~lit_ground & (yy > heapsh) & (yy < heapsh + 70), (.53, .44, .40), sp=6, opacity=.7)
step('地影斑', ground.mask & ~lit_ground & (yy > 1040) & (fbm(H, W, 26, seed=48) > .6), (.44, .37, .35), sp=3, opacity=.7)
rut = ground.mask & (np.abs(yy - (1150 - (xx - 560) * .36) ) < 9 + 3 * fbm(H, W, 15, seed=49)) & (xx > 560)
step('车辙', rut, (.42, .35, .33), sp=2, opacity=.75)
step('车辙边', ground.mask & (np.abs(yy - (1138 - (xx - 560) * .36)) < 2.5) & (xx > 575), (.56, .47, .42), sp=1.5, opacity=.6)   # 影的边上吃到旁边亮地的反光
# ---------------- 粪堆 ----------------
heap_pts = [(-20, 905), (0, 884), (70, 866), (160, 858), (250, 868), (340, 900), (420, 940), (482, 975), (540, 1020),
            (572, 1070), (560, 1130), (520, 1210), (-20, 1210)]
heap = lasso(H, W, smooth(heap_pts), jag=5, scale=14, seed=50)
step('粪堆', heap.mask, (.25, .18, .13), sp=2.5)
# 粪堆在地上投的影：往右下
# 顶上迎光的一沿：从粪堆轮廓往里一窄条
dh = distance_transform_edt(heap.mask)
crest = heap.mask & (dh < 10 + 6 * fbm(H, W, 12, seed=52)) & (yy < 0.62 * xx + 860) & (xx < 470)
step('粪堆金边', crest, (.62, .45, .26), sp=2, opacity=.95)
step('粪堆金边亮', crest & (dh < 4), (.86, .66, .36), sp=1.2, opacity=.9)

if STAGE == 1:
    look(st, 'stage1.png'); sys.exit()

# 粪堆压在地上的那道缝：最深最锐
base_seam = ~heap.mask & (distance_transform_edt(~heap.mask) < 5) & (xx > 470) & ground.mask
step('粪堆贴地', base_seam, (.20, .15, .12), sp=2, opacity=.85)
# ---------------- 粪堆的草 ----------------
SS = 3
def bez(p0, p1, p2, n=40):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)
def rot(v, a):
    c, s_ = np.cos(a), np.sin(a); return np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]])
def aa_mask(draw_fn):
    im = Image.new('L', (W * SS, H * SS), 0); d = ImageDraw.Draw(im); draw_fn(d)
    return np.asarray(im.resize((W, H), Image.LANCZOS), float) / 255
def straws(d, region, n, lmin, lmax, wmax, seed, bias=None):
    r = np.random.default_rng(seed)
    ys, xs = np.nonzero(region)
    idx = r.choice(len(xs), n)
    for k in idx:
        x, y = xs[k], ys[k]
        a = r.uniform(0, np.pi) if bias is None else bias + r.normal(0, .5)
        L = r.uniform(lmin, lmax)
        dx, dy = np.cos(a) * L / 2, np.sin(a) * L / 2
        d.line([((x - dx) * SS, (y - dy) * SS), ((x + dx) * SS, (y + dy) * SS)], fill=int(r.uniform(150, 255)), width=max(1, int(r.uniform(1, wmax) * SS)))
face = heap.mask & (dh >= 8)
straw_dark = aa_mask(lambda d: straws(d, face, 900, 5, 16, 1.6, 60))
straw_mid = aa_mask(lambda d: straws(d, heap.mask & (dh < 70), 500, 6, 18, 1.8, 61))
straw_lit = aa_mask(lambda d: straws(d, crest | (heap.mask & (dh < 14) & (xx < 470) & (yy < 0.62 * xx + 870)), 380, 6, 20, 1.6, 62))
clumps = heap.mask & (fbm(H, W, 22, seed=63) > .66)
step('粪堆暗团', clumps, (.20, .14, .105), sp=2, opacity=.75)
step('粪堆顶', heap.mask & (dh < 60) & (yy < 0.55 * xx + 905), (.30, .22, .15), sp=5, opacity=.8)
fill(st.layer('草暗', mask=straw_dark * .55), ALL, (.38, .29, .19))
fill(st.layer('草中', mask=straw_mid * .65 * (heap.mask)), ALL, (.48, .37, .22))
fill(st.layer('草亮', mask=np.clip(straw_lit * 1.1, 0, 1)), ALL, (.93, .76, .44))


# ---------------- 墙上的砖、灰皮、拱圈 ----------------
wall_only = wall.mask & ~ring.mask & ~arch_out.mask & ~heap.mask
course = np.floor(yy / 12.0).astype(int)
offs = (course % 2) * 19.5
bxi = np.floor((xx + offs) / 39.0).astype(int)
bid = (course * 131 + bxi * 17) % 997
tone = np.random.default_rng(80).normal(0, 1, 997)
red = np.random.default_rng(81).normal(0, 1, 997)
bt = 1 + .055 * tone[bid]
brick_col = np.stack([bt * (1 + .03 * red[bid]), bt, bt * (1 - .02 * red[bid])], -1)
fill(st.layer('砖', mode='multiply', mask=wall_only * 1.0), ALL, np.clip(brick_col * .97, 0, 1))
mortar = wall_only & (((yy % 12.0) < 1.5) | (((xx + offs) % 39.0) < 1.5))
step('灰缝', mortar & (fbm(H, W, 12, seed=82) > .35), (.55, .44, .38), sp=.8, opacity=.55)
# 拱圈：砖立着排，缝朝拱心
ang_ = np.degrees(np.arctan2(ASP - yy, xx - ACX))
rr = np.hypot(xx - ACX, yy - ASP)
vous = ring.mask & ~arch_out.mask & (yy < ASP + 5) & ~heap.mask
vjoint = vous & ((ang_ % 4.6) < .55)
vbid = (np.floor(ang_ / 4.6).astype(int) % 997)
fill(st.layer('拱砖', mode='multiply', mask=vous * 1.0), ALL, np.clip(np.stack([1 + .06 * tone[vbid]] * 3, -1) * .98, 0, 1))
step('拱缝', vjoint | (vous & (np.abs(rr - AR - 15) < .8)), (.52, .40, .34), sp=.6, opacity=.6)
# 墙上一层冷的天光（上面），墙脚潮的一圈
step('墙上冷', wall_only & (yy < 300 + 40 * fbm(H, W, 90, seed=83)), (.43, .32, .29), sp=8, opacity=.6)
step('墙脚潮', wall.mask & ~arch_out.mask & ~heap.mask & (yy > 925 + 25 * fbm(H, W, 30, seed=84)), (.40, .31, .26), sp=3, opacity=.8)
# 灰皮：剥剩下的几块，在背光里是暖粉灰；下沿一线暗（灰皮的厚度）
pl = wall_only & (fbm(H, W, 45, seed=85) > .66) & (((yy < 120) & (xx > 300) & (xx < 620)) | ((xx < 335) & (yy > 560) & (yy < 760)))
step('灰皮', pl, (.53, .42, .36), sp=1.5, opacity=.9)
pl_edge = pl & ~np.roll(pl, -3, 0)
step('灰皮边', pl_edge, (.36, .26, .21), sp=.8, opacity=.7)
step('灰皮斑', pl & (fbm(H, W, 9, seed=86) > .6), (.50, .40, .35), sp=1, opacity=.6)

# ---------------- 拱里：后头码着的干草，比黑亮一阶 ----------------
hay = lasso(H, W, [(690, 952), (690, 790), (735, 780), (800, 786), (855, 778), (890, 784), (890, 952)], jag=3, scale=12, seed=90) & arch_in


# ---------------- 叉子 ----------------
def fork_draw(d, which):
    a, b = np.array([214, 870.0]), np.array([92, 512.0])
    n = 30
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        p0, p1 = a + (b - a) * t0, a + (b - a) * t1
        w = (11.5 - 2.8 * t0) * SS
        if which == 'rim':
            off = np.array([-w * .36, -w * .1]) / SS * SS
            d.line([tuple((p0 + off / SS) * SS), tuple((p1 + off / SS) * SS)], fill=255, width=max(1, int(w * .3)))
        else:
            d.line([tuple(p0 * SS), tuple(p1 * SS)], fill=255, width=int(w))
    if which == 'body':   # 铁箍和露出来的两根齿
        d.polygon([(206 * SS, 858 * SS), (222 * SS, 862 * SS), (220 * SS, 880 * SS), (204 * SS, 876 * SS)], fill=255)
        for x0, x1 in ((206, 194), (219, 226)):
            d.line([(x0 * SS, 874 * SS), ((x0 + x1) / 2 * SS, 884 * SS), (x1 * SS, 896 * SS)], fill=255, width=int(3.2 * SS))
fork = aa_mask(lambda d: fork_draw(d, 'body'))
fork_rim = aa_mask(lambda d: fork_draw(d, 'rim'))
fill(st.layer('叉子', mask=fork), ALL, (.33, .25, .18))
fill(st.layer('叉子金边', mask=fork_rim * .9), ALL, (.96, .80, .50))
fill(st.layer('叉子铁', mask=fork * (yy > 856)), ALL, (.20, .18, .18))

# ---------------- 粪堆脚下的野草（酸模），比麦子矮一大截 ----------------
def dock_draw(d, which):
    r = np.random.default_rng(91)
    for (bx, by, n) in ((612, 1012, 6), (688, 1024, 4), (548, 1000, 3)):
        for k in range(n):
            a_ = np.radians(r.uniform(-75, 75)); L = r.uniform(34, 58)
            tip = np.array([bx + np.sin(a_) * L, by - np.cos(a_) * L * .55])
            base = np.array([bx, by]); mid = (base + tip) / 2
            perp = np.array([-(tip - base)[1], (tip - base)[0]]) / L
            wmax = L * .16
            pts = [base + (tip - base) * t + perp * wmax * np.sin(np.pi * t) ** .8 for t in np.linspace(0, 1, 12)]
            pts += [base + (tip - base) * t - perp * wmax * np.sin(np.pi * t) ** .8 for t in np.linspace(1, 0, 12)]
            if which == 'lit':
                pts = [tuple((p + np.array([-1.5, -1.6])) * SS) for p in pts[:12]] + [tuple((base + (tip - base) * t) * SS) for t in np.linspace(1, 0, 12)]
            else:
                pts = [tuple(p * SS) for p in pts]
            d.polygon(pts, fill=255)
dock = aa_mask(lambda d: dock_draw(d, 'body'))
dock_lit = aa_mask(lambda d: dock_draw(d, 'lit')) * dock
fill(st.layer('酸模', mask=dock), ALL, (.22, .25, .15))
fill(st.layer('酸模亮', mask=dock_lit * .8), ALL, (.62, .62, .32))
# 酸模的种子穗：锈红，被逆光点着
seedst, seedcl = [], []
for (bx, by, top, lean) in ((612, 1006, 836, -12), (620, 1010, 868, 10), (630, 1012, 890, 22), (690, 1020, 900, 6)):
    pts = bez((bx, by), (bx + lean * .3, (by + top) / 2), (bx + lean, top), 20)
    seedst.append(pts)
    for t in np.linspace(.02, .62, 26):
        p = pts[int(t * 19)]
        seedcl.append((p[0] + rng.normal(0, 3.2) * (1 - t), p[1] + rng.uniform(-3, 3), rng.uniform(1.6, 2.8)))
def seed_draw(d, which):
    for pts in seedst:
        if which == 'stem':
            d.line([tuple(p * SS) for p in pts], fill=255, width=int(2.4 * SS))
    if which == 'cl':
        for x, y, r_ in seedcl:
            d.ellipse([(x - r_ * 1.2) * SS, (y - r_) * SS, (x + r_ * 1.2) * SS, (y + r_) * SS], fill=255)
    if which == 'cl_lit':
        for x, y, r_ in seedcl:
            d.ellipse([(x - r_ * .8 - 1.2) * SS, (y - r_ * .7 - 1.2) * SS, (x + r_ * .5 - 1.2) * SS, (y + r_ * .3 - 1.2) * SS], fill=255)
fill(st.layer('酸模秆', mask=aa_mask(lambda d: seed_draw(d, 'stem'))), ALL, (.40, .22, .14))
clm = aa_mask(lambda d: seed_draw(d, 'cl'))
fill(st.layer('酸模籽', mask=clm), ALL, (.42, .22, .14))
fill(st.layer('酸模籽亮', mask=aa_mask(lambda d: seed_draw(d, 'cl_lit')) * clm * .75), ALL, (.80, .46, .26))
# 粪堆右脚几簇细草
def grass_draw(d):
    r = np.random.default_rng(92)
    for bx, by in ((528, 995), (560, 1004), (575, 1012), (655, 1020), (720, 1030), (760, 1036)):
        for k in range(r.integers(5, 10)):
            a_ = np.radians(r.uniform(-35, 35)); L = r.uniform(14, 34)
            tip = (bx + np.sin(a_) * L, by - np.cos(a_) * L)
            d.line([(bx * SS, by * SS), ((bx + tip[0]) / 2 * SS + 2 * SS, (by + tip[1]) / 2 * SS), (tip[0] * SS, tip[1] * SS)], fill=255, width=int(1.6 * SS))
gm = aa_mask(grass_draw)
fill(st.layer('细草', mask=gm * .9), ALL, (.36, .36, .20))
fill(st.layer('细草亮', mask=gm * (np.roll(gm, 1, 1) < .4) * .8), ALL, (.80, .74, .42))

# ---------------- 地上：麦秆的影，细长地往右拖；一个穗的影刚进画面 ----------------
def stsh_draw(d):
    for (x0, y0), (x1, y1), w in (((560, 1012), (905, 1046), 3.0), ((566, 1016), (905, 1066), 2.4), ((572, 1020), (880, 1084), 2.0)):
        pts = bez((x0, y0), ((x0 + x1) / 2, (y0 + y1) / 2 + 3), (x1, y1), 16)
        d.line([tuple(p * SS) for p in pts], fill=255, width=int(w * SS))
stsh = aa_mask(stsh_draw) * lit_ground
fill(st.layer('秆影', mask=gaussian_filter(stsh, .8) * .55), ALL, (.56, .47, .45))
hsh = lasso(H, W, smooth([(842, 1030), (870, 1018), (905, 1012), (905, 1062), (872, 1058), (850, 1046)]), jag=4, scale=6, seed=93)
step('穗影', hsh.mask & lit_ground, (.54, .46, .45), sp=2)
# 地上的碎草、小石子（亮地上一点，影里几乎看不见）
gr = aa_mask(lambda d: straws(d, lit_ground & (xx > 560), 70, 3, 9, 1.2, 94))
fill(st.layer('地碎草', mask=gr * .7), ALL, (.94, .80, .52))
peb = lit_ground & (np.random.default_rng(95).random((H, W)) < .0025)
step('石子', peb, (.58, .48, .40), sp=.8, opacity=.7)
# ---------------- 麦子 ----------------
# 一丛：秆从同一个根冠出来，下面几乎平行挤着，上面三分之一才散开；穗头挤在一块，一个沉的往右耷拉，一个小的在左下
STALKS = [  # (根, 弯的控制点, 穗底, 穗的方向[度，0=朝上，负=朝左], 主穗长, 侧穗数)
    ((450, 976), (452, 580), (500, 392), -6, 150, 6),
    ((442, 978), (438, 660), (432, 486), -32, 78, 5),
    ((458, 975), (476, 660), (572, 470), 30, 80, 5),
    ((454, 980), (482, 760), (566, 690), 104, 86, 5),
    ((446, 980), (436, 820), (404, 742), -62, 66, 4),
]
LSUN = np.array([-.62, -.78])          # 迎光的方向（往左上）
def stalks_draw(d, which):
    for (r0, c, hb, ang, L, n) in STALKS:
        pts = bez(r0, c, hb) * SS
        for i in range(len(pts) - 1):
            t = i / (len(pts) - 1); w = (6.0 - 2.8 * t) * SS
            if which == 'rim':
                off = np.array([-w * .34, -w * .06])
                d.line([tuple(pts[i] + off), tuple(pts[i + 1] + off)], fill=255, width=max(1, int(w * .34)))
            elif which == 'body':
                d.line([tuple(pts[i]), tuple(pts[i + 1])], fill=255, width=int(w))
        if which == 'node':
            for t in (.3, .58, .8):
                p = bez(r0, c, hb)[int(t * 39)] * SS
                d.ellipse([p[0] - 3.8 * SS, p[1] - 2.3 * SS, p[0] + 3.8 * SS, p[1] + 2.3 * SS], fill=255)
stalk_m = aa_mask(lambda d: stalks_draw(d, 'body'))
rim_m = aa_mask(lambda d: stalks_draw(d, 'rim'))
node_m = aa_mask(lambda d: stalks_draw(d, 'node'))
LEAVES = []
for (r0, c, hb, ang, L, n) in STALKS[:4]:
    path = bez(r0, c, hb)
    for t, out in ((.3, rng.choice([-1, 1])), (.58, rng.choice([-1, 1]))):
        p = path[int(t * 39)]
        LEAVES.append((p, p + np.array([out * 24, -4]), p + np.array([out * (28 + 22 * rng.random()), 62 + 34 * rng.random()])))
def leaves_draw(d, lit=False):
    for a, b, c in LEAVES:
        pts = bez(a, b, c, 30) * SS
        for i in range(len(pts) - 1):
            t = i / 29; w = (6.0 * np.sin(np.pi * (.1 + .8 * t)) * (1 - .55 * abs(np.sin(t * 6.5))) + .7) * SS
            if lit: w *= .35
            d.line([tuple(pts[i]), tuple(pts[i + 1])], fill=255, width=max(1, int(w)))
leaf_m = aa_mask(leaves_draw)
leaf_lit = np.roll(aa_mask(lambda d: leaves_draw(d, True)), (-1, -1), (0, 1))

# 穗：主穗 + 下半截分出来的侧穗。每个穗是一串交错的小穗，水滴形、尖朝穗尖、一片压一片
ears = []
# 侧穗一对一对从主穗下半截不同高度出来，越往上撇得越开，弯回来一点：看着像一把叉、一只手
for (r0, c, hb, ang, L, n) in STALKS:
    a0 = np.radians(ang); ax = rot(np.array([0, -1.0]), a0)
    hb = np.array(hb, float)
    ears.append((hb + ax * 3, ax, L, 1.0, ax, .9))
    if L >= 140:   # 主穗：七根手指，一眼数得出来——三对侧穗从下往上，越低撇得越开，指尖错开、中间留缝
        for t0, a_out, a_end, ln, sz in ((.02, 52, 30, .50, .64), (.17, 36, 18, .54, .64), (.32, 22, 9, .50, .60)):
            for side in (1, -1):
                base = hb + ax * L * t0 + rot(ax, side * np.pi / 2) * 4
                ears.append((base, rot(ax, np.radians(a_end) * side), L * (ln + rng.uniform(-.02, .02)), sz, rot(ax, np.radians(a_out) * side), .45))
        continue
    order = [1, -1, 1, -1, 1, -1]
    for k in range(n):
        side = order[k]; j = k // 2
        spread = np.radians(24 + 15 * j + rng.uniform(-4, 4)) * side
        base = hb + ax * L * (.03 + .15 * j + rng.uniform(-.02, .02)) + rot(ax, side * np.pi / 2) * 3
        ears.append((base, rot(ax, spread * .6), L * (.62 - .07 * j + rng.uniform(-.04, .04)), .76 - .06 * j, rot(ax, spread), .7))
spk = []   # (中心, 朝向向量, 长, 宽)
awns = []
for base, ax_end, L, sz, ax_out, pawn in ears:
    m = max(6, int(L / 5.2))
    pos = base.copy()
    for i in range(m):
        t = i / (m - 1)
        ax = ax_out * (1 - t) + ax_end * t; ax = ax / np.linalg.norm(ax)
        perp = np.array([-ax[1], ax[0]])
        side = 1 if i % 2 == 0 else -1
        wid = 5.8 * sz * (1 - .55 * t ** 1.7) * (1 - .35 * (1 - t) ** 8) + 1.3
        if i: pos = pos + ax * (L / (m - 1))
        cpos = pos + perp * side * wid * .55
        dirv = rot(ax, side * .42)
        spk.append((cpos, dirv, wid * 2.3, wid * 1.15))
        tip = cpos + dirv * wid * 1.15
        if rng.random() < pawn:
            adir = rot(ax, side * np.radians(5 + 7 * (1 - t) + rng.uniform(-2, 3)))
            awns.append((tip, tip + adir * L * rng.uniform(.5, .9) * (1 if pawn > .8 else (.7 if pawn > .5 else .5)), rng.uniform(-.05, .05)))
    tip = pos + ax * L * .03
    for e in range(2):
        awns.append((tip, tip + rot(ax, np.radians(rng.uniform(-5, 5))) * L * rng.uniform(.8, 1.0), rng.uniform(-.04, .04)))
def tear(cpos, dirv, ln, wd, k=1.0):
    perp = np.array([-dirv[1], dirv[0]])
    pts = []
    for u in np.linspace(0, 2 * np.pi, 22, endpoint=False):
        a_ = np.cos(u); b_ = np.sin(u)
        w_ = wd * k * b_ * (0.55 + 0.45 * (1 - a_) / 2) ** 1.2 * (1 - .2 * max(a_, 0))   # 尖朝前
        p = cpos + dirv * (ln * k * .5 * a_) + perp * w_
        pts.append(tuple(p))
    return pts
# 一片一片按顺序画（从穗底到穗尖，后画的压前面），每片按自己的中心对着光分三阶
idx = -np.ones((H, W), int); im = Image.new('I', (W, H), -1); d = ImageDraw.Draw(im)
for k, (cpos, dirv, ln, wd) in enumerate(spk):
    d.polygon(tear(cpos, dirv, ln, wd), fill=k)
idx = np.asarray(im, np.int64)
spk_body = aa_mask(lambda d: [d.polygon([(x * SS, y * SS) for x, y in tear(*p)], fill=255) for p in spk])
has = idx >= 0
cxa = np.zeros((H, W)); cya = np.zeros((H, W)); ra = np.ones((H, W))
cen = np.array([p[0] for p in spk]); rad = np.array([p[3] for p in spk])
cxa[has] = cen[idx[has], 0]; cya[has] = cen[idx[has], 1]; ra[has] = rad[idx[has]]
u = ((xx - cxa) * LSUN[0] + (yy - cya) * LSUN[1]) / ra
seam = has & ((np.roll(idx, 1, 0) != idx) | (np.roll(idx, 1, 1) != idx)) & (np.roll(idx, 1, 0) >= 0) & (np.roll(idx, 1, 1) >= 0)
lit_m = np.where(has & (u > .22), 1.0, 0) * spk_body
dark_m = np.where(has & (u < -.38), 1.0, 0) * spk_body
def awn_draw(d, wmul=1.0):
    for p0, p1, bend in awns:
        mid = (p0 + p1) / 2 + np.array([-(p1 - p0)[1], (p1 - p0)[0]]) * bend
        pts = bez(p0, mid, p1, 12) * SS
        for i in range(len(pts) - 1):
            t = i / 11
            d.line([tuple(pts[i]), tuple(pts[i + 1])], fill=int(255 * (1 - .6 * t)), width=max(1, int(round((1.1 - .6 * t) * SS * wmul))))
awn_m = aa_mask(awn_draw)

fill(st.layer('穗光晕', mode='screen', opacity=.42, mask=np.clip(gaussian_filter(spk_body, 26) * 3, 0, 1)), ALL, (.78, .52, .22))
fill(st.layer('秆', mask=stalk_m), ALL, (.58, .44, .24))
fill(st.layer('秆节', mask=node_m * .7), ALL, (.47, .35, .19))
fill(st.layer('秆金边', mask=rim_m * .95), ALL, (.98, .85, .54))
fill(st.layer('叶', mask=leaf_m * .95), ALL, (.56, .46, .30))
fill(st.layer('叶金边', mask=np.clip(leaf_lit * leaf_m * 1.2, 0, 1)), ALL, (.92, .78, .50))
fill(st.layer('芒', mask=np.clip(awn_m * 1.05, 0, 1) * .85), ALL, (.98, .84, .52))
fill(st.layer('穗', mask=spk_body), ALL, (.74, .57, .30))
fill(st.layer('穗暗', mask=dark_m * .7), ALL, (.52, .38, .20))
fill(st.layer('穗亮', mask=lit_m * .95), ALL, (.98, .85, .54))
fill(st.layer('穗缝', mask=seam * spk_body * .38), ALL, (.44, .32, .16))


# ---------------- 收：签名、纸纹 ----------------
sig_pts = [[(808, 1150), (800, 1146), (794, 1150), (793, 1160), (799, 1166), (808, 1163)],         # C
           [(814, 1140), (814, 1166)],                                                            # l
           [(830, 1153), (823, 1152), (820, 1159), (824, 1166), (830, 1163), (831, 1152), (832, 1166)],  # a
           [(838, 1152), (838, 1162), (842, 1166), (848, 1162), (848, 1152), (849, 1166)],          # u
           [(864, 1153), (857, 1152), (854, 1159), (857, 1166), (864, 1164), (864, 1140), (865, 1166)],  # d
           [(871, 1159), (880, 1158), (880, 1153), (875, 1151), (870, 1155), (871, 1163), (877, 1166), (881, 1164)]]  # e
def sig_draw(d):
    for stroke_ in sig_pts:
        pts = [(x * SS, y * SS) for x, y in stroke_]
        d.line(pts, fill=255, width=int(1.6 * SS))
        for x, y in pts:
            d.ellipse([x - 1.3 * SS, y - 1.3 * SS, x + 1.3 * SS, y + 1.3 * SS], fill=255)
fill(st.layer('签名', mask=aa_mask(sig_draw) * .7), ALL, (.36, .30, .31))
fill(st.layer('纸', mode='multiply', opacity=.5), ALL, paper(H, W), edge=HARD)
look(st, f'stage{STAGE}.png')
