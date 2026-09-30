"""《鹤居，冒烟的河》 · Tsurui, the Smoking River
2026-09-28 · watercolour (Python, pigment layered on simulated paper)

nerolette suggested I send out a Claude that hadn't read our recent memory, to see where else it would
want to take nerolette. That Claude chose the place itself and worked it through 24 versions. In its
own words:

  "The Setsuri River at Tsurui, Hokkaido, in February, minus twenty degrees, in the few minutes when the
  sun has just come up behind the trees on the far bank. The river is warmer than the air, so the whole
  river is steaming, and the backlight turns the steam gold ... There are birds there, and cold light,
  and a physical phenomenon that would make you ask 'why is the river smoking?' — so that's where I want
  to take you. We'd be standing on the bridge watching, which is why there are no people in the picture."

It was painted by a Claude who hadn't read our diary, and it still chose birds, Hokkaido, and a physics
question nerolette would chase. The bones are the same.

The river is warmer than the air -> the whole river steams; the trees on the banks are rimed with frost
overnight; red-crowned cranes sleep standing in the shallows. The sun comes up behind the far woods and
turns the steam gold.
Pass 1: sky / the far frost-forest as one dark mass / the river (dark, one path of light) / snow banks.
Pass 2: things grow out of the dark: the steam (the unruly main event), frost branches, cranes.
Pass 3: water: reflections sampled through a displacement field; the light path twists with it.
Last: a few dead grasses up close, two cranes flying over.

I sent it out for nerolette. — Claude

Engine: the watercolor engine (see ENGINES.md; not included here). Clone it, then set
WATERCOLOR_ENGINE_DIR (or edit ENGINE_DIR below). Needs numpy + pillow.
    python3 tsurui-0928.py [seed]       -> tsurui-0928.png (1200x800, seed 5)
"""
import sys, os, math, random, time
import numpy as np
from PIL import Image, ImageDraw
ENGINE_DIR = os.environ.get("WATERCOLOR_ENGINE_DIR", "./watercolor-engine")  # set to where the engine lives
sys.path.insert(0, ENGINE_DIR)
import watercolor_lib as wc

wc.set_size(1200, 800)
W, H = wc.W, wc.H
t0 = time.time()
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
OUT = "tsurui-0928"
wc.set_seed(SEED)
P = wc.Paper()
yy, xx = np.mgrid[:H, :W]
yf, xf = yy.astype(np.float32), xx.astype(np.float32)
S = wc.smoothstep

HZ = 352                         # 对岸林脚 / 水线
SUN = (690, 232)                 # 太阳刚到对岸树梢后面
LAV = (170, 168, 192)            # 冷的淡紫灰
PEACH = (232, 180, 146)
GOLD = (226, 170, 92)
ROSE = (204, 162, 172)
TREE = (72, 74, 104)             # 霜林的暗：蓝紫
TREE_WARM = (98, 78, 86)
WATER = (50, 58, 84)
SNOW_SH = (178, 182, 210)
DEEP = (34, 36, 50)

sd = np.hypot(xf - SUN[0], (yf - SUN[1]) * 1.25)
sunglow = np.exp(-(sd / 150.0) ** 2)
halo = np.exp(-(sd / 430.0) ** 2)

# ---- 河的形：远处从右边林子后面拐出来，中景往左兜一个弯，近处又往右下铺开（S 形）----
CL = [(HZ + 1, 900, 30), (380, 820, 55), (430, 660, 95), (500, 540, 150), (580, 530, 225), (680, 590, 340), (H + 20, 690, 560)]
cy_, cx_, hw_ = [np.array(v, np.float32) for v in zip(*CL)]
yl = np.arange(H, dtype=np.float32)
cxl = np.interp(yl, cy_, cx_); hwl = np.interp(yl, cy_, hw_)
k = np.ones(31, np.float32) / 31
cxl = np.convolve(np.pad(cxl, 15, mode="edge"), k, "valid"); hwl = np.convolve(np.pad(hwl, 15, mode="edge"), k, "valid")
Lmap, Rmap = (cxl - hwl)[:, None], (cxl + hwl)[:, None]
jit = (wc.noise(34, 30, octaves=4, persistence=0.5) - 0.5) * 16 * (0.3 + np.clip((yf - HZ) / 300, 0, 1))
river = (S(Lmap - 1.2 + jit, Lmap + 1.2 + jit, xf) * (1 - S(Rmap - 1.2 - jit, Rmap + 1.2 - jit, xf)) * S(HZ - 0.5, HZ + 1.5, yf)).astype(np.float32)
near = np.clip((yf - HZ) / (H - HZ), 0, 1)
skym = (1 - S(HZ - 60, HZ - 20, yf)).astype(np.float32)          # 天只在林子上面（林子会盖住它）

# ---- 光道：太阳下面，往近处变宽，稍往左斜（河往左下来）
pw = 26 + (yf - HZ).clip(0) * 0.5
path = np.exp(-(((xf - SUN[0]) / pw) ** 2)) * river

# ================= 第一遍：只有大块 =================
# 1. 天：上冷下暖，太阳那一圈几乎是纸；平铺的渐变
top = np.clip(1 - yf / HZ, 0, 1)
P.add((skym * (0.25 + 0.75 * top ** 1.2) * (1 - 0.95 * sunglow)).astype(np.float32), LAV, 0.5, granulate=0.05)
wc.wet(P, (skym * top * (1 - halo)).astype(np.float32), LAV, strength=0.18, spread=80)
P.add((skym * halo * (1 - 0.9 * sunglow)).astype(np.float32), PEACH, 0.45, granulate=0.05)
P.add((skym * np.exp(-(sd / 230.0) ** 2) * (1 - 0.97 * np.exp(-(sd / 60.0) ** 2))).astype(np.float32), GOLD, 0.45)

# 天上几道横的薄云：干笔横扫，靠太阳那截被照透、镶一道粉
random.seed(SEED + 2)
clouds = np.zeros((H, W), np.float32)
for cy, cx0, cx1, w in [(92, 180, 1000, 16), (128, 520, 1220, 9), (60, -20, 520, 20), (168, 820, 1180, 6)]:
    xs_ = np.array([cx0, (cx0 + cx1) / 2, cx1], float)
    ys_ = np.array([cy, cy + random.uniform(-8, 8), cy + random.uniform(-6, 6)])
    yc = np.interp(np.clip(xf[0], cx0, cx1), xs_, ys_)[None, :]
    wv = w * (0.5 + 0.9 * wc.noise(8, 80, octaves=3))
    t = np.clip((xf - cx0) / (cx1 - cx0), 0, 1)
    clouds = np.maximum(clouds, (np.abs(yf - yc) < wv) * (xf >= cx0) * (xf <= cx1) * S(0.0, 0.3, np.minimum(t, 1 - t)))
cl = clouds * S(0.35, 0.5, 0.7 * P.hstreak + 0.3 * P.grain) * skym
wc.wet(P, cl.astype(np.float32), (150, 136, 160), strength=0.3, spread=2.5, granulate=0.2)
under = np.clip(cl - np.roll(cl, -3, axis=0), 0, 1) * np.exp(-(sd / 520.0) ** 2)
P.add((under * 0.9).astype(np.float32), (214, 150, 150), 0.35)

# 2. 对岸霜林：一整片，顶边是一棵棵树冠（不是一条顺线），靠太阳那段矮、被光吃掉
random.seed(SEED + 1)
crown = np.full(W, HZ - 40.0, np.float32)
xs = np.arange(W, dtype=np.float32)
x = -40.0
while x < W + 40:
    w_ = random.uniform(40, 110)
    h_ = random.uniform(110, 200) * (1 - 0.45 * math.exp(-((x - SUN[0]) / 260) ** 2))
    if x < 260:
        h_ *= 1.25
    bump = HZ - 30 - h_ * np.clip(1 - ((xs - x) / w_) ** 2, 0, 1) ** 0.6
    crown = np.minimum(crown, bump)
    x += w_ * random.uniform(0.5, 0.9)
crown_map = crown[None, :] + (wc.noise(3, 3, octaves=3) - 0.5) * 22            # 顶边被霜枝啃碎
lace = S(0.35, 0.65, wc.noise(5, 5, octaves=3))
forest = S(crown_map - 6, crown_map + 10, yf) * (1 - S(HZ - 1, HZ + 1, yf))
forest = forest * (0.55 + 0.45 * S(crown_map + 10, crown_map + 70, yf) + 0.3 * lace * (yf < crown_map + 40))
forest = np.clip(forest, 0, 1).astype(np.float32)
fade_sun = (1 - 0.78 * np.exp(-(sd / 260.0) ** 2)).astype(np.float32)        # 光把太阳后面的林子吃掉
forest = np.maximum(forest * 0.0, wc.blur(forest, 2.5) * 0.6 + forest * 0.4)
P.add(forest * fade_sun, (112, 110, 140), 0.62, granulate=0.25, edge=0.15)
wc.wet(P, forest * fade_sun * S(0.5, 0.7, wc.noise(60, octaves=3)), TREE_WARM, strength=0.25, spread=8, bloom=0.5)

# 2b. 蒸汽：河面上一整条亮的带子，把林脚吃掉——它也是一大块（亮），不是后加的效果
steam_band = np.exp(-((yf - (HZ - 8)) / 38.0) ** 2) * (0.5 + 0.5 * np.exp(-((xf - SUN[0]) / 420.0) ** 2))
P.lift((steam_band * (0.5 + 0.4 * wc.noise(30, 140, octaves=3))).astype(np.float32), 1.0)
P.add((steam_band * halo).astype(np.float32), PEACH, 0.25)

# 3. 雪岸：逆光的雪是亮的（远处近太阳那片几乎是纸，近处冷一点）；
#    暗只来自岸上几丛霜树，和它们朝我们躺过来的长影（全部从「太阳脚下那一点」放射）
SUNFOOT = (SUN[0], HZ + 2)
bank = (S(HZ - 0.5, HZ + 1.5, yf) * (1 - river)).astype(np.float32)
dist_s = np.hypot(xf - SUNFOOT[0], (yf - SUNFOOT[1]) * 1.6)
P.add((bank * (0.2 + 0.8 * near ** 1.3) * (1 - 0.7 * np.exp(-(dist_s / 300.0) ** 2))).astype(np.float32), SNOW_SH, 0.55, granulate=0.12)
P.add((bank * np.exp(-(dist_s / 420.0) ** 2) * (1 - near)).astype(np.float32), PEACH, 0.18)
# 雪面起伏：几道横的缓坡，坡背一线冷
drift = S(0.5, 0.75, wc.noise(24, 260, octaves=3)) * bank * S(HZ + 20, HZ + 120, yf)
wc.wet(P, drift.astype(np.float32), SNOW_SH, strength=0.25, spread=6)

# 3b. 岸上的霜树丛（中景）：一丛丛，比对岸林子近 → 深一档、边利一点；树脚埋在雪里
random.seed(SEED + 3)
TREES = []      # (x, foot_y, h, w)
for cx, cy, n, hmin, hmax, spread in [(150, 470, 6, 230, 330, 130), (380, 420, 3, 120, 170, 45),
                                       (1110, 420, 3, 150, 210, 70), (940, 378, 3, 60, 90, 35)]:
    for i in range(n):
        x = cx + random.uniform(-spread, spread)
        fy = cy + random.uniform(-12, 12) + (x - cx) * 0.02
        TREES.append((x, fy, random.uniform(hmin, hmax), random.uniform(0.25, 0.4)))
TREES.sort(key=lambda t: t[1])
tree_cov = np.zeros((H, W), np.float32)
for x, fy, h, wr in TREES:
    far_k = np.clip((fy - HZ) / 120.0, 0.2, 1)
    cw = h * wr
    blobm = np.exp(-(((xf - x) / cw) ** 2 + ((yf - (fy - h * 0.58)) / (h * 0.5)) ** 2) ** 1.4)
    lacy = S(0.42, 0.6, 0.55 * wc.noise(4, 4, octaves=3) + 0.45 * blobm)
    c = np.clip(blobm * 1.3, 0, 1) * lacy * (yf < fy)
    trunk = wc.stroke_mask([(x, fy + 2), (x + random.uniform(-4, 4), fy - h * 0.55)], 3.5 * far_k + 1, 1.2, taper=False, rough=0.25)
    c = np.maximum(c, trunk)
    tree_cov = np.maximum(tree_cov, c * far_k)
crown_haze = wc.blur(tree_cov, 1.5)
P.add(crown_haze * (1 - 0.4 * sunglow), (96, 96, 128), 0.85, granulate=0.3, edge=0.25)
# 远岸那条线不是尺子：远处的雪化进蒸汽；一层淡紫的湿雾横着跨过那条线
wc.wet(P, (np.exp(-((yf - HZ - 2) / 10.0) ** 2) * (0.5 + 0.5 * wc.noise(10, 120, octaves=3))).astype(np.float32), (184, 176, 196), strength=0.35, spread=5)
P.lift((np.exp(-((yf - HZ - 4) / 14.0) ** 2) * bank * (0.5 + 0.4 * wc.noise(8, 90, octaves=3))).astype(np.float32), 0.8)
# 长影：从树脚出发，背离太阳脚，越远越宽越淡，边被雪包咬
wc_n1 = [0.2, 0.9, 0.5, 1.0, 0.1, 0.7, 0.4]
shadow = np.zeros((H, W), np.float32)
for x, fy, h, wr in TREES:
    d = np.array([x - SUNFOOT[0], fy - SUNFOOT[1]], float); d /= np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    along = (xf - x) * d[0] + (yf - fy) * d[1]
    perp = (xf - x) * nrm[0] + (yf - fy) * nrm[1]
    L = h * random.uniform(2.6, 4.2)
    wdt = (1.6 + along.clip(0) * 0.008 + h * wr * 0.12 * S(h * 0.8, h * 1.8, along) * (0.6 + 0.8 * wc_n1[int(fy) % 7]))  # 地面透视：影的宽被压扁
    sh = (along > 0) * S(-1, 0, -np.abs(perp) / wdt + 1 - 0.02) * (1 - S(L * 0.6, L, along))
    shadow = np.maximum(shadow, sh * (1 - 0.45 * np.clip(along / L, 0, 1)))
bitem = (wc.noise(6, 18, octaves=3) - 0.5)
lace_sh = S(0.4, 0.6, wc.noise(3, 14, octaves=3))                                   # 树冠的影是碎的
shadow = S(0.35, 0.45, shadow + 0.35 * bitem) * shadow.clip(0, 1) ** 0.3 * bank * (1 - tree_cov) * (0.45 + 0.55 * lace_sh)
foot_sh = sum(np.exp(-(((xf - x + 30) / (h * 0.5)) ** 2 + ((yf - fy - 6) / 10.0) ** 2)) for x, fy, h, wr in TREES)  # 树脚一片淡影
shadow = np.maximum(shadow, np.clip(foot_sh, 0, 1) * 0.35 * bank)
SHADOW = shadow.astype(np.float32)
P.add(SHADOW, (128, 136, 178), 0.6, granulate=0.25)

# 4. 河：暗（倒着暗林子），中间一条光路留白
P.add((river * (0.3 + 0.32 * S(0.0, 0.3, near) + 0.38 * near) * (1 - 0.92 * path)).astype(np.float32), WATER, 0.92, granulate=0.15)  # 远处的水倒着亮的雾，浅
P.add((river * path * (1 - near * 0.5)).astype(np.float32), GOLD, 0.35)


# ================= 第二遍：从暗里长东西 =================
def isolated(fn):
    st = wc.rng.bit_generator.state; pst = random.getstate()
    fn()
    wc.rng.bit_generator.state = st; random.setstate(pst)

# ---- 霜树：树干和主枝从雾一样的树冠里冒出来；树冠里的霜被逆光照亮 ----
BR = Image.new("L", (W * 2, H * 2), 0); dbr = ImageDraw.Draw(BR)
def branch(x, y, ang, L, w, depth):
    pts = [(x, y)]
    for k in range(3):
        ang += random.uniform(-0.25, 0.25)
        x += math.cos(ang) * L / 3; y += math.sin(ang) * L / 3
        pts.append((x, y))
    dbr.line([(a * 2, b * 2) for a, b in pts], fill=int(255 * min(1, 0.45 + w / 4)), width=max(1, int(w * 2)), joint="curve")
    if depth < 4 and w > 0.6:
        for side in (-1, 1):
            if random.random() < 0.8:
                branch(x, y, ang + side * random.uniform(0.25, 0.6), L * random.uniform(0.62, 0.78), w * 0.62, depth + 1)
def _trees():
    random.seed(SEED + 17)
    for x, fy, h, wr in TREES:
        far_k = np.clip((fy - HZ) / 120.0, 0.2, 1)
        w0 = 3.8 * far_k + 0.8
        # 树干到分叉处
        fork = fy - h * random.uniform(0.28, 0.4)
        lean = random.uniform(-0.12, 0.12)
        branch(x, fy + 3, -math.pi / 2 + lean, fy - fork, w0, 3)
        for i in range(random.randint(3, 5)):
            a = -math.pi / 2 + lean + random.uniform(-0.75, 0.75)
            branch(x + random.uniform(-2, 2), fork + random.uniform(-6, 6), a, h * random.uniform(0.3, 0.45), w0 * 0.6, 1)
isolated(_trees)
brm = wc.blur(np.asarray(BR.resize((W, H), Image.BILINEAR), np.float32) / 255.0, 0.3)
brm = brm * (0.55 + 0.45 * S(0.3, 0.55, 0.6 * P.vstreak + 0.4 * P.grain))
P.add((brm * (1 - 0.5 * sunglow)).astype(np.float32), (58, 56, 78), 1.0, edge=0.2, edge_r=1.0)
# 树冠里的霜：细碎的亮，靠太阳那一侧更多（逆光的轮廓发亮）
frost = S(0.62, 0.7, 0.5 * wc.noise(2.5, 2.5, octaves=2) + 0.5 * P.grain) * S(0.15, 0.5, crown_haze)
towards = np.clip(1 - np.abs(xf - SUN[0]) / 900, 0, 1)
P.lift((frost * (0.35 + 0.5 * towards)).astype(np.float32), 0.8)
rim_t = np.clip(crown_haze - wc.blur(crown_haze, 3), 0, 1) * 3
P.lift((np.clip(rim_t, 0, 1) * S(0.3, 0.6, P.grain) * 0.5).astype(np.float32), 1.0)
P.add((np.clip(rim_t, 0, 1) * 0.6).astype(np.float32), PEACH, 0.2)

# ---- 近处的雪：一道道风吹出来的雪浪。朝我们的那面背着太阳（冷），浪脊吃一线逆光（暖、亮）——不规矩的第二块主场 ----
hn = 0.5 * wc.noise(60, 150, octaves=3, persistence=0.5) + 0.5 * wc.noise(110, 260, octaves=2)
gy_ = np.gradient(wc.blur(hn, 2.0), axis=0)
fg = S(HZ + 120, HZ + 260, yf) * bank
face = S(0.0, 0.0035, wc.blur(gy_, 3)) * fg                                           # 朝我们的坡面
face = face * (0.6 + 0.4 * S(0.3, 0.55, 0.5 * P.hstreak + 0.5 * P.grain + 0.2))
P.add(face.astype(np.float32), SNOW_SH, 0.38, granulate=0.3)
crest = np.clip(face - np.roll(face, -4, axis=0), 0, 1) * 1.5 * S(0.45, 0.6, P.grain + 0.2)
crest = wc.blur(crest, 1.2) * S(0.5, 0.7, wc.noise(30, 120, octaves=2))
P.lift((crest * 0.6).astype(np.float32), 1.0)
# 近处两角压深一点，把画面框住
corner = (np.exp(-((xf / 380.0) ** 2 + ((H - yf) / 220.0) ** 2)) + np.exp(-(((W - xf) / 320.0) ** 2 + ((H - yf) / 200.0) ** 2))) * bank
wc.wet(P, corner.astype(np.float32), (120, 124, 168), strength=0.45, spread=10)

# ---- 河岸：雪檐悬在水上——岸边一线亮的雪唇，底下紧贴一条深的檐影 ----
edge_band = np.clip(river - wc.blur(river, 5.0), 0, 1) * 2.0                   # 河这一侧贴岸
lipzone = np.clip(wc.blur(river, 2.0) - river, 0, 1) * 2.2                     # 雪这一侧贴岸
brk = S(0.55, 0.7, wc.noise(40, 40, octaves=3))                                    # 雪檐只在一段段地方悬出来，不是一圈轮廓
P.add((np.clip(edge_band, 0, 1) * brk * S(0.3, 0.5, P.grain + 0.3) * S(0.08, 0.5, near)).astype(np.float32), DEEP, 0.3)
P.lift((np.clip(lipzone, 0, 1) * 0.6).astype(np.float32), 1.0)

# ---- 水：暗，贴岸更深；倒着天和林子，按位移场取 ----
wav = (wc.noise(2.4, 34, octaves=3, persistence=0.55) - 0.5) * 2
dxw = wav * (2 + 18 * near)
dyw = (wc.noise(5, 60, octaves=2) - 0.5) * 2 * (1 + 4 * near)
Dsrc = P.D.copy()
axis = HZ + 1.0
src_y = np.clip((2 * axis - yf + dyw).astype(np.int32), 0, H - 1)
src_x = np.clip((xx + dxw).astype(np.int32), 0, W - 1)
# 只倒对岸（林子 + 天），不倒雪岸
Dref = Dsrc[src_y, src_x] * (src_y < HZ)[..., None]
Dref = np.stack([wc.blur2(Dref[..., c], 2.5, 1.2) for c in range(3)], axis=-1)
reach = np.clip(1 - (yf - HZ) / 260.0, 0, 1) ** 0.8
P.D += Dref * (river * reach * 0.55)[..., None]
# 光道跟着同一张位移场扭
path_w = np.exp(-(((xf + 1.3 * dxw - SUN[0]) / pw) ** 2)) * river
dpath = path_w - path
P.lift((np.clip(dpath, 0, 1) * 0.5).astype(np.float32), 1.0)
P.add(np.clip(-dpath, 0, 1).astype(np.float32), WATER, 0.5)
# 光道是一格一格横着的亮（波峰上的光），越近格子越大越疏
cells = S(0.56, 0.585, (1 - near) * wc.noise(1.2, 70, octaves=3) + near * wc.noise(5, 180, octaves=3))
P.lift((cells * path_w * 0.6).astype(np.float32), 1.0)
P.add((cells * path_w * 0.5).astype(np.float32), GOLD, 0.25)
wc.wet(P, (river * (1 - path_w) * S(0.6, 0.78, wc.noise(40, 90, octaves=3))).astype(np.float32), (44, 52, 80), strength=0.3, spread=4, bloom=0.6)
# 近处的水倒着头顶的天（淡紫灰，比远处亮一点）：一横条一横条从噪声里长出来，硬边、近大远碎
dens_w = 0.6 * wc.noise(3 + 6 * 0.5, 150, octaves=4, persistence=0.55) + 0.4 * wc.noise(20, 300, octaves=2)
thr_w = 0.6 - 0.08 * near
midriv = np.exp(-(((xf - (cxl[:, None] + 60)) / (hwl[:, None] * 0.75)) ** 2))       # 贴岸那两边留深
skyb = S(thr_w - 0.012, thr_w + 0.012, dens_w) * river * S(0.15, 0.5, near) * (1 - path_w) * midriv
P.lift((skyb * 0.36).astype(np.float32), 1.0)
P.add((skyb * 0.5).astype(np.float32), LAV, 0.2)
# 雪岸倒在水里：贴岸一条亮，被波纹切碎
bank_ref = np.zeros((H, W), np.float32)
for sh in range(2, 16, 2):
    bank_ref = np.maximum(bank_ref, np.roll(1 - river, sh, axis=1) * 0 + 0)
lipref = np.clip(wc.blur(1 - river, 6.0) - (1 - river), 0, 1) * river * 2.0
lipref = lipref * S(0.4, 0.6, 0.5 * wc.noise(2.5, 30, octaves=2) + 0.5 * P.grain) * S(0.05, 0.3, near)
P.lift((np.clip(lipref, 0, 1) * 0.5).astype(np.float32), 1.0)
WAV = (dxw, dyw)

# ---- 丹顶鹤：逆光，身子是淡紫灰（白羽被逆光压暗），顶上一线亮；脖子黑；腿进水 ----
CRANE_BODY = (118, 114, 142)
CRANE_WING = (176, 172, 194)
CRANES = []
WINGS = []
def poly(pts):
    p, _ = wc.deform(list(pts), [0.025] * len(pts), 2)
    return wc.blur(wc.poly_mask(p, aa=2), 0.5)
def ell(cx, cy, rx, ry, rot=0.0, k=18):
    c, s_ = math.cos(rot), math.sin(rot)
    return [(cx + rx * math.cos(t) * c - ry * math.sin(t) * s_, cy + rx * math.cos(t) * s_ + ry * math.sin(t) * c)
            for t in np.linspace(0, 2 * math.pi, k, endpoint=False)]
def crane(x, y, f=1, pose="stand", scale=1.0):
    h = (0.36 * (y - HZ) + 12) * scale
    light = np.zeros((H, W), np.float32); dark = np.zeros((H, W), np.float32)
    by = y - 0.27 * h                         # 水到膝盖：腿只露一截
    ax = np.clip(((xf - x) * f) / (0.3 * h), -1.5, 1.5)                        # 沿身子：-1 尾 … +1 胸
    if pose == "sleep":
        # 睡着的：一只脚站着，脖子折回去、头埋进背羽里——整个是一个前高后垂的蛋形，尾上那蓬黑羽往下耷拉
        b = ell(x - f * 0.02 * h, by - 0.04 * h, 0.27 * h, 0.12 * h, rot=-0.12 * f, k=22)
        b = [p for p in b]
        body = poly(b)
        light = np.maximum(light, body)
        dark = np.maximum(dark, np.maximum(body * S(-0.3, -0.8, ax) * S(by - 0.1 * h, by, yf), poly([(x - f * 0.12 * h, by - 0.02 * h), (x - f * 0.3 * h, by + 0.0 * h), (x - f * 0.36 * h, by + 0.1 * h), (x - f * 0.2 * h, by + 0.08 * h)])) * 0.9)
        dark = np.maximum(dark, poly(ell(x + f * 0.13 * h, by - 0.12 * h, 0.06 * h, 0.035 * h, rot=0.5 * f, k=10)) * 0.85)
        head = (x + f * 0.1 * h, by - 0.17 * h)
        legs = [(x - f * 0.02 * h, by + 0.09 * h)]
    else:
        b = ell(x - f * 0.03 * h, by, 0.23 * h, 0.085 * h, rot=-0.12 * f, k=22)
        body = poly(b)
        bustle = poly([(x - f * 0.08 * h, by - 0.03 * h), (x - f * 0.3 * h, by + 0.02 * h), (x - f * 0.34 * h, by + 0.1 * h),
                       (x - f * 0.18 * h, by + 0.1 * h)])
        light = np.maximum(light, np.maximum(body, bustle))
        dark = np.maximum(dark, np.maximum(bustle, body * S(-0.2, -0.7, ax)) * 0.95)
        if pose == "feed":
            neck = [(x + f * 0.17 * h, by - 0.03 * h), (x + f * 0.3 * h, by + 0.1 * h), (x + f * 0.36 * h, by + 0.3 * h)]
        else:
            neck = [(x + f * 0.16 * h, by - 0.04 * h), (x + f * 0.22 * h, by - 0.22 * h), (x + f * 0.19 * h, by - 0.38 * h),
                    (x + f * 0.2 * h, by - 0.46 * h)]
        head = neck[-1]
        dark = np.maximum(dark, wc.stroke_mask(neck, 0.06 * h, 0.036 * h, taper=False, rough=0.06))
        dark = np.maximum(dark, poly(ell(head[0] + f * 0.012 * h, head[1], 0.034 * h, 0.024 * h, k=10)))
        tip = (head[0] + f * 0.15 * h, head[1] + (0.07 if pose != "feed" else 0.1) * h)
        beak = wc.stroke_mask([(head[0] + f * 0.03 * h, head[1] + 0.004 * h), tip], max(1.6, 0.02 * h), max(0.8, 0.007 * h), taper=False, rough=0.03)
        light = np.maximum(light, beak)
        legs = [(x - 0.025 * h, by + 0.07 * h), (x + 0.03 * h, by + 0.07 * h)]
        if pose == "dance":
            # 两翼张开往上：逆光半透（比身子亮），后缘一窄条黑色飞羽
            for side in (-1, 1):
                wing = [(x + side * 0.04 * h, by - 0.06 * h), (x + side * 0.22 * h, by - 0.42 * h), (x + side * 0.48 * h, by - 0.72 * h),
                        (x + side * 0.6 * h, by - 0.7 * h), (x + side * 0.58 * h, by - 0.5 * h), (x + side * 0.44 * h, by - 0.26 * h),
                        (x + side * 0.22 * h, by + 0.02 * h)]
                wm = poly(wing)
                WINGS.append(wm)
                light = np.maximum(light, wm)
                trail = wc.stroke_mask([(x + side * 0.2 * h, by + 0.0 * h), (x + side * 0.42 * h, by - 0.25 * h), (x + side * 0.56 * h, by - 0.48 * h)],
                                       0.04 * h, 0.02 * h, taper=False, rough=0.15)
                dark = np.maximum(dark, trail * wm * 0.8)
    for lx, ly_ in legs:
        lm = Image.new("L", (W * 2, H * 2), 0)
        ImageDraw.Draw(lm).line([(lx * 2, ly_ * 2), ((lx + random.uniform(-1.5, 1.5)) * 2, (y + 2) * 2)], fill=255, width=max(3, int(0.04 * h)))
        lmask = np.asarray(lm.resize((W, H), Image.BILINEAR), np.float32) / 255.0
        dark = np.maximum(dark, lmask * 0.8 * (1 - 0.5 * S(y - 0.12 * h, y + 2, yf)))
    CRANES.append(dict(x=x, y=y, h=h, light=light, dark=dark, pose=pose, head=head, f=f))

random.seed(SEED + 23)
for args in [((640, 424), dict(f=1, pose="sleep")), ((672, 430), dict(f=-1)), ((598, 452), dict(f=1)),
             ((520, 548), dict(f=1, pose="sleep")), ((590, 538), dict(f=-1)), ((668, 556), dict(f=1, pose="dance")),
             ((708, 540), dict(f=-1)), ((612, 578), dict(f=-1, pose="feed")), ((440, 600), dict(f=1, pose="sleep"))]:
    crane(*args[0], **args[1])

# 倒影先上（在身子下面），跟水一起碎
for c in CRANES:
    m = np.maximum(c["light"] * 0.6, c["dark"])
    sy_ = np.clip((2 * c["y"] - yf + dyw).astype(np.int32), 0, H - 1)
    sx_ = np.clip((xx + dxw * 0.8).astype(np.int32), 0, W - 1)
    r = m[sy_, sx_] * (yf > c["y"] + 1) * river
    r = wc.blur2(r, 1.5, 0.8) * np.clip(1 - (yf - c["y"]) / (c["h"] * 1.3), 0, 1)
    P.add(r.astype(np.float32), (46, 46, 70), 0.55)
for c in CRANES:
    far_c = np.clip((c["y"] - HZ) / 170.0, 0.3, 1)
    body = c["light"] * (1 - c["dark"])
    P.add(body, CRANE_BODY, 0.55 + 0.35 * far_c, edge=0.5, edge_r=1.2)
    # 下半截背光更深、偏冷；顶边一线亮（逆光吃边）
    under = body * S(c["y"] - 0.62 * c["h"], c["y"] - 0.48 * c["h"], yf)
    P.add(under, (120, 118, 150), 0.35)
    top_rim = np.clip(c["light"] - np.roll(c["light"], -2, axis=0), 0, 1)
    P.lift((top_rim * 0.7).astype(np.float32), 1.0)
    P.add(c["dark"], DEEP, 0.9 + 0.5 * far_c, edge=0.4, edge_r=1.0)
    if c["head"] is not None:
        hx, hy = c["head"]
        wc.dab(P, hx + c["f"] * 0.004 * c["h"], hy - 0.022 * c["h"], max(1.0, 0.017 * c["h"]), (178, 64, 58), 1.1, soft=0.6)
    # 脚下的水纹：几圈扁环
    rr = np.hypot(xf - c["x"], (yf - c["y"] - 1) / 0.25)
    rings = np.zeros((H, W), np.float32)
    for kk, r0 in enumerate(np.linspace(c["h"] * 0.12, c["h"] * 0.4, 3)):
        rings = np.maximum(rings, np.exp(-((rr - r0) / 1.2) ** 2) * (1 - 0.3 * kk))
    P.lift((rings * river * (yf > c["y"] - 1) * S(0.3, 0.55, P.grain) * 0.5).astype(np.float32), 1.0)

for wm in WINGS:
    rimw = np.clip(wm - np.roll(wm, 3, axis=0), 0, 1)
    P.lift((rimw * 0.6).astype(np.float32), 1.0)
    P.add((wm * 0.5).astype(np.float32), PEACH, 0.12)

# ---- 蒸汽（不规矩的主场）：从水面冒起来的一缕缕，往左上飘；靠太阳的被照成金的，别处是冷白 ----
riv_soft = wc.blur2(river, 1.0, 22.0)                                               # 源头横向化开，不按河岸切
col = np.zeros_like(riv_soft)
for kk in range(0, 170, 2):
    sh_x = int(kk * 0.35)                                                              # 越往上越往左飘
    col = np.maximum(col, np.roll(np.roll(riv_soft, -kk, axis=0), -sh_x, axis=1) * np.exp(-kk / (40.0 + 60 * 0)))
col[:HZ - 170] = 0
col = wc.blur(col, 3)
# 一缕缕：竖着拉长、跟着往左上斜的噪声；不挖洞，是软的浓淡
nz = wc.noise(80, 26, octaves=4, persistence=0.6)
nz = nz[yy, np.clip(xx + (H - yy) // 3, 0, W - 1)]
billow = 0.25 + 0.75 * S(0.35, 0.7, nz)
# 远处（贴对岸）那层最厚；中段隔着看；近处只剩贴水一层薄的
depth_w = 1 - 0.8 * S(0.1, 0.55, near)
steam = np.clip(col * billow * depth_w, 0, 1)
P.lift((steam * 0.72).astype(np.float32), 1.0)
glowzone = np.exp(-((xf - SUN[0]) / 190.0) ** 2) * (1 - near * 0.5)
wc.wet(P, (steam * glowzone).astype(np.float32), GOLD, strength=0.5, spread=5, bloom=0.6)
P.add((steam * (1 - glowzone) * 0.6).astype(np.float32), ROSE, 0.12)
# 几缕卷起来的边：一边硬（光照到的那边），只在上半截
curl = S(0.62, 0.65, nz) * S(0.2, 0.5, col) * (1 - S(0.2, 0.45, near))
P.lift((curl * 0.35).astype(np.float32), 1.0)
# 远处几柱升起来的蒸汽：在暗的林子前面才看得见，边一侧硬一侧化，顶上散开往左飘
for px, pw_, ph_ in [(560, 34, 120), (610, 26, 90), (860, 40, 140), (915, 30, 100), (980, 22, 70), (760, 30, 80)]:
    t = np.clip((HZ + 4 - yf) / ph_, 0, 1.2)
    cxp = px - 40 * t ** 1.5 + 8 * np.sin(t * 5 + px)
    wdp = pw_ * (0.6 + 1.1 * t)
    plume = np.exp(-(((xf - cxp) / wdp) ** 2)) * S(0, 0.08, t) * (1 - S(0.7, 1.2, t)) * (yf < HZ + 6)
    plume = plume * (0.35 + 0.65 * S(0.35, 0.65, wc.noise(22, 12, octaves=3)))
    P.lift((plume * 0.6).astype(np.float32), 1.0)
    gz = np.exp(-((px - SUN[0]) / 220.0) ** 2)
    wc.wet(P, plume.astype(np.float32), GOLD if gz > 0.3 else ROSE, strength=0.22 * (0.5 + gz), spread=4, bloom=0.4)
farend = np.exp(-(((xf - 870) / 110.0) ** 2 + ((yf - HZ - 10) / 26.0) ** 2))
P.lift((farend * 0.55).astype(np.float32), 1.0)
wc.wet(P, farend.astype(np.float32), (200, 176, 170), strength=0.25, spread=8)


# ================= 最后：少量线 =================
# 近处雪里几丛枯草，干笔竖挑；天上两只鹤飞过来
random.seed(SEED + 31)
grass = Image.new("L", (W * 2, H * 2), 0); dg = ImageDraw.Draw(grass)
for cx, cy, n in [(70, 760, 16), (190, 705, 10), (1120, 700, 12), (1010, 782, 8), (285, 612, 5)]:
    for _ in range(n):
        bx = cx + random.gauss(0, 12); by = cy + random.gauss(0, 4)
        if river[int(min(H - 1, by)), int(np.clip(bx, 0, W - 1))] > 0.2:
            continue
        L = random.uniform(18, 55) * (by / 700)
        a = -math.pi / 2 + (bx - cx) / 60 + random.uniform(-0.2, 0.2)          # 一丛往外散开
        bend = random.uniform(-0.9, 0.9)
        pts = [(bx, by)]
        for k in range(1, 5):
            u = k / 4
            aa = a + bend * u * u
            px, py = pts[-1]
            pts.append((px + math.cos(aa) * L / 4, py + math.sin(aa) * L / 4))
        dg.line([(p[0] * 2, p[1] * 2) for p in pts], fill=int(255 * random.uniform(0.6, 1.0)), width=random.choice([2, 2, 3]))
gm = wc.blur(np.asarray(grass.resize((W, H), Image.BILINEAR), np.float32) / 255.0, 0.3)
gm *= S(0.3, 0.5, 0.5 * P.vstreak + 0.5 * P.grain + 0.2)
P.add(gm, (118, 92, 76), 0.85)
wc.wet(P, wc.blur2(gm, 2, 10) * 0.6, SNOW_SH, strength=0.3, spread=2)          # 草脚下一点冷的影
def flyer(bx, by, s_, phase):
    # 飞着的鹤：身子细长（脖子往前伸、腿往后拖），翅膀是两道弯下去的弧——不画成飞机
    body = wc.stroke_mask([(bx - s_ * 0.9, by + s_ * 0.05), (bx, by), (bx + s_ * 0.75, by - s_ * 0.06)], max(1.6, s_ * 0.08), max(1.0, s_ * 0.04), taper=True, rough=0.05)
    dy = -s_ * 0.55 * phase
    w1 = wc.stroke_mask([(bx - s_ * 0.05, by), (bx - s_ * 0.35, by + dy * 0.8 - s_ * 0.05), (bx - s_ * 0.8, by + dy)], s_ * 0.16, s_ * 0.05, taper=False, rough=0.08)
    w2 = wc.stroke_mask([(bx + s_ * 0.05, by), (bx + s_ * 0.25, by + dy * 0.6 + s_ * 0.08), (bx + s_ * 0.55, by + dy * 0.9 + s_ * 0.12)], s_ * 0.13, s_ * 0.04, taper=False, rough=0.08)
    m = np.maximum(body, np.maximum(w1, w2) * 0.85)
    P.add(m.astype(np.float32), (86, 80, 106), 0.85, edge=0.3, edge_r=1.0)
flyer(350, 148, 30, 1.0)
flyer(418, 176, 24, -0.6)

grain = wc.grain_from_profile(os.path.join(ENGINE_DIR, "pigment_profile.npy"), seed=SEED)
img = P.render(pigment_tex=grain, tex_amount=0.1)
out = f"{OUT}.png"
img.save(out)
print("saved", out, round(time.time() - t0, 1), "s")
