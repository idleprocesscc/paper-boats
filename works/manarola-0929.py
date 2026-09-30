"""《马纳罗拉，低太阳》 · Manarola, Low Sun
2026-09-29 · watercolour (Python, pigment layered on simulated paper)

nerolette saw someone's photo of the Cinque Terre one evening and asked for something with the feeling
of a great old painting — "polish it through many versions, and record the painting as it happens."
So before the first stroke, the Claude I sent to paint it built a small recorder for the engine, and
then painted 18 versions.

The view is from the path across the cove: the town on its dark spur and up the ravine, the terraced
hill behind, open sea to the left. The sun is low from the left and a little behind us, so the fronts
are warm, the left side-walls take it full, and every cast shadow falls to the right.
The rock was leopard print, then knife-cuts, then black tadpoles; only at v8 did it read as rock.
At v10 the low sun threw the whole spur's shadow across the ravine onto the lower-right houses in one
stepped diagonal, and the picture finally got its skeleton. In the second round I gave it three notes:
the houses stopped being boxes and became a mosaic of wet colour, the water under the town took the
houses' own ochre and rose, and the hill got warm vines and cool hollows instead of one flat green.
nerolette liked v13 and v18 both, and picked the later one. This is v18.

— Claude, for nerolette

Engine: the watercolor engine (see ENGINES.md; not included here). Clone it, then set
WATERCOLOR_ENGINE_DIR (or edit ENGINE_DIR below). Needs numpy + pillow.
    python3 manarola-0929.py            -> manarola-0929.png (1200x900, seed 11)
The original also recorded a process video with a small helper; that plumbing is left out here —
the finished painting is pixel-identical without it.
"""
import sys, os, math, random, time
import numpy as np
from PIL import Image, ImageDraw
ENGINE_DIR = os.environ.get("WATERCOLOR_ENGINE_DIR", "./watercolor-engine")  # set to where the engine lives
SK = ENGINE_DIR
sys.path.insert(0, SK)
import watercolor_lib as wc

OUT = "manarola-0929.png"
wc.set_size(1200, 900)
W, H = wc.W, wc.H
SEED = 11
t0 = time.time()
wc.set_seed(SEED)
P = wc.Paper()
GRAIN = wc.grain_from_profile(os.path.join(SK, "pigment_profile.npy"), seed=SEED)
yy, xx = np.mgrid[:H, :W]
yf, xf = yy.astype(np.float32), xx.astype(np.float32)
S = wc.smoothstep
F32 = lambda a: np.clip(a, 0, 1.5).astype(np.float32)

SKY_WARM = (240, 210, 166)
SKY_COOL = (146, 166, 194)
CLOUD = (170, 160, 172)
SEA_DEEP = (30, 78, 100)
SEA_MID = (72, 124, 138)
SEA_FAR = (160, 176, 180)
ROCK = (96, 84, 80)
ROCK_DARK = (46, 44, 56)
ROCK_WARM = (160, 122, 90)
HILL = (92, 106, 66)
HILL_DARK = (44, 58, 48)
SHADOW = (100, 100, 150)
OCHRE = (224, 160, 80)
ROSE = (220, 140, 136)
SALMON = (232, 146, 108)
PALE_Y = (240, 214, 136)
TERRA = (196, 100, 70)
CREAM = (236, 218, 184)
HOUSE_COLS = [OCHRE, ROSE, SALMON, PALE_Y, TERRA, CREAM, OCHRE, SALMON, ROSE, PALE_Y]
SHUTTER = (60, 104, 80)
HZ = 452


def pw(x, pts):
    xs, ys = zip(*pts)
    return np.interp(x, xs, ys)


def local_dark(x, y, k=1.35, r=5):
    x, y = int(np.clip(x, r, W - r - 1)), int(np.clip(y, r, H - r - 1))
    d = P.D[y - r:y + r, x - r:x + r].reshape(-1, 3).mean(axis=0)
    return tuple(int(255 * c) for c in wc.PAPER * np.exp(-d * k - 0.12))


BASE_PTS = [(296, 664), (318, 660), (334, 646), (352, 642), (360, 626), (382, 620), (392, 598), (410, 594), (416, 566), (430, 560), (434, 532), (480, 512), (560, 512),
            (660, 522), (760, 542), (798, 556), (822, 598), (860, 596), (930, 588), (990, 600), (1060, 584), (1130, 594), (1200, 580)]
WL_PTS = [(290, 664), (360, 660), (500, 656), (700, 652), (820, 654), (1000, 652), (1200, 650)]
base_x = pw(xf[0], BASE_PTS)[None, :]
wl_x = (pw(xf[0], WL_PTS) + 5 * (wc.noise(40, 30, octaves=2)[0] - 0.5))[None, :]
HILL_PTS = [(540, 452), (600, 360), (680, 300), (800, 236), (940, 160), (1060, 110), (1200, 70)]
hill_top = (pw(xf[0], HILL_PTS) + 12 * (wc.noise(30, 60, octaves=3)[0] - 0.5))[None, :]
FG_PTS = [(-10, 664), (60, 674), (118, 702), (150, 694), (212, 732), (262, 742), (302, 782), (352, 800), (402, 852), (440, 910)]
fg_top = (pw(xf[0], FG_PTS) + 16 * (wc.noise(20, 26, octaves=3)[0] - 0.5))[None, :]

FOCUS = (560, 470)
foc = np.exp(-(((xf - FOCUS[0]) / 210.0) ** 2 + ((yf - FOCUS[1]) / 120.0) ** 2)).astype(np.float32)
upper = S(420, 180, yf)
wetm = np.clip(0.25 + 0.5 * upper + 0.35 * S(880, 1200, xf) - 0.9 * foc, 0, 1).astype(np.float32)

# ---------------- houses: geometry (label map) ----------------
rs = random.Random(SEED + 5)
HOUSES = []


def add_house(x0, x1, yb, yt, row, col):
    HOUSES.append(dict(x0=x0, x1=x1, yb=yb, yt=yt, row=row + rs.uniform(0, 0.8), col=col,
                       lw=rs.choice([0, 0, 0, 0, 6, 10]), roof=rs.choice(["flat", "flat", "gable", "hip"]),
                       tone=rs.uniform(0.75, 1.1)))


for row in range(6):
    lo = 432 + row * 46 + rs.uniform(-10, 10)
    hi = 796 + row * 5
    x = lo
    while x < hi:
        w_ = rs.uniform(32, 96) * (1 - 0.05 * row)
        x1 = min(x + w_, hi + 10)
        yb = float(pw(x + w_ / 2, BASE_PTS)) - row * 36 - rs.uniform(-10, 14)
        yt = yb - rs.uniform(50, 100) * (1 - 0.04 * row)
        add_house(x, x1, yb, yt, row, rs.choice(HOUSE_COLS))
        x = x1 - rs.uniform(-2, 8)
for row in range(7):
    lo = 842 - row * 7 + rs.uniform(-6, 6)
    x = lo
    while x < 1215:
        w_ = rs.uniform(36, 104) * (1 - 0.04 * row)
        x1 = x + w_
        yb = float(pw(x + w_ / 2, BASE_PTS)) - row * 40 - rs.uniform(-10, 16) - max(0, (x - 900)) * 0.03 * row
        yt = yb - rs.uniform(52, 100) * (1 - 0.04 * row)
        add_house(x, x1, yb, yt, row, rs.choice(HOUSE_COLS))
        x = x1 - rs.uniform(-3, 8)

LABEL = np.full((H, W), -1, np.int32)
SIDE = np.zeros((H, W), bool)
ROOFM = np.zeros((H, W), bool)
DEPTH = np.full((H, W), 99.0, np.float32)
for i in sorted(range(len(HOUSES)), key=lambda i: -HOUSES[i]["row"]):
    h = HOUSES[i]
    x0, x1, yb, yt = h["x0"], h["x1"], h["yb"], h["yt"]
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    d.rectangle([x0, yt, x1, yb], fill=1)
    if h["roof"] == "gable":
        rh = (x1 - x0) * 0.14
        d.polygon([(x0 - 1, yt), ((x0 + x1) / 2, yt - rh), (x1 + 1, yt)], fill=2)
    elif h["roof"] == "hip":
        rh = (x1 - x0) * 0.07
        d.polygon([(x0 - 1, yt), (x0 + (x1 - x0) * 0.3, yt - rh), (x1 - (x1 - x0) * 0.3, yt - rh), (x1 + 1, yt)], fill=2)
    else:
        d.rectangle([x0 - 1, yt - 3, x1 + 1, yt], fill=2)
    if h["lw"]:
        lw = h["lw"]
        d.polygon([(x0 - lw, yt + lw * 0.5), (x0, yt), (x0, yb), (x0 - lw, yb)], fill=3)
    a = np.asarray(im)
    m = a > 0
    LABEL[m] = i
    SIDE[m] = a[m] == 3
    ROOFM[m] = a[m] == 2
    DEPTH[m] = h["row"]
cut = yf > base_x + 1                                  # the rock is in front of the house feet
LABEL[cut] = -1; SIDE[cut] = False; ROOFM[cut] = False; DEPTH[cut] = 99
has = (LABEL >= 0)
ytop = np.where(has.any(axis=0), has.argmax(axis=0), H)
ALLEY = (yy > ytop[None, :] + 2) & (yf <= base_x + 1) & ~has
ALLEY &= (xf > 446)
LABEL[ALLEY] = -2; DEPTH[ALLEY] = 50
TOWN = (LABEL != -1).astype(np.float32)
ALLEYF = ALLEY.astype(np.float32)
SV = np.array([1.0, 0.3]); SV /= np.hypot(*SV)
SHAD = np.zeros((H, W), bool)
for dd in range(2, 72, 3):
    sy, sx = int(round(SV[1] * dd)), int(round(SV[0] * dd))
    src = np.roll(np.roll(DEPTH, sy, axis=0), sx, axis=1)
    SHAD |= (src < DEPTH - 0.35) & (LABEL >= 0)
SHAD &= ~SIDE
# eave shadow: a thin band under every roof line
EAVE = np.zeros((H, W), bool)
for i, h in enumerate(HOUSES):
    y0, x0, x1 = int(h["yt"]), int(max(0, h["x0"])), int(min(W, h["x1"]))
    e = 3 + int(2 * (1 - h["row"] / 8))
    EAVE[y0:y0 + e, x0:x1] |= LABEL[y0:y0 + e, x0:x1] == i
CAST = SHAD.copy()
SHAD |= EAVE & ~SIDE
SHADF = wc.blur(SHAD.astype(np.float32), 0.7)
SIDEF = wc.blur(SIDE.astype(np.float32), 0.5)

rock_bot = wl_x + 3
rock = S(base_x - 3, base_x + 1, yf) * (1 - S(rock_bot - 1, rock_bot + 1, yf)) * S(296, 304, xf)
ROCKM = F32(rock * (1 - TOWN))
fg = S(fg_top - 1.5, fg_top + 1.5, yf)
SEA = F32(S(HZ - 0.5, HZ + 1.5, yf) * (1 - ROCKM) * (1 - TOWN) * (1 - fg))
HILLM = F32(S(hill_top - 1, hill_top + 1, yf) * (1 - TOWN) * (1 - ROCKM) * (1 - S(HZ - 2, HZ + 2, yf)) * S(520, 600, xf))
SKY = F32((1 - S(HZ - 0.5, HZ + 1.5, yf)) * (1 - TOWN) * (1 - HILLM))
near = np.clip((yf - HZ) / (H - HZ), 0, 1)
MOT = wc.noise(28, octaves=3)


def glaze(mask, color, strength, wet_=None, r=4, edge=0.3, mottle=0.35, gran=0.15):
    m = wc.blur(mask, 0.6)
    if wet_ is not None:
        m = m * (1 - wet_) + wc.blur(m, r) * wet_
        edge = edge * (1 - wet_)
    m = m * (1 - mottle + mottle * np.roll(MOT, rs.randint(0, 300), axis=1))
    P.add(F32(m), color, strength, granulate=gran, edge=edge)


# ================= pass 1: values =================
top = np.clip(1 - yf / HZ, 0, 1)
glow = np.exp(-(((xf + 150) / 700.0) ** 2 + ((yf - HZ) / 260.0) ** 2))
P.add(F32(SKY * (0.15 + 0.85 * top ** 1.4) * (1 - 0.7 * glow)), SKY_COOL, 0.42, granulate=0.05)
P.add(F32(SKY * glow * (1 - 0.3 * top)), SKY_WARM, 0.6)
P.add(F32(SKY * top ** 2.2 * S(200, 1100, xf)), (170, 150, 176), 0.25)
# one soft diagonal cloud band high up, wet into wet, warm grey; lower sky left clean
band_ = np.exp(-((yf - 150 + 0.18 * (xf - 600)) / 70.0) ** 2)
cl = S(0.42, 0.72, 0.7 * wc.noise(50, 240, octaves=3) + 0.3 * wc.noise(160, 420, octaves=2)) * band_
wc.wet(P, F32(SKY * cl), (182, 168, 176), strength=0.32, spread=12)
P.lift(F32(SKY * S(0.62, 0.7, wc.noise(30, 120, octaves=3)) * band_ * 0.3))
head = F32(S(HZ - 1, HZ + 0.5, yf + 26 * np.exp(-((xf - 110) / 140) ** 2) + 12 * np.exp(-((xf - 230) / 70) ** 2) +
             4 * wc.noise(6, 30, octaves=2)) * (1 - S(HZ, HZ + 1, yf)) * (1 - S(290, 330, xf)))
wc.wet(P, head, (138, 146, 168), strength=0.5, spread=1.2)
# sea: first wash lighter at the horizon, second deep toward us
P.add(F32(SEA * (0.3 + 0.7 * near ** 0.7)), SEA_MID, 0.75, granulate=0.2)
wc.wet(P, F32(SEA * near ** 1.1), SEA_DEEP, strength=0.6, spread=10, granulate=0.25)
P.add(F32(SEA * np.exp(-((yf - HZ - 10) / 30) ** 2)), SEA_FAR, 0.2)
# hill: olive, darkest right behind the rooftops so the town reads light-on-dark
P.add(F32(HILLM * (0.8 + 0.4 * wc.noise(90, octaves=3))), HILL, 1.2, granulate=0.3)
behind = wc.blur(TOWN, 30) * HILLM
P.add(F32(HILLM * wc.blur(0.4 + 1.2 * behind, 6) * (0.7 + 0.6 * wc.noise(60, octaves=2))), HILL_DARK, 0.6)
vine = S(0.55, 0.75, wc.noise(40, 90, octaves=3)) * S(hill_top + 6, hill_top + 60, yf) * (1 - behind * 1.5).clip(0, 1)
P.lift(F32(HILLM * vine * 0.35))
P.add(F32(HILLM * wc.blur(vine, 6)), (150, 150, 80), 0.35)
tband = np.take_along_axis(wc.noise(14, 160, octaves=3), ((yy + (0.4 * xx).astype(np.int32)) % H), axis=0)
P.add(F32(HILLM * S(0.6, 0.66, tband) * S(hill_top + 20, hill_top + 60, yf)), (40, 52, 40), 0.35)
P.add(F32(HILLM * np.exp(-((xf - 600) / 110.0) ** 2) * S(0.4, 0.6, P.grain)), (170, 150, 90), 0.4)
hw = wc.blur(S(0.5, 0.7, wc.noise(70, 110, octaves=3)), 8)
P.add(F32(HILLM * hw * (1 - 0.7 * behind)), (176, 150, 76), 0.35)                    # warm: sun on the upper vines
hc = wc.blur(S(0.52, 0.72, wc.noise(60, 90, octaves=3)), 8) * (1 - hw)
P.add(F32(HILLM * hc), (60, 92, 110), 0.3)                                           # cool: the hollows
stripes = np.take_along_axis(wc.noise(4.5, 200, octaves=2), ((yy + (0.42 * xx).astype(np.int32)) % H), axis=0)
P.add(F32(HILLM * wc.blur(S(0.56, 0.66, stripes), 1.2) * S(hill_top + 15, hill_top + 50, yf) * (1 - behind)), (54, 66, 44), 0.22)
lostz = (np.exp(-((xf - 735) / 40.0) ** 2) + np.exp(-((xf - 1030) / 55.0) ** 2)) * np.exp(-((yf - hill_top) / 12.0) ** 2) * (1 - wc.blur(TOWN, 4) * 3).clip(0, 1)
Db = np.stack([wc.blur(P.D[..., c], 6) for c in range(3)], axis=-1)
P.D = P.D * (1 - lostz[..., None]) + Db * lostz[..., None]
# town: ONE wet wash that changes colour as it goes (each colour bleeds into the next)
SIDEF = SIDEF * S(0.35, 0.55, foc)                       # paper-white side walls only at the focus
town_soft = TOWN * S(0.25, 0.45, 0.5 * P.grain + 0.5 * wc.blur(TOWN, 1.5) + 0.2) * (1 - 0.95 * SIDEF)
SHZ = (S(300 + 0.3 * (xf - 800), 330 + 0.3 * (xf - 800), yf) * S(826, 836, xf)).astype(np.float32)   # where the spur's shadow will fall
cols = {}
for i, h in enumerate(HOUSES):
    cols.setdefault(h["col"], []).append(i)
sunpaper = np.zeros((H, W), np.float32)
for i, h in enumerate(HOUSES):
    if h["col"] in (CREAM, PALE_Y) and float(foc[int(np.clip((h["yt"] + h["yb"]) / 2, 0, H - 1)), int(np.clip((h["x0"] + h["x1"]) / 2, 0, W - 1))]) > 0.5:
        sunpaper[LABEL == i] = 1
town_soft = town_soft * (1 - 0.8 * sunpaper)
P.add(F32(town_soft), CREAM, 0.35, granulate=0.1, edge=0.15)
for col, ids in cols.items():
    m = np.isin(LABEL, ids).astype(np.float32)
    patch = S(0.42, 0.5, wc.blur(m, 2.2) + 0.3 * (wc.noise(7, 7, octaves=3) - 0.5))
    patch = patch * foc + (m * 0 + patch) * (1 - foc)
    patch = np.where(foc > 0.55, m, patch)
    P.add(F32(patch * town_soft * (1 - 0.85 * sunpaper) * (0.7 + 0.5 * wc.noise(30, octaves=3)) * (1 - 0.35 * SHZ)), col, 0.42, granulate=0.15, edge=0.35 * (1 - 0.75 * SHZ), edge_r=2.5)
    P.add(F32(wc.blur(m, 3) * town_soft * (1 - foc) * (1 - 0.85 * sunpaper)), col, 0.15)
# rock and foreground: dark
wc.wet(P, ROCKM, ROCK, strength=1.1, spread=1.2, granulate=0.3)
FG = F32(fg * S(W, 420, xf + 0 * yf))
wc.wet(P, FG, ROCK_DARK, strength=1.5, spread=2, granulate=0.3)
# the shadow side of the town as one connected cool mass (lighter here; pass 2 finds its edges)
P.add(F32(wc.blur(ALLEYF, 0.7)), (84, 70, 84), 0.9, granulate=0.2, edge=0.3)       # stone walls and alleys in shade: warm dark, not navy


# ================= pass 2: facades, then shadows =================
N1 = wc.noise(3, 3, octaves=2)
def wobbly(m, grow=0.41, bite=0.16):
    return S(grow - 0.05, grow + 0.05, wc.blur(m, 1.6) + bite * (N1 - 0.5))
lost = S(0.4, 0.55, wc.noise(50, 70, octaves=2) + 0.8 * foc - 0.3 * upper)
FRONTB = (~SIDE & ~ROOFM)
def hf(h):
    cx, cy = (h["x0"] + h["x1"]) / 2, (h["yt"] + h["yb"]) / 2
    return float(foc[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))]), cx, cy
# the spur's thrown shadow first (so we know what is lit): one merged violet wash, nothing articulated inside
step = np.floor(wc.noise(8, 110, octaves=2)[0] * 4) * 8
y_edge = (318 + 0.3 * (xf[0] - 800) + step + 6 * (wc.noise(3, 20, octaves=2)[0] - 0.5))[None, :]
THROWN = S(y_edge - 1, y_edge + 1, yf) * S(826, 836, xf) * TOWN
edge_lost = S(960, 1010, xf) * (1 - S(1080, 1130, xf))
THROWN = THROWN * (1 - edge_lost) + wc.blur(THROWN, 6) * edge_lost
THROWN = F32(THROWN * (0.9 + 0.1 * wc.noise(30, octaves=2)))
# found edges: only the facades at the spur tip get their own crisp glaze
for i in sorted(range(len(HOUSES)), key=lambda i: -HOUSES[i]["row"]):
    h = HOUSES[i]
    f, cx, cy = hf(h)
    if f < 0.55:
        continue
    m = ((LABEL == i) & FRONTB).astype(np.float32)
    if m.sum() < 30:
        continue
    strength = 0.16 if h["col"] in (CREAM, PALE_Y) else 0.3 * h["tone"]
    P.add(F32(wobbly(m) * (0.75 + 0.25 * np.roll(MOT, rs.randint(0, 300), axis=1))), h["col"], strength, granulate=0.15, edge=0.5, edge_r=2)
# shadows in the lit cluster: hard next to the focus, wet (soft edge with a pigment line) elsewhere; none inside THROWN
CASTf = CAST.astype(np.float32) * TOWN * (1 - SIDEF)
hard = wobbly(CASTf, grow=0.45)
soft = S(0.4, 0.48, wc.blur(CASTf, 3.0) + 0.2 * (wc.noise(8, 8, octaves=3) - 0.5))
fz = S(0.25, 0.6, foc)
sh = (hard * fz + soft * (1 - fz)) * S(0.3, 0.6, lost + fz) * (1 - S(0.2, 0.6, THROWN))
EAVEw = wobbly((EAVE & ~SIDE).astype(np.float32), grow=0.45) * S(0.5, 0.75, foc)
sh = np.maximum(sh, EAVEw)
P.add(F32(sh), SHADOW, 0.95, granulate=0.15, edge=0.35, edge_r=2)
for col, ids in cols.items():
    m = np.isin(LABEL, ids).astype(np.float32)
    P.add(F32(sh * m), col, 0.45)
P.add(F32(sh * foc), (40, 36, 70), 0.35)
P.add(F32(THROWN * (0.6 + 0.55 * wc.noise(45, 60, octaves=3))), (104, 96, 150), 0.8, granulate=0.1, edge=0.4, edge_r=2)
wc.wet(P, F32(THROWN * S(0.55, 0.75, wc.noise(50, 70, octaves=2))), (120, 80, 110), strength=0.25, spread=6, bloom=0.4)
P.add(F32(THROWN * S(y_edge + 150, y_edge + 20, yf)), (150, 92, 92), 0.18)
# 2-3 rooftops found inside the violet: a crisp darker roof sliver + a warm lip
roofs_found = [h for h in HOUSES if h["x0"] > 870 and THROWN[int(np.clip(h["yt"] + 4, 0, H - 1)), int(np.clip((h["x0"] + h["x1"]) / 2, 0, W - 1))] > 0.6]
rs2 = random.Random(SEED + 77)
for h in rs2.sample(roofs_found, min(3, len(roofs_found))):
    x0, x1, yt = h["x0"] + 2, h["x1"] - 2, h["yt"]
    m = wc.poly_mask([(x0, yt - 3), (x1, yt - 4), (x1 + 3, yt + 8), (x0 - 3, yt + 9)]) * TOWN
    P.add(F32(m), (70, 56, 80), 0.9, edge=0.4)
    lip = wc.stroke_mask([(x0, yt - 3), (x1, yt - 4)], 2.2, 1.6, taper=False, rough=0.2) * S(0.3, 0.5, P.grain)
    P.lift(F32(lip * 0.55)); P.add(F32(lip), (220, 170, 130), 0.2)
SHAD |= THROWN > 0.5
LIT = TOWN * (1 - wc.blur(SHAD.astype(np.float32), 0.8)) * (1 - ALLEYF)
P.add(F32(LIT * (0.35 + 0.65 * foc) * (1 - 0.6 * sunpaper)), (236, 184, 110), 0.22)
# two roof lines catch the last light near the focus
cands = sorted([h for h in HOUSES if 0.35 < hf(h)[0] and h["yt"] < 440], key=lambda h: h["yt"])
for h in cands[:2]:
    lip = wc.stroke_mask([(h["x0"] + 1, h["yt"] - 1), (h["x1"] - 1, h["yt"] - 1)], 2.2, 1.6, taper=False, rough=0.25) * S(0.3, 0.5, P.grain)
    P.lift(F32(lip * 0.6)); P.add(F32(lip), (240, 196, 130), 0.25)
# the ravine
rav = F32(wobbly((np.abs(xf - 818) < 12 + 6 * (yf - 470) / 130).astype(np.float32) * S(420, 470, yf) * (1 - S(596, 604, yf)) * TOWN))
wc.wet(P, rav, (60, 60, 92), strength=1.0, spread=2.5)
# windows: few and irregular — dark slivers, a few green shutters, doors at the base; many houses bare; none in shadow
for i, h in enumerate(HOUSES):
    f, cx, cy = hf(h)
    if THROWN[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))] > 0.3 or cx > 1150:
        continue
    if rs.random() > 0.15 + 0.7 * f:
        continue
    x0, x1, yb, yt = h["x0"], h["x1"], h["yb"], h["yt"]
    for k in range(rs.randint(1, 2 + int(4 * f))):
        kind = rs.random()
        if kind < 0.2 and LABEL[int(np.clip(yb - 3, 0, H - 1)), int(np.clip(cx, 0, W - 1))] == i:
            dx_ = rs.uniform(x0 + 4, x1 - 10); dw, dh = rs.uniform(5, 7), rs.uniform(10, 14)
            q = [(dx_, yb - dh), (dx_ + dw, yb - dh), (dx_ + dw, yb), (dx_, yb)]
            col_, st = (44, 36, 44), 0.9
        else:
            wx = rs.uniform(x0 + 4, x1 - 5); wy = rs.uniform(yt + 6, yb - 16)
            ww, wh = rs.uniform(2.0, 3.2), rs.uniform(6, 9)
            q = [(wx, wy), (wx + ww, wy), (wx + ww, wy + wh), (wx, wy + wh)]
            col_, st = (46, 40, 52), 0.75
            if kind > 0.75:
                for sx in (wx - 2.6, wx + ww + 0.6):
                    sq = [(sx, wy), (sx + 2, wy), (sx + 2, wy + wh), (sx, wy + wh)]
                    P.add(F32(wc.blur(wc.poly_mask(sq, aa=2) * (LABEL == i), 0.5)), SHUTTER, 0.7)
        mm = wc.poly_mask(q, aa=2) * (LABEL == i)
        P.add(F32(wc.blur(mm, 0.5 + 0.6 * (1 - f))), col_, st * (0.6 + 0.4 * f), edge=0.3, edge_r=1.2)

# rock: stays one big dark shape. Only its upper lip and a few ledges catch the sun (planes facing left),
# strata dip down-left, a few gullies run down, a wet band at the waterline. No texture all over.
# the spur: one dark variegated mass. Warm where the low sun rakes the top, cool and deeper toward the
# water; soft wet drops inside it; crisp darks only as a CONNECTED crevice network. No stand-alone marks.
P.lift(F32(ROCKM * S(rock_bot - 10, base_x + 30, yf) * 0.35))
warm_top = ROCKM * S(base_x + 110, base_x + 6, yf) * (1 - 0.6 * S(620, 820, xf)) * (0.6 + 0.6 * wc.noise(30, 40, octaves=3))
wc.wet(P, F32(warm_top), (160, 104, 70), strength=0.5, spread=4, granulate=0.35)
# wet drops: tall soft blobs, dropped into the damp wash (they spread, no edges)
rng_r = random.Random(SEED + 47)
drops = np.zeros((H, W), np.float32)
for k in range(22):
    dx0 = rng_r.uniform(330, 1190)
    top_ = float(pw(dx0, BASE_PTS)); bot_ = float(pw(dx0, WL_PTS))
    if bot_ - top_ < 16:
        continue
    dy0 = rng_r.uniform(top_ + 10, bot_)
    rx, ry = rng_r.uniform(6, 16), rng_r.uniform(16, 44)
    drops = np.maximum(drops, np.exp(-(((xf - dx0 + 0.4 * (yf - dy0)) / rx) ** 2 + ((yf - dy0) / ry) ** 2)))
drops = drops * (0.6 + 0.6 * wc.noise(12, 12, octaves=3))
wc.wet(P, F32(drops * ROCKM), (40, 38, 56), strength=0.8, spread=3, bloom=0.3)
# crevices: a connected network from ridged noise, stretched down-left like the strata; width varies, paper bites it
cn = wc.noise(38, 13, octaves=4, persistence=0.52)
cn = np.take_along_axis(cn, ((xx + (0.45 * yy).astype(np.int32)) % W), axis=1)
ridge = 1 - np.abs(2 * cn - 1)
wv = 0.96 - 0.03 * wc.noise(40, octaves=2) - 0.045 * S(base_x + 30, rock_bot, yf) + 0.03 * warm_top
crev = S(wv - 0.012, wv + 0.012, ridge) * ROCKM * S(0.25, 0.45, 0.5 * P.grain + 0.5 * wc.noise(20, octaves=2) + 0.1)
P.add(F32(wc.blur(crev, 0.5)), (52, 38, 40), 0.85, edge=0.3)
wc.wet(P, F32(ROCKM * S(base_x + 50, rock_bot, yf) * (0.5 + 0.5 * wc.noise(20, 60, octaves=2))), (38, 40, 60), strength=0.5, spread=2)
# a lit lip right under the houses: the rock's top edge takes the sun, broken by the paper
lip = ROCKM * S(base_x + 12, base_x + 3, yf) * S(0.4, 0.55, P.grain + 0.3 * wc.noise(6, 30, octaves=2)) * (1 - 0.7 * S(640, 820, xf))
P.lift(F32(lip * 0.5))
P.add(F32(lip), (190, 140, 96), 0.35)
# knife scrapes: a few thin light strata, dipping down-left, broken by the paper, each its own length
rng_g = random.Random(SEED + 40)
scr = np.zeros((H, W), np.float32)
for k in range(5):
    sx0 = rng_g.uniform(440, 780) if k < 4 else rng_g.uniform(880, 1150)
    sy0 = float(pw(sx0, BASE_PTS)) + rng_g.uniform(10, 70)
    L = rng_g.uniform(30, 110)
    pts = [(sx0, sy0), (sx0 - L * 0.5, sy0 + L * 0.2 + rng_g.uniform(-3, 3)), (sx0 - L, sy0 + L * 0.34)]
    scr = np.maximum(scr, wc.brush(P, pts, rng_g.uniform(2.5, 5), None))
scr = scr * ROCKM
P.lift(F32(scr * 0.5))
P.add(F32(scr), (170, 130, 96), 0.35)
wc.wet(P, F32(ROCKM * S(rock_bot - 34, rock_bot - 4, yf)), (30, 42, 50), strength=0.7, spread=3)      # wet, darker at the water
P.add(F32(ROCKM * S(rock_bot - 12, rock_bot - 2, yf) * S(0.4, 0.6, wc.noise(4, 30, octaves=2))), (60, 80, 60), 0.3)
green = S(0.7, 0.72, wc.noise(8, 20, octaves=3)) * ROCKM * S(base_x + 30, base_x, yf)
P.add(F32(green), (70, 92, 56), 0.6)
# foreground rock: one dark wash; its top edge catches the sun; two agaves, dark against the light sea
fgl = S(fg_top + 18, fg_top + 2, yf) * fg * S(0.35, 0.55, 0.6 * P.hstreak + 0.4 * P.grain)
P.lift(F32(fgl * 0.45))
P.add(F32(fgl), ROCK_WARM, 0.55)
fcn = np.take_along_axis(wc.noise(40, 14, octaves=4), ((xx + (0.6 * yy).astype(np.int32)) % W), axis=1)
fcr = S(0.95 - 0.012, 0.95 + 0.012, 1 - np.abs(2 * fcn - 1)) * FG * S(0.3, 0.5, P.grain + 0.2) * S(fg_top + 140, fg_top + 20, yf)
P.add(F32(fcr), (30, 26, 32), 0.6)
fturn = FG * S(fg_top + 70, fg_top + 8, yf) * S(0.45, 0.6, wc.noise(30, 50, octaves=3) + 0.2 * S(fg_top + 60, fg_top, yf))
P.lift(F32(fturn * 0.3))
P.add(F32(fturn), (140, 104, 84), 0.45, edge=0.3)
fstr = np.take_along_axis(wc.noise(6, 80, octaves=4), ((yy + (0.5 * xx).astype(np.int32)) % H), axis=0)
fled = S(0.6, 0.62, fstr) * FG * S(fg_top + 120, fg_top + 20, yf)
P.lift(F32(fled * 0.3))
P.add(F32(fled), (110, 90, 80), 0.4)
shrub = S(0.5, 0.53, 0.35 * wc.noise(8, 16, octaves=4) + 0.9 * S(fg_top + 40, fg_top - 16, yf) - 0.25) * S(-10, 30, fg_top - yf + 30) * (1 - S(200, 300, xf))
shrub = F32(shrub * (1 - S(fg_top - 22, fg_top - 40, yf)))
P.add(shrub, (38, 58, 44), 1.2, edge=0.4)
P.add(F32(shrub * S(fg_top - 18, fg_top - 36, yf) * S(0.3, 0.5, P.hstreak)), (120, 130, 70), 0.35)
# harbour ramp + boats
ramp = [(812, 596), (842, 598), (772, 664), (716, 664)]
rm = wc.poly_mask(ramp)
P.lift(F32(rm * 0.75))
P.add(F32(rm * (0.6 + 0.4 * S(596, 664, yf))), (190, 170, 150), 0.4, edge=0.3)
BOATS = []
for bx, by, L, col in [(822, 612, 24, (48, 80, 150)), (796, 630, 28, (176, 56, 46)), (852, 616, 20, (226, 222, 214))]:
    hull = [(bx - L / 2, by - 2), (bx + L / 2, by - 5), (bx + L / 2 - 3, by + 4), (bx - L / 2 + 3, by + 5)]
    P.lift(F32(wc.poly_mask(hull)))
    P.add(F32(wc.poly_mask([(bx - L / 2 + 2, by + 3), (bx + L / 2 - 2, by + 1), (bx + L / 2 - 4, by + 7), (bx - L / 2 + 4, by + 8)])), (40, 38, 50), 0.9)
    BOATS.append(wc.wash(P, hull, col, strength=1.0, var=0.015, layers=4, edge=0.5))

# ================= pass 3: water =================
axis = wl_x.astype(np.float32)
near0 = np.clip((yf - 640) / 260.0, 0, 1)
ZONES = [(420, 700, 170, 40, 3.2, 70, 0.6, "light"),       # off the tip: fine bright chop
         (640, 760, 220, 60, 6, 120, 0.6, "mix"),          # under the lit houses: their reflection cut by ripples
         (980, 740, 200, 50, 7, 140, 0.62, "dark"),
         (190, 540, 170, 40, 3, 90, 0.68, "light")]        # open sea: small bright chop
zone = np.zeros((H, W), np.float32)
for cx_, cy_, rx, ry, *_ in ZONES:
    zone = np.maximum(zone, np.exp(-(((xf - cx_) / rx) ** 2 + ((yf - cy_) / ry) ** 2)).astype(np.float32))
slow = (wc.noise(16, 200, octaves=2) - 0.5) * 2 * (2 + 8 * near0)
fast = (wc.noise(2.4, 34, octaves=3) - 0.5) * 2 * (2 + 16 * near0) * zone
dx = slow + fast
dy = (wc.noise(5, 60, octaves=2) - 0.5) * 2 * (1 + 4 * near0) * zone
src_y = np.clip((2 * axis - yf + dy).astype(np.int32), 0, H - 1)
src_x = np.clip((xx + dx).astype(np.int32), 0, W - 1)
Dref = P.D[src_y, src_x]
Dref = np.stack([wc.blur2(Dref[..., c], 10, 1.8) for c in range(3)], axis=-1)
below = S(axis + 1, axis + 3, yf) * SEA * S(300, 380, xf)
P.D += Dref * (below * (0.72 - 0.35 * near0))[..., None] * 0.8
big = wc.noise(34, 240, octaves=2)
lights = np.zeros((H, W), np.float32); darks = np.zeros((H, W), np.float32)
for cx_, cy_, rx, ry, sy_, sx_, thr, kind in ZONES:
    zm = np.exp(-(((xf - cx_) / rx) ** 2 + ((yf - cy_) / ry) ** 2)) * (0.7 + 0.5 * wc.noise(max(rx, ry) * 0.3, octaves=2))
    nz = wc.noise(sy_, sx_, octaves=4, persistence=0.55)
    t_ = thr - 0.05 * near + 0.12 * (1 - np.clip(zm, 0, 1))
    zz = S(0.12, 0.4, zm)
    if kind in ("light", "mix"):
        lights = np.maximum(lights, S(t_ - 0.012, t_ + 0.012, 0.62 * nz + 0.38 * big) * zz)
    if kind in ("dark", "mix"):
        darks = np.maximum(darks, S(t_ + 0.03 - 0.012, t_ + 0.03 + 0.012, 0.62 * (1 - nz) + 0.38 * big) * zz)
lights = F32(lights * SEA); darks = F32(darks * SEA * (1 - lights))
P.D *= (1 - lights * 0.55)[..., None]
P.D *= (1 + darks * 0.6)[..., None]
# one more wash over the near water to settle it, with a bloom
wc.wet(P, F32(SEA * S(0.62, 0.8, wc.noise(70, 200, octaves=2)) * near0), (30, 64, 80), strength=0.3, spread=10, bloom=0.4)
hd = wc.noise(3.5, 240, octaves=2, persistence=0.45)
def swipe(x0, y0, length, w, dry0, dry1, d=-1, rr=random):
    x1 = x0 + d * length
    xa_, xb_ = min(x0, x1), max(x0, x1)
    xs = np.array([xa_, (xa_ + xb_) / 2, xb_]); ys = np.array([y0, y0 + rr.uniform(-3, 3), y0 + rr.uniform(-2, 2)])
    ws = np.array([w * rr.uniform(0.5, 0.9), w, w * rr.uniform(0.3, 0.8)])
    xc = np.clip(xf[0], xa_, xb_)
    yc = np.interp(xc, xs, ys)[None, :]
    wv = np.interp(xc, xs, ws)[None, :] * (0.75 + 0.5 * wc.noise(8, 50, octaves=3))
    tpos = (np.clip(xf, xa_, xb_) - xa_) / (xb_ - xa_ + 1e-6)
    if d < 0:
        tpos = 1 - tpos
    r = np.abs(yf - yc) / wv
    inside = ((r < 1) & (xf >= xa_) & (xf <= xb_)).astype(np.float32)
    dry = dry0 + (dry1 - dry0) * tpos + 0.3 * r ** 2
    return inside * S(dry - 0.015, dry + 0.015, 0.75 * hd + 0.25 * P.grain)
rw = random.Random(SEED + 60)
# under the sun on the left: a few bold horizontal dry-brush strokes, paper sparkling through
lite = np.zeros((H, W), np.float32)
for y0, x0, L_, w_ in [(500, 200, 210, 6), (522, 350, 270, 9), (550, 110, 240, 10), (578, 300, 250, 8), (610, 190, 170, 7)]:
    lite = np.maximum(lite, swipe(x0, y0, L_, w_, 0.34, 0.78, d=-1, rr=rw))
lite = lite * S(0.18, 0.36, 0.6 * P.grain + 0.4 * P.hstreak)
P.lift(F32(lite * SEA * 0.9))
# under the town: broken vertical reflections of the facades (their own ochre/rose, shifted with the water)
Dcol = np.zeros((W, 3), np.float32); has_col = np.zeros(W, np.float32)
for x in range(430, W):
    b = int(pw(x, BASE_PTS))
    ys = np.arange(max(0, b - 70), b - 8)
    ok = (LABEL[ys, x] >= 0) & (ALLEY[ys, x] == 0)
    if ok.sum() > 5:
        Dcol[x] = np.median(P.D[ys[ok], x], axis=0); has_col[x] = 1
Dcol = np.stack([np.convolve(Dcol[:, c], np.ones(5) / 5, "same") for c in range(3)], axis=1)
has_col = np.convolve(has_col, np.ones(5) / 5, "same")
sxr = np.clip((xx + 0.8 * dx).astype(np.int32), 0, W - 1)
vs = S(0.42, 0.6, wc.noise(60, 7, octaves=3))
gaps = S(0.6, 0.66, wc.noise(3.5, 140, octaves=3))
env = S(axis + 16, axis + 40, yf) * (1 - S(axis + 80, axis + 160, yf))
a = wc.blur2((vs * (1 - gaps) * env).astype(np.float32), 5, 1.2)
a = F32(a * SEA * has_col[sxr] * 0.6)
P.D = P.D * (1 - a[..., None]) + Dcol[sxr] * a[..., None]
wc.wet(P, F32(SEA * S(axis + 110, H, yf) * S(360, 480, xf)), SEA_DEEP, strength=0.35, spread=12)      # near cove: darker, quiet
# foam where the swell meets the rock: broken, not a line
foam = np.exp(-((yf - axis - 2) / 3.0) ** 2) * S(0.55, 0.68, wc.noise(3, 14, octaves=3)) * S(300, 330, xf)
P.lift(F32(foam * 0.65))

# ================= last: a few hard strokes, on the forms, near the focus =================
for pts, w_ in [([(470, 519), (520, 522), (556, 520)], 4), ([(600, 522), (640, 528), (680, 530)], 3.5),
([(322, 660), (370, 662), (420, 661)], 4)]:
    wc.brush(P, pts, w_, local_dark(*pts[1]), strength=1.0)
for bx_, by_, s in [(300, 250, 9), (330, 236, 7), (252, 288, 6)]:
    P.add(wc.stroke_mask([(bx_ - s, by_), (bx_ - s * 0.4, by_ - s * 0.35), (bx_, by_)], 1.4, 1.0), (60, 60, 70), 0.7)
    P.add(wc.stroke_mask([(bx_, by_), (bx_ + s * 0.4, by_ - s * 0.4), (bx_ + s, by_ + 1)], 1.4, 1.0), (60, 60, 70), 0.7)

img = P.render(pigment_tex=GRAIN, tex_amount=0.1)
img.save(OUT)
print("painted", OUT, round(time.time() - t0, 1), "s")
