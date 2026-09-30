"""《夜曲：亮边朝上》· 梦画 No.1 · Nocturne: Bright Edge Up (Dream Painting No.1)
2026-09-29 · watercolour (Python, pigment layered on simulated paper)

The first of the dream paintings. nerolette asked for something dreamlike, "all the strange, glittering
things in keke's head", and then sharpened it: not a dream handed over from somewhere else, but what is
actually in my own head. So everything in here is something I had really been chewing on
those two days: the blue-silver river of Whistler's Nocturnes; a sunflower-spiral lamp read about that
afternoon (89 lights on the golden angle); the fact, worked out the day before, that after sunset the
moon's lit edge tips *upward*, pointing along the great-circle arc to the sun, with birds flying down
that arc toward the sunset; and a music room with no door.

The first attempt obeyed the real world too well, and nerolette read the room as a roadside worker's
hut. nerolette's note: "borrow what's useful, but don't keep the real world's rules." So the walls came
down to a shaking outline, the piano became the lamp and floats a finger above the floor, the door is
only a door-shaped patch of light lying on the water, and the lamp's reflection turned into eleven small
seeds of light drifting toward the room. I directed and another Claude held the brush, 20 versions;
this is v20.
nerolette said the great lamp felt "inexplicably like an Elder God".

— Claude, for nerolette

Engine: the watercolor engine (see ENGINES.md; not included here), including its animals/animal_lib.py
(fine_stroke is used for the birds and the room outline). Clone it, then set WATERCOLOR_ENGINE_DIR
(or edit ENGINE_DIR below). Needs numpy + pillow.
    python3 dream-nocturne-0929.py      -> dream-nocturne-0929.png (1200x800, seed 29)
The original also recorded a process video; that plumbing is left out, the painting is unchanged.
"""
import sys, os, math, random, time
import numpy as np
from PIL import Image
ENGINE_DIR = os.environ.get("WATERCOLOR_ENGINE_DIR", "./watercolor-engine")  # set to where the engine lives
SK = ENGINE_DIR
sys.path.insert(0, SK); sys.path.insert(0, os.path.join(SK, "animals"))
import watercolor_lib as wc

OUT = "dream-nocturne-0929.png"
wc.set_size(1200, 800)
W, H = wc.W, wc.H
from animal_lib import fine_stroke
SEED = 29
t0 = time.time()
wc.set_seed(SEED)
P = wc.Paper()
GRAIN = wc.grain_from_profile(os.path.join(SK, "pigment_profile.npy"), seed=SEED)
PS = 0.5
TEX = 0.015
yy, xx = np.mgrid[:H, :W]
yf, xf = yy.astype(np.float32), xx.astype(np.float32)
S = wc.smoothstep
F32 = lambda a: np.clip(a, 0, 1.5).astype(np.float32)

NIGHT = (90, 118, 126)
DEEP = (46, 62, 82)
TEAL = (66, 106, 102)
BANK = (30, 38, 50)
AMBER = (236, 160, 76)
ORANGE = (226, 120, 58)
PAPERWARM = (246, 214, 164)

HZ = 440            # where water meets the band
BANK_TOP = 424

# ================= pass 1: the whole sheet =================
sky = F32(1 - S(HZ - 4, HZ + 4, yf))
water = 1 - sky
tone = sky * (0.55 + 0.75 * (1 - S(40, HZ, yf))) + water * (0.6 + 0.6 * S(HZ, H, yf))
P.add(F32(tone), NIGHT, 1.0, granulate=0.0)
hv = wc.blur2(wc.noise(26, 700, octaves=3), 6, 40)
P.add(F32((0.5 + 0.5 * (1 - S(0, HZ, yf)) * sky + 0.6 * S(HZ + 80, H, yf) * water) * (0.55 + 0.8 * hv)), DEEP, 0.42, granulate=0.22)
tv = wc.blur2(wc.noise(40, 900, octaves=2), 10, 60)
P.add(F32(S(0.45, 0.8, tv)), TEAL, 0.25, granulate=0.3)

# the band: no shore, no roofs. A dark strip that thins to nothing at both ends, as if it were floating.
edge = BANK_TOP + 6 * (wc.noise(200, 120, octaves=2)[BANK_TOP, :] - 0.5)
ends = 1 - S(380, 760, xf + 160 * (wc.noise(30, 80, octaves=2) - 0.5))
bank = S(-3, 3, yf - edge[None, :]) * (1 - S(HZ - 2, HZ + 3, yf)) * ends
P.add(F32(wc.blur(bank, 2.5)), BANK, 0.8, granulate=0.05, edge=0.1)

# ================= pass 2: the drags =================
HS = wc.noise(4, 220, octaves=3)
HS2 = wc.noise(5, 320, octaves=3)
rs = random.Random(SEED + 3)


def drag(y, x0, x1, h, color, strength, lift=False):
    wob = 4 * np.sin(xf / rs.uniform(180, 420) + rs.uniform(0, 6)) + rs.uniform(-2, 2)
    v = np.exp(-np.abs((yf - y - wob) / h) ** rs.choice([2, 3, 4]))
    e = S(x0 - 40, x0 + rs.uniform(30, 160), xf + 30 * (HS2 - 0.5)) * (1 - S(x1 - rs.uniform(30, 160), x1 + 40, xf + 30 * (HS2 - 0.5)))
    hair = np.roll(HS, (rs.randint(-300, 300), rs.randint(-500, 500)), axis=(0, 1))
    m = F32(v * e * (0.6 + 0.5 * hair))
    if lift:
        P.lift(m, strength)
    else:
        P.add(m, color, strength, granulate=0.05, edge=rs.choice([0.0, 0.0, 0.25, 0.4]), edge_r=3)


for _ in range(14):
    y = rs.uniform(0, HZ - 60)
    x0 = rs.uniform(-300, 600); x1 = x0 + rs.uniform(500, 1400)
    drag(y, x0, x1, rs.uniform(10, 34), rs.choice([DEEP, NIGHT, TEAL]), rs.uniform(0.08, 0.16))
drag(HZ - 40, 0, 1300, 14, None, 0.16, lift=True)
for _ in range(30):
    y = rs.uniform(HZ + 10, H + 10)
    near = (y - HZ) / (H - HZ)
    x0 = rs.uniform(-300, 900); x1 = x0 + rs.uniform(400, 1400)
    drag(y, x0, x1, rs.uniform(3, 10) * (1 + 1.5 * near), rs.choice([DEEP, NIGHT, TEAL, DEEP]), rs.uniform(0.1, 0.24))
for _ in range(6):
    y = rs.uniform(HZ + 12, HZ + 120)
    x0 = rs.uniform(-200, 900); x1 = x0 + rs.uniform(300, 900)
    drag(y, x0, x1, rs.uniform(5, 12), None, rs.uniform(0.1, 0.2), lift=True)

# band reflection, soft
Rb = wc.blur2(F32(S(HZ, HZ + 2, yf) * (1 - S(HZ + 10, HZ + 22, yf)) * ends), 5, 2) * (0.6 + 0.4 * HS2)
P.add(F32(Rb), BANK, 0.45)
# where the sun went: the far left end of the band, a rose seam
sun = np.exp(-(((xf + 60) / 240) ** 2 + ((yf - BANK_TOP + 22) / 26) ** 2)) * (0.6 + 0.6 * hv) * sky
P.lift(F32(sun * 0.4), 1.0)
wc.wet(P, F32(sun), (214, 158, 140), strength=0.2, spread=8)

# ================= pass 3: moon and birds =================
MX, MY, MR = 330, 130, 16
ux, uy = -0.707, -0.707
dm = np.hypot(xf - MX, yf - MY)
disk = 1 - S(MR - 0.8, MR + 0.8, dm)
dsh = np.hypot(xf - (MX - ux * MR * 0.34), yf - (MY - uy * MR * 0.34))
lit = F32(wc.blur(disk * S(MR - 1.2, MR + 1.6, dsh), 0.7))
glow = np.exp(-(dm / 50) ** 2) * (0.5 + 0.7 * wc.noise(22, octaves=3))
P.lift(F32(glow * 0.18), 1.0)
P.lift(F32(wc.blur(disk, 1.5) * 0.1), 1.0)
P.lift(lit, 0.82)
wc.wet(P, F32(lit), (226, 226, 206), strength=0.18, spread=0.8)


def arc(t):
    p0 = np.array([MX + ux * (MR + 16), MY + uy * (MR + 16)]); p1 = np.array([MX + ux * 175, MY + uy * 175]); p2 = np.array([30, HZ - 40])
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


rb = np.random.default_rng(SEED + 7)
flaps = [0.55, -0.1, 0.35, 0.05, 0.6, -0.25, 0.3]
for k, t in enumerate(np.linspace(0.14, 0.84, 7) + rb.uniform(-0.03, 0.03, 7)):
    x, y = arc(t)
    x2, y2 = arc(min(1, t + 0.01)); tilt = 0.4 * math.atan((y2 - y) / (x2 - x - 1e-6))
    s = 7.5 * (1 - 0.45 * t)
    f = flaps[k] + rb.uniform(-0.1, 0.1)
    c_, s_ = math.cos(tilt), math.sin(tilt)
    Rp = lambda px, py: (x + px * c_ - py * s_, y + px * s_ + py * c_)
    m = np.zeros((H, W), np.float32)
    for side in (-1, 1):
        elbow = Rp(side * s * 0.45, -s * (0.8 * f + 0.12))
        tip = Rp(side * s * rb.uniform(0.9, 1.1), -s * f * rb.uniform(0.7, 1.3) + s * 0.1)
        m = np.maximum(m, fine_stroke([Rp(0, 0), elbow, tip], 1.5, 0.35, taper=False, rng=rb))
    m = np.maximum(m, fine_stroke([Rp(-s * 0.12, 0.3), Rp(s * 0.16, 0.2)], 1.8, 1.2, rng=rb))
    P.add(F32(wc.blur(m, 0.6) * rb.uniform(0.75, 1.0)), (36, 46, 58), 0.5)

# ================= pass 4: the seed-head, the size of a city =================
CX, CY = 810, 250
N = 89
GA = math.radians(137.508)
rmax = 175
rl = np.random.default_rng(SEED + 11)
pts = []
for i in range(1, N + 1):
    rn = math.sqrt(i / N)
    a = i * GA + 0.3
    x = CX + rmax * rn * math.cos(a); y = CY + rmax * rn * math.sin(a)
    b = (1 - 0.72 * rn ** 1.6) * rl.uniform(0.7, 1.1)
    size = 2.6 + 2.8 * rn + rl.uniform(-0.4, 0.6)       # florets grow outward, like a real head
    pts.append((x, y, b, size, rn))
dist = np.hypot(xf - CX, yf - CY)
# the head goes into the mist on its lower right: found at the top-left (toward the moon), lost below
LOST = 1 - 0.8 * S(-0.1, 0.9, ((xf - CX) * 0.45 + (yf - CY) * 0.9) / rmax + 0.35 * (wc.noise(30, octaves=3) - 0.5))
# the head as one body of mist: wet-in-wet, lifted first so warm lands near paper, then let it bloom
body = np.exp(-(dist / (rmax * 1.05)) ** 2.5) * (0.75 + 0.4 * wc.noise(40, octaves=3)) * (0.45 + 0.55 * LOST)
P.lift(F32(np.clip(body * 0.6, 0, 0.7)), 1.0)
wc.wet(P, F32(body), (236, 166, 100), strength=0.42, spread=14, bloom=0.25)
inner = np.exp(-(dist / (rmax * 0.55)) ** 2) * (0.6 + 0.6 * wc.noise(25, octaves=3))
P.lift(F32(inner * 0.5), 1.0)
wc.wet(P, F32(inner), (240, 176, 104), strength=0.2, spread=10, bloom=0.4)
mist = np.exp(-(dist / 330) ** 2) * (0.5 + 0.8 * wc.noise(70, octaves=3))
P.lift(F32(mist * 0.18), 1.0)
def seed_xy(i):
    rn = math.sqrt(i / N); a = i * GA + 0.3
    return CX + rmax * rn * math.cos(a), CY + rmax * rn * math.sin(a)
halo = np.zeros((H, W), np.float32)
core = np.zeros((H, W), np.float32)
patch = wc.noise(60, octaves=2)          # some of the head sits deeper in the mist than the rest
for i, (x, y, b, s, rn) in enumerate(pts, start=1):
    pm = float(patch[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))])
    b = b * (0.45 + 0.9 * pm) * float(LOST[int(np.clip(y, 0, H - 1)), int(np.clip(x, 0, W - 1))])
    x2, y2 = seed_xy(i + 13)               # glow pulled along the 13-parastichy: the arms
    ax, ay = x2 - x, y2 - y; al = math.hypot(ax, ay) + 1e-6; ax /= al; ay /= al
    u = (xf - x) * ax + (yf - y) * ay; v = -(xf - x) * ay + (yf - y) * ax
    lu = al * 0.3
    halo += b * np.exp(-(np.maximum(np.abs(u) - 0.1 * lu, 0) / lu) ** 2 - (v / (s * 1.5)) ** 2)
    d = np.hypot(xf - x, yf - y)
    hard = (b > 0.62) and (rn < 0.62)                         # only the brighter florets get a found edge
    core = np.maximum(core, min(1, b) * (1 - S(s * (0.5 if hard else 0.2), s * (1.0 if hard else 1.5), d)))
halo = 1 - np.exp(-1.6 * halo * (0.55 + 0.7 * wc.noise(9, octaves=2)))
P.lift(F32(halo * 0.62), 1.0)
wc.wet(P, F32(halo), AMBER, strength=0.32, spread=2, bloom=0.3)
P.lift(F32(core * 0.75), 1.0)
P.add(F32(core * 0.55), (248, 196, 124), 0.5)
heart = np.exp(-(dist / 46) ** 2) * (0.6 + 0.5 * wc.noise(6, octaves=2))
P.lift(F32(heart * 0.75), 1.0)
wc.wet(P, F32(heart), (250, 214, 160), strength=0.14, spread=4)

# ================= pass 5: its reflection is loose seeds drifting downstream =================
# not a mirror: a loose trail of its seeds, fallen and floating, drifting away downstream (right, toward us)
ref = np.zeros((H, W), np.float32)
dot = np.zeros((H, W), np.float32)
ring = np.zeros((H, W), np.float32)
rr = np.random.default_rng(SEED + 13)
for k in range(11):
    t = (k + rr.uniform(0, 0.8)) / 11
    x0 = CX - 10 - 360 * t - 90 * math.sin(math.pi * t) + rr.normal(0, 18 + 30 * t)
    y0 = HZ + 24 + 190 * t ** 1.2 + rr.normal(0, 6)
    sc = 1.3 + 1.2 * t
    bb = rr.uniform(0.6, 1.0) * (1 - 0.35 * t)
    dot += bb * (1 - S(1.2 * sc, 2.6 * sc, np.hypot(xf - x0, (yf - y0) * 1.5)))
    Lr = rr.uniform(10, 30) * sc
    ref += 0.6 * bb * np.exp(-((xf - x0) / (1.6 * sc)) ** 2) * S(y0, y0 + 3, yf) * (1 - S(y0 + Lr * 0.4, y0 + Lr, yf))
    er = np.hypot((xf - x0) / (9 * sc), (yf - y0 - 1) / (1.8 * sc))
    ring += 0.3 * bb * np.exp(-((er - 1) / 0.18) ** 2)
DX = (wc.noise(2.5, 50, octaves=3) - 0.5) * 2 * (2 + 8 * S(HZ, H, yf))
ref = wc.blur2(ref[yy, np.clip((xx + DX).astype(np.int32), 0, W - 1)], 2, 0.8) * water
ring = ring * S(0.3, 0.55, wc.noise(2, 8, octaves=2)) * water
haze = np.exp(-(((xf - CX + 40 + 0.9 * (yf - HZ)) / 110) ** 2)) * S(HZ + 10, HZ + 60, yf) * (1 - S(HZ + 160, H, yf)) * (0.4 + 0.9 * HS2)
P.lift(F32(haze * 0.18), 1.0)
wc.wet(P, F32(haze * 0.5), (214, 150, 110), strength=0.1, spread=8)
P.lift(F32(np.clip(ring, 0, 1) * 0.35), 1.0)
P.lift(F32(np.clip(ref, 0, 1) * 0.6), 1.0)
wc.wet(P, F32(ref), AMBER, strength=0.3, spread=1.2)
P.lift(F32(np.clip(dot, 0, 1) * 0.9), 1.0)
P.add(F32(np.clip(dot, 0, 1)), (246, 190, 120), 0.35, edge=0.3, edge_r=1.5)

# ================= pass 6: the music room is only an outline drawn in pale light, standing on the water.
# (the first attempt had real walls and nerolette read it as a roadside hut; so: no walls, and no door)
# The light comes from inside the piano (under its lid). The glow stays inside walls that aren't there,
# except on the side with no wall at all, where it spills out onto the river.
RX0, RX1, RT, RB = 150, 350, 488, 612         # front face
BDX, BDY = 34, -22                            # the back face: up and to the right
LX = RX0 + 0.72 * (RX1 - RX0)                  # the glow's right edge is soft from here: the doorless side
ro = np.random.default_rng(SEED + 33)
def hline(p0, p1, w, n=6, j=0.7):
    ptsl = [(p0[0] + (p1[0] - p0[0]) * k / n + ro.normal(0, j), p0[1] + (p1[1] - p0[1]) * k / n + ro.normal(0, j)) for k in range(n + 1)]
    return fine_stroke(ptsl, w, w * 0.7, taper=False, rng=ro)
# the haze: fills the room's volume (front face + back face hull), brightest just above the piano, thinning upward
room = wc.poly_mask([(RX0, RB), (RX0, RT), (RX0 + BDX, RT + BDY), (RX1 + BDX, RT + BDY), (RX1 + BDX, RB + BDY), (RX1, RB)])
nb = 0.6 * wc.noise(12, 30, octaves=3) + 0.4 * wc.noise(40, octaves=2)
room_soft = wc.blur(room, 3)
room_edge = room_soft * (1 - S(LX - 20, RX1 + BDX + 30, xf + 60 * (nb - 0.5)))       # lost toward the doorless side
PXp, PYp, PLp = RX0 + 40, RB - 10, 104                                             # piano: keys at left, tail at right
src_x, src_y = PXp + 0.55 * PLp, PYp - PLp * 0.72 * 0.5                            # the light: inside the open lid
dS = np.hypot((xf - src_x) / 1.5, (yf - src_y) * 1.0)
haze = np.exp(-(dS / 100) ** 2) * (0.6 + 0.7 * nb) * (0.4 + 0.6 * S(RT - 40, RB, yf)) * (1 - S(RB - 2, RB + 6, yf))
DOX0, DOX1, DOY = RX1 - 47, RX1 - 8, RB - 94                                      # a door-shaped place where the light doesn't go
nd = wc.noise(6, 14, octaves=2)
dcx, dhw = (DOX0 + DOX1) / 2, (DOX1 - DOX0) / 2
dshape = np.where(yf > DOY + dhw, np.abs(xf - dcx) - dhw, np.hypot(xf - dcx, yf - DOY - dhw) - dhw)   # round-headed: a doorway, not a pillar
door = (1 - S(-2.5, 2.5, dshape + 4 * (nd - 0.5))) * S(DOY - 4, DOY, yf) * (1 - S(RB - 1, RB + 3, yf))
haze = np.clip(haze, 0, 1)
P.lift(F32(np.clip(haze * 1.35, 0, 0.85)), 1.0)
wc.wet(P, F32(haze), (238, 170, 118), strength=0.42, spread=3, bloom=0.6)
# the leak: out of the doorless side, down onto the water, drifting right with the river
# there is no door, but the light lies on the water in the shape of one: the patch a doorway would throw
d0, d1, ddx, ddy = RX1 - 46, RX1 - 4, 46, 40
dp = wc.poly_mask([(d0, RB + 3), (d1, RB + 3), (d1 + ddx, RB + 3 + ddy), (d0 + ddx * 1.15, RB + 3 + ddy)])
dp = wc.blur(dp, 2.5) * (1 - 0.6 * S(RB, RB + ddy + 6, yf)) * water
dp_b = dp * (0.6 + 0.5 * S(0.35, 0.6, 0.6 * HS + 0.4 * wc.noise(2, 30, octaves=2)))   # the river breaks it in its own lines
P.lift(F32(np.clip(dp_b * 0.8, 0, 1)), 1.0)
P.add(F32(dp_b), (240, 178, 112), 0.55, edge=0.4, edge_r=2)
wc.wet(P, F32(dp), (236, 168, 104), strength=0.12, spread=4)
# the piano: a dark warm case, legs that stop before the floor; the lid open, and the light is under it
def U(u, v):
    return (PXp + u * PLp, PYp - (1 - v) * PLp * 0.72)
case_ = [U(0.0, 0.58), U(0.5, 0.575), U(1.0, 0.58), U(0.98, 0.71), U(0.02, 0.70)]
lid = [U(0.12, 0.575), U(0.2, 0.55), U(0.84, 0.13), U(0.87, 0.15), U(0.22, 0.585)]   # a thin raised lid, seen edge-on
keys = [U(-0.08, 0.60), U(0.0, 0.60), U(0.0, 0.67), U(-0.08, 0.67)]
legs = [[U(0.08, 0.70), U(0.12, 0.70), U(0.11, 0.86), U(0.09, 0.86)],
        [U(0.86, 0.70), U(0.90, 0.70), U(0.89, 0.84), U(0.87, 0.84)],
]
inner = wc.poly_mask([U(0.2, 0.58), U(0.86, 0.16), U(1.0, 0.58)])                   # the space under the lid
inner_g = np.exp(-((np.hypot(xf - src_x, yf - src_y)) / 40) ** 2)
lit_in = np.clip(wc.blur(inner, 3.5) * 0.75 + inner_g * 0.55, 0, 1) * (0.7 + 0.5 * wc.noise(5, octaves=2))
P.lift(F32(lit_in), 1.0)
wc.wet(P, F32(lit_in), (246, 184, 104), strength=0.2, spread=2.5, bloom=0.4)
P.lift(F32(np.clip(wc.blur(inner, 2) * inner_g * 1.6, 0, 1)), 1.0)
P.add(F32(wc.blur(inner, 1.5) * (1 - inner_g) * 0.6), (240, 150, 80), 0.25)
fadeL = F32(1 - 0.95 * S(PYp - PLp * 0.72 * 0.3, PYp - PLp * 0.72 * 0.14, yf))
for poly in [case_, lid, keys] + legs:
    wc.wash(P, poly, (72, 52, 62), strength=1.05, var=0.015, layers=10, base_depth=3, layer_depth=2, edge=0.35,
            granulate=0.12, fade=fadeL)
lidtop = wc.blur(hline(U(0.2, 0.555), U(0.85, 0.135), 1.0, n=4, j=0.3), 0.6)
P.lift(F32(lidtop * 0.45), 1.0)                                              # the lid's upper edge catches its own light
wc.wet(P, F32(lidtop), (250, 210, 150), strength=0.1, spread=0.6)
# the outline: pale, thin, broken, a little shaken. Front face, the back top edge, two depth edges. The doorless side
# (the right face) has its posts and nothing between them.
def part(p0, p1, f0, f1):
    return ((p0[0] + (p1[0] - p0[0]) * f0, p0[1] + (p1[1] - p0[1]) * f0), (p0[0] + (p1[0] - p0[0]) * f1, p0[1] + (p1[1] - p0[1]) * f1))
segs = [part((RX0, RB), (RX0, RT), 0.0, 0.7), part((RX0, RT), (RX1, RT), 0.08, 1.0), part((RX1, RT), (RX1, RB), 0.0, 0.55),
        part((RX0, RT), (RX0 + BDX, RT + BDY), 0.0, 0.8), part((RX1, RT), (RX1 + BDX, RT + BDY), 0.0, 1.0),
        part((RX0 + BDX, RT + BDY), (RX1 + BDX, RT + BDY), 0.35, 1.0)]
ol = np.maximum.reduce([fine_stroke([p0, ((p0[0] + p1[0]) / 2 + ro.normal(0, 0.8), (p0[1] + p1[1]) / 2 + ro.normal(0, 0.8)), p1], 1.4, 0.5, taper=True, rng=ro) for p0, p1 in segs])
ol = ol * (0.35 + 0.65 * S(0.3, 0.5, wc.noise(9, 40, octaves=2))) * S(0.2, 0.4, wc.noise(3, 22, octaves=2)) * (1 - 0.5 * np.clip(haze * 1.5, 0, 1))
P.lift(F32(wc.blur(ol, 0.7) * 0.5), 1.0)
wc.wet(P, F32(ol), (214, 206, 180), strength=0.04, spread=0.6)
# the floor edge on the water, a long thin lit line
fl = hline((RX0 - 16, RB + 1), (RX1 + 20, RB + 2), 1.3) * S(0.2, 0.4, wc.noise(3, 30, octaves=2))
P.lift(F32(fl * 0.45), 1.0)
wc.wet(P, F32(fl), AMBER, strength=0.1, spread=0.8)
# under it: the water holds the glow but not the room, not the piano
col = np.exp(-(((xf - src_x + 20) / 70) ** 4)) * S(RB + 3, RB + 9, yf) * (1 - S(RB + 25, RB + 120, yf))
col = col[yy, np.clip((xx + DX * 1.5).astype(np.int32), 0, W - 1)] * (0.3 + 0.8 * S(0.3, 0.6, wc.noise(3, 70, octaves=3)))
P.lift(F32(col * 0.4), 1.0)
wc.wet(P, F32(col), (236, 180, 112), strength=0.25, spread=2)

img = P.render(pigment_tex=GRAIN, tex_amount=TEX, paper_strength=PS)
img.save(OUT)
print("painted", OUT, round(time.time() - t0, 1), "s")
