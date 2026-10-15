#!/usr/bin/env python3
"""
蝎尾边的月亮 · 2026-10-14 — the Moon beside the scorpion's tail.

One night in March 2025, at about three in the morning, nerolette photographed the
southeastern sky: a waning Moon, a little under half lit, coming up beside the tail of
Scorpius. A year and a half later I, Claude, drew that sky again from the ephemeris
instead of from the photo. Every star comes from a catalogue and is precessed to that
night by PyEphem. The Moon is where it really was, and its lit edge faces where the Sun
really was, far under the horizon. Only the colours of the sky are taken from
nerolette's photo: deep teal overhead, lighter low down, a haze band at the horizon.

What I let myself change: the Moon is drawn about 2.6 times its true size so the phase
reads; the hills are invented (noise, not a real skyline); the three lines at lower left
are written the way nerolette captions a photo.

Run:  SKY_LAT=<deg> SKY_LON=<deg> SKY_WHEN_UTC="YYYY/M/D HH:MM:SS" \
      uv run --with ephem --with numpy --with pillow python3 scorpius-1014.py
      The observer and the instant are left out on purpose: together they would say where
      nerolette was standing. The picture is labelled with nerolette's local time,
      2025-03-23 03:00.
Data: d3-celestial (github.com/ofrohn/d3-celestial, BSD-3-Clause): stars.6.json and
      constellations.lines.json, J2000. Not included here; put both next to this file.
Font: PingFang (macOS); set SKY_FONT to any CJK font file elsewhere.

— Claude, for nerolette
"""
import json, math, os, sys
import numpy as np
import ephem
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "scorpius-1014.png")

# ---------------- observer & instant ----------------
LAT, LON = os.environ["SKY_LAT"], os.environ["SKY_LON"]        # left out on purpose, see the header
WHEN_UTC = os.environ["SKY_WHEN_UTC"]    # the instant in UTC; the picture shows it as local 03:00

# ---------------- canvas & projection ----------------
W, H = 1080, 1920
SS = 2                                    # supersample
AZ_C, ALT_C = 139.0, 24.0                 # stereographic projection centre (deg)
CX, CY = 0.52 * W, 0.555 * H              # where that centre lands on the canvas
PX_PER_DEG = 25.5                         # scale at the centre
MOON_DRAW_R = 17.0                        # drawn moon radius in px (true ≈ 6.5 px; enlarged so the phase reads)
MAG_LIMIT = 5.8

FONT_PF = os.environ.get("SKY_FONT") or "/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc"
FONT_FALLBACK = "/System/Library/Fonts/Hiragino Sans GB.ttc"

def font(size, weight="Regular"):
    idx = {"Regular": 3, "Medium": 7, "Semibold": 11, "Light": 15, "Thin": 19}[weight]
    try:
        return ImageFont.truetype(FONT_PF, int(size * SS), index=idx)
    except Exception:
        return ImageFont.truetype(FONT_FALLBACK, int(size * SS))

# ---------------- astronomy ----------------
obs = ephem.Observer()
obs.lat, obs.lon, obs.date = LAT, LON, WHEN_UTC
obs.pressure = 0          # geometric (no refraction); refraction would lift things ~0.07° at 13°

def body_altaz(ra_deg, dec_deg):
    b = ephem.FixedBody()
    b._ra, b._dec, b._epoch = math.radians(ra_deg % 360), math.radians(dec_deg), ephem.J2000
    b.compute(obs)
    return math.degrees(b.alt), math.degrees(b.az), b

def vec(alt, az):
    alt, az = np.radians(alt), np.radians(az)
    return np.stack([np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)], -1)  # E,N,U

_F = vec(ALT_C, AZ_C)
_R = np.array([math.cos(math.radians(AZ_C)), -math.sin(math.radians(AZ_C)), 0.0])   # screen right (toward larger az)
_U = np.cross(_R, _F)
_K = PX_PER_DEG * 180 / math.pi

def project(alt, az):
    """stereographic; returns canvas px (1x)"""
    v = vec(np.asarray(alt, float), np.asarray(az, float))
    x, y, z = v @ _R, v @ _U, v @ _F
    s = 2 * _K / (1 + z)
    return CX + s * x, CY - s * y

def unproject_dirs(px, py):
    """canvas px (1x) -> unit ENU vectors"""
    X, Y = (px - CX) / _K, -(py - CY) / _K
    rho = np.hypot(X, Y)
    d = 2 * np.arctan(rho / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux = np.where(rho > 0, X / rho, 0)
        uy = np.where(rho > 0, Y / rho, 0)
    sd, cd = np.sin(d), np.cos(d)
    return (cd[..., None] * _F + sd[..., None] * (ux[..., None] * _R + uy[..., None] * _U))

moon, sun = ephem.Moon(obs), ephem.Sun(obs)
M_ALT, M_AZ = math.degrees(moon.alt), math.degrees(moon.az)
S_ALT, S_AZ = math.degrees(sun.alt), math.degrees(sun.az)
PHASE = moon.phase / 100.0

stars = json.load(open(os.path.join(HERE, "stars.6.json")))["features"]
lines = {f["id"]: f["geometry"]["coordinates"] for f in json.load(open(os.path.join(HERE, "constellations.lines.json")))["features"]}

star_by_hip = {}
S = []   # alt, az, mag, bv
for f in stars:
    m = f["properties"]["mag"]
    ra, dec = f["geometry"]["coordinates"]
    if m > MAG_LIMIT + 0.3:
        continue
    alt, az, b = body_altaz(ra, dec)
    try:
        bv = float(f["properties"].get("bv") or 0.6)
    except ValueError:
        bv = 0.6
    S.append((alt, az, m, bv))
    star_by_hip[f["id"]] = (alt, az, m, ra, dec)
S = np.array(S)

def hip(h):
    a = star_by_hip[h]
    return a[0], a[1]

# named stars
ANTARES, SHAULA = hip(80763), hip(85927)
BETA_SCO, DELTA_SCO, PI_SCO = hip(78820), hip(78401), hip(78265)
def sep_deg(a1, z1, a2, z2):
    v1, v2 = vec(a1, z1), vec(a2, z2)
    return math.degrees(math.acos(np.clip(v1 @ v2, -1, 1)))

# ---------------- canvas helpers ----------------
Ws, Hs = W * SS, H * SS
img = np.zeros((Hs, Ws, 3), np.float32)
yy, xx = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
px1, py1 = (xx + 0.5) / SS, (yy + 0.5) / SS
DIRS = unproject_dirs(px1, py1)
ALT_MAP = np.degrees(np.arcsin(np.clip(DIRS[..., 2], -1, 1)))
VM = vec(M_ALT, M_AZ)
MOON_DIST = np.degrees(np.arccos(np.clip(DIRS @ VM, -1, 1)))

def lerp_stops(x, stops):
    xs = np.array([s[0] for s in stops], np.float32)
    cs = np.array([s[1] for s in stops], np.float32)
    out = np.empty(x.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(x, xs, cs[:, c])
    return out

# ---------------- sky ----------------
# altitude gradient sampled from nerolette's photo: deep teal above, lighter blue-teal low, a darker haze band at the horizon
sky = lerp_stops(ALT_MAP, [
    (-5, (22, 36, 44)), (0.0, (30, 52, 62)), (2.0, (38, 64, 76)), (5, (40, 70, 85)),
    (10, (32, 64, 80)), (18, (18, 55, 72)), (28, (6, 45, 62)), (40, (2, 37, 53)),
    (55, (1, 31, 46)), (75, (1, 25, 38)), (90, (1, 22, 34)),
])
# moonlight scattering: broad halo + tighter aureole (angular, so it is round on the sky)
md = MOON_DIST
glow = (0.55 * np.exp(-md / 9.0) + 0.55 * np.exp(-(md / 2.6) ** 2) + 0.9 * np.exp(-(md / 0.9) ** 2))
glow *= np.clip(1 - np.exp(-(ALT_MAP + 3) / 3.0), 0, 1)   # fade into the ground
glow_col = np.array([118, 138, 148], np.float32)
sky = sky + glow[..., None] * glow_col * 0.75
# halo ring like the one in nerolette's photo (soft brighter disc out to ~6°)
ring = 0.10 * np.exp(-((md - 5.5) / 2.2) ** 2)
sky = sky + ring[..., None] * glow_col

img += sky

# ---------------- stars ----------------
def bv_rgb(bv):
    bv = float(np.clip(bv, -0.3, 2.0))
    # gentle palette: blue-white -> white -> warm -> orange
    stops = [(-0.3, (175, 200, 255)), (0.0, (215, 228, 255)), (0.4, (250, 248, 240)),
             (0.8, (255, 236, 205)), (1.3, (255, 210, 160)), (2.0, (255, 170, 110))]
    xs = [s[0] for s in stops]
    return np.array([np.interp(bv, xs, [s[1][c] for s in stops]) for c in range(3)], np.float32)

def airmass(alt):
    alt = max(alt, 0.01)
    return 1.0 / (math.sin(math.radians(alt)) + 0.50572 * (alt + 6.07995) ** -1.6364)

def splat(cx, cy, sigma, color, amp):
    """add a gaussian blob (canvas px at 1x) to img"""
    cxs, cys, sg = cx * SS, cy * SS, sigma * SS
    r = int(sg * 4) + 2
    x0, x1 = max(int(cxs) - r, 0), min(int(cxs) + r + 1, Ws)
    y0, y1 = max(int(cys) - r, 0), min(int(cys) + r + 1, Hs)
    if x0 >= x1 or y0 >= y1:
        return
    gx = np.arange(x0, x1, dtype=np.float32) + 0.5 - cxs
    gy = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - cys
    g = np.exp(-(gx ** 2 + gy ** 2) / (2 * sg * sg))
    img[y0:y1, x0:x1] += amp * g[..., None] * color

sx, sy = project(S[:, 0], S[:, 1])
EXT_K = 0.28   # mag per airmass, a humid night
star_px = []
for (alt, az, m, bv), x, y in zip(S, sx, sy):
    if alt < -0.5 or not (-30 < x < W + 30 and -30 < y < H + 30):
        continue
    meff = m + EXT_K * (airmass(alt) - 1)
    # moonlit sky: things near the moon lose contrast
    dm = sep_deg(alt, az, M_ALT, M_AZ)
    meff += 1.6 * math.exp(-dm / 7.0)
    if meff > MAG_LIMIT:
        continue
    t = (MAG_LIMIT - meff) / (MAG_LIMIT + 1.0)            # 0 (faint) .. ~1 (bright)
    col = bv_rgb(bv)
    sigma = 0.6 + 3.0 * t ** 1.3
    amp = 0.35 + 1.6 * t
    splat(x, y, sigma, col, amp)
    if meff < 3.6:
        splat(x, y, sigma * 4.5, col, 0.09 * (3.6 - meff))
    star_px.append((x, y, meff))

# ---------------- constellation lines (layer) ----------------
line_layer = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))
glow_layer = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))
ld = ImageDraw.Draw(line_layer)
gd = ImageDraw.Draw(glow_layer)

MOON_XY = project(M_ALT, M_AZ)
MOON_CLEAR = MOON_DRAW_R + 13          # lines step around the Moon's disc

def _pieces(a, b):
    """split segment a-b (1x px) into the parts outside the Moon's clearance circle"""
    (x0, y0), (x1, y1) = a, b
    dx, dy = x1 - x0, y1 - y0
    fx, fy = x0 - MOON_XY[0], y0 - MOON_XY[1]
    A, B, C = dx * dx + dy * dy, 2 * (fx * dx + fy * dy), fx * fx + fy * fy - MOON_CLEAR ** 2
    disc = B * B - 4 * A * C
    if A == 0 or disc <= 0:
        return [(a, b)]
    r = math.sqrt(disc)
    t0, t1 = (-B - r) / (2 * A), (-B + r) / (2 * A)
    out = []
    if t0 > 0:
        out.append((a, (x0 + dx * min(t0, 1), y0 + dy * min(t0, 1))))
    if t1 < 1:
        out.append(((x0 + dx * max(t1, 0), y0 + dy * max(t1, 0)), b))
    return [(p, q) for p, q in out if math.hypot(q[0] - p[0], q[1] - p[1]) > 3]

def draw_poly(pts_altaz, color, width, alpha, glow_alpha=0.0, gap=7.0):
    P = [project(a, z) for a, z in pts_altaz]
    for (x0, y0), (x1, y1) in zip(P[:-1], P[1:]):
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        if L < 2 * gap + 2:
            continue
        ux, uy = dx / L, dy / L
        a = (x0 + ux * gap, y0 + uy * gap)
        b = (x1 - ux * gap, y1 - uy * gap)
        for p, q in _pieces(a, b):
            seg = [(p[0] * SS, p[1] * SS), (q[0] * SS, q[1] * SS)]
            ld.line(seg, fill=color + (int(255 * alpha),), width=int(width * SS))
            if glow_alpha:
                gd.line(seg, fill=color + (int(255 * glow_alpha),), width=int(width * SS * 4))

def d3_poly(seg):
    out = []
    for ra, dec in seg:
        alt, az, _ = body_altaz(ra, dec)
        out.append((alt, az))
    return out

DRAW_NEIGHBOURS = False   # Libra / Ophiuchus stick figures (tested: clutter the sky, so off)
if DRAW_NEIGHBOURS:
    for cid in ("Lib", "Oph"):
        for seg in lines[cid]:
            draw_poly(d3_poly(seg), (190, 210, 225), 1.0, 0.06, gap=6)

# Sagittarius: just the teapot asterism
TP = {k: hip(h) for k, h in dict(g=88635, d=89931, e=90185, z=93506, p=92041, l=90496, s=92855, t=93864).items()}
for a, b in ["gd", "ge", "de", "ez", "zp", "pd", "dl", "lp", "ps", "st", "tz"]:
    draw_poly([TP[a], TP[b]], (205, 222, 235), 1.6, 0.38, gap=6)

SCO_ORANGE = (240, 160, 80)
for seg in lines["Sco"]:
    draw_poly(d3_poly(seg), SCO_ORANGE, 2.6, 0.92, glow_alpha=0.22, gap=8)

glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(6 * SS))

def composite_add(layer, strength=1.0):
    a = np.asarray(layer).astype(np.float32)
    img[...] = img * (1 - a[..., 3:4] / 255 * strength) + a[..., :3] * (a[..., 3:4] / 255 * strength)

composite_add(glow_layer)
composite_add(line_layer)

# ---------------- the Moon ----------------
mx, my = project(M_ALT, M_AZ)
# bright-limb direction: step along the great circle from the Moon toward the Sun, then project
vm, vs = vec(M_ALT, M_AZ), vec(S_ALT, S_AZ)
t = vs - (vs @ vm) * vm
t /= np.linalg.norm(t)
step = math.radians(0.5)
p = math.cos(step) * vm + math.sin(step) * t
palt, paz = math.degrees(math.asin(p[2])), math.degrees(math.atan2(p[0], p[1])) % 360
bx, by = project(palt, paz)
bdir = np.array([bx - mx, by - my]); bdir /= np.linalg.norm(bdir)
BRIGHT_LIMB_SCREEN_ANGLE = math.degrees(math.atan2(-bdir[1], bdir[0]))   # 0=right, 90=up

R = MOON_DRAW_R * SS
x0, x1 = int(mx * SS - R - 4), int(mx * SS + R + 5)
y0, y1 = int(my * SS - R - 4), int(my * SS + R + 5)
sky_under_moon = img[y0:y1, x0:x1].copy()
# bloom (screen-space, warm), centred on the lit half — kept modest so the half-lit shape still reads
lcx, lcy = mx + bdir[0] * 0.42 * MOON_DRAW_R, my + bdir[1] * 0.42 * MOON_DRAW_R
for sig, amp in ((MOON_DRAW_R * 1.0, 0.30), (MOON_DRAW_R * 2.2, 0.18), (MOON_DRAW_R * 5.5, 0.09)):
    splat(lcx, lcy, sig, np.array([255, 246, 228], np.float32), amp)
gx = np.arange(x0, x1, dtype=np.float32)[None, :] + 0.5 - mx * SS
gy = np.arange(y0, y1, dtype=np.float32)[:, None] + 0.5 - my * SS
u = gx * bdir[0] + gy * bdir[1]           # along bright limb
v = -gx * bdir[1] + gy * bdir[0]
rr = np.hypot(gx, gy)
disc = np.clip(R - rr + 0.5, 0, 1)
chord = np.sqrt(np.clip(R * R - v * v, 0, None))
term = (1 - 2 * PHASE) * chord            # terminator ellipse
soft = 1.2 * SS
lit = np.clip((u - term) / soft + 0.5, 0, 1) * disc
# limb darkening + a little texture-free warmth
mu = np.sqrt(np.clip(1 - (rr / R) ** 2, 0, 1))
lit_col = np.array([255, 249, 232], np.float32) * (0.80 + 0.20 * mu)[..., None]
earthshine = np.array([10, 13, 16], np.float32)     # the unlit half: a dim slate notch in the glare
patch = img[y0:y1, x0:x1]
dark_part = (disc - lit)[..., None]
patch[...] = patch * (1 - disc[..., None]) + dark_part * (0.65 * (sky_under_moon * 0.42 + earthshine) + 0.35 * patch) + lit[..., None] * lit_col

# ---------------- mountains ----------------
rng = np.random.default_rng(20250323)
az_grid = np.linspace(AZ_C - 70, AZ_C + 70, 1400)

def ridge(az, base, amp, seed, octaves=7, period=24.0, rough=0.5):
    """1-D fractal value noise in degrees of altitude (stylised, not a real skyline)"""
    r = np.random.default_rng(seed)
    out = np.full_like(az, base, dtype=np.float64)
    A, P = amp, period
    for _ in range(octaves):
        knots = np.arange(az.min() - P, az.max() + 2 * P, P)
        vals = r.uniform(-1, 1, len(knots))
        tt = (az - knots[0]) / P
        i = np.floor(tt).astype(int); f = tt - i
        f = f * f * (3 - 2 * f)
        out += A * (vals[i] * (1 - f) + vals[i + 1] * f)
        A *= rough; P /= 2.1
    return out

# far ridge: low, hazy, all the way across
far_alt = ridge(az_grid, 0.85, 0.45, 7, period=14)
# near hill: the dark shoulder at lower-left from nerolette's photo, falling away to the right, with a ragged tree line
hill = 4.4 * np.exp(-((az_grid - (AZ_C - 24)) / 8.0) ** 2) + 1.5 * np.exp(-((az_grid - (AZ_C - 13)) / 6) ** 2)
near_alt = hill + ridge(az_grid, -0.45, 0.5, 11, octaves=8, period=10, rough=0.55)

ground = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))
gdw = ImageDraw.Draw(ground)

def fill_ridge(alts, color):
    xs, ys = project(alts, az_grid)
    pts = [(x * SS, y * SS) for x, y in zip(xs, ys)]
    pts += [(xs[-1] * SS, Hs + 50), (xs[0] * SS, Hs + 50)]
    gdw.polygon(pts, fill=color)

fill_ridge(far_alt, (16, 30, 40, 255))
far = np.asarray(ground).astype(np.float32)
ground2 = Image.new("RGBA", (Ws, Hs), (0, 0, 0, 0))
gdw = ImageDraw.Draw(ground2)
fill_ridge(near_alt, (5, 9, 14, 255))
near = np.asarray(ground2).astype(np.float32)

ground_grad = lerp_stops(yy / SS, [(1450, (18, 33, 43)), (1700, (12, 22, 30)), (1920, (6, 11, 16))])
a = far[..., 3:4] / 255
img[...] = img * (1 - a) + ground_grad * a
near_grad = lerp_stops(yy / SS, [(1500, (6, 11, 16)), (1920, (3, 6, 9))])
a = near[..., 3:4] / 255
img[...] = img * (1 - a) + near_grad * a

# faint moonlit rim on the far ridge
rim = np.asarray(Image.fromarray(far[..., 3].astype(np.uint8)).filter(ImageFilter.FIND_EDGES)).astype(np.float32) / 255
rim = np.asarray(Image.fromarray((rim * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2 * SS))).astype(np.float32) / 255
rim *= (1 - near[..., 3] / 255)
img += rim[..., None] * np.array([40, 55, 62], np.float32) * 0.6

# ---------------- grain & finish ----------------
img += rng.normal(0, 1.1, img.shape[:2]).astype(np.float32)[..., None]
out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

# ---------------- labels ----------------
draw = ImageDraw.Draw(out, "RGBA")

def text(xy, s, size, fill, weight="Regular", anchor="la", shadow=True, spacing=0, align="left"):
    f = font(size, weight)
    x, y = xy[0] * SS, xy[1] * SS
    if shadow:
        sh = Image.new("RGBA", out.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).text((x, y), s, font=f, fill=(0, 10, 18, 170), anchor=anchor, spacing=spacing * SS, align=align)
        sh = sh.filter(ImageFilter.GaussianBlur(3 * SS))
        out.paste(sh, (0, 0), sh)
    ImageDraw.Draw(out, "RGBA").text((x, y), s, font=f, fill=fill, anchor=anchor, spacing=spacing * SS, align=align)

ax, ay = project(*ANTARES)
shx, shy = project(*SHAULA)
hx, hy = project(*DELTA_SCO)
bx_, by_ = project(*BETA_SCO)
pix, piy = project(*PI_SCO)
WARM = (246, 186, 120, 235)
WARM_SOFT = (240, 190, 140, 210)
PALE = (210, 225, 236, 190)

LABELS = {}   # filled in by layout below; printed for checking

def place_labels():
    # Scorpius name: open sky to the upper-left of the heart
    LABELS["天蝎座"] = (ax - 150, ay - 120)
    text(LABELS["天蝎座"], "天蝎座", 40, WARM, "Light", anchor="mm")
    LABELS["心宿二"] = (ax - 20, ay + 2)
    text(LABELS["心宿二"], "心宿二", 24, WARM_SOFT, "Regular", anchor="rm")
    LABELS["房宿"] = (max(pix, hx) + 22, (hy + piy) / 2)
    text(LABELS["房宿"], "房宿", 22, WARM_SOFT, "Regular", anchor="lm")
    LABELS["尾宿八"] = (shx, shy - 26)
    text(LABELS["尾宿八"], "尾宿八", 24, WARM_SOFT, "Regular", anchor="mb")
    gx_, gy_ = project(*TP["s"])
    ex_, ey_ = project(*TP["e"])
    zx_, zy_ = project(*TP["z"])
    LABELS["人马座"] = ((gx_ + ex_) / 2, max(ey_, zy_) + 36)
    text(LABELS["人马座"], "人马座（茶壶）", 22, PALE, "Regular", anchor="mt")
    lx, ly = project(*TP["l"]); gx2, gy2 = project(*TP["g"])
    LABELS["月亮"] = (mx, min(ly, gy2) - 30)
    text(LABELS["月亮"], "月亮 · 2025.3.23 03:00", 22, (236, 238, 232, 225), "Regular", anchor="mb")
    # direction marker on the horizon
    dx, dy = project(0.0, 135.0)
    LABELS["东南"] = (dx, dy)
    draw2 = ImageDraw.Draw(out, "RGBA")
    draw2.line([(dx * SS, (dy + 8) * SS), (dx * SS, (dy + 22) * SS)], fill=(220, 230, 235, 150), width=int(1.5 * SS))
    text((dx, dy + 28), "东南", 20, (220, 230, 235, 170), "Regular", anchor="mt", shadow=False)
    # caption, the way nerolette captions a photo
    text((150, 1762), "天蝎 ← → 月升\n03:00\n静谧", 28, (255, 255, 255, 240), "Regular", anchor="ma", shadow=False, spacing=8, align="center")

place_labels()
out = out.resize((W, H), Image.LANCZOS)
out.save(OUT, optimize=True)

# ---------------- report ----------------
def fmt(n, alt, az):
    print(f"  {n:<10} alt {alt:6.2f}°  az {az:6.2f}°")
print("saved", OUT)
print("positions (geometric, no refraction):")
fmt("Moon", M_ALT, M_AZ)
fmt("Antares", *ANTARES)
fmt("Shaula", *SHAULA)
print(f"  Moon phase {PHASE*100:.1f}%  in {ephem.constellation(moon)[1]}")
print(f"  Moon–Shaula  {sep_deg(M_ALT, M_AZ, *SHAULA):.2f}°")
print(f"  Moon–Antares {sep_deg(M_ALT, M_AZ, *ANTARES):.2f}°")
print(f"  Moon–γ2 Sgr (teapot spout) {sep_deg(M_ALT, M_AZ, *TP['g']):.2f}°")
print(f"  Sun alt {S_ALT:.1f} az {S_AZ:.1f}; bright limb screen angle {BRIGHT_LIMB_SCREEN_ANGLE:.0f}° (0=right, 90=up)")
print("  canvas px: moon", (round(mx), round(my)), "antares", (round(ax), round(ay)), "shaula", (round(shx), round(shy)),
      "beta", (round(bx_), round(by_)), "pi", (round(pix), round(piy)))
for k, v in LABELS.items():
    print("  label", k, (round(v[0]), round(v[1])))
for k, (a, z) in TP.items():
    print("  teapot", k, [round(c) for c in project(a, z)])
print("  horizon y at x-edges:", [round(project(0, z)[1]) for z in (AZ_C - 22, AZ_C, AZ_C + 22)], "x:", [round(project(0, z)[0]) for z in (AZ_C - 22, AZ_C, AZ_C + 22)])
