# Part of hkmap-1103.py (Roads Walked): writes masks.npz (water, parks) for the map.
# Input ref/osm.png is not included: OpenStreetMap standard-layer tiles at zoom 16, x 53547-53553,
# y 28599-28606, stitched into one 1792 x 2048 image. Map data © OpenStreetMap contributors (ODbL).
# 从 OSM 拼图里取水和公园的形，概括一遍（糊开再切，像手描时省掉的小弯），存成画布大小的布尔图
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from hkmap_geo import CROP, W, H
from PIL import Image
from scipy import ndimage as nd
im = np.asarray(Image.open('ref/osm.png').convert('RGB').crop(CROP).resize((W, H), Image.BILINEAR)).astype(float)
def near(c, tol):
    return np.sqrt(((im - np.array(c)) ** 2).sum(-1)) < tol
water = near((170, 211, 223), 22)
yy, xx = np.mgrid[0:H, 0:W].astype(float)
def seg(p, q, r):
    p, q = np.array(p, float), np.array(q, float); d = q - p
    t = np.clip(((xx - p[0]) * d[0] + (yy - p[1]) * d[1]) / (d @ d), 0, 1)
    return np.hypot(xx - p[0] - t * d[0], yy - p[1] - t * d[1]) < r
water |= seg((345, 615), (-10, 1010), 16)                                          # 西隧那道线底下是海
water = nd.binary_closing(water, nd.generate_binary_structure(2, 1), iterations=4)   # 航线虚线、字
lab, n = nd.label(water); sizes = nd.sum(water, lab, range(1, n + 1))
water = np.isin(lab, 1 + np.nonzero(sizes > 20000)[0])                               # 只留海和避风塘，公园里的池子不要
water = nd.binary_fill_holes(~water) == 0 if False else water
land = ~water
lab, n = nd.label(land); sizes = nd.sum(land, lab, range(1, n + 1))
for i, s in enumerate(sizes):
    if s < 400: land[lab == i + 1] = False                                         # 水里的字被当成小岛的，抹掉
water = ~land
w = nd.gaussian_filter(water.astype(float), 2.2) > 0.5
park = near((200, 250, 204), 14) | near((205, 235, 176), 14) | near((173, 209, 158), 14) | near((170, 224, 203), 14)
park = nd.binary_opening(nd.binary_closing(park, iterations=3), iterations=2)
lab, n = nd.label(park); sizes = nd.sum(park, lab, range(1, n + 1))
keep = np.zeros_like(park)
for i, s in enumerate(sizes):
    if s > 1500: keep |= lab == i + 1
park = nd.gaussian_filter(keep.astype(float), 2.5) > 0.5
park &= ~w
np.savez_compressed('masks.npz', water=w, park=park)
Image.fromarray((np.stack([w * 120, park * 200, w * 255], -1)).astype(np.uint8)).save('ref/masks.png')
print(w.mean(), park.mean())
