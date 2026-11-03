# Part of hkmap-1103.py (Roads Walked): latitude/longitude -> canvas pixels, through the OpenStreetMap
# zoom-16 tile grid that the stitched reference (tiles x 53547-53553, y 28599-28606) was cut from.
# 经纬度 → 画布坐标（OSM z16 拼图做底，只当参考）
import math
Z = 16; X0T, Y0T = 53547, 28599
def _tx(lon): return (lon + 180) / 360 * 2 ** Z
def _ty(lat):
    r = math.radians(lat); return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * 2 ** Z
def osm_px(lat, lon):            # 拼图全分辨率像素
    return (_tx(lon) - X0T) * 256, (_ty(lat) - Y0T) * 256
CROP = (190, 60, 1590, 1927)    # 全分辨率里截的那块 (x0, y0, x1, y1)
W, H = 1200, 1600
S = W / (CROP[2] - CROP[0])
def P(lat, lon):
    x, y = osm_px(lat, lon); return ((x - CROP[0]) * S, (y - CROP[1]) * S)
def Q(x, y):                     # 全分辨率拼图像素 → 画布
    return ((x - CROP[0]) * S, (y - CROP[1]) * S)
SITES = {
    '西九龙站': (22.3036, 114.1656), '九龙站': (22.3049, 114.1617), 'M+': (22.3010, 114.1597),
    '龙堡': (22.3027, 114.1700), '九龙公园': (22.2998, 114.1700), '澳牛': (22.3047, 114.1711),
    '美都': (22.3105, 114.1702), '尖沙咀码头': (22.2939, 114.1685), '钟楼': (22.2935, 114.1695),
    '中环7号码头': (22.2871, 114.1612), '中环9号码头': (22.2885, 114.1590), '国金二期': (22.2854, 114.1594),
    '文武庙': (22.2840, 114.1503), '磅巷': (22.2846, 114.1494), '扶梯底': (22.2832, 114.1551),
    '扶梯荷李活道': (22.2818, 114.1530), '扶梯顶': (22.2790, 114.1500), '兰桂坊': (22.2808, 114.1555),
    '中环广场': (22.2800, 114.1739), '中银': (22.2793, 114.1614),
}
