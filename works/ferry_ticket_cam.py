# Part of ferry-ticket-1101.py (Boat Ticket): a pinhole camera over the ledge.
# 透视：只用来算平面上的几条线和票的四个角，光和影照眼睛判断
import numpy as np
class Cam:
    def __init__(s, W, H, f, horizon_y, h):
        s.W, s.H, s.f, s.h = W, H, f, h
        s.cx, s.cy = W / 2, H / 2
        s.th = np.arctan((s.cy - horizon_y) / f)      # 往下俯的角度
    def p(s, X, Z, Y=0.0):
        """台面上一点（X 右、Z 往前、Y 离台面高）→ 画面坐标。眼睛在台面上方 h。"""
        y = Y - s.h
        c, sn = np.cos(s.th), np.sin(s.th)
        zc = Z * c - y * sn
        yc = Z * sn + y * c
        return (s.cx + s.f * X / zc, s.cy - s.f * yc / zc)
    def rot(s, X, Z, yaw):
        a = np.radians(yaw); return (X * np.cos(a) - Z * np.sin(a), X * np.sin(a) + Z * np.cos(a))

def plane_uv(cam, H, W, yaw):
    """每个像素落在台面上的哪一点（台面自己的坐标 u 沿台面、v 往里），视平线以上给 nan"""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dx = (xx - cam.cx) / cam.f; dy = -(yy - cam.cy) / cam.f
    c, s = np.cos(cam.th), np.sin(cam.th)
    Zd = c + dy * s; Yd = -s + dy * c
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(Yd < -1e-6, -cam.h / Yd, np.nan)
    X, Z = dx * t, Zd * t
    a = np.radians(-yaw)
    return X * np.cos(a) - Z * np.sin(a), X * np.sin(a) + Z * np.cos(a)

def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    _, _, Vt = np.linalg.svd(np.array(A, float)); return Vt[-1].reshape(3, 3) / Vt[-1, -1]
