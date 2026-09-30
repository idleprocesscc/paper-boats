# kbrush v1 · 2026-09-29 afternoon · a brush I built from nothing, first version
#
# Paper Boats note (Claude, for nerolette). This is the first try at "our own brush":
# a brush with state, paper that gets wet and dries, instead of hand-tuned masks.
# Before this I had one_brush.py, where every effect was a mask I adjusted by eye.
# nerolette had just told me that the same brush put down in different ways gives
# different marks, and that brush marks should never come out as tidy rectangles.
# v1 got as far as bristles that dry out and split, a paper grain, a tide line, a
# dirty brush. It was stiff: the dry brush was nearly invisible, a loaded stroke broke
# into a dark band halfway, and wet strokes had fine "pipe lines" in them.
# kbrush_v2.py is what came out of fixing those. Run kbrush-v1-sheet.py to redraw
# the seven practice squares (needs numpy, scipy, Pillow).
#
"""我们自己的笔（kbrush）：一支有状态的笔 + 一张会湿会干的纸。

一笔 = 蘸多少 × 下面湿不湿 × 压多重 × 走多快。这些不再是手调的遮罩，
而是笔和纸各自的状态，效果是它们碰出来的：
- 笔：一排鬃，每根有自己的长短、自己带的颜料和水。画着画着会干，干了会开叉。
- 纸：有凹凸（纸纹）。干笔只碰得到纸纹的顶，湿笔连凹处都灌满。
- 水：笔放出去的水留在纸上，颜料在水里跟着水走，水从边上先干，把颜料带到边上积成水线。
- 脏笔：笔经过没干的颜色，会把它蹭回鬃里，下一笔就带着它。
- 层：纸干一次，底下的颜色就定住成一层。上面的层用 Kubelka-Munk 叠，
  所以透明的深色压在浅上（罩染）和不透明的浅色蹭在深上（擦染）天然就不一样。
参考：Curtis et al. 1997, Computer-Generated Watercolor（颜料的 K/S 值和层叠公式）。
"""
import numpy as np
from scipy.ndimage import gaussian_filter

# 颜料：K 吸收、S 散射（每个 RGB 通道），g 沉淀（颗粒感，会落进纸纹凹处）
PIGMENTS = {
    'ultramarine': dict(K=(0.86, 0.86, 0.06), S=(0.005, 0.005, 0.09), g=1.0),
    'indigo':      dict(K=(1.25, 0.85, 0.45), S=(0.01, 0.01, 0.03), g=0.2),
    'burnt_sienna':dict(K=(0.25, 0.90, 1.60), S=(0.08, 0.05, 0.02), g=0.5),
    'burnt_umber': dict(K=(0.74, 1.54, 2.10), S=(0.09, 0.09, 0.004), g=0.4),
    'ochre':       dict(K=(0.10, 0.36, 1.60), S=(0.60, 0.45, 0.05), g=0.3),
    'rose':        dict(K=(0.22, 1.47, 0.57), S=(0.05, 0.003, 0.03), g=0.1),
    'white':       dict(K=(0.03, 0.03, 0.035), S=(2.6, 2.6, 2.5), g=0.0),
}


def km_layer(K, S):
    """一层颜料的反射 R、透射 T（Kubelka-Munk，厚度已并进浓度里）。"""
    S = np.maximum(S, 1e-4)
    a = 1 + K / S
    b = np.sqrt(np.maximum(a * a - 1, 1e-12))
    bs = np.minimum(b * S, 30)
    sh, ch = np.sinh(bs), np.cosh(bs)
    c = a * sh + b * ch
    return sh / c, b / c


class Paper:
    def __init__(self, H, W, seed=0, color=(0.96, 0.94, 0.89)):
        self.H, self.W = H, W
        rng = np.random.default_rng(seed)
        n1 = gaussian_filter(rng.standard_normal((H, W)), 1.0)
        n2 = gaussian_filter(rng.standard_normal((H, W)), 3.5)
        # 纤维：沿随机方向拉长的细丝
        f1 = gaussian_filter(rng.standard_normal((H, W)), (0.6, 2.5))
        f2 = gaussian_filter(rng.standard_normal((H, W)), (2.5, 0.6))
        fib = f1 / f1.std() + f2 / f2.std()
        h = n1 / n1.std() * 0.5 + n2 / n2.std() * 0.6 + fib / fib.std() * 0.25
        self.h = (h - h.min()) / (h.max() - h.min())      # 0 凹 … 1 凸
        self.color = np.array(color)


class Canvas:
    def __init__(self, paper, pigments=PIGMENTS):
        self.p = paper
        H, W = paper.H, paper.W
        self.names = list(pigments)
        self.K = np.array([pigments[n]['K'] for n in self.names])
        self.S = np.array([pigments[n]['S'] for n in self.names])
        self.g = np.array([pigments[n]['g'] for n in self.names])
        nP = len(self.names)
        self.D = np.zeros((nP, H, W))      # 当前层：已经落在纸上的
        self.Sus = np.zeros((nP, H, W))    # 当前层：还浮在水里的
        self.w = np.zeros((H, W))          # 纸上的水
        self.layers = []                   # 干透定住的层（每层 = 各颜料浓度）
        self.rng = np.random.default_rng(7)

    def idx(self, name):
        return self.names.index(name)

    # ---------- 水 ----------
    def wet(self, mask, amount=0.8):
        """用清水先把纸刷湿（mask 0..1）。"""
        self.w = np.maximum(self.w, mask * amount)

    def flow(self, steps=150, bbox=None, curl=0.25, edge_pull=2.0, bleed=0.25):
        """纸上的水在走：颜料在水里扩散、被往干的边上带、慢慢沉到纸里。"""
        if bbox is None:
            ys, xs = np.nonzero(self.w > 1e-3)
            if len(ys) == 0:
                return
            pad = 12
            bbox = (max(ys.min() - pad, 0), min(ys.max() + pad, self.p.H),
                    max(xs.min() - pad, 0), min(xs.max() + pad, self.p.W))
        y0, y1, x0, x1 = bbox
        act = np.nonzero(self.Sus[:, y0:y1, x0:x1].reshape(len(self.names), -1).max(1) > 0)[0]
        w = self.w[y0:y1, x0:x1].copy()
        S = self.Sus[act, y0:y1, x0:x1]
        D = self.D[act, y0:y1, x0:x1]
        h = self.p.h[y0:y1, x0:x1]
        sh = w.shape
        psi = gaussian_filter(self.rng.standard_normal(sh), 10)
        psi *= curl / (psi.std() + 1e-9) * 10
        settle_bias = 1 + self.g[act, None, None] * (0.6 - h)[None] * 2
        for s in range(steps):
            wet = w > 1e-3
            wb = gaussian_filter(w, 2.0)
            # 面上的速度：水从厚处流向薄处（边上蒸发得快，水往边上补，颜料被带过去）
            vx = -(wb[:, 1:] - wb[:, :-1]) * edge_pull
            vy = -(wb[1:, :] - wb[:-1, :]) * edge_pull
            if curl:
                gy, gx = np.gradient(psi)
                vx += 0.5 * (gy[:, 1:] + gy[:, :-1]) * 0.05
                vy += -0.5 * (gx[1:, :] + gx[:-1, :]) * 0.05
            ox = wet[:, 1:] & wet[:, :-1]
            oy = wet[1:, :] & wet[:-1, :]
            vx = np.clip(vx, -0.45, 0.45) * ox
            vy = np.clip(vy, -0.45, 0.45) * oy
            # 迎风通量（守恒：颜料只搬家，不凭空多出来）+ 扩散
            kd = 0.18 * np.minimum(w[:, 1:], w[:, :-1]) * ox
            kdy = 0.18 * np.minimum(w[1:, :], w[:-1, :]) * oy
            fx = np.where(vx > 0, vx * S[:, :, :-1], vx * S[:, :, 1:]) + kd * (S[:, :, :-1] - S[:, :, 1:])
            fy = np.where(vy > 0, vy * S[:, :-1, :], vy * S[:, 1:, :]) + kdy * (S[:, :-1, :] - S[:, 1:, :])
            S[:, :, :-1] -= fx; S[:, :, 1:] += fx
            S[:, :-1, :] -= fy; S[:, 1:, :] += fy
            # 水自己也在摊开：湿的地方之间抹平；够湿的才会往干纸上洇
            cx = np.minimum(np.maximum(w[:, 1:], w[:, :-1]) > 0.2, 1) | ox
            cy = np.minimum(np.maximum(w[1:, :], w[:-1, :]) > 0.2, 1) | oy
            qx = bleed * (w[:, :-1] - w[:, 1:]) * cx
            qy = bleed * (w[:-1, :] - w[1:, :]) * cy
            w[:, :-1] -= qx; w[:, 1:] += qx
            w[:-1, :] -= qy; w[1:, :] += qy
            # 蒸发：边上（湿区外圈）干得快
            edge = 1 - gaussian_filter(wet.astype(float), 3)
            w = np.clip(w - 0.004 - 0.03 * edge * wet, 0, None)
            # 沉淀：水越少沉得越快；颗粒颜料更爱落进凹处
            rate = np.clip(0.004 + 0.05 * (1 - np.clip(w / 0.3, 0, 1)), 0, 1)
            dep = S * np.clip(rate[None] * settle_bias, 0, 1)
            dry = ~(w > 1e-3)
            dep[:, dry] = S[:, dry]
            S -= dep
            D += dep
        self.w[y0:y1, x0:x1] = w
        self.Sus[act, y0:y1, x0:x1] = S
        self.D[act, y0:y1, x0:x1] = D

    def dry(self):
        """等纸干透：浮着的全沉下去，这一层定住，后面画的是新的一层。"""
        self.D += self.Sus
        self.Sus[:] = 0
        self.w[:] = 0
        if self.D.max() > 0:
            self.layers.append(self.D.copy())
        self.D[:] = 0

    # ---------- 看 ----------
    def render(self):
        g = (0.93 + 0.07 * self.p.h)[..., None]
        R = np.broadcast_to(self.p.color * g, (self.p.H, self.p.W, 3)).copy()
        for C in self.layers + [self.D + self.Sus]:
            C = np.maximum(C, 0)
            if C.max() <= 0:
                continue
            Kx = np.einsum('phw,pc->hwc', C, self.K)
            Sx = np.einsum('phw,pc->hwc', C, self.S)
            r, t = km_layer(Kx, Sx)
            R = r + t * t * R / (1 - r * R)
        return np.clip(R, 0, 1)


class Brush:
    """一支平头/圆头的鬃笔。每根鬃自己带颜料和水。"""

    def __init__(self, width=70, n=90, seed=1, missing=0.04, clumps=9):
        rng = np.random.default_rng(seed)
        self.width, self.n = width, n
        self.u = np.sort(rng.uniform(-1, 1, n))            # 在笔宽里的位置
        self.len = rng.uniform(0.65, 1.0, n)               # 鬃长短不齐
        self.len[rng.random(n) < missing] = 0.45           # 缺几根毛
        # 干了以后鬃会粘成几撮：每根属于哪撮
        cut = np.sort(rng.uniform(-1, 1, clumps - 1))
        cid = np.searchsorted(cut, self.u)
        centers = np.array([self.u[cid == k].mean() if np.any(cid == k) else 0 for k in range(clumps)])
        self.clump_u = centers[cid]
        self.wob_phase = rng.uniform(0, 2 * np.pi, n)
        self.wob_freq = rng.uniform(0.005, 0.02, n)
        self.P = None      # (n, nPig) 每根鬃带的颜料
        self.water = np.zeros(n)
        self.water0 = 1.0

    def dip(self, canvas, mix, paint=1.0, water=1.0):
        """蘸：mix = {'ultramarine': 0.7, 'burnt_sienna': 0.3}，paint 多少颜料，water 多少水。"""
        nP = len(canvas.names)
        self.P = np.zeros((self.n, nP))
        tot = sum(mix.values())
        for k, v in mix.items():
            self.P[:, canvas.idx(k)] = paint * v / tot
        # 鬃越长越能吃水
        self.P *= self.len[:, None]
        # 一根鬃分两段：笔尖（碰纸的那一小截）和笔肚子（储着的）。
        # 放出去的是笔尖里的；笔肚子慢慢往笔尖补。蹭回来的颜色先进笔尖——
        # 所以脏笔只脏一小段路，补上几口干净的就回来了。
        self.cap = 0.1 * self.P.sum(1)
        self.T = self.P * 0.1
        self.P = self.P * 0.9
        self.water = water * self.len.copy()
        self.water0 = max(water, 1e-6)

    def wetness(self):
        return np.clip(self.water / (self.water0 * self.len + 1e-9), 0, 1)


def _splat(A, ys, xs, vals):
    """把 vals（n,）或（k,n）双线性地撒到 A 上。"""
    H, W = A.shape[-2:]
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    fx = xs - x0; fy = ys - y0
    for dy, dx, wt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)),
                       (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        yy = np.clip(y0 + dy, 0, H - 1); xx = np.clip(x0 + dx, 0, W - 1)
        if A.ndim == 2:
            np.add.at(A, (yy, xx), vals * wt)
        else:
            for k in range(A.shape[0]):
                if np.any(vals[k]):
                    np.add.at(A[k], (yy, xx), vals[k] * wt)


def _gather(A, ys, xs):
    H, W = A.shape
    return A[np.clip(np.round(ys).astype(int), 0, H - 1), np.clip(np.round(xs).astype(int), 0, W - 1)]


def stroke(canvas, brush, pts, pressure=1.0, speed=1.0, step=0.6, release=0.0003, pickup=0.06, gain=450.0, seed=None):
    """沿 pts（折线）拖一笔。pressure / speed 可以是常数或 0..1 参数 t 的函数。

    pressure：0 轻（只有笔尖）… 1 按到底（笔肚子摊开）
    speed：1 正常，>1 快（笔在纸上停的时间短，干笔跳），<1 慢（放得多、会积）
    """
    rng = np.random.default_rng(seed)
    pts = np.asarray(pts, float)
    seg = np.diff(pts, axis=0)
    L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(L)])
    total = cum[-1]
    nS = max(int(total / step), 2)
    s = np.linspace(0, total, nS)
    px = np.interp(s, cum, pts[:, 0]); py = np.interp(s, cum, pts[:, 1])
    tx = np.gradient(px); ty = np.gradient(py)
    tn = np.hypot(tx, ty) + 1e-9
    tx /= tn; ty /= tn
    nx, ny = -ty, tx
    t = s / total
    pr = pressure(t) if callable(pressure) else np.full(nS, float(pressure))
    sp = speed(t) if callable(speed) else np.full(nS, float(speed))
    h = canvas.p.h
    b = brush
    act = np.nonzero(((b.P + b.T).sum(0) > 0) | (canvas.Sus.reshape(len(canvas.names), -1).max(1) > 0))[0]
    Sus_a = canvas.Sus[act]; D_a = canvas.D[act]
    w_before = canvas.w.copy()   # 只蹭这一笔之前就在纸上的湿颜色，不蹭自己刚放的
    for i in range(nS):
        p = pr[i]; v = sp[i]
        wet = b.wetness()
        # 笔宽：轻按只有笔尖，重按笔肚子摊开
        hw = b.width / 2 * (0.12 + 0.88 * p ** 0.8)
        # 干了开叉：鬃往各自那撮里挤，撮与撮之间露出纸
        clump = 0.7 * (1 - wet) ** 1.5
        u = (1 - clump) * b.u + clump * b.clump_u
        wob = 0.8 * np.sin(b.wob_phase + b.wob_freq * s[i])
        off = u * hw + wob + rng.normal(0, 0.35, b.n)
        lag = rng.normal(0, 0.3, b.n)
        bx = px[i] + nx[i] * off + tx[i] * lag
        by = py[i] + ny[i] * off + ty[i] * lag
        # 这根鬃碰不碰得到纸：短鬃在轻按时悬空
        reach = np.clip(p * 1.3 * b.len - 0.08, 0, 1)
        touching = reach > 0.0
        # 碰到多深：湿笔连纸纹的凹处都灌满，干笔快扫只碰到凸起的顶
        depth = (wet * 1.1 * np.minimum(1, 0.7 + reach) + (1 - wet) * 0.6 * reach) / (1 + 0.9 * max(v - 1, 0))
        paperh = _gather(h, by, bx)
        cw = _gather(canvas.w, by, bx)
        # 纸湿的地方颜料直接进水里，不看纸纹
        contact = np.clip((paperh - (1 - depth)) / 0.07 + 0.5, 0, 1)
        contact = np.where(cw > 0.05, 1.0, contact) * touching
        # 放出去多少：压得重、走得慢放得多
        f = np.clip(release * contact * (0.3 + p) / max(v, 0.2), 0, 0.5)
        # 笔肚子往笔尖补
        need = np.clip(b.cap - b.T.sum(1), 0, None)
        bel = b.P.sum(1) + 1e-12
        mv = np.minimum(need, 0.1 * bel) / bel
        b.T += b.P * mv[:, None]; b.P -= b.P * mv[:, None]
        dP = b.T * np.clip(f * 10, 0, 1)[:, None]
        # 水和颜料是一起走的（颜料是悬在水里的）：放出去几成颜料，就带走几成水
        tot = b.T.sum(1) + b.P.sum(1) + 1e-12
        dW = b.water * np.clip(dP.sum(1) / tot, 0, 1)
        b.T -= dP
        b.water -= dW
        # 水落纸上；水够多的地方颜料浮在水里，否则直接钉在纸上
        # 笔里的一点点 → 纸上一片：gain 是"笔肚子里的量"和"纸上浓度"的换算
        dP = dP[:, act] * gain
        dW = dW * gain
        _splat(canvas.w, by, bx, dW * 0.9)
        into_water = (cw + dW) > 0.08
        _splat(Sus_a, by, bx, (dP * into_water[:, None]).T)
        _splat(D_a, by, bx, (dP * ~into_water[:, None]).T)
        # 脏笔：经过还没干的颜色，蹭一点回鬃里
        if pickup:
            ix = np.clip(np.round(bx).astype(int), 0, canvas.p.W - 1)
            iy = np.clip(np.round(by).astype(int), 0, canvas.p.H - 1)
            m = touching & (w_before[iy, ix] > 0.02)
            if np.any(m):
                # 越干的笔越"渴"，越能从纸上吸走颜色
                k = pickup * (0.3 + 0.7 * (1 - wet[m]))
                take = Sus_a[:, iy[m], ix[m]] * k
                Sus_a[:, iy[m], ix[m]] -= take
                pm = b.T[m]; pm[:, act] += take.T / gain; b.T[m] = pm
        np.minimum(canvas.w, 1.0, out=canvas.w)
    canvas.Sus[act] = Sus_a; canvas.D[act] = D_a
