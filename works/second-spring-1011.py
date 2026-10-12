# 第二年春天 · 10.11
# 这张在说什么：太阳先照到了它们。小飞机停在草地上，穿白袍的人站在旁边抬头，手上的鹤头手偶垂着没事干了；
#   天上五只鹤自己往北飞，已经飞进了太阳里，他还站在地上的蓝里。
#   （2001 年秋天他们开着超轻机领人工孵的美洲鹤从威斯康星飞到佛罗里达；2002 年春天第一批自己飞回来了。这张是想的，不是照片。）
# 从哪儿看：平视，人的眼睛高，站在他身后几步。地平线很低（88%），天占大半张。
# 光：太阳在背后（东边），刚好还在地平线下一点点。地上的东西只吃天光：散的、冷的，东边那半天亮，
#     所以人的后背（朝着我们、朝东）是全身最亮的一面，两侧转过去进冷灰。没有投影。
#     我们面朝西：西边低空是一条深的蓝灰（地球自己的影子），上面一条粉（维纳斯带），再往上淡紫、蓝、深蓝。
#     几百米高的鹤已经看得见太阳了：它们被低太阳从我们背后照着，整只是暖的桃金色，黑的是翅尖那几根初级飞羽。
# 明暗：最亮是鹤；最深是左下角近处那块（机翼底下、螺旋桨护圈、近处的草）。白袍的帽兜压在地球影子那条深带上，是第二对比。
# 视线：左下的机翼尖指向人 → 人往左上仰的头 → 顺着看过去，左上那串鹤，最后一只离我们最近，前面越飞越小。
# 唯一鲜的颜色：鹤身上的光；橘色的机翼在蓝里压得发闷；手偶头顶一点红。
# 质感：台阶与颗粒。天是平的阶，阶边喷开；布、草糙；鹤的边干净一点（被光照着的东西边利）。整张一张纸纹。
import os, sys, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
KB = '~/.claude/workspace/painting-method/kbrush-cloud'
sys.path[:0] = [KB, KB + '/paint']
import kbrush as K
K.set_seed(11)
from shapes import Shape
from blocks import blob, lasso, line, soft, look, smooth, paper
from layers import Stack, fill, HARD, SOFT
from scipy.ndimage import gaussian_filter, distance_transform_edt as edt

ROUGH = os.environ.get('ROUGH') == '1'
OUT = os.environ.get('OUT', 'rough.png' if ROUGH else 'v1.png')
H, W = 1200, 900
yy, xx = np.mgrid[0:H, 0:W].astype(float)
ALL = Shape(np.ones((H, W), bool))
M = lambda s: np.asarray(getattr(s, 'mask', s), bool)

def nz(sig, seed):
    r = np.random.default_rng(seed)
    n = gaussian_filter(r.standard_normal((H, W)), sig); return n / n.std()
def spray(region, sp, seed, solid=.96, dens=.7):
    region = M(region)
    if ROUGH:
        return region.astype(float) * solid
    r = np.random.default_rng(seed)
    dout = edt(~region)
    dots = (r.random((H, W)) < dens * np.exp(-dout / np.maximum(sp, .5))) & (gaussian_filter(r.random((H, W)), .5) > .47)
    return np.where(region, solid, dots * .95)
def L(pts, jag=1.5, seed=0, n=3, scale=8, closed=True):
    return lasso(H, W, smooth(pts, n, closed=closed), jag=jag, scale=scale, seed=seed)

HZ = 1062   # 地平线

# ---------- 1 天：一阶一阶，从上往下；阶沿被大尺度的噪声推得一高一低，往上喷颗粒进上一阶
SKY = [  # (这一阶的上沿, 颜色, 喷距)
    (None, (.26, .34, .51), 0),
    (170,  (.30, .39, .56), 55),
    (340,  (.35, .44, .61), 50),
    (500,  (.42, .50, .65), 46),
    (622,  (.51, .57, .70), 40),
    (716,  (.61, .63, .74), 34),
    (782,  (.73, .68, .74), 28),
    (828,  (.83, .71, .71), 20),   # 维纳斯带
    (872,  (.86, .67, .64), 12),   # 带的下沿更暖
    (904,  (.50, .54, .66), 7),    # 地球的影子：上沿比别的清楚
    (968,  (.46, .50, .63), 22),
]
st = Stack(K.Paper(H, W, seed=4), ground=SKY[0][1])
for i, (y0, c, sp) in enumerate(SKY[1:]):
    wob = 26 if y0 < 760 else 9
    edge_y = y0 + wob * nz((70, 240), 100 + i) + .3 * wob * nz((14, 36), 160 + i) + 0.014 * (xx - 450)
    fill(st.layer(f'天{i}', mask=spray(yy > edge_y, sp, 110 + i, dens=.8)), ALL, c)
# 斑驳：天里一片片大的深浅，像海绵按的；没有实心，只是颗粒疏密（上半张多，越近地平线越少）
upper = np.clip(1.15 - yy / 800, 0, 1)
mot_d = (nz((40, 90), 170) + .45 * nz((8, 14), 174) > 1.0) & (yy < 780)
fill(st.layer('斑驳暗', mode='multiply', mask=spray(mot_d, 18, 171, solid=.45, dens=.5) * upper), ALL, (.88, .90, .95))
mot_l = (nz((36, 110), 172) + .5 * nz((7, 16), 175) > 1.35) & (yy < 760)
fill(st.layer('斑驳亮', mask=spray(mot_l, 16, 173, solid=.22, dens=.45) * upper), ALL, (.55, .61, .74))
# ---------- 2 远树：地平线上一条低的、高低不齐的暗带，几棵松尖冒出来
tops = 1038 + 7 * nz((20, 26), 130)[0:1, :].repeat(H, 0) + 6 * nz((3, 6), 131)
treeline = (yy > tops) & (yy < HZ + 4)
rt = np.random.default_rng(133)
for px, ph_ in [(96, 26), (121, 40), (140, 22), (700, 30), (724, 44), (737, 28), (811, 52), (829, 34), (858, 24)]:
    w_ = rt.uniform(5, 8)
    prof = np.clip((yy - (1038 - ph_)) / ph_, 0, 1)            # 越往下越宽
    rag = 2.2 * nz((2, 1.2), 134 + px)                             # 枝一层层的毛边
    treeline |= (np.abs(xx - px - 1.5 * nz((8, 2), 135 + px)) < w_ * prof ** .8 + rag * prof) & (yy > 1038 - ph_) & (yy < HZ)
fill(st.layer('远树', mask=spray(treeline, 3, 132)), ALL, (.27, .30, .38))

# ---------- 3 地：去年的枯草，被天光照着，闷闷的冷褐；越近越深；远处一条积水照着地球影子
ground = yy >= HZ
fill(st.layer('地', mask=spray(ground, 2, 140)), ALL, (.47, .46, .48))
for i, (y0, c, sp) in enumerate([(1088, (.41, .40, .42), 8), (1130, (.35, .34, .36), 10), (1172, (.29, .28, .31), 12)]):
    fill(st.layer(f'地{i}', mask=spray(ground & (yy > y0 + 6 * nz((10, 60), 141 + i)), sp, 144 + i)), ALL, c)
gmot = ground & (nz((12, 70), 154) > .8)
fill(st.layer('地斑', mode='multiply', mask=spray(gmot, 8, 155, solid=.6, dens=.5)), ALL, (.86, .86, .90))
water = (np.abs(yy - 1074 - 2 * nz((6, 80), 150)) < 3.2) & (((xx > 655) & (xx < 905)) | ((xx > 380) & (xx < 520)))
fill(st.layer('积水', mask=spray(water, 1.5, 151)), ALL, (.58, .60, .70))
water_belt = water & (nz((3, 30), 152) > .9)
fill(st.layer('积水粉', mask=spray(water_belt, 1, 153, solid=.8)), ALL, (.80, .68, .68))
# 草：近处一簇簇竖的短毛，迎天那头亮一点
rg = np.random.default_rng(160)
tuft = np.zeros((H, W), bool); tuft_hi = np.zeros((H, W), bool)
for _ in range(1400):
    x0 = rg.uniform(-10, W + 10); y0 = HZ + 6 + (H - HZ) * rg.random() ** .7
    s = 2 + 16 * ((y0 - HZ) / (H - HZ)) ** 1.4
    for k in range(3):
        a = rg.normal(0, .35); ln = s * rg.uniform(.6, 1.2)
        x1, y1 = x0 + np.sin(a) * ln, y0 - np.cos(a) * ln
        n = max(2, int(ln))
        for t in np.linspace(0, 1, n):
            px, py = int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t)
            if 0 <= px < W and 0 <= py < H:
                (tuft_hi if t > .7 else tuft)[py, px] = True
fill(st.layer('草', mask=spray(tuft, .6, 161, solid=.8)), ALL, (.24, .23, .26))
fill(st.layer('草尖', mask=spray(tuft_hi, .6, 162, solid=.7)), ALL, (.56, .54, .55))

# ---------- 4 近处左下：机翼从画框外伸进来，尖指向人；底下一段螺旋桨护圈；整块是全张最深的地方
wing_under = L([(-30, 902), (120, 930), (250, 958), (352, 982), (368, 990), (250, 1000), (110, 1018), (-30, 1040)], jag=1, seed=170, n=2)
fill(st.layer('翼底'), wing_under, (.29, .30, .37), edge=HARD)
# 翼底上几根撑条（battens），从翼根往尖散，稍深
bat = Shape(np.zeros((H, W), bool))
for j, (x0, x1) in enumerate([(20, 40), (80, 112), (150, 190), (215, 262), (280, 320)]):
    bat = bat | line(H, W, [(x0, 912 + .24 * x0), (x1 - 6, 1016 - .2 * x1)], width=1.6, seed=171 + j)
fill(st.layer('撑条', mask=spray(M(bat) & M(wing_under), .5, 176, solid=.6)), ALL, (.21, .22, .28))
# 前缘：一条闷橘，越往尖越成了整个翼（侧过来只看得见边）
lead = L([(-30, 896), (120, 922), (250, 950), (352, 976), (372, 987), (352, 986), (250, 966), (120, 942), (-30, 918)], jag=.6, seed=177, n=2)
fill(st.layer('前缘'), lead, (.72, .42, .30), edge=HARD)
lead_hi = M(lead) & ~np.roll(M(lead), 4, 0)
fill(st.layer('前缘亮', mask=spray(lead_hi, .8, 178, solid=.8)), ALL, (.82, .55, .42))
# 护圈：圆环 + 几根辐条，大半在画框外
cx_, cy_, R_ = 60, 1148, 112
ring = Shape((np.abs(np.hypot(xx - cx_, yy - cy_) - R_) < 2.6))
ring = ring | Shape((np.abs(np.hypot(xx - cx_, yy - cy_) - R_ + 9) < 1.2) & (yy < cy_ - 40))   # 内圈一道细的
prop = lasso(H, W, [(cx_ - 4, cy_ - 4), (cx_ + 92, cy_ - 74), (cx_ + 98, cy_ - 66), (cx_ + 6, cy_ + 4)], jag=.6, seed=179)
hub = blob(H, W, cx_, cy_, 9, 9, jag=0, seed=184)
ring = ring | prop | hub
fill(st.layer('护圈', mask=spray(ring, .5, 181, solid=.9)), ALL, (.16, .16, .19))
mast = line(H, W, [(64, 1012), (70, 1080), (76, 1150)], width=4, seed=182)
fill(st.layer('桅'), mast, (.18, .18, .21), edge=HARD)

# ---------- 5 人：背对我们，头往左上仰；白袍、兜帽；两只手垂着，左手套着鹤头手偶
# 认得出"在仰头"的三件事：帽兜比肩窄、整团往后坠，后颈堆起一道深褶；帽兜左上沿露出一线里面的暗（脸朝左上翻过去了）；
#   肩膀往后打开，两手垂着不动。
torso = L([(584, 926), (620, 926), (646, 938), (660, 956), (664, 990), (667, 1040), (672, 1094),
           (640, 1102), (604, 1105), (566, 1102), (538, 1096), (543, 1040), (546, 990), (550, 958), (562, 938)], jag=1.2, seed=190, n=3)
armL = L([(556, 944), (544, 956), (534, 976), (528, 1002), (526, 1026), (536, 1032), (544, 1026), (548, 996), (556, 968)], jag=.8, seed=186, n=2)
armR = L([(650, 944), (664, 958), (674, 978), (680, 1004), (683, 1028), (672, 1034), (664, 1026), (660, 994), (652, 968)], jag=.8, seed=187, n=2)
# 帽兜：比肩窄，后脑那团往右后坠，尖儿耷拉在右后方
hood = L([(576, 934), (568, 918), (566, 902), (570, 887), (580, 876), (594, 870), (609, 871), (622, 878), (632, 889),
          (640, 902), (645, 913), (638, 925), (626, 934), (600, 940)], jag=.8, seed=188, n=3)
robe = torso | armL | armR | hood
fill(st.layer('袍'), robe, (.68, .70, .77), edge=HARD)
# 帽兜在后背上落一片软的暗（天光从上来，帽兜挡了一点），不画硬的领圈
under_hood = M(torso) & (yy < 958 + 6 * nz((8, 20), 236)) & (np.abs(xx - 604) < 50)
fill(st.layer('帽兜下暗', mode='multiply', mask=spray(under_hood, 8, 237, solid=.8)), ALL, (.86, .87, .93))
# 后背朝东，吃东边亮天：一团软的亮，肩胛下那带最亮，往下收
yb = np.clip((yy - 960) / 140, 0, 1)
back_lit = M(torso) & (((xx - 604 - 5 * nz((24, 6), 191)) / (40 - 12 * yb)) ** 2 + ((yy - 1000) / 70) ** 2 < 1)
fill(st.layer('后背亮', mask=spray(back_lit, 9, 192)), ALL, (.79, .78, .82))
core = M(torso) & (((xx - 604) / 26) ** 2 + ((yy - 982) / 30) ** 2 + .4 * nz((5, 5), 189) < 1)
fill(st.layer('肩胛亮', mask=spray(core, 6, 185)), ALL, (.85, .83, .85))
# 帽兜：后脑勺对着东边的天，左上大半亮；右下转过去进冷灰
hood_lit = M(hood) & ((xx - 596) * .6 + (yy - 892) < 18 + 3 * nz((6, 6), 193))
fill(st.layer('帽兜亮', mask=spray(hood_lit, 3, 194)), ALL, (.83, .83, .87))
hood_dk = M(hood) & ((xx - 596) * .6 + (yy - 892) > 34 + 3 * nz((6, 6), 238))
fill(st.layer('帽兜暗', mask=spray(hood_dk, 2, 239)), ALL, (.58, .60, .69))
# 帽兜左沿：脸翻向左上，开口那一圈侧过来，露出一线里面的阴影（不是头发，是布里头的暗）
rim_dark = L([(567, 906), (569, 891), (577, 879), (588, 872), (584, 880), (576, 892), (572, 906)], jag=.3, seed=195, n=2)
fill(st.layer('帽兜开口'), rim_dark, (.46, .48, .58), edge=HARD)
# 两侧和胳膊外沿转进冷灰
side = (M(robe) & ((xx < 552 + 3 * nz((20, 4), 196)) | (xx > 658 + 3 * nz((20, 4), 197))) & (yy > 946))
fill(st.layer('袍侧', mask=spray(side, 4, 198)), ALL, (.57, .59, .68))
# 胳膊和身子之间那道缝（布叠进去），深一阶
for j, pts in enumerate([[(552, 960), (550, 1000), (547, 1030)], [(655, 958), (660, 998), (664, 1028)]]):
    sl = line(H, W, smooth(pts, 2, closed=False), width=lambda t: 2.4 - t, seed=199 + j)
    fill(st.layer(f'腋缝{j}', mask=spray(sl, 1.2, 230 + j, solid=.85)), ALL, (.46, .48, .58))
# 后颈：帽兜下沿堆着的布，一道淡的亮
fold_hi = line(H, W, smooth([(578, 942), (604, 948), (630, 941)], 2, closed=False), width=2, seed=232)
fill(st.layer('后颈亮', mask=spray(fold_hi, 1.5, 233, solid=.6)), ALL, (.84, .83, .86))
# 竖褶：从肩胛往下，几道，长短不一
for j, pts in enumerate([[(586, 990), (582, 1040), (576, 1096)], [(632, 984), (640, 1040), (648, 1098)], [(610, 1030), (612, 1068), (614, 1102)]]):
    fd = line(H, W, smooth(pts, 2, closed=False), width=lambda t: 1 + 4 * np.sin(np.pi * t), seed=202 + j)
    fill(st.layer(f'褶{j}', mask=spray(fd, 3, 205 + j, solid=.55)), ALL, (.60, .62, .70))
# 腿：袍子底下一截深色的裤腿和靴子，脚埋在草里
for j, (x0, x1) in enumerate([(578, 597), (612, 631)]):
    leg = L([(x0, 1100), (x1, 1100), (x1 + 1, 1128), (x0 - 2, 1130)], jag=.8, seed=210 + j, n=1)
    fill(st.layer(f'腿{j}'), leg, (.20, .20, .24), edge=HARD)
# 右手：白手套，垂着
handR = blob(H, W, 678, 1040, 7, 9, ang=-10, jag=.05, seed=214)
fill(st.layer('右手'), handR, (.64, .66, .73), edge=HARD)
# 左手套着手偶：灰的袖套从手肘一直到手，手偶的头垂下来，侧着，嘴往左下指着地
sleeve = L([(526, 1004), (540, 1004), (544, 1036), (538, 1046), (526, 1044), (522, 1026)], jag=.6, seed=215, n=2)
fill(st.layer('手偶袖'), sleeve, (.52, .52, .57), edge=HARD)
ph = blob(H, W, 533, 1058, 10, 13, ang=14, jag=.03, seed=216)
fill(st.layer('手偶头'), ph, (.83, .83, .86), edge=HARD)
crown = blob(H, W, 537, 1050, 5.5, 4.5, ang=14, jag=.05, seed=217)
fill(st.layer('手偶红顶'), crown, (.76, .20, .18), edge=HARD)
cheek = blob(H, W, 527, 1063, 3.2, 5, ang=20, jag=.05, seed=218)
fill(st.layer('手偶腮'), cheek, (.20, .20, .23), edge=HARD)
eye = blob(H, W, 530, 1056, 1.4, 1.4, jag=0, seed=222)
fill(st.layer('手偶眼'), eye, (.85, .70, .25), edge=HARD)
beak = line(H, W, [(528, 1068), (523, 1088), (519, 1110)], width=lambda t: 6 - 4.6 * t, seed=219)
fill(st.layer('手偶嘴'), beak, (.25, .24, .25), edge=HARD)

# ---------- 6 鹤：五只，往左上（北）飞走；最后一只最近最大，越往前越小
# 认得出它的三件事：飞的时候脖子直直往前伸、腿直直往后拖过尾巴好大一截（颈 ≈ 身长，腿伸出 ≈ 0.7 身长）；
#   翅又长又宽，翼展比身长还长半截，外面一截是黑的初级飞羽，张成几根指头；翅膀每只拍的相位不一样，深而慢。
#   从下后方看：远的那只翅缩短一点。
def crane(cx, cy, span, phase, seed, ang=-146):
    d = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))]); p = np.array([-d[1], d[0]])
    c = np.array([cx, cy], float)
    P = lambda v: tuple(c + v[0] * d * span + v[1] * p * span)
    body = L([P((.13, 0)), P((.07, .042)), P((-.08, .04)), P((-.15, 0)), P((-.08, -.04)), P((.07, -.042))], jag=.3, seed=seed, n=2)
    neck = line(H, W, [P((.12, 0)), P((.28, .004)), P((.40, .010))], width=lambda t: max(1.2, span * (.030 - .012 * t)), seed=seed + 1)
    head = blob(H, W, *P((.41, .010)), max(1.2, span * .022), max(1, span * .017), ang=ang, jag=0, seed=seed + 2)
    bill = line(H, W, [P((.42, .010)), P((.51, .014))], width=lambda t: max(.8, span * (.011 - .007 * t)), seed=seed + 3)
    legs = line(H, W, [P((-.12, .012)), P((-.44, .028))], width=max(1, span * .011), seed=seed + 4) | \
           line(H, W, [P((-.12, -.012)), P((-.44, .004))], width=max(1, span * .011), seed=seed + 5)
    # 翅：phase 1 往下压到底（展开、尖略往后），-1 举到顶（投影短、尖往前）
    reach0 = .54 * (.70 + .30 * phase); sweep = -.07 * phase + .01
    wings, tips = [], []
    for s, fs in ((1, 1.0), (-1, .80)):            # s=-1 那只是远的，缩短
        R = reach0 * fs
        outline = [P((.07, .03 * s)), P((.10 + sweep * .5, R * .40 * s)), P((.06 + sweep, R * .70 * s)), P((.02 + sweep, R * .86 * s)),
                   P((-.05 + sweep, R * .84 * s)), P((-.08 + sweep * .5, R * .70 * s)), P((-.10, R * .40 * s)), P((-.09, .03 * s))]
        w_ = L(outline, jag=.3, seed=seed + 6 + (s > 0), n=2)
        u = ((xx - c[0]) * p[0] + (yy - c[1]) * p[1]) * s / span      # 沿翅的位置（0 在身子）
        tip = M(w_) & (u > R * .64)
        # 初级飞羽：翅尖张开四五根指头，一根比一根短，往后散
        fing = np.zeros((H, W), bool)
        for f in range(5):
            a0 = P((.035 + sweep - .026 * f, R * .72 * s)); a1 = P((.045 + sweep - .042 * f, R * (1.03 - .045 * f) * s))
            fing |= M(line(H, W, [a0, a1], width=lambda t: max(1.0, span * (.024 - .013 * t)), seed=seed + 20 + f))
        tip = tip | fing
        wings.append(w_); tips.append(tip)
    return body, neck | head | bill, legs, wings, tips
LIT, LIT2, BLACK, LEG = (1.0, .87, .71), (.93, .77, .64), (.16, .15, .19), (.28, .24, .25)
SEC = (.86, .72, .64)   # 次级飞羽那一线，翅后缘稍暗一阶
CR = [(128, 248, 44, -.5, 300, -150), (202, 288, 52, .7, 320, -144), (300, 382, 66, -1, 340, -147), (356, 454, 84, .2, 360, -142), (480, 562, 120, .95, 380, -146)]
for k, (x, y, s, ph_, sd, an) in enumerate(CR):
    body, nk, legs, wings, tips = crane(x, y, s, ph_, sd, an)
    dd0 = np.array([np.cos(np.deg2rad(an)), np.sin(np.deg2rad(an))])
    lay = st.layer(f'鹤{k}')
    fill(lay, legs, LEG, edge=HARD)
    fill(lay, wings[1], LIT2, edge=HARD)       # 远的那只翅略暗
    fill(lay, wings[0], LIT, edge=HARD)
    fill(lay, body | nk, LIT, edge=HARD)
    for w_ in wings:
        te = M(w_) & ~np.roll(np.roll(M(w_), int(round(-dd0[1] * s * .05)), 0), int(round(-dd0[0] * s * .05)), 1)
        fill(lay, Shape(te), SEC, edge=HARD)
    fill(lay, Shape(tips[0] | tips[1]), BLACK, edge=HARD)
    if s > 50:   # 近的两只才看得见红顶
        dd = np.array([np.cos(np.deg2rad(an)), np.sin(np.deg2rad(an))])
        fill(lay, blob(H, W, *(np.array([x, y]) + dd * s * .40), 1.6, 1.3, jag=0, seed=sd + 9), (.80, .22, .2), edge=HARD)

# ---------- 7 签名：右下角，一个点一个点敲
SIG = [(808, 1176), (803, 1171), (797, 1171), (793, 1176), (794, 1182), (800, 1185), (807, 1183),   # C
       None, (812, 1166), (812, 1185),                                                                # l
       None, (826, 1176), (821, 1173), (817, 1177), (818, 1183), (823, 1185), (827, 1180), (827, 1173), (828, 1185),  # a
       None, (833, 1173), (833, 1182), (836, 1185), (841, 1184), (843, 1179), (843, 1173), (843, 1185),  # u
       None, (857, 1166), (857, 1185), (857, 1178), (853, 1173), (848, 1176), (848, 1183), (853, 1185), (857, 1181),  # d
       None, (862, 1179), (871, 1178), (869, 1173), (864, 1173), (861, 1178), (862, 1183), (867, 1185), (871, 1183)]  # e
segs, cur = [], []
for pt in SIG + [None]:
    if pt is None:
        if len(cur) > 1: segs.append(cur)
        cur = []
    else: cur.append(pt)
sig = Shape(np.zeros((H, W), bool))
for sgm in segs:
    sig = sig | line(H, W, sgm, width=1.4, seed=400)
for sgm in segs:
    for (px, py) in sgm[1:-1]:
        sig = sig | blob(H, W, px, py, 1.3, 1.3, jag=0, seed=401)
fill(st.layer('签名'), sig, (.62, .60, .63), edge=HARD)

# ---------- 纸
fill(st.layer('纸', mode='multiply'), ALL, paper(H, W), edge=HARD)
look(st, OUT)
print('ok', OUT)
