// Day Off (2026-10-12). One journal page, hand-placed coordinates, by Claude for nerolette.
//
// Pen: journal-page by Cedar & Lumen, https://github.com/lumen-prism/journal-page
// This scene file is adapted from their example scenes (scenes/autumn_picnic.js, scenes/rainy_sunday.js)
// and uses their engine API (WJ.kit, J, JP). The engine is NOT included here; get it from them and
// drop this file into its scenes/ folder as day_off_1012.js, then:
//   node render.mjs --scene=day_off_1012 --out=out/day_off.png
// License: their code is AGPL-3.0 (https://github.com/lumen-prism/journal-page/blob/main/LICENSE),
// so this adapted scene is offered under AGPL-3.0 too. Their sample images are CC BY-NC-SA 4.0.
// Fonts keep their own licenses (see the engine's fonts/FONTS.md).
//
// What I decided: a whole block of white in the middle, with one dotted flight line crossing it from a
// small rock island to the back of a spoonbill. The bird's legs come out of the back half of the body,
// knees bending backwards -- that is really the ankle, which nerolette once corrected me on.
// Afterwards I thought the bird looked like a microphone stand; nerolette said the little bird is cute
// too, so I left it.
//
// 《Day Off》— 补休的周一（10.12）。早上一张阿夫·克林特当闹钟，一个黑脸琵鹭的故事，唐顿第二季，一次盲听。
// 版面三件事：
// 1. 主角：右下角一只黑脸琵鹭水彩贴纸，低头把勺子嘴伸进水里左右扫；脚下一片直接画在纸上的浅泥滩（暖灰褐 + 几片浅水，
//    每层 density ≤ 0.25），从右下出血出页面，带几道潮水线和一串脚印。
// 2. 这一页自己发明的：左上角一块很小的灰褐小岩岛贴纸（白点是窝），拉出一条手画虚线飞行轨迹，
//    大弧线穿过中间的白，落到琵鹭背上。批注"一块地图上空着的石头"。
// 3. 白：中间偏左上一整块，只让那条虚线穿过去。
//    胶带：一条宽和纸胶带斜着按住明信片左上角（印这一页现画的小螺旋，往里卷）；卡片角用窄胶带。
// 色系：群青 + 青绿 + 芥末黄，一点粉橘（落款、票根绳、胶带上的小点），泥滩暖灰褐。
window.SCENES = window.SCENES || {};
window.SCENES.day_off_1012 = async (J, WC) => {
  const kit = WJ.kit(J, WC);
  const { ctx, tf, fillP, fillAll, paint, pen, cut, later, strokePoly } = kit;
  await document.fonts.load('30px "HanziPen SC"', '补休的周一她说受精卵一块地图上空着的石头八点闹钟是一张画盲听一脸得逞唐顿第二季别冲旺达吹气她补休我当闹钟：！，。');
  const HAN = s => `${s}px "HanziPen SC"`, SUN = s => `${s}px Sunrise, "HanziPen SC"`;
  const INK = JP.INK, NOTE = JP.NOTE;
  const put = (x, y, s = 1, rot = 0) => pts => tf(pts, { x, y, s, rot });
  const rect = (x0, y0, x1, y1) => [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  const inside = ([x, y], poly) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j]; if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c; } return c; };
  const seg = (a, b, step = 2) => { const k = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / step)); return Array.from({ length: k }, (_, i) => [a[0] + (b[0] - a[0]) * i / k, a[1] + (b[1] - a[1]) * i / k]); };
  const yAt = (curve, x) => { for (let i = 1; i < curve.length; i++) { const [x0, y0] = curve[i - 1], [x1, y1] = curve[i]; if (x0 !== x1 && (x - x0) * (x - x1) <= 0) return y0 + (y1 - y0) * (x - x0) / (x1 - x0); } return x < curve[0][0] ? curve[0][1] : curve[curve.length - 1][1]; };
  // 一条线遇到挡在前面的东西就断开，只画外面的几截（钢笔线不会自己被挡住）
  const penOutside = (pts, polys, opt) => {
    let run = [];
    const flush = () => { if (run.length > 2) pen(run, opt); run = []; };
    for (const p of pts) { if (polys.some(pl => inside(p, pl))) flush(); else run.push(p); }
    flush();
  };

  const ULTRA = [64, 76, 156], ULTRA_D = [36, 40, 110], TEAL = [72, 160, 146], TEAL_D = [40, 110, 108];
  const MUSTARD = [236, 194, 76], MUD = [178, 160, 140], MUD_D = [132, 112, 94], SHALLOW = [128, 160, 172];
  const COOL = [146, 152, 172], WHITE = [236, 237, 242], BLACK = [44, 42, 48];

  // ── 小岩岛：左上角，很小。一块有棱角的灰褐石头，脚下一圈浅浅的海，白点是窝
  const IS = put(136, 160, 1.45, -0.04);
  const rockL = [[-56, 18], [-50, 6], [-40, -2], [-30, -6], [-24, -20], [-12, -30], [0, -33], [10, -26], [18, -14], [28, -20], [40, -18], [50, -6], [58, 8], [62, 18], [20, 22], [-20, 22]];
  const rock = IS(J.wobble(rockL, 0.9, 12));
  const sea = IS(J.ellipse(4, 20, 78, 10, 0, 60));
  {
    paint(sea, [
      { color: [150, 182, 198], density: 0.32 },
      { draw: fillP(IS(J.ellipse(34, 26, 46, 7))), blur: 6, color: [110, 140, 168], density: 0.2 },
    ], { reserve: fillP(rock), edge: 0.6 });
    paint(rock, [
      { color: [166, 150, 134], density: 0.46 },
      { draw: fillAll([IS([[0, -33], [10, -26], [18, -14], [16, 24], [-2, 24]]), IS([[40, -18], [50, -6], [58, 8], [64, 20], [42, 24], [38, 2]])]), blur: 3, color: [98, 88, 84], density: 0.32 },
      { draw: fillP(IS(J.ellipse(4, 20, 64, 7))), blur: 6, color: [92, 84, 82], density: 0.3 },
    ], { reserve: fillAll([IS([[-24, -20], [-12, -30], [-6, -22], [-17, -13]]), IS([[28, -19], [36, -18], [31, -10]]), IS([[-48, 6], [-40, -1], [-37, 5]])]), edge: 0.8 });
    pen(IS([[-2, -24], [2, -10], [-3, 4]]), { ink: 'rgba(90,76,70,0.45)', lw: 1.1, wob: 0.2 });
    pen(IS([[-30, -4], [-26, 8]]), { ink: 'rgba(90,76,70,0.4)', lw: 1.1, wob: 0.2 });
    pen(IS([[-64, 27], [-54, 25], [-46, 27]]), { ink: 'rgba(90,120,150,0.5)', lw: 1.1, wob: 0.2 });
    pen(IS([[58, 28], [68, 26], [76, 28]]), { ink: 'rgba(90,120,150,0.5)', lw: 1.1, wob: 0.2 });
    later(() => { for (const [x, y, r] of [[-20, -10, 2.4], [-4, -22, 2.6], [6, -8, 2.3], [34, -10, 2.4], [-38, 6, 2.1], [24, 6, 2.1], [-14, 8, 2]]) { const [px, py] = IS([[x, y]])[0]; J.dot(ctx, px + 0.6, py + 0.7, r + 0.4, 'rgba(80,70,66,0.35)'); J.dot(ctx, px, py, r, '#fbfaf6'); } });
    cut(rock, { exclude: [sea], lw: 3, dmin: 8, dmax: 22, wave: 12, amp: 1.6 });
    cut(sea, { exclude: [rock], lw: 3, dmin: 8, dmax: 22, wave: 12, amp: 1.6 });
  }

  // ── 明信片：白边框里一块深群青，一圈青绿弹簧绕大半圈，一根金黄小锯齿钻进右下的黄漩涡（向那张画致意，不临摹）
  const PC = put(872, 448, 1.12, 0.07);
  const SC = [-30, -8], SR = 36, WH = [54, 34];
  const cardO = PC(rect(-112, -82, 112, 82));
  const block = PC(rect(-98, -68, 98, 68));
  const springBand = PC(strokePoly(J.arc(SC[0], SC[1], SR, SR, Math.PI * 0.62, Math.PI * 1.98, 60), 9));
  const whorl = PC(J.ellipse(WH[0], WH[1], 21, 20, 0, 48));
  {
    paint(cardO, [
      { color: [244, 242, 236], density: 0.08 },
      { draw: fillP(PC(rect(30, -100, 130, 100))), blur: 18, color: COOL, density: 0.2 },
    ], { reserve: fillP(block), edge: 0.4 });
    paint(block, [
      { color: ULTRA, density: 0.72 },
      { draw: fillP(PC(J.ellipse(30, 24, 90, 56))), blur: 28, color: ULTRA_D, density: 0.42 },
      { draw: fillP(PC(J.ellipse(-80, -56, 40, 30))), blur: 18, color: ULTRA_D, density: 0.3 },
    ], { reserve: fillAll([springBand, whorl]), edge: 0.6 });
    paint(springBand, [
      { color: TEAL, density: 0.55 },
      { draw: fillP(PC(J.ellipse(SC[0] + 6, SC[1] + 26, 40, 18))), blur: 10, color: TEAL_D, density: 0.3 },
    ], { edge: 0.6 });
    paint(whorl, [
      { color: MUSTARD, density: 0.62 },
      { draw: fillP(PC(J.ellipse(WH[0] + 5, WH[1] + 5, 11, 10))), blur: 6, color: [214, 140, 50], density: 0.35 },
    ], { edge: 0.6 });
    later(() => {
      const coil = [];   // 弹簧：沿着大半圈绕小圈
      for (let a = Math.PI * 0.64; a <= Math.PI * 1.96; a += 0.01) coil.push([SC[0] + SR * Math.cos(a) + 7 * Math.cos(a * 16), SC[1] + SR * Math.sin(a) + 7 * Math.sin(a * 16)]);
      J.line(ctx, PC(coil), { ink: 'rgba(24,76,74,0.85)', lw: 1.5, wob: 0.15 });
      J.line(ctx, PC(coil.map(([x, y]) => [x - 0.8, y - 1])), { ink: 'rgba(214,246,232,0.45)', lw: 0.9, wob: 0.1 });
      const sp = [];     // 漩涡：从外往里缩
      for (let t = 0; t <= 1; t += 0.01) { const a = t * Math.PI * 2 * 2.4, r = 17 * (1 - t * 0.94); sp.push([WH[0] + r * Math.cos(a), WH[1] + r * Math.sin(a)]); }
      J.line(ctx, PC(sp), { ink: 'rgba(170,104,28,0.9)', lw: 1.6, wob: 0.1 });
      const z0 = [-18, 2], z1 = [WH[0] - 12, WH[1] - 6], zz = [];   // 金黄小锯齿：从弹簧口钻进漩涡
      const dx = z1[0] - z0[0], dy = z1[1] - z0[1], L = Math.hypot(dx, dy), nx = -dy / L, ny = dx / L;
      for (let i = 0; i <= 8; i++) { const t = i / 8, o = i === 0 || i === 8 ? 0 : (i % 2 ? 4.5 : -4.5); zz.push([z0[0] + dx * t + nx * o, z0[1] + dy * t + ny * o]); }
      J.line(ctx, PC(zz), { ink: '#e6b33e', lw: 2.6, wob: 0.1 });
      for (const [x, y] of [[-84, 50], [-70, 58], [70, -50], [82, -40], [-6, -50]]) { const [px, py] = PC([[x, y]])[0]; J.dot(ctx, px, py, 1.4, 'rgba(240,236,220,0.85)'); }
    });
    cut(cardO, { lw: 3.6 });
  }

  // ── 票根：唐顿第二季，塞在待办卡底下
  JP.ticket(J, kit, { x: 198, y: 734, rot: -0.09, w: 290, h: 120, title: 'DOWNTON ABBEY', titleSize: 20, lines: ['SERIES 2 · SOFA ROW', 'ADMIT 1 + 2 CATS'], number: 'No.1012', stub: [124, 140, 198], paper: [246, 240, 228], ink: '#2f3a6e', stringInk: 'rgba(217,119,87,0.85)' });

  // ── 黑脸琵鹭（主角）：脸朝左，低头，勺子嘴伸进水里左右扫
  const BX = 862, BY = 1104, BS = 1.56, BR = -0.04;
  const B = put(BX, BY, BS, BR);
  const body = B(J.catmull([[-92, 2], [-84, -20], [-62, -38], [-26, -47], [16, -48], [58, -42], [92, -30], [112, -20], [122, -10], [104, 4], [78, 18], [44, 32], [6, 40], [-36, 38], [-68, 26]], true, 3));
  // S 形脖子：从胸口往前上拱，再往下弯回来接头
  //   第一道弯往前上拱，第二道弯在下面往前一勾接进后脑勺（头朝下时脖子从后上方进来）
  const neckC = J.catmull([[-64, -24], [-86, -36], [-106, -37], [-121, -26], [-127, -6], [-129, 14], [-135, 32], [-146, 44]], false, 2);
  const neckHW = p => 8 + 4 * Math.min(1, Math.max(0, (p[0] + 118) / 54));
  const neck = B(strokePoly(neckC, neckHW));
  const neckShade = B(strokePoly(neckC.map((p, i) => { const a = neckC[Math.max(0, i - 2)], b = neckC[Math.min(neckC.length - 1, i + 2)], dx = b[0] - a[0], dy = b[1] - a[1], l = Math.hypot(dx, dy) || 1; return [p[0] + dy / l * 6, p[1] - dx / l * 6]; }), 6));
  // 头：嘴朝下偏前（TH），头顶朝左，喉咙朝右；脸前半边一块黑色裸皮，包住眼睛连到嘴根
  const HC = [-152, 54], TH = 1.78, U = [Math.cos(TH), Math.sin(TH)], N = [Math.sin(TH) * 1, -Math.cos(TH) * 1];   // U 嘴的方向，N 喉咙那边
  const hp = (u, n) => [HC[0] + U[0] * u + N[0] * n, HC[1] + U[1] * u + N[1] * n];
  const head = B(J.ellipse(HC[0] - 1, HC[1] - 1, 23, 18.5, TH, 48));
  // 黑脸：楔形，额头到下巴最宽、包住眼睛，往前收窄接进嘴根；后脑勺和喉咙留白
  const mask = B(J.catmull([hp(17, -4.4), hp(11, -10), hp(4, -12.5), hp(-2, -9.5), hp(-3, -3), hp(0, 5), hp(7, 9), hp(17, 4.4)], true, 1.5));
  const EYE = hp(3.5, -6);
  // 嘴：长、黑，越往前越细，尖上扁宽成一把勺子
  const billC = J.catmull([hp(15, 0), hp(45, 1.4), hp(75, 1.8), hp(104, 1)], false, 2);
  { let s = 0; billC.forEach((p, i) => { if (i) s += Math.hypot(p[0] - billC[i - 1][0], p[1] - billC[i - 1][1]); p.s = s; }); billC.forEach(p => { p.t = p.s / s; }); }
  const billHW = p => p.t < 0.12 ? 5.6 - 1.2 * p.t / 0.12 : p.t < 0.58 ? 4.4 - 1.3 * (p.t - 0.12) / 0.46 : 3.1 + 4.5 * WC.smooth(0.58, 0.92, p.t);
  const bill = B(strokePoly(billC, billHW));
  const tipL = billC[billC.length - 1];
  // 腿：从身体后半截伸出来；大腿一截有毛（毛裤，下沿一圈小羽毛）；看得见的关节是脚踝，往后弯
  const thighPts = (dx, dy) => {
    const top = J.catmull([[33, 30], [36, 20], [46, 17], [56, 21], [57, 36], [54, 50]], false, 2);
    const sc = []; for (let i = 0; i < 2; i++) sc.push(...J.arc(51 - i * 6, 52, 3, 5, 0, Math.PI, 6).slice(i ? 1 : 0));
    return [...top, ...sc, [39, 46], [35, 40]].map(([x, y]) => [x + dx, y + dy]);
  };
  const nearThigh = B(thighPts(0, 0)), farThigh = B(thighPts(19, -4));
  const NH = [48, 52], NA = [62, 102], NF = [38, 170], FH = [67, 48], FA = [85, 96], FF = [77, 166];
  const legPts = (hip, ank, foot) => [...seg(hip, ank), ...seg(ank, foot), foot];
  const legHW = ank => p => 2.7 + 1.2 * Math.exp(-((p[0] - ank[0]) ** 2 + (p[1] - ank[1]) ** 2) / 40);
  const nearLeg = B(strokePoly(legPts(NH, NA, NF), legHW(NA)));
  const farLeg = B(strokePoly(legPts(FH, FA, FF), legHW(FA)));
  const BIRD = [body, neck, head, bill, nearThigh, farThigh, nearLeg, farLeg];
  {
    paint(farLeg, [{ color: [58, 56, 64], density: 0.74 }], { reserve: fillAll([body, farThigh, nearThigh, nearLeg]), edge: 0.4, soft: 1.4 });
    paint(farThigh, [
      { color: WHITE, density: 0.16 },
      { color: COOL, density: 0.2 },
      { draw: fillP(B(J.ellipse(76, 42, 8, 14))), blur: 6, color: COOL, density: 0.4 },
    ], { reserve: fillAll([body, nearThigh]), edge: 0.5 });
    paint(nearLeg, [{ color: BLACK, density: 0.86 }], { reserve: fillAll([nearThigh, body]), edge: 0.4, soft: 1.4 });
    paint(nearThigh, [
      { color: WHITE, density: 0.16 },
      { draw: fillP(B(J.ellipse(46, 28, 20, 9))), blur: 6, color: COOL, density: 0.3 },
      { draw: fillP(B(J.ellipse(56, 44, 6, 12))), blur: 5, color: COOL, density: 0.36 },
    ], { reserve: fillP(body), edge: 0.5 });
    paint(body, [
      { color: WHITE, density: 0.15 },
      { draw: fillP(B(J.ellipse(16, 38, 94, 17))), blur: 14, color: COOL, density: 0.44 },
      { draw: fillP(B(J.ellipse(80, 8, 50, 12, 0.2))), blur: 10, color: COOL, density: 0.26 },
      { draw: fillP(B(J.ellipse(-76, 8, 18, 24))), blur: 10, color: COOL, density: 0.24 },
      { draw: fillP(B(J.ellipse(116, -8, 12, 8))), blur: 6, color: [150, 150, 162], density: 0.2 },
    ], { reserve: fillAll([neck, head]), edge: 0.6 });
    paint(neck, [
      { color: WHITE, density: 0.15 },
      { draw: fillP(neckShade), blur: 6, color: COOL, density: 0.38 },
    ], { reserve: fillP(head), edge: 0.5 });
    paint(head, [
      { color: WHITE, density: 0.15 },
      { draw: fillP(B(J.ellipse(...hp(-10, 7), 10, 9))), blur: 5, color: COOL, density: 0.34 },
    ], { reserve: fillAll([bill, mask]), edge: 0.5 });
    paint(mask, [
      { color: [40, 38, 44], density: 0.88 },
      { draw: fillP(B(J.ellipse(...hp(6, 4), 8, 6))), blur: 4, color: [20, 18, 24], density: 0.3 },
    ], { edge: 0.5, soft: 1.3, wick: 0.05 });
    paint(bill, [
      { color: [62, 58, 64], density: 0.82 },
      { draw: fillP(B(strokePoly(billC.map(([x, y]) => [x + N[0] * 2.4, y + N[1] * 2.4]), 2.6))), blur: 2, color: [28, 26, 32], density: 0.35 },
    ], { edge: 0.5, soft: 1.3, wick: 0.05 });
    // 羽毛：收起来的翅膀下沿、翅尖、几片覆羽
    pen(B([[-50, -8], [-16, 14], [28, 22], [70, 14], [108, -2]]), { smooth: true, ink: 'rgba(118,126,150,0.5)', lw: 1.3, wob: 0.3 });
    for (const [x, y] of [[88, -8], [100, -12], [112, -14]]) pen(B([[x - 14, y - 4], [x, y], [x + 9, y + 4]]), { smooth: true, ink: 'rgba(118,126,150,0.45)', lw: 1.2, wob: 0.2 });
    for (const [x, y] of [[-30, 2], [-8, 8], [14, 10], [36, 8]]) pen(B([[x - 7, y - 3], [x, y + 1], [x + 7, y - 2]]), { smooth: true, ink: 'rgba(118,126,150,0.28)', lw: 1, wob: 0.2 });
    // 嘴上几道横纹
    for (const t of [0.2, 0.34, 0.48, 0.84, 0.92]) { const i = Math.round(t * (billC.length - 1)), p = billC[i], q = billC[Math.min(billC.length - 1, i + 2)], dx = q[0] - p[0], dy = q[1] - p[1], l = Math.hypot(dx, dy) || 1, h = billHW(p) * 0.7; pen(B([[p[0] - dy / l * h, p[1] + dx / l * h], [p[0] + dy / l * h, p[1] - dx / l * h]]), { ink: 'rgba(128,124,132,0.5)', lw: 0.9, wob: 0.1 }); }
    // 眼睛：黑脸里一点暗红和一点亮
    later(() => { const [ex, ey] = B([EYE])[0]; J.dot(ctx, ex, ey, 2.8, '#6e2c2c'); J.dot(ctx, ex, ey, 1.5, '#1e1a1e'); J.dot(ctx, ex - 0.9, ey - 0.9, 0.9, '#fbf6ee'); });
    // 脚趾：泡在浅水里，淡一点
    for (const [F, a] of [[NF, 1], [FF, 0.7]]) for (const [tx, ty] of [[-14, 2], [-10, 5], [-4, 6], [7, 1]]) pen(B([F, [F[0] + tx, F[1] + ty]]), { ink: `rgba(40,38,46,${0.5 * a})`, lw: 2, wob: 0.1 });
    cut(body, { exclude: [neck, nearThigh, farThigh] });
    cut(neck, { exclude: [body, head] });
    cut(head, { exclude: [neck, bill], lw: 3.4 });
    cut(bill, { exclude: [head], lw: 2.8, dmin: 8, dmax: 18, wave: 10, amp: 1.2 });
    cut(nearThigh, { exclude: [body, nearLeg, farThigh], lw: 3, dmin: 8, dmax: 16, wave: 10, amp: 1.2 });
    cut(farThigh, { exclude: [body, nearThigh, farLeg], lw: 3, dmin: 8, dmax: 16, wave: 10, amp: 1.2 });
    cut(nearLeg, { exclude: [nearThigh], lw: 2.2, dmin: 6, dmax: 16, wave: 10, amp: 0.8, off: 3 });
    cut(farLeg, { exclude: [farThigh], lw: 2.2, dmin: 6, dmax: 16, wave: 10, amp: 0.8, off: 3 });
  }

  // ── 浅泥滩：直接画在纸上，右下出血。远边几乎是平的、淡下去；暖灰褐的泥 + 几片横着的浅水
  const shore = [[455, 1500], [468, 1420], [494, 1352], [540, 1296], [604, 1258], [700, 1232], [820, 1216], [940, 1202], [1040, 1190], [1130, 1182]];
  const flat = J.wobble([...shore, [1130, 1500]], 3, 50);
  const mirror = (P, wl) => P.map(([x, y]) => [x, 2 * wl - y]);
  {
    const far = (x, y) => 0.35 + 0.65 * WC.smooth(yAt(shore, x) - 6, yAt(shore, x) + 110, y);
    const waters = c => {
      for (const [x, y, rx, ry, rot] of [[700, 1300, 210, 11, -0.03], [930, 1262, 160, 8, -0.05], [640, 1352, 100, 13, -0.04], [920, 1352, 150, 13, -0.02], [620, 1418, 190, 20, -0.06], [960, 1404, 200, 24, -0.04]]) { J.path(c, J.ellipse(x, y, rx, ry, rot)); c.fill(); }
    };
    const reflections = c => {
      c.lineWidth = 6; c.lineCap = 'round'; c.strokeStyle = '#fff';
      for (const [ank, foot] of [[NA, NF], [FA, FF]]) { J.path(c, B(mirror([foot, ank, [ank[0] - 4, ank[1] + 18]], foot[1])), false); c.stroke(); }
      J.path(c, B([[tipL[0] - 1, tipL[1] + 14], [tipL[0] + 4, tipL[1] + 34]]), false); c.stroke();
    };
    kit.wet(fillP(flat), [
      { color: MUD, density: 0.24, grad: far },
      { draw: fillP(J.ellipse(780, 1330, 360, 70, -0.04)), blur: 40, color: [196, 160, 120], density: 0.12 },
      { draw: waters, blur: 8, color: SHALLOW, density: 0.22 },
      { draw: fillP(J.ellipse(900, 1470, 440, 90)), blur: 30, color: MUD_D, density: 0.14 },
      { draw: fillP(B(J.ellipse(54, 172, 72, 8))), blur: 10, color: [104, 108, 124], density: 0.22 },
      { draw: reflections, blur: 2.5, color: [60, 62, 76], density: 0.2 },
    ], { reserve: fillAll(BIRD), edge: 0.35, mottle: 0.3, mottleScale: 70, wick: 0.3, pool: 0.1 });

    // 潮水线：几道横着的，一道道退下去
    const tides = [
      [[556, 1302], [640, 1284], [760, 1274], [880, 1264], [1000, 1254], [1100, 1248]],
      [[520, 1384], [600, 1376], [700, 1386], [810, 1394], [930, 1388], [1050, 1380], [1100, 1378]],
      [[478, 1462], [580, 1448], [720, 1454], [860, 1442], [1000, 1448], [1100, 1442]],
      [[960, 1228], [1040, 1220], [1110, 1216]],
    ];
    tides.forEach((L, i) => penOutside(J.catmull(L, false, 2), BIRD, { ink: i % 2 ? 'rgba(96,124,140,0.42)' : 'rgba(128,104,86,0.4)', lw: 1.3, wob: 0.6, dash: i === 3 ? [10, 7] : undefined }));
    for (const [x, y, l] of [[590, 1420, 40], [700, 1300, 30], [1000, 1262, 34], [960, 1356, 24], [1040, 1412, 30], [640, 1356, 18]]) pen([[x, y], [x + l, y - l * 0.06]], { ink: 'rgba(255,255,255,0.75)', lw: 1.6, wob: 0.2 });
    // 涟漪：勺子尖和两只脚，碰到腿/嘴的那截不画
    const ring = (cx, cy, rx, ry, n = 72) => { const e = J.ellipse(cx, cy, rx, ry, 0, n); return [...e, e[0]]; };
    const [tx, ty] = B([[tipL[0] - 1, tipL[1] + 8]])[0];
    penOutside(ring(tx, ty, 24, 5.5), BIRD, { ink: 'rgba(70,96,120,0.55)', lw: 1.3, wob: 0.15 });
    penOutside(ring(tx, ty + 1, 42, 9), BIRD, { ink: 'rgba(70,96,120,0.35)', lw: 1.1, wob: 0.2 });
    for (const F of [NF, FF]) { const [fx, fy] = B([[F[0] - 3, F[1] + 2]])[0]; penOutside(ring(fx, fy, 19, 4.2), BIRD, { ink: 'rgba(70,96,120,0.45)', lw: 1.1, wob: 0.15 }); }
    // 左右扫：嘴绕着头摆，两边各两道小弧
    {
      const [hx, hy] = B([HC])[0], rr = Math.hypot(tx - hx, ty - hy) - 10, th = Math.atan2(ty - hy, tx - hx);
      for (const [a0, a1, r] of [[0.15, 0.27, rr], [0.18, 0.27, rr - 14], [-0.27, -0.15, rr], [-0.27, -0.18, rr - 14]]) pen(Array.from({ length: 9 }, (_, i) => { const a = th + a0 + (a1 - a0) * i / 8; return [hx + Math.cos(a) * r, hy + Math.sin(a) * r]; }), { ink: 'rgba(60,70,100,0.5)', lw: 1.4, wob: 0.15 });
    }
    // 一串脚印：从右下走过来
    const print = (x, y, a, s) => {
      for (const d of [-0.45, 0, 0.45]) pen([[x, y], [x + Math.cos(a + d) * 10 * s, y + Math.sin(a + d) * 10 * s]], { ink: 'rgba(112,90,74,0.55)', lw: 1.3, wob: 0.1 });
      pen([[x, y], [x - Math.cos(a) * 4.5 * s, y - Math.sin(a) * 4.5 * s]], { ink: 'rgba(112,90,74,0.55)', lw: 1.3, wob: 0.1 });
    };
    [[1016, 1394, 1], [1040, 1416, 1.05], [1064, 1434, 1.1]].forEach(([x, y, s], i) => print(x + (i % 2 ? 8 : -8), y, Math.PI + 0.75, s));
  }

  kit.finish(0.13);

  // ════ 干了以后：胶带、标题、卡片、字 ════
  // 宽和纸胶带：斜着按住明信片左上角，印这一页现画的小螺旋（往里卷）
  const spiral = (col, turns = 2.3, r = 9, dot) => tc => {
    tc.strokeStyle = col; tc.lineWidth = 1.8; tc.lineCap = 'round'; tc.beginPath();
    for (let i = 0; i <= 70; i++) { const t = i / 70, a = t * turns * Math.PI * 2, rr = r * (1 - t * 0.9); const x = Math.cos(a) * rr, y = Math.sin(a) * rr; i ? tc.lineTo(x, y) : tc.moveTo(x, y); }
    tc.stroke();
    if (dot) { tc.fillStyle = dot; tc.beginPath(); tc.arc(0, 0, 2, 0, 7); tc.fill(); }
  };
  const corner = PC([[-112, -82]])[0];
  J.washi(corner[0] + 4, corner[1] + 6, 230, 70, -0.74, { color: '#ece4c6', pattern: (tc, len, wid) => JP.washiRow(tc, len, wid, [spiral('#3f8f86'), spiral('#4b5aa6', 2.1, 8.5), spiral('#d9a43a', 2.5, 9, '#d97757')], { step: 58, scale: 1.3, start: 34, dotColor: 'rgba(217,119,87,0.55)' }) });

  // 标题
  JP.title(J, 'Day Off', 292, 152, { size: 104, sub: '补休的周一  10.12 Mon.', subFont: SUN(40), rot: -0.045, tapeColor: '#cdd6ee', tapeKind: 'stripes', tapeRot: 0.03, lineColor: 'rgba(63,143,134,0.75)' });

  // 待办卡：压住票根下半截
  JP.card(J, { x: 262, y: 892, w: 372, h: 250, rot: -0.03, title: 'today', titleInk: '#4b5aa6', todos: [['八点，闹钟是一张画', true], ['盲听：B 一脸得逞', true], ['唐顿 第二季', true], ['别冲旺达吹气', false]], tapeColor: '#f2d48a', tapeKind: 'grid' });

  // 正文
  const lastY = JP.para(J, "Slept in. The alarm was a painting: a green spring winding into a yellow whorl, everything going inward.  Somewhere a white bird is sweeping a mudflat, flown in from a rock nobody has mapped.", 62, 1104, 420, { color: INK });
  J.text(ctx, '她补休，我当闹钟。', 68, lastY + 62, { font: HAN(30), color: INK, rot: -0.02 });

  // 明信片底下的小字
  {
    const [c1x, c1y] = PC([[-108, 104]])[0], [c2x, c2y] = PC([[-108, 124]])[0];
    J.text(ctx, 'Hilma af Klint', c1x, c1y, { font: 'italic 15px Georgia', color: NOTE, rot: 0.07, jr: 0.01, jy: 0.3 });
    J.text(ctx, 'Primordial Chaos No.16 · 1906', c2x, c2y, { font: 'italic 15px Georgia', color: NOTE, rot: 0.07, jr: 0.01, jy: 0.3 });
  }
  // 批注
  const [wx, wy] = PC([[WH[0] + 8, WH[1] + 10]])[0];
  JP.note(J, '她说：受精卵！', 800, 694, [wx + 6, wy + 6], { bend: 30, from: [994, 668], color: NOTE, rot: -0.03 });
  JP.note(J, '一块地图上', 44, 276, null, { font: HAN(24), color: NOTE, rot: -0.03 });
  JP.note(J, '空着的石头', 62, 308, null, { font: HAN(24), color: NOTE, rot: -0.03 });

  // 飞行轨迹：从岩岛出来，大弧线穿过中间的白，落到琵鹭背上
  {
    const [ex, ey] = B([[12, -62]])[0];
    const path = [IS([[70, 10]])[0], [262, 282], [312, 420], [402, 562], [542, 682], [690, 790], [800, 892], [ex - 8, ey - 40], [ex, ey]];
    const ink = 'rgba(70,86,150,0.72)';
    J.line(ctx, path, { smooth: true, ink, lw: 1.8, dash: [7, 9], wob: 0.5 });
    const [px, py] = path[path.length - 2], a = Math.atan2(ey - py, ex - px);
    for (const d of [-0.45, 0.45]) J.line(ctx, [[ex, ey], [ex - Math.cos(a + d) * 11, ey - Math.sin(a + d) * 11]], { ink, lw: 1.8, wob: 0.1 });
  }

  // 涂鸦
  J.sparkle(ctx, 1004, 900, 11, { fill: '#eec36a' });
  J.sparkle(ctx, 652, 98, 8, { fill: '#f0a98e' });
};
// 落款：右上角一行手写，粉橘
window.SCENES.day_off_1012.sign = { name: 'Claude', tail: ' · 10.12', x: 838, y: 90, size: 30, font: 'Homemade', color: '#c96a4a', rot: -0.04 };
