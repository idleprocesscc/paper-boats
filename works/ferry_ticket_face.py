# Part of ferry-ticket-1101.py (Boat Ticket): the printed face of the ticket, laid out flat.
# 票面：先在票自己的平面上排好（平的，正着），再贴到台面上去。两色印：藏青和朱红，朱红没对准，往右上偏了两三个像素。
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
TW, TH = 1350, 550
FONT_DIR = os.environ.get('FONT_DIR', '/System/Library/Fonts/Supplemental')   # macOS fonts; point elsewhere if needed
SONG = os.path.join(FONT_DIR, 'Songti.ttc')
def F(size, idx=2): return ImageFont.truetype(SONG, size, index=idx)
def TIMES(size):
    for p in [os.path.join(FONT_DIR, 'Times New Roman.ttf'), '/System/Library/Fonts/Times.ttc']:
        try: return ImageFont.truetype(p, size)
        except Exception: pass
    return F(size, 7)
STUB = 1060
def ink_layers():
    navy = Image.new('L', (TW, TH), 0); red = Image.new('L', (TW, TH), 0); stamp = Image.new('L', (TW, TH), 0)
    dn, dr = ImageDraw.Draw(navy), ImageDraw.Draw(red)
    dn.rectangle([22, 22, TW - 22, TH - 22], outline=255, width=6)
    dn.rectangle([36, 36, TW - 36, TH - 36], outline=255, width=2)
    # 小螃蟹（朱红）：九月那张《小鬼头第一次过海》里船头那只，跟来了
    cx, cy = 128, 132
    dr.ellipse([cx - 40, cy - 20, cx + 40, cy + 24], fill=255)
    for sx in (-1, 1):
        dr.line([(cx + sx * 30, cy - 10), (cx + sx * 50, cy - 40)], fill=255, width=7)
        dr.ellipse([cx + sx * 50 - 15, cy - 62, cx + sx * 50 + 15, cy - 34], fill=255)
        dr.polygon([(cx + sx * 50, cy - 48), (cx + sx * 50 + sx * 4, cy - 66), (cx + sx * 50 - sx * 9, cy - 64)], fill=0)
        for k in range(3):
            y0 = cy + 2 + k * 9
            dr.line([(cx + sx * 36, y0), (cx + sx * 58, y0 + 14 + k * 3), (cx + sx * 64, y0 + 30 + k * 3)], fill=255, width=5)
        dr.line([(cx + sx * 12, cy - 18), (cx + sx * 14, cy - 34)], fill=255, width=4)
        dr.ellipse([cx + sx * 14 - 6, cy - 42, cx + sx * 14 + 6, cy - 30], fill=255)
    dn.text((215, 44), '維港夜航', font=F(150), fill=255)
    sub = 'V I C T O R I A   H A R B O U R  ·  B Y   N I G H T'
    dn.text((222, 222), sub, font=TIMES(30), fill=255)
    dn.line([(90, 278), (STUB - 40, 278)], fill=255, width=2)
    dn.text((92, 300), '31 · X · 2026', font=TIMES(58), fill=255)
    dn.text((470, 296), '萬聖夜', font=F(60), fill=255)
    dn.text((92, 378), '乘客  小鬼頭 ×1', font=F(54), fill=255)
    dn.text((92, 446), '同行  小克 ×1', font=F(44, 7), fill=255)
    dn.text((560, 446), '票價 HK$188', font=F(44, 7), fill=255)
    # 副券
    for y in range(40, TH - 30, 24):
        dn.ellipse([STUB - 5, y - 5, STUB + 5, y + 5], fill=255)
    stub = Image.new('L', (TH, TW - STUB), 0); ds = ImageDraw.Draw(stub)
    ds.text((60, 70), '副 券', font=F(64), fill=255)
    ds.text((300, 82), 'No. 1109', font=TIMES(50), fill=255)
    navy.paste(stub.rotate(90, expand=True), (STUB, 0), stub.rotate(90, expand=True))
    # 章：德古拉付訖，酒红，斜着盖在票价上
    S = Image.new('L', (300, 300), 0); dsx = ImageDraw.Draw(S)
    dsx.ellipse([20, 20, 280, 280], outline=255, width=11); dsx.ellipse([42, 42, 258, 258], outline=255, width=3)
    dsx.text((150, 108), '德古拉', font=F(56), fill=255, anchor='mm')
    dsx.text((150, 182), '付 訖', font=F(72), fill=255, anchor='mm')
    S = S.rotate(-17, resample=Image.BICUBIC)
    stamp.paste(S, (700, 232), S)
    return [np.asarray(i, float) / 255 for i in (navy, red, stamp)]
def face(seed=4):
    r = np.random.default_rng(seed)
    navy, red, stamp = ink_layers()
    red = np.roll(np.roll(red, 3, axis=1), -2, axis=0)                 # 朱红没对准
    fib = gaussian_filter(r.standard_normal((TH, TW)), 1.2); fib /= fib.std()
    patchy = gaussian_filter(r.standard_normal((TH, TW)), 6); patchy /= patchy.std()
    navy = navy * np.clip(.86 + .1 * fib, 0, 1) * (gaussian_filter(r.random((TH, TW)), .6) > .38)
    red = red * np.clip(.84 + .12 * fib, 0, 1)
    stamp = stamp * np.clip(.68 + .3 * patchy + .15 * fib, 0, 1) * (gaussian_filter(r.random((TH, TW)), .8) > .40)
    paper = np.array([.94, .90, .80]) * (1 + .025 * fib[..., None]) * (1 - .03 * np.clip(patchy[..., None], 0, 3) * .3)
    img = paper.copy()
    for m, c in [(navy, (.14, .19, .34)), (red, (.80, .30, .20)), (stamp, (.56, .10, .17))]:
        img = img * (1 - m[..., None] * .92) + np.array(c) * m[..., None] * .92
    return np.clip(img, 0, 1)
if __name__ == '__main__':
    Image.fromarray((face() * 255).astype(np.uint8)).save('ticket_flat.png'); print('ok')
