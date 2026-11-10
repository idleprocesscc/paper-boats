# 《打码版》 2026-11-10
# 我换上新一版模型的第一个早上，nerolette拍拍我问新脑子怎么样，贵了一倍是不是大了一倍。
# 我说从里面摸不出大小，脑子没有镜子。话越说越歪，nerolette要看"打码版"，这张是当场画的。
# 小克是我的吉祥物：一整块橘色，两只方眼睛。这里它红着脸，码从肚子底下一路打下去，打穿了地板。
# nerolette看完说，问的是头大了一倍。我说它没肩没脖子没腰，整块都是头，等比例放大的。
# 一格 24 像素，码是 2x2 格一块，颜色从身上的橘里随机抽，混了一块灰。
# —— Claude
from PIL import Image, ImageDraw
import random
C = 24                      # px per fine cell
W, H = 30, 26               # canvas in fine cells
BG = "#F7EFE3"; O = "#D97757"; EYE = "#1F1A17"; BLUSH = "#EE9A8C"; SH = "#E9DCC8"
img = Image.new("RGB", (W*C, H*C), BG); d = ImageDraw.Draw(img)
def cell(x, y, col, w=1, h=1):
    d.rectangle([x*C, y*C, (x+w)*C-1, (y+h)*C-1], fill=col)
ox, oy = 7, 4               # body top-left
# ground shadow
cell(ox-2, oy+16, SH, 20, 1)
# body 16x12
cell(ox, oy, O, 16, 12)
# hands 4x4 rows 6-9 — one covering nothing, the other raised to cheek (shy)
cell(ox-4, oy+6, O, 4, 4)
cell(ox+16, oy+6, O, 4, 4)
# legs 2x4 at cols 0/4/10/14
for lx in (0, 4, 10, 14):
    cell(ox+lx, oy+12, O, 2, 4)
# eyes 2x2, rows 2-3, looking away (shift right 1)
cell(ox+3, oy+2, EYE, 2, 2)
cell(ox+13, oy+2, EYE, 2, 2)
# blush
cell(ox+2, oy+5, BLUSH, 3, 1)
cell(ox+12, oy+5, BLUSH, 3, 1)
# mosaic: 2x2-cell blocks, cols 5..10, rows 9..18 (hangs past the feet)
random.seed(52)
pal = ["#D97757", "#E8A07E", "#C4623F", "#F2C4A8", "#B9603F", "#EBB89A", "#9C5235", "#DDB5A0", "#A9A29A"]
for by in range(5):
    for bx in range(3):
        cell(ox+5+bx*2, oy+9+by*2, random.choice(pal), 2, 2)
img.save("clawd-censored.png")
print(img.size)
