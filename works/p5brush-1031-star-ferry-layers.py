# 《小鬼头第一次过海》 · Little Devil's First Crossing — the compositor (Claude, 2026-09-23).
# Renders each step of p5brush-1031-star-ferry.html on its own (?only=N), then multiplies the layers onto
# the paper; the lit windows (5), the lanterns (9) and the figure (10) are laid over instead, because orange
# multiplied onto deep blue disappears. Step 10 is masked to the bow: the brush now and then flings a
# stroke to the canvas origin. s4_final.png is the picture.
#   python3 p5brush-1031-star-ferry-layers.py --render   # opens a browser window (needs a GUI session)
#   python3 p5brush-1031-star-ferry-layers.py            # re-composite existing layers/L*.png only
import os, sys, numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
HERE=os.path.dirname(os.path.abspath(__file__))
base='file://'+os.path.join(HERE,'p5brush-1031-star-ferry.html')
out=os.path.join(HERE,'layers')+'/'
os.makedirs(out, exist_ok=True)
steps=range(11)
with sync_playwright() as p:
    b=p.chromium.launch(args=["--use-angle=metal","--ignore-gpu-blocklist"], headless=False)
    for s in (steps if '--render' in sys.argv else []):
        pg=b.new_page(viewport={'width':900,'height':1200})
        pg.goto(f'{base}?only={s}')
        pg.wait_for_function("document.title==='DONE'", timeout=240000); pg.wait_for_timeout(400)
        pg.screenshot(path=f'{out}L{s}.png'); pg.close(); print('L',s,flush=True)
    b.close()
paper=np.array([0xf2,0xec,0xe0],dtype=np.float64)
acc=np.ones((1200,900,3))
stages={2:'s1_sky',5:'s2_shore',7:'s3_water',10:'s4_final'}
for s in steps:
    L=np.asarray(Image.open(f'{out}L{s}.png').convert('RGB'),dtype=np.float64)
    if s in (5,9,10):
        out_=acc*paper
        a=np.clip(np.abs(L-paper).max(axis=2,keepdims=True)/55.0,0,1)
        if s==10:  # BBOX10: 只留船头那一块，笔刷偶尔把一两笔甩到原点坐标去
            m=np.zeros_like(a); m[840:960,600:760]=1; a=a*m
        out_=out_*(1-a)+L*a
        acc=out_/paper
    else:
        acc*=np.clip(L/paper,0,1.2)
    if s in stages:
        Image.fromarray(np.clip(acc*paper,0,255).astype(np.uint8)).save(f'{out}{stages[s]}.png'); print('stage',stages[s])
