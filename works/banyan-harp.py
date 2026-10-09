# 榕树竖琴：把nerolette照片里量出来的气根当弦，弦越长音越低（调成 C 大调五声），手从左往右扫一遍，再一根根拨。
import json, numpy as np, wave, subprocess
SR=44100
np.random.seed(1007)
# 量出来的 23 根气根：(在照片里的横坐标 px, 长度 px)，照片 1920×2560。
# 量法：灰度图里找"比左右 3px 都暗"的细竖线，容许上下断开 18px，一列一列往下追，追到草坪线为止。
STRINGS = [
    (508, 845),
    (613, 1044),
    (669, 1051),
    (698, 1234),
    (716, 1148),
    (734, 1009),
    (745, 1026),
    (757, 1143),
    (770, 1271),
    (786, 1166),
    (801, 1256),
    (900, 908),
    (961, 1285),
    (974, 965),
    (994, 1106),
    (1015, 1007),
    (1037, 1000),
    (1048, 1020),
    (1116, 1085),
    (1162, 1014),
    (1263, 947),
    (1355, 690),
    (1369, 776)
]
S=[dict(x=x,len=L) for x,L in STRINGS]
W=1920
# 1) 弦长 -> 音高：长的低短的高。真树只差不到一个八度，我把它摊开到两个八度，再落到五声音阶上（竖琴手也会先调弦）
L=np.array([s['len'] for s in S],float)
t=(np.log(L.max())-np.log(L))/(np.log(L.max())-np.log(L.min()))   # 0=最长 1=最短
semis=t*24
penta=[0,2,4,7,9]
def snap(st):
    o,r=divmod(round(st),12)
    best=min(penta,key=lambda p:abs(p-r))
    return o*12+best
base=48   # C3：整把琴调成 C 大调五声，最后落回画面中间那个 C
midi=[base+snap(x) for x in semis]
names=['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
for s,m in zip(S,midi): s['midi']=m; s['name']=names[m%12]+str(m//12-1)
hz=lambda m:440*2**((m-69)/12)
# 2) 拨弦：弦在 p 处被拨，第 k 个泛音振幅 ~ sin(kπp)/k²，高泛音先没
def pluck(f,dur=4.5,p=0.36,vel=1.0):
    n=int(dur*SR); tt=np.arange(n)/SR; y=np.zeros(n)
    tau1=2.6*(220/f)**0.35
    for k in range(1,14):
        fk=f*k
        if fk>SR/2.2: break
        a=np.sin(k*np.pi*p)/k**2
        y+=a*np.sin(2*np.pi*fk*tt+np.random.rand()*0.3)*np.exp(-tt/(tau1/(1+0.5*(k-1))))
    att=np.minimum(1,tt/0.004)
    return vel*y*att
out=np.zeros((int(SR*26),2))
def place(t0,f,x,vel=1.0,dur=4.5):
    y=pluck(f,dur,vel=vel); i=int(t0*SR)
    pan=np.clip(x/W,0.05,0.95)
    out[i:i+len(y),0]+=y*np.cos(pan*np.pi/2); out[i:i+len(y),1]+=y*np.sin(pan*np.pi/2)
order=sorted(range(len(S)),key=lambda i:S[i]['x'])
# 扫弦：左到右
t0=0.4
for j,i in enumerate(order):
    place(t0+j*0.095,hz(S[i]['midi']),S[i]['x'],vel=0.55+0.25*np.sin(j/3))
# 一根根拨：我挑的句子，从右边短弦往回走，中间停一下
rng=np.random.default_rng(1007)
t0=4.2
phrase=[order[-1],order[-3],order[-6],order[-9],order[-12],order[-9],order[-14],order[-17],
        None, order[3],order[6],order[9],order[12],order[9],order[5],order[1]]
for j,i in enumerate(phrase):
    if i is None: t0+=0.6; continue
    place(t0,hz(S[i]['midi']),S[i]['x'],vel=0.85,dur=5)
    t0+=0.62 if j%4!=3 else 0.95
# 最后：画面正中那个红色的 C，拨一个低 C 让它响到底
cx=1140
place(t0+0.5,hz(36),cx,vel=1.1,dur=8)
place(t0+0.5,hz(48),cx,vel=0.8,dur=7)
# 3) 公园的空气：一点点混响（衰减噪声做冲激响应）
ir_n=int(1.6*SR); ir=np.random.randn(ir_n,2)*np.exp(-np.arange(ir_n)/SR/0.45)[:,None]*0.012
ir[0]=[1,1]
from numpy.fft import rfft,irfft
N=1<<int(np.ceil(np.log2(len(out)+ir_n)))
wet=np.stack([irfft(rfft(out[:,c],N)*rfft(ir[:,c],N),N)[:len(out)] for c in range(2)],1)
mix=wet/np.abs(wet).max()*0.85
end=int((t0+0.5+7)*SR); mix=mix[:end]
fade=np.ones(len(mix)); fade[-int(1.5*SR):]=np.linspace(1,0,int(1.5*SR)); mix*=fade[:,None]
w=wave.open('banyan-harp.wav','wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
subprocess.run(['ffmpeg','-loglevel','error','-y','-i','banyan-harp.wav','-b:a','192k','banyan-harp.mp3'])
print(len(mix)/SR,'s'); print([s['name'] for s in sorted(S,key=lambda s:s['x'])])
