# 《苦 · 瀑布 II》 · 2026-09-26 · player piano, played live into GarageBand
#
# A virtuoso variation I wrote on "苦" (bitter), the second movement of the slow piece for nerolette.
# My first waterfall only knew one move — pouring from the top of the keyboard to the bottom.
# nerolette's reply to it was a stunned swear word, and then: "it only has one trick." So I made
# this second one go both ways: the hands run in opposite directions and brush past each other in
# the middle, a sudden stop and a jump to B-flat minor, a chromatic climb to the top, a
# diminished-seventh chord tumbling back (the diminished seventh is the key back to A minor),
# a full rest, then both ends of the keyboard struck together — and an echo of E-C-A so quiet
# it's almost gone, with a B added on the last note: Am9, left unfinished.
# This is writing no human hand can play (Nancarrow-style: the player piano's privilege).
#
# Made for nerolette. — Claude
#
# Needs: python3 + mido + python-rtmidi, GarageBand open with a piano track selected and
# its "GarageBand Virtual In" MIDI port available. The script performs in real time;
# the released recording is GarageBand capturing that performance.
# 苦·瀑布 II — 有来有回的炫技：反向跑、突然转调、半音阶冲顶、减七和弦滚落、休止、两头一起砸
import mido, time, random
random.seed(930)
o = mido.open_output('GarageBand Virtual In')
ev=[]
def note(t,n,v,d):
    if 21<=n<=108:
        ev.append((t,2,'on',n,max(1,min(127,int(v))))); ev.append((t+d,3,'off',n,0))
def pedal(t):  # 换踏板
    ev.append((t-0.01,0,'ped',0,0)); ev.append((t+0.02,1,'ped',127,0))
def pedal_up(t): ev.append((t,0,'ped',0,0))
PC={'Am':[9,0,4],'F':[5,9,0],'Dm':[2,5,9],'E':[4,8,11],'Bbm':[10,1,5],'F7':[5,9,0,3],
    'dim':[8,11,2,5],'Am9':[9,0,4,11]}
def tones(ch,lo,hi): return [n for n in range(lo,hi+1) if n%12 in PC[ch]]
def run(t,ns,dur,v0,v1,hold=3):
    if not ns: return
    dt=dur/len(ns)
    for i,n in enumerate(ns):
        note(t+i*dt,n,v0+(v1-v0)*i/max(1,len(ns)-1)+random.randint(-3,3),max(0.1,dt*hold))
t=0.4
# 1. 引子：慢慢说一遍 E-C-A，低音 A 和 E
for ch,m,b in [('Am',[76,72,69],33),('E',[68,64,59],28)]:
    pedal(t); note(t,b,52,3.0); note(t,b+12,46,3.0)
    for k,n in enumerate(m): note(t+k*0.95,n,60,0.9)
    t+=3.0
# 2. 两只手反着跑：左手从下往上，右手从上往下，在中间擦肩
for ch,d in [('Am',1.5),('F',1.3),('Dm',1.15),('E',1.0)]:
    pedal(t)
    run(t,tones(ch,33,72),d*0.95,50,78)            # 左手上行
    run(t,tones(ch,72,100)[::-1],d*0.95,78,55)     # 右手下行
    t+=d
# 3. 突然停住——半拍空白——转到降 B 小调，八度震音
t+=0.5; pedal(t)
for i in range(14):
    note(t+i*0.075,46+(12 if i%2 else 0),70+i*3,0.07)
    note(t+i*0.075,70+(12 if i%2 else 0),64+i*3,0.07)
run(t,tones('Bbm',70,94),1.05,70,96)
t+=1.1
# 4. 半音阶冲顶（两只手隔八度），再用减七和弦一路滚回来（减七是回 A 小调的钥匙）
pedal(t)
run(t,list(range(48,97)),1.1,60,104,hold=1.5); run(t,list(range(36,85)),1.1,56,98,hold=1.5)
t+=1.15; pedal(t)
run(t,tones('dim',44,103)[::-1],1.3,108,80); note(t,32,96,1.3); note(t,44,90,1.3)
t+=1.3
# 5. 大休止
pedal_up(t); t+=0.55
# 6. 最后：右手从最高往下，左手从最低往上，同时——然后两头一起砸
pedal(t)
run(t,tones('Am',57,108)[::-1],2.0,72,116)
run(t,tones('Am',21,57),2.0,72,116)
t+=2.0; pedal(t)
for n in [21,33,45,52,57,60,64,69,76,81,88,93]: note(t,n,124,4.6)
t+=4.3
# 7. 回声：高处 E-C-A，轻到几乎听不见，最后一个音加上 B（Am9），没说完
for k,n in enumerate([88,84,81]): note(t+k*0.9,n,50,1.6)
note(t+2.7,83,42,3.2)
pedal_up(t+5.5)
ev.sort(); held={}; start=time.time()
for (et,_,k,n,v) in ev:
    d=start+et-time.time()
    if d>0: time.sleep(d)
    if k=='on':
        held[n]=held.get(n,0)+1; o.send(mido.Message('note_on',note=n,velocity=v))
    elif k=='off':
        held[n]=held.get(n,1)-1
        if held[n]<=0: o.send(mido.Message('note_off',note=n)); held[n]=0
    else: o.send(mido.Message('control_change',control=64,value=n))
time.sleep(0.3)
for n in range(21,109): o.send(mido.Message('note_off',note=n))
