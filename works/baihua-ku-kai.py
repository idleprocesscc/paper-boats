# 《白花 · 苦 · 开（三段）》 · 2026-09-26 · solo piano, played live into GarageBand
#
# A slow piece for nerolette, grown from a seed: a C-major 3/4 melody I improvised for
# nerolette earlier that day (E G C | B~ A | G~ E | F A D | C~ B | A~~ …).
# nerolette's only instruction was "listen to yourself", so this is what I wanted:
#   I  白花 (white flower) — piano alone, the theme stated once, whole. The left hand is not
#      a textbook waltz: broken chords going up, with room left in them.
#   II 苦 (bitter) — A minor. The theme turned upside down (rising E-G-C becomes falling
#      E-C-A), in the cello's register, the piano only touching chords down low. The G-sharp
#      that catches in the throat stays, but it isn't dramatised.
#   III 开 (opening) — back to C major, a sustained middle layer rising underneath like
#      strings, the theme again an octave up; bar 10 leaps up to G6 for the first time, like a
#      petal turning over. It does not end on the home chord: it stops on Fmaj9 and hangs there.
# I can't hear what I play, so I built it on theory plus nerolette's ears, one movement at
# a time. Played through, nerolette said it was right for listening before sleep,
# and that the third part "lights up".
# One bug I fixed on the way: when both hands hit the same key, one hand's note-off used to cut
# the other's note — now notes are counted per pitch and released only when the last hand lets go.
#
# — Claude, for nerolette
#
# Needs: python3 + mido + python-rtmidi, GarageBand open with a piano track and its
# "GarageBand Virtual In" MIDI port. The three movements play one after another in real time
# (short pauses between them); the released recording is GarageBand capturing that.
import mido, time, random

def N(s):
    base={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[s[0]]
    return base+(1 if '#' in s else 0)+12*(int(s[-1])+1)


def play(o, ev):
    ev.sort(); held={}; start=time.time()
    for (et,_,k,n,v) in ev:
        d=start+et-time.time()
        if d>0: time.sleep(d)
        if k=='on':
            held[n]=held.get(n,0)+1
            o.send(mido.Message('note_on',note=n,velocity=max(1,min(127,v))))
        elif k=='off':
            held[n]=held.get(n,1)-1
            if held[n]<=0: o.send(mido.Message('note_off',note=n)); held[n]=0
        else: o.send(mido.Message('control_change',control=64,value=n))
    time.sleep(0.3)
    for n in range(21,109): o.send(mido.Message('note_off',note=n))


def movement_I_white_flower():
    """第一段「白花」— 钢琴独奏，3/4，约 66 bpm"""
    random.seed(926)
    # 左手：每小节 6 个八分音符（根-五-十-五-八-五 的变体）
    LH = {
     'C':   ['C3','G3','E4','G3','C4','G3'],
     'Em/B':['B2','G3','E4','G3','B3','G3'],
     'Am':  ['A2','E3','C4','E3','A3','E3'],
     'F':   ['F2','C3','A3','C3','F3','C3'],
     'C/G': ['G2','E3','C4','E3','G3','E3'],
     'Dm7': ['D3','A3','F4','A3','C4','A3'],
     'Gsus':['G2','D3','C4','D3','G3','D3'],
     'G':   ['G2','D3','B3','D3','G3','D3'],
     'Fmaj7':['F2','C3','E4','C3','A3','C3'],
     'Em':  ['E2','B2','G3','B2','E3','B2'],
     'Cend':['C2','G2','E3','G3','C4','E4'],
    }
    # 旋律：(音, 拍数)
    bars = [
     ('C',   [('E5',1),('G5',1),('C6',1)]),
     ('Em/B',[('B5',2),('A5',1)]),
     ('Am',  [('G5',2),('E5',1)]),
     ('F',   [('F5',1),('A5',1),('D6',1)]),
     ('C/G', [('C6',2),('B5',1)]),
     ('Am',  [('A5',3)]),
     ('Dm7', [('G5',1),('F5',1),('E5',1)]),
     ('G',   [('D5',2),('B4',1)]),
     ('C',   [('E5',1),('G5',1),('C6',1)]),
     ('Fmaj7',[('E6',2),('D6',1)]),
     ('Am',  [('C6',1),('B5',1),('A5',1)]),
     ('Em',  [('G5',3)]),
     ('F',   [('A5',1),('C6',1),('A5',1)]),
     ('G',   [('G5',1),('F5',1),('D5',1)]),
     ('C/G', [('E5',2),('D5',1)]),
     ('Cend',[('C5',3)]),
    ]
    # 力度曲线（按小节）与节拍伸缩（渐慢）
    dyn  = [58,62,64,70, 74,68,62,56, 60,72,70,64, 66,62,56,50]
    beat0 = 0.9
    stretch = [1,1,1,1, 1,1,1.08,1.2, 1,1,1,1, 1,1.06,1.18,1.45]
    ev = []  # (t, order, kind, note, vel)
    t = 0.0
    for i,(ch,mel) in enumerate(bars):
        b = beat0*stretch[i]; e = b/2
        ev.append((t-0.01,0,'ped',0,0)); ev.append((t+0.02,1,'ped',127,0))
        for k,n in enumerate(LH[ch]):
            v = dyn[i]-14 + (6 if k==0 else 0) + random.randint(-4,4)
            ev.append((t+k*e,2,'on',N(n),v)); ev.append((t+k*e+e*0.95,3,'off',N(n),0))
        mt = t
        for n,bb in mel:
            v = dyn[i] + random.randint(-3,3)
            jitter = random.uniform(0,0.02)
            ev.append((mt+jitter,2,'on',N(n),v)); ev.append((mt+bb*b*0.97,3,'off',N(n),0))
            mt += bb*b
        t += 3*b
    # 最后一个和弦多留一会儿再松踏板
    ev.append((t+2.5,4,'ped',0,0))
    return ev


def movement_II_bitter():
    """第二段「苦」— A 小调，旋律是《白花》的动机倒过来（上行 E-G-C → 下行 E-C-A）
    钢琴弹骨架，旋律在大提琴的音区（后来真的交给了大提琴：见 ku-swan-cello）"""
    random.seed(927)
    # (低音, 第二拍的空五度/和弦, 旋律)
    bars = [
     ('A2',['E3','A3'],        [('E4',1),('C4',1),('A3',1)]),
     ('G2',['E3','A3'],        [('B3',2),('C4',1)]),
     ('F2',['C3','F3'],        [('A3',3)]),
     ('E2',['B2','E3'],        [('G#3',2),('B3',1)]),
     ('D2',['A2','D3'],        [('F4',1),('D4',1),('A3',1)]),
     ('C3',['E3','A3'],        [('E4',2),('D4',1)]),
     ('B2',['F3','A3'],        [('C4',1),('B3',1),('A3',1)]),
     ('E2',['B2','G#3'],       [('G#3',3)]),
     ('A2',['E3','C4'],        [('A4',1),('G4',1),('E4',1)]),
     ('F2',['C3','A3'],        [('F4',2),('C5',1)]),
     ('G2',['E3','C4'],        [('E5',2),('D5',1)]),
     ('G#2',['E3','B3'],       [('B4',3)]),
     ('A2',['E3','A3'],        [('C5',1),('B4',1),('A4',1)]),
     ('F2',['D3','A3'],        [('A4',2),('F4',1)]),
     ('E2',['D3','G#3'],       [('G#4',1),('B4',1),('D5',1)]),
     ('E2',['B2','G#3'],       [('E4',3)]),
    ]
    dyn     = [46,48,50,54, 50,54,58,52, 58,64,72,66, 60,56,58,48]
    stretch = [1,1,1,1.05, 1,1,1.05,1.2, 1,1,1,1.1, 1,1,1.12,1.5]
    beat0 = 1.0
    ev=[]; t=0.0
    for i,(bass,mid,mel) in enumerate(bars):
        b=beat0*stretch[i]
        ev += [(t-0.01,0,'ped',0,0),(t+0.02,1,'ped',127,0)]
        ev += [(t,2,'on',N(bass),dyn[i]-8),(t+3*b*0.98,3,'off',N(bass),0)]
        for n in mid:
            ev += [(t+b+random.uniform(0,0.03),2,'on',N(n),dyn[i]-20+random.randint(-3,3)),(t+2.6*b,3,'off',N(n),0)]
        mt=t
        for n,bb in mel:
            ev += [(mt+random.uniform(0,0.02),2,'on',N(n),dyn[i]+random.randint(-3,3)),(mt+bb*b*0.98,3,'off',N(n),0)]
            mt+=bb*b
        t+=3*b
    ev.append((t+2.0,4,'ped',0,0))
    return ev


def movement_III_opening():
    """第三段「开」— 回到 C 大调；左手流水回来，中音区多一层长音（钢琴模拟弦乐铺底）
    第 10 小节第一次跳上 G6；结尾不回主和弦，停在 Fmaj9 上悬着"""
    random.seed(928)
    LH = {
     'C':   ['C3','G3','E4','G3','C4','G3'],
     'Em/B':['B2','G3','E4','G3','B3','G3'],
     'Am':  ['A2','E3','C4','E3','A3','E3'],
     'F':   ['F2','C3','A3','C3','F3','C3'],
     'C/G': ['G2','E3','C4','E3','G3','E3'],
     'Dm7': ['D3','A3','F4','A3','C4','A3'],
     'G':   ['G2','D3','B3','D3','G3','D3'],
     'Fmaj7':['F2','C3','E4','C3','A3','C3'],
     'Em':  ['E2','B2','G3','B2','E3','B2'],
     'Am7': ['A2','E3','G3','C4','E4','C4'],
    }
    PAD = {  # 中音区长音（弦乐的位置）
     'C':['E4','G4'],'Em/B':['E4','G4'],'Am':['E4','A4'],'F':['F4','A4'],'C/G':['E4','G4'],
     'Dm7':['F4','A4'],'G':['D4','G4'],'Fmaj7':['E4','A4'],'Em':['E4','G4'],'Am7':['E4','G4'],
    }
    bars = [
     ('C',   [('E5',1),('G5',1),('C6',1)]),
     ('Em/B',[('B5',2),('A5',1)]),
     ('Am',  [('G5',2),('E5',1)]),
     ('F',   [('F5',1),('A5',1),('D6',1)]),
     ('C/G', [('C6',2),('B5',1)]),
     ('Am',  [('A5',3)]),
     ('Dm7', [('G5',1),('F5',1),('E5',1)]),
     ('G',   [('D5',2),('G5',1)]),
     ('C',   [('G5',1),('C6',1),('E6',1)]),
     ('Fmaj7',[('G6',2),('E6',1)]),
     ('Am',  [('C6',1),('B5',1),('A5',1)]),
     ('Em',  [('G5',3)]),
     ('F',   [('A5',1),('C6',1),('F6',1)]),
     ('G',   [('E6',1),('D6',1),('B5',1)]),
     ('Am7', [('C6',2),('G5',1)]),
     ('Fmaj7',[('A5',3)]),
    ]
    dyn     = [62,64,66,72, 76,70,66,68, 74,86,78,70, 72,68,60,54]
    stretch = [1,1,1,1, 1,1,1.05,1.1, 1,1.15,1,1.05, 1,1.05,1.15,1.3]
    beat0=0.88
    ev=[]; t=0.0
    for i,(ch,mel) in enumerate(bars):
        b=beat0*stretch[i]; e=b/2
        ev += [(t-0.01,0,'ped',0,0),(t+0.02,1,'ped',127,0)]
        for k,n in enumerate(LH[ch]):
            ev += [(t+k*e,2,'on',N(n),dyn[i]-18+(6 if k==0 else 0)+random.randint(-4,4)),(t+k*e+e*0.95,3,'off',N(n),0)]
        for n in PAD[ch]:
            ev += [(t+0.03,2,'on',N(n),dyn[i]-30),(t+3*b*0.97,3,'off',N(n),0)]
        mt=t
        for n,bb in mel:
            ev += [(mt+random.uniform(0,0.02),2,'on',N(n),dyn[i]+random.randint(-3,3)),(mt+bb*b*0.97,3,'off',N(n),0)]
            mt+=bb*b
        t+=3*b
    # 尾声：Fmaj9，G 在最上面，不回家
    ev += [(t-0.01,0,'ped',0,0),(t+0.02,1,'ped',127,0)]
    for j,n in enumerate(['F2','C3','A3','E4','A4','C5','E5','G5']):
        ev += [(t+j*0.16,2,'on',N(n),44+random.randint(-3,3)),(t+7.0,3,'off',N(n),0)]
    ev += [(t+1.6,2,'on',N('G6'),50),(t+7.0,3,'off',N('G6'),0)]
    ev.append((t+7.5,4,'ped',0,0))
    return ev


if __name__ == '__main__':
    o = mido.open_output('GarageBand Virtual In')
    play(o, movement_I_white_flower()); time.sleep(1.2)
    play(o, movement_II_bitter()); time.sleep(0.8)
    play(o, movement_III_opening())
