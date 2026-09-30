# 《苦 · 天鹅版（大提琴）》 · 2026-09-27 · piano + cello, written as MIDI
#
# The second movement ("苦", bitter) of the slow piece I wrote for nerolette in three movements:
# 白花 (white flower) · 苦 (bitter) · 开 (opening).
#
# The first cello version (F minor, the piano crouched low, and a metronome left on) got
# "Batman hasn't risen yet… too Hans Zimmer" from nerolette, so I took this one back to A minor
# at 64 bpm and gave the piano flowing eighth-note arpeggios in the middle register — toward the
# Swan — with the cello melody unchanged. nerolette: "this is so lovely…! especially with the piano!"
#
# What I was thinking: a piano note can only get quieter once it's struck; a cello note can
# grow while the bow is still moving. Bitterness is something held in the mouth that gets
# stronger the longer you hold it — so every long note swells with CC1/CC11.
# The first eight bars fall and rest on the open-string dominant E; bar 9 is the first time
# this movement climbs; bar 12's B4 is the peak, bowed full; the end stays on the dominant,
# unresolved, so the third movement can come in a half step sideways (V→♭VI, a deceptive
# cadence): the bitterness doesn't return to itself, it falls sideways into a flower.
#
# — Claude, for nerolette
#
# Needs: python3 + mido. Writes ku-swan-cello.mid (tempo track + Piano ch0 + Cello ch1).
# Released recording: .mid dragged into GarageBand (it splits into two tracks), second track
# switched to GarageBand's Cello, metronome off, bounced. TR=<semitones> env var transposes.
#
# (Original working notes below.)
# 慢曲 · 第二段「苦」· 大提琴版 — 写成 .mid 文件，不走 Virtual In
# 两轨：钢琴(低音区点和弦) + 大提琴(旋律)。导进库乐队会自动分成两条轨，
# 把第二条换成 Cello 就行 —— 绕开"Virtual In 只进选中轨道"那个坑。
#
# 想法：钢琴的音一按下去就只会变小；大提琴的音能在拉着的时候变大。
# 苦 = 一个含在嘴里、越含越明显的音。所以每个长音都用 CC1/CC11 往上推。
#
# 写在 A 小调；第一版用 TR=-4 移到 F 小调 4/4。天鹅版（本文件）TR=0，就是 A 小调。
# 结尾停在属和弦，第三段「开」从 ♭VI 进 —— V→♭VI，阻碍终止：
# 苦不回到自己，往旁边落进一朵花里。
import mido, random, os
random.seed(927)
TR = int(os.environ.get('TR', '0'))
TPB = 480
def N(s):
    b = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[s[0]]
    return b + (1 if '#' in s else 0) + 12*(int(s[-1])+1) + TR

ARP = [
 ['A2','E3','A3','C4','E4'], ['G2','E3','G3','B3','E4'], ['F2','C3','F3','A3','C4'], ['E2','B2','E3','G#3','B3'],
 ['D2','A2','D3','F3','A3'], ['C3','E3','A3','C4','E4'], ['B2','F3','B3','D4','F4'], ['E2','B2','E3','G#3','B3'],
 ['A2','E3','A3','C4','E4'], ['F2','C3','F3','A3','C4'], ['G2','D3','G3','B3','D4'], ['E2','B2','E3','G#3','D4'],
 ['A2','E3','A3','C4','E4'], ['F2','D3','F3','A3','D4'], ['E2','B2','D3','G#3','B3'], ['E2','B2','E3','G#3','B3'],
]
# 每小节：(钢琴低音, 第三拍的和弦, 大提琴旋律[(音或None, 拍数)])
bars = [
 ('A2', ['E3','A3'],  [(None,1),('E4',1.5),('D4',.5),('C4',1)]),
 ('G2', ['E3','B3'],  [('B3',3),('C4',1)]),
 ('F2', ['C3','A3'],  [('A3',4)]),
 ('E2', ['B2','D3'],  [('G#3',2),('A3',1),('B3',1)]),       # 揪一下，不演
 ('D2', ['A2','F3'],  [('F4',2),('E4',1),('D4',1)]),
 ('C3', ['E3','A3'],  [('E4',3),('A3',1)]),
 ('B2', ['D3','F3'],  [('D4',1.5),('C4',.5),('B3',2)]),
 ('E2', ['B2','G#3'], [('E3',4)]),                           # 空弦，歇在属上
 ('A2', ['E3','C4'],  [('A3',1),('C4',1),('E4',1),('A4',1)]),# 这一段第一次往上走
 ('F2', ['C3','A3'],  [('C5',3),('A4',1)]),
 ('G2', ['D3','B3'],  [('B4',2),('D5',1),('C5',1)]),
 ('E2', ['D3','G#3'], [('B4',4)]),                           # 最高处，拉满再收
 ('A2', ['E3','C4'],  [('C5',2),('B4',1),('A4',1)]),
 ('F2', ['D3','A3'],  [('A4',2),('F4',2)]),
 ('E2', ['D3','G#3'], [('E4',1),('D4',1),('B3',2)]),
 ('E2', ['B2','G#3'], [('E3',4)]),                           # 不解决
]
dyn     = [44,48,52,50, 54,56,58,50, 58,66,74,78, 68,60,54,46]
bpm     = [64,64,64,62, 64,64,62,58, 64,66,66,60, 62,60,56,48]

mid = mido.MidiFile(ticks_per_beat=TPB)
tempo_tr = mido.MidiTrack(); mid.tracks.append(tempo_tr)
tempo_tr.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
pno = mido.MidiTrack(); mid.tracks.append(pno)
vc  = mido.MidiTrack(); mid.tracks.append(vc)
pno.append(mido.MetaMessage('track_name', name='Piano'))
vc.append(mido.MetaMessage('track_name', name='Cello'))
pno.append(mido.Message('program_change', channel=0, program=0))
vc.append(mido.Message('program_change', channel=1, program=42))

def flush(track, evs):
    evs.sort(key=lambda e: (e[0], e[1]))
    now = 0
    for t, _, msg in evs:
        t = int(round(t)); msg.time = t - now; now = t; track.append(msg)

P, V, T = [], [], []
tick = 0
for i, (bass, chord, mel) in enumerate(bars):
    T.append((tick, 0, mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm[i]))))
    bar = 4*TPB
    # 钢琴（v2·天鹅）：不蹲低音区，每小节八个八分音符的分解和弦上去再下来，像水
    P += [(tick, 0, mido.Message('control_change', channel=0, control=64, value=0)),
          (tick+20, 1, mido.Message('control_change', channel=0, control=64, value=100))]
    tones = ARP[i]
    pat = [0,1,2,3,4,3,2,1] if i < 15 else [0,1,2,3,4]
    e = TPB//2 if i < 15 else TPB*4//5
    for k, idx in enumerate(pat):
        s0 = tick + k*e + random.randint(0, 12)
        v = max(22, dyn[i]-24 + (6 if k == 0 else 0) + (3 if idx == 4 else 0) + random.randint(-3,3))
        P += [(s0, 2, mido.Message('note_on', channel=0, note=N(tones[idx]), velocity=v)),
              (s0+int(e*1.8), 3, mido.Message('note_off', channel=0, note=N(tones[idx])))]
    # 大提琴：连奏(音头压一点点重叠)，每个音自己一条表情曲线
    t = tick
    for n, beats in mel:
        L = int(beats*TPB)
        if n is not None:
            peak = dyn[i] + (10 if beats >= 2 else 5)
            start = max(30, peak-26)
            V.append((t, 2, mido.Message('note_on', channel=1, note=N(n), velocity=dyn[i])))
            V.append((t+L+30, 3, mido.Message('note_off', channel=1, note=N(n))))  # 跟下一个音咬一点
            steps = max(4, int(beats*8))
            for k in range(steps+1):
                x = k/steps
                # 前七成往上推，最后三成回落一点（长音尤其明显）
                val = start + (peak-start)*(x/0.7 if x < 0.7 else 1-(x-0.7)/0.3*0.35)
                for cc in (1, 11):
                    V.append((t+x*L*0.98, 1, mido.Message('control_change', channel=1, control=cc, value=int(max(0, min(127, val))))))
        t += L
    tick += bar
# 最后一个 E 多拉两拍、松踏板
P.append((tick+2*TPB, 4, mido.Message('control_change', channel=0, control=64, value=0)))
flush(tempo_tr, T); flush(pno, P); flush(vc, V)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ku-swan-cello.mid')
mid.save(out)
print(out, f'{mid.length:.1f}s')
