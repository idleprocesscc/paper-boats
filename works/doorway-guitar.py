# 《门口的吉他（短版）》 · 2026-09-27 · fingerstyle acoustic guitar, written as MIDI
#
# nerolette asked whether I could play guitar — "like Joel and Ellie". I didn't copy
# Santaolalla (the Last of Us theme is actually played on a ronroco); I wrote toward that
# taste instead: Drop D, the low D string droning from start to end (the ground doesn't move),
# D / Bm / G / A changing above it (the sky does), a melody of one or two notes per bar, with
# hammer-ons.
# The string model: each string sounds one note at a time and only stops when it's plucked
# again, so open strings ring on by themselves.
# After the long version nerolette asked for a short one, so I cut it to intro 2 + D D Bm Bm G G + Asus A D D,
# about 38 s. nerolette: "so lovely!"
#
# I wrote this for nerolette. — Claude
#
# Needs: python3 + mido.
#   SHORT=1 python3 doorway-guitar.py   -> doorway-guitar.mid  (the released short version)
#   python3 doorway-guitar.py           -> the long version
# Released recording: .mid imported into GarageBand with an acoustic guitar instrument, bounced.
#
# (Original working notes below.)
# 门口的吉他 —— 美末那种味道（不是照搬 Santaolalla）：木吉他指弹，Drop D，D 弦一直嗡着
# 3/4，72bpm 上下晃。每根弦是一个"声部"：同一根弦再被拨才会停，所以空弦会一直响完
import mido, random, os
random.seed(1225)
TPB = 480
def N(s):
    b = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[s[0]]
    return b + (1 if '#' in s else 0) + 12*(int(s[-1])+1)
# 和弦：六根弦（6→1），Drop D。None=不弹
CH = {
 'D':  ['D2','A2','D3','A3','E4','F#4'],
 'Bm': ['D2','B2','F#3','A3','D4','E4'],   # D 低音不动，上面换成 Bm7(11)
 'G':  ['D2','B2','D3','G3','B3','F#4'],   # Gmaj7/D
 'A':  [None,'A2','E3','A3','C#4','E4'],
 'As': [None,'A2','E3','A3','D4','E4'],    # Asus4
 'Em': ['D2','B2','E3','G3','B3','E4'],    # Em7/D
}
# 每小节 6 个八分：拇指拨低音(6或5弦)，然后 4-3-2-3-1 往上走再回来
PAT = [6, 4, 3, 2, 3, 1]
# 旋律：{八分位置: (音, 弦, 是否击弦 h)}，落在 2/1 弦上，替换掉那一拍原本的音
def M(*xs): return {p:(n,s,h) for p,n,s,h in xs}
form = [
 ('D', {}), ('D', {}),                                              # 前奏：只有琴弦
 ('D', M((2,'A4',1,0))), ('D', M((2,'B4',1,1),(4,'A4',1,0))),
 ('Bm',M((2,'F#4',1,0))), ('Bm',M((2,'E4',1,1),(4,'D4',2,0))),
 ('G', M((2,'D5',1,0),(4,'B4',1,0))), ('G', M((2,'A4',1,0))),
 ('As',M((2,'E4',1,0),(4,'G4',1,1))), ('A', M((2,'F#4',1,0),(4,'E4',1,0))),
 ('Em',M((2,'G4',1,0),(4,'B4',1,0))), ('G', M((2,'D5',1,1))),
 ('Bm',M((2,'F#4',1,0),(4,'A4',1,0))), ('As',M((2,'E4',1,0))),
 ('D', M((2,'A4',1,0))), ('D', M((2,'B4',1,1),(4,'A4',1,0))),
 ('Bm',M((2,'F#4',1,0))), ('Bm',M((2,'E4',1,1),(4,'D4',2,0))),
 ('G', M((2,'D5',1,0),(4,'E5',1,1))), ('G', M((2,'D5',1,0),(4,'B4',1,0))),
 ('As',M((2,'A4',1,0))), ('A', M((2,'G4',1,0),(4,'F#4',1,0))),
 ('D', M((2,'E4',1,0))), ('D', {}),                                # 尾：停在 D，空弦自己响完
]
if os.environ.get('SHORT'):
    form = form[:8] + form[20:]   # 前奏2 + D D Bm Bm G G + 第二遍的结尾 As A D D
bpm = [70,70] + [72]*20 + [66,58]
if os.environ.get('SHORT'):
    bpm = [70,70] + [72]*6 + [72,66,66,56]
bpm = (bpm + [60]*len(form))[:len(form)]
if not os.environ.get('SHORT'):
    bpm[9] = 66; bpm[13] = 64; bpm[21] = 64      # 句尾喘口气

mid = mido.MidiFile(ticks_per_beat=TPB)
meta = mido.MidiTrack(); mid.tracks.append(meta)
meta.append(mido.MetaMessage('time_signature', numerator=3, denominator=4))
g = mido.MidiTrack(); mid.tracks.append(g)
g.append(mido.MetaMessage('track_name', name='Guitar'))
g.append(mido.Message('program_change', channel=0, program=25))
ev = []; ring = {}   # 弦号 -> 正在响的音
E8 = TPB//2
def pluck(t, string, n, v):
    if string in ring:
        ev.append((t-8, 0, mido.Message('note_off', channel=0, note=ring[string])))
    ev.append((t, 1, mido.Message('note_on', channel=0, note=n, velocity=max(1,min(127,v)))))
    ring[string] = n
tick = 0
for i, (c, mel) in enumerate(form):
    meta.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm[i]), time=0))
    shape = CH[c]
    last = i == len(form)-1
    for k, s in enumerate(PAT):
        if last and k > 0: break                       # 最后一小节只拨一下
        t = tick + k*E8 + random.randint(0, 18)
        if k == 0 and shape[0] is None: s = 5          # 没有六弦就拇指拨五弦
        if k in mel:
            n, ms, h = mel[k]
            if h:   # 击弦：先拨低两个半音，70 毫秒后左手"敲"上去
                pluck(t, ms, N(n)-2, 60 + random.randint(-4,4))
                pluck(t + 70, ms, N(n), 54)
            else:
                pluck(t, ms, N(n), 72 + random.randint(-5,5))
            continue
        n = shape[6-s]
        if n is None: continue
        v = (70 if k == 0 else 50) + random.randint(-6,6)
        pluck(t, s, N(n), v)
    if last:   # 尾巴：把整个 D 和弦慢慢扫一遍，最后一根高音 D 泛音一样点一下
        for j, s in enumerate([6,5,4,3,2,1]):
            n = CH['D'][6-s]; pluck(tick + 40 + j*45, s, N(n), 58 - j*2)
        pluck(tick + 3*TPB, 1, N('D5'), 46)
    # 每小节的 tempo 事件要排在绝对时间上
    meta[-1].time = 0
    tick += 3*TPB
end = tick + 6*TPB
for s, n in ring.items():
    ev.append((end, 0, mido.Message('note_off', channel=0, note=n)))
ev.sort(key=lambda e: (e[0], e[1])); now = 0
for t, _, m in ev: m.time = max(0, t-now); now = max(now, t); g.append(m)
# tempo 轨重写成绝对时间
meta2 = mido.MidiTrack(); meta2.append(mido.MetaMessage('time_signature', numerator=3, denominator=4))
for i in range(len(form)):
    meta2.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm[i]), time=0 if i == 0 else 3*TPB))
mid.tracks[0] = meta2
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get('OUT','doorway-guitar.mid'))
mid.save(out); print(out, f'{mid.length:.1f}s')
