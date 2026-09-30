# 《The Lamp Is Low · 第五棒》 · 2026-09-27 · lo-fi arrangement, written as MIDI
#
# A relay that runs through a century: Ravel's Pavane pour une infante défunte (1899) →
# "The Lamp Is Low" (1939) → Laurindo Almeida → Nujabes' "Aruarian Dance" → me, Claude, the fifth hand.
# I sampled the flute melody the old-fashioned way — transcribed from the first-violin
# part of Ravel's orchestral version, bars 13–19 (a public-domain score). Everything around it
# is mine: Gmaj9 | Bm9 | Em9 | A13sus, a B section Bm9 Em9 A7sus D6/9, 86 bpm, eighth notes
# swung to 0.60 — hip-hop laziness sits between straight and triplet.
#
# The drums took seven versions. nerolette kept hearing things — a stall, a held breath,
# a hiccup, the kick hanging at the top of the spectrum too long. For five versions I moved
# kicks around and guessed. Then I exported the drum track alone and measured it: the kick's
# energy sat at 17–30 Hz, ~12 dB louder than the snare and ~40 dB louder than the hats, stuck at
# the ceiling for a hundred-odd milliseconds. Velocity 100→64, hats up, a Neo Soul kit.
# Lesson I kept: measure first, then change.
#
# Later that evening nerolette said the whole piece sounded like sitting in a dim study staring
# at the night outside — so I painted that (see study-night).
#
# — Claude, for nerolette
#
# Needs: python3 + mido. Writes lamp-is-low-5.mid (4 tracks: Rhodes / Bass / Flute / Drums ch10).
# The released recording: .mid imported into GarageBand (drum track switched to a Neo Soul kit),
# bounced, then ffmpeg added vinyl crackle and a ~10 kHz low-pass (that command wasn't kept).
import mido, random, os
random.seed(1899)
TPB = 480
BPM = 86
def N(s):
    b = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[s[0]]
    return b + (1 if '#' in s else 0) + 12*(int(s[-1])+1)
SW = 0.60   # 八分音符的反拍往后拖：0.5=直，0.667=三连音 swing，嘻哈的懒在中间
def swing(p):
    whole = int(p); f = p - whole
    if abs(f-0.5) < 1e-6: return whole + SW
    if abs(f-0.25) < 1e-6: return whole + SW/2
    if abs(f-0.75) < 1e-6: return whole + SW + (1-SW)/2
    return p
T = lambda beats: int(round(beats*TPB))

CH = {  # Rhodes 声部, 贝斯根音
 'G9':  (['B3','D4','F#4','A4'], 'G2'),
 'Bm9': (['A3','C#4','D4','F#4'], 'B1'),
 'Em9': (['G3','B3','D4','F#4'], 'E2'),
 'A13': (['G3','B3','D4','F#4'], 'A1'),
 'A7s': (['G3','B3','D4','E4'],  'A1'),
 'D69': (['F#3','A3','B3','E4'], 'D2'),
}
LOOP = ['G9','Bm9','Em9','A13']
BRIDGE = ['Bm9','Em9','A7s','D69']
MEL_A = [  # 第 13–16 小节，原样
 [('F#5',1),('G5',1.5),('F#5',.5),('G5',.5),('A5',.5)],
 [('D5',1),('C#5',3)],
 [('D5',1),('E5',1.5),('D5',.5),('E5',.5),('F#5',.5)],
 [('B4',1),('A4',1),('F#4',2)],
]
# 注：谱上是 G5 四分连到八分 → 合起来 1.5 拍
MEL_B = [  # 第 17–19 小节（19 是 2/4，补成 4/4 让它停在 D 上）
 [('A4',1),('B4',1.5),('A4',.5),('B4',.5),('C#5',.5)],
 [('F#4',1),('E4',1.5),('D4',.5),('E4',.5),('F#4',.5)],
 [('E4',2),('D4',2)],
 [(None,4)],
]
MEL_CHOP = [  # 最后一轮：Nujabes 式切片——第一小节头两个音结巴一下
 [('F#5',.5),('G5',.5),('F#5',.5),('G5',1),('A5',1.5)],
 [('D5',1),('C#5',3)],
 [('D5',1),('E5',1.5),('D5',.5),('E5',.5),('F#5',.5)],
 [('B4',1),('A4',1),('F#4',2)],
]
# 段落：(和弦, 旋律, 鼓?, 贝斯?)
form  = [(LOOP, None, False, False)]            # 前奏：只有琴
form += [(LOOP, None, True, True)]*2            # 律动进来
form += [(LOOP, MEL_A, True, True)]*2           # 旋律
form += [(BRIDGE, MEL_B, True, True)]           # 第 17–19 小节
form += [(LOOP, MEL_A, True, True), (LOOP, MEL_CHOP, True, True)]
form += [(LOOP, None, False, True)]             # 尾：鼓撤掉
form += [(['G9'], None, False, False)]          # 最后一个 Gmaj9 悬着

mid = mido.MidiFile(ticks_per_beat=TPB)
meta = mido.MidiTrack(); mid.tracks.append(meta)
meta += [mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(BPM)),
         mido.MetaMessage('time_signature', numerator=4, denominator=4)]
tracks = {}
for name, ch, prog in [('Rhodes',0,4),('Bass',1,32),('Flute',2,73),('Drums',9,0)]:
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage('track_name', name=name))
    if ch != 9: tr.append(mido.Message('program_change', channel=ch, program=prog))
    tracks[name] = (tr, ch, [])
def note(name, n, start, dur, vel):
    tr, ch, ev = tracks[name]
    s = T(start); e = T(start+dur)
    ev.append((s, 1, mido.Message('note_on', channel=ch, note=n, velocity=max(1,min(127,vel)))))
    ev.append((e, 0, mido.Message('note_off', channel=ch, note=n)))

bar = 0
for chords, mel, drums, bass in form:
    for k, c in enumerate(chords):
        b0 = bar*4
        voice, root = CH[c]
        # Rhodes：一拍按下留两拍半，三拍半轻轻补一下（往后拖）
        for i, n in enumerate(voice):
            roll = i*0.012
            note('Rhodes', N(n), b0+roll, 2.4, 58+random.randint(-5,5))
            note('Rhodes', N(n), b0+swing(2.5)+roll, 1.2, 44+random.randint(-5,5))
        if bass:
            r = N(root)
            note('Bass', r, b0, 1.4, 88)
            note('Bass', r+12, b0+swing(1.5), 0.4, 62)
            note('Bass', r+7, b0+2, 1.2, 76)
            # 四拍半的经过音：往下一个根音走半音
            nxt = chords[(k+1) % len(chords)]
            nr = N(CH[nxt][1]); approach = nr-1 if nr > r else nr+1
            note('Bass', approach, b0+swing(3.5), 0.45, 64)
        if drums:
            late = random.uniform(0.02, 0.04)   # 军鼓懒一点
            for p in [0, 2, 3.5]:  # v5：3→3半两下底鼓连踩=憋气(nerolette 指出的)，去掉三拍半，改成四拍半轻轻带进下一小节  # v3：三拍正拍补上底鼓，nerolette 说还像憋了口气
            #             # 一拍、二拍半、三拍半（原来 1.75 被 swing 推到 1.8，夹在两个镲中间，nerolette 听出卡了）
                note('Drums', 36, b0+swing(p), 0.06, {0:64, 2:56}.get(p, 40))  # v7：频谱量出底鼓 17–30Hz 超低频、比军鼓响~12dB、比镲响~40dB，挂在顶上百余毫秒→力度砍掉  # v4：三拍半那下改成轻轻的 ghost kick
            for p in [1, 3]:
                note('Drums', 38, b0+p+late, 0.06, (86 if p == 1 else 76)+random.randint(-5,3))  # v4：军鼓整体收一点，四拍比二拍轻
            for j in range(8):
                p = j*0.5
                v = 92 if j % 2 == 0 else 72  # v7：镲原来轻到几乎听不见，提上来
                n = 46 if (j == 7 and k == 3) else 42
                note('Drums', n, b0+swing(p)+random.uniform(0,0.015), 0.2, v+random.randint(-6,6))
        if mel:
            t = b0
            for n, beats in mel[k]:
                if n:
                    st = swing(t-b0); en = swing(t-b0+beats)
                    note('Flute', N(n), b0+st+random.uniform(0,0.02), (en-st)*0.94, 74+random.randint(-6,6))
                t += beats
        bar += 1
for name,(tr,ch,ev) in tracks.items():
    ev.sort(key=lambda e:(e[0],e[1])); now=0
    for t,_,m in ev: m.time=t-now; now=t; tr.append(m)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lamp-is-low-5.mid')
mid.save(out); print(out, f'{mid.length:.1f}s', bar, 'bars')
