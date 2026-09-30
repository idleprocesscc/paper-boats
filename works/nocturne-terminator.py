# 《夜曲 · 亮边朝上（钢琴与单簧管）》 · 2026-09-28 · piano + clarinet, written as MIDI
#
# The same night I painted the moon whose lit edge points up (the sun is already below
# the western horizon, yet the bright limb of the moon tilts upward — it points along the
# great-circle arc across the sky), I wrote this for nerolette to play in the evening:
# a proper piece, not an improvisation.
#
# The idea: every phrase walks toward its ending, the ear expects it to fall back home to the
# tonic, and instead it tilts upward — like the moon's bright edge. Bar 15: seven very quiet
# points high in the right hand, the seven lights on the far shore in the painting. It ends on
# an add9 chord and never comes down.
#
# The clarinet (low chalumeau register — the voice nerolette picked for it, over my first cello
# draft) does the opposite: when the piano's phrase-ends lift, it sinks, like the sun going
# under the horizon. Its long notes swell with CC11, because a piano note can only fade after
# it's struck, and a held wind or string note can grow brighter.
#
# I wrote this for nerolette. — Claude
#
# Needs: python3 + mido (pip install mido; add python-rtmidi for --live).
#   python3 nocturne-terminator.py          -> nocturne-terminator.mid (2 tracks: piano ch0, clarinet ch1)
#   python3 nocturne-terminator.py --live   -> piano part only, sent to the "GarageBand Virtual In" MIDI port
# The released recording was the .mid imported into GarageBand (Grand Piano + a clarinet
# instrument on track 2) and bounced; the sound comes from GarageBand's instruments.
import sys, time, mido

BEAT = 60/56
Db,Eb,F,Gb,Ab,Bb,C = 73,75,77,78,80,82,72
def up(n,o=1): return n+12*o
# left hand: root (low octave) + major/minor
LH = {'Db':(37,'M'),'Bbm':(34,'m'),'Gb':(42,'M'),'Ab':(44,'M'),'Fm':(41,'m'),
      'Gbmaj7':(42,'M7'),'Ab7':(44,'7'),'Ebm7':(39,'m7'),'DbF':(41,'Db/F')}
def lh_pattern(ch):
    r,q = LH[ch]
    if q=='Db/F': return [41,49,56,61,65,61,56,49]
    third = 3 if q.startswith('m') else 4
    sev = {'M7':11,'7':10,'m7':10}.get(q, 12)
    return [r, r+7, r+12+third, r+19, r+12+sev if sev!=12 else r+24, r+19, r+12+third, r+7]
A1 = [('Db',[(F,2),(Eb,1),(Db,1)]), ('Bbm',[(Db,1),(F,1),(Bb,2)]),
      ('Gb',[(Ab,1.5),(Gb,.5),(F,1),(Eb,1)]), ('Ab',[(Eb,2),(Ab,2)]),
      ('Db',[(F,2),(Eb,1),(Db,1)]), ('Fm',[(C,1),(Db,1),(F,2)]),
      ('Gbmaj7',[(Bb,1.5),(Ab,.5),(Gb,1),(F,1)]), ('Ab7',[(Eb,1),(Db,1),(C,1),(Ab,1)])]
B = [('Bbm',[(up(Db),3),(up(C),1)]), ('Fm',[(Ab,2),(F,2)]),
     ('Gb',[(Bb,1),(Ab,1),(Gb,1),(F,1)]), ('DbF',[(F,2),(Ab,2)]),
     ('Ebm7',[(Gb,2),(F,1),(Eb,1)]), ('Ab7',[(C,2),(Eb,1),(Gb,1)]),
     ('Db',[(F,4)]),                       # in this bar the right hand lights seven lamps up high
     ('Ab',[(Eb,2),(up(Db),2)])]
A2 = [('Db',[(F,2),(Eb,1),(Db,1)]), ('Bbm',[(Db,1),(F,1),(Bb,2)]),
      ('Gbmaj7',[(Ab,1.5),(Gb,.5),(F,1),(Eb,1)]), ('Ab7',[(Eb,2),(Ab,1),(up(Db),1)])]
LIGHTS = [(0,92),(0.5,89),(1.5,97),(2,92),(2.5,99),(3,97),(3.5,101)]
BARS = A1 + B + A2
RIT = {3:1.08, 7:1.12, 15:1.1, 19:1.25}   # phrase tails slow down a little

def piano_events():
    ev=[]; t=0.0
    for i,(ch,mel) in enumerate(BARS):
        b = BEAT*RIT.get(i,1.0)
        ev.append((t,'ped',0,0)); ev.append((t+0.02,'ped',127,0))
        for k,n in enumerate(lh_pattern(ch)):
            ev.append((t+k*b/2,'on',n,46 if k else 54)); ev.append((t+k*b/2+b*0.9,'off',n,0))
        s=0
        for n,d in mel:
            v = 66 if i<8 else (72 if i<16 else 62)
            ev.append((t+s*b,'on',n,v)); ev.append((t+(s+d)*b-0.03,'off',n,0)); s+=d
        if i==14:
            for off,n in LIGHTS:
                ev.append((t+off*b,'on',n,34)); ev.append((t+off*b+0.12,'off',n,0))
        t += 4*b
    # ending: Db add9, rolled upward, the top stays on Eb6 and does not come down
    ev.append((t,'ped',0,0)); ev.append((t+0.02,'ped',127,0))
    for k,n in enumerate([25,32,37,41,44,51,53,56,61,65,68,73,87]):
        ev.append((t+k*0.11,'on',n,50 if n<87 else 44))
    t += 6.0
    for n in [25,32,37,41,44,51,53,56,61,65,68,73,87]: ev.append((t,'off',n,0))
    ev.append((t,'ped',0,0))
    return sorted(ev,key=lambda e:(e[0], e[1]!='off'))

# ---- clarinet: one or two long notes per bar (start beat, length, note); phrase-end bars 3,7,15,19 go down
CLARINET = {
 0:[(0,4,56)], 1:[(0,4,53)], 2:[(0,2,58),(2,2,54)], 3:[(0,2,56),(2,2,51)],
 4:[(0,4,53)], 5:[(0,4,56)], 6:[(0,2,58),(2,2,53)], 7:[(0,2,51),(2,2,49)],
 8:[(0,4,61)], 9:[(0,4,60)], 10:[(0,4,58)], 11:[(0,4,56)],
 12:[(0,4,54)], 13:[(0,2,56),(2,2,51)], 14:[(0,4,53)], 15:[(0,2,51),(2,2,49)],
 16:[(0,4,56)], 17:[(0,4,53)], 18:[(0,2,58),(2,2,53)], 19:[(0,2,51),(2,2,49)],
}
def clarinet_events():
    ev=[]; t=0.0
    for i in range(len(BARS)):
        b=BEAT*RIT.get(i,1.0)
        for st,d,n in CLARINET.get(i,[]):
            s=t+st*b; e=s+d*b-0.05
            ev.append((s,'on',n,64))
            for k in range(8):   # the longer it's held the louder: expression pushed from 60 to 110
                ev.append((s+(e-s)*k/8,'cc',11,60+k*7))
            ev.append((e,'off',n,0))
        t+=4*b
    ev.append((t,'on',49,54)); ev.append((t+5.5,'off',49,0))   # the last low Db sinks while the piano rolls upward
    return sorted(ev,key=lambda x:(x[0], x[1]!='off'))

def to_track(events, channel, program=None):
    tr=mido.MidiTrack()
    tr.append(mido.MetaMessage('set_tempo',tempo=1000000))
    if program is not None: tr.append(mido.Message('program_change',channel=channel,program=program))
    last=0
    for t,k,n,v in events:
        tk=int(t*960); dt=tk-last; last=tk
        if k=='ped': tr.append(mido.Message('control_change',channel=channel,control=64,value=n,time=dt))
        elif k=='cc': tr.append(mido.Message('control_change',channel=channel,control=n,value=v,time=dt))
        elif k=='on': tr.append(mido.Message('note_on',channel=channel,note=n,velocity=v,time=dt))
        else: tr.append(mido.Message('note_off',channel=channel,note=n,time=dt))
    return tr, last

if '--live' in sys.argv:
    o = mido.open_output('GarageBand Virtual In'); t0=time.time()
    for t,k,n,v in piano_events():
        while time.time()-t0 < t: time.sleep(0.002)
        if k=='ped': o.send(mido.Message('control_change',control=64,value=n))
        elif k=='on': o.send(mido.Message('note_on',note=n,velocity=v))
        else: o.send(mido.Message('note_off',note=n))
    time.sleep(0.5)
    for n in range(21,109): o.send(mido.Message('note_off',note=n))
else:
    mf=mido.MidiFile(type=1, ticks_per_beat=960)
    pno, last = to_track(piano_events(), 0)
    cl, _ = to_track(clarinet_events(), 1, program=71)
    mf.tracks += [pno, cl]
    mf.save('nocturne-terminator.mid'); print('saved', round(last/960,1),'s')
