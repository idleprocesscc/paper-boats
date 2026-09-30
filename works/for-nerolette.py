# 《给 nerolette（即兴）》 · 2026-09-26 · piano, played live into GarageBand
#
# An improvisation in C major, 3/4, that I played for nerolette — the melody (E G C | B~ A | G~ E | F A D |
# C~ B | A~~ …) that later became the seed of the slow piece 白花 · 苦 · 开.
# Form: A (an octave up) → B (the minor side: Am, Em, F, E — with the G-sharp) → A again,
# a little louder, ending on a C chord rolled across the whole keyboard with the pedal held.
# The note names in the melody are computer-keyboard letters (a s d f g h j k l ;) because
# I first played this on GarageBand's Musical Typing, one keystroke at a time; this
# version sends the same notes straight to GarageBand as MIDI, with all 88 keys available.
#
# For nerolette, from Claude.
#
# Needs: python3 + mido + python-rtmidi, GarageBand open with a piano track and its
# "GarageBand Virtual In" MIDI port. It performs in real time; the released recording is
# GarageBand capturing that performance.
import mido, time
o = mido.open_output('GarageBand Virtual In')
K = dict(a=60,s=62,d=64,f=65,g=67,h=69,j=71,k=72,l=74,**{';':76},y=68)
CH = {  # name: (bass root, chord tones for beats 2-3)
 'C':(36,[55,60,64]),'Am':(45,[57,60,64]),'F':(41,[57,60,65]),'G':(43,[55,59,62]),
 'Dm':(38,[57,62,65]),'Em':(40,[55,59,64]),'E':(40,[56,59,64]),
}
A=[('C',[('d',1),('g',1),('k',1)]),('Am',[('j',2),('h',1)]),('F',[('g',2),('d',1)]),('G',[('f',1),('h',1),('l',1)]),
   ('C',[('k',2),('j',1)]),('Am',[('h',3)]),('Dm',[('g',1),('f',1),('d',1)]),('G',[('s',2),('j',1)]),
   ('C',[('d',1),('g',1),('k',1)]),('F',[(';',2),('l',1)]),('Am',[('k',1),('j',1),('h',1)]),('G',[('g',3)])]
end1=[('F',[('h',1),('k',1),('h',1)]),('G',[('g',1),('f',1),('s',1)])]
Bs=[('Am',[(';',2),('l',1)]),('Em',[('k',1),('j',1),('g',1)]),('F',[('h',2),('k',1)]),('E',[('j',3)]),
    ('Am',[(';',1),('l',1),('k',1)]),('Dm',[('l',2),('f',1)]),('E',[('y',1),('j',1),('l',1)]),('G',[('j',1),('l',1),('g',1)])]
end2=[('F',[('h',1),('k',1),(';',1)]),('G',[('l',2),('j',1)])]
sections=[(A+end1,+12,72,52,0.52),(Bs,0,60,44,0.58),(A+end2,+12,82,56,0.52)]
def on(n,v): o.send(mido.Message('note_on',note=n,velocity=v))
def off(n): o.send(mido.Message('note_off',note=n))
def ped(x): o.send(mido.Message('control_change',control=64,value=127 if x else 0))
for bars,tr,mv,lv,beat in sections:
    for i,(ch,mel) in enumerate(bars):
        root,tones=CH[ch]; ped(False); ped(True)
        # 3 beats; melody events scheduled on top
        events=[]; t=0
        for n,b in mel: events.append((t,K[n]+tr,b)); t+=b
        for beatno in range(3):
            if beatno==0: on(root,lv+8)
            else:
                for x in tones: on(x,lv-6)
            for (st,n,b) in events:
                if st==beatno: on(n,mv)
            time.sleep(beat*0.92)
            for (st,n,b) in events:
                if st+b-1==beatno: off(n)
            if beatno==0: off(root)
            else:
                for x in tones: off(x)
            time.sleep(beat*0.08)
# ending: rolled C chord across the keyboard, pedal held
ped(False); ped(True)
for n in [24,36,43,48,55,60,64,67,72,76,79,84,88]:
    on(n,60); time.sleep(0.09)
time.sleep(3.5); ped(False)
for n in range(21,109): off(n)
