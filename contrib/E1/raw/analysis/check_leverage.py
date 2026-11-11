"""Check 2: the leverage of the 60 s point in the four-point fit."""
import json
from pathlib import Path
import numpy as np
M = Path('work/E1/measurements'); GiB=1024**3

def pts(arm, enc, secs):
    t=[];y=[]
    for s in secs:
        d=json.load(open(M/f'{arm}-{enc}-{s:03d}s.json'))
        t.append(d['frames']); y.append(d['hwm_inference']-d['rss_loaded'])
    return np.array(t,float), np.array(y,float)

print("=== peak increase per run (GiB), xeus ===")
for arm in ('before','after'):
    t,y=pts(arm,'xeus',[15,30,45,60])
    print(f"  {arm:7s}", "  ".join(f"{v/GiB:.3f}" for v in y), f"  -> quad={np.polyfit(t,y,2)[0]:.2f}")

print()
print("=== check 2: is 84 against 128 an effect of the range or of one run? ===")
tb,yb = pts('before','xeus',[15,30,45,60])
ta,ya = pts('after','xeus',[15,30,45,60])
print(f"  xeus before 15-60 : quad = {np.polyfit(tb,yb,2)[0]:7.2f}")
print(f"  xeus after  15-60 : quad = {np.polyfit(ta,ya,2)[0]:7.2f}")
# swap only the 60 s point of the after arm for the one of the before arm
ya2 = ya.copy(); ya2[-1] = yb[-1]
print(f"  xeus after with the 60 s point of before: quad = {np.polyfit(ta,ya2,2)[0]:7.2f}")
yb2 = yb.copy(); yb2[-1] = ya[-1]
print(f"  xeus before with the 60 s point of after : quad = {np.polyfit(tb,yb2,2)[0]:7.2f}")
d60 = (ya[-1]-yb[-1])
print(f"\n  difference at the 60 s point: {d60/GiB:+.3f} GiB between replicates of the SAME code path")

print()
print("=== front-end term: peak increase after block 0 (linear in T?) ===")
for arm,enc in (('after','phoneticxeus'),('after','xeus')):
    print(f"  {arm}-{enc}")
    prev=None
    for s in (15,30,45,60,90,120):
        p=M/f'{arm}-{enc}-{s:03d}s.json'
        if not p.is_file(): continue
        d=json.load(open(p))
        inc0=d['layers'][0]['hwm']-d['rss_loaded']
        step = f"  step={+(inc0-prev)/GiB:+.3f}" if prev is not None else ""
        print(f"    {s:4d}s T={d['frames']:5d} inc@block0={inc0/GiB:6.3f}GiB  MB/s audio={inc0/1e6/s:6.1f}{step}")
        prev=inc0

print()
print("=== front end (linear, 21.9 MiB/s) against attention (128*T^2), in GiB ===")
print(f"{'dur':>5s} {'front end':>9s} {'attention':>9s} {'larger':>12s}")
for s in (15,30,45,60,90,120):
    T=(s*16000-400)//320+1
    fe = 21.9e6*s; at = 128*T*T
    print(f"{s:4d}s {fe/GiB:8.3f}G {at/GiB:8.3f}G {'front end' if fe>at else 'attention':>12s}")
