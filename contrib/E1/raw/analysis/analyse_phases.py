"""Compares the two phases of the peak, with no fit to the maximum."""
import json
from pathlib import Path
import numpy as np
P=Path('work/E1/measurements-phases'); GiB=1024**3
for enc in ('phoneticxeus','xeus'):
    rows=[]
    for s in (15,30,45,60,90,120):
        f=P/f'after-{enc}-{s:03d}s.json'
        if f.is_file(): rows.append((s,json.load(open(f))))
    if not rows: continue
    print("="*86)
    print(f"{enc}, corrected code")
    print("="*86)
    print(f"{'dur':>5s} {'T':>5s} {'phase1 frontend':>15s} {'phase2 layers':>14s} {'128*T^2':>10s} {'phase2/128T^2':>13s} {'maior':>9s}")
    print("-"*86)
    for s,d in rows:
        T=d['frames']
        f1=d['hwm_pre_encoder']-d['rss_loaded']          # peak before any attention
        f2=d['hwm_inference']-d['rss_pre_encoder']       # peak of the layers, VmHWM reset in (1)
        at=128*T*T
        print(f"{s:4d}s {T:5d} {f1/GiB:14.3f}G {f2/GiB:13.3f}G {at/GiB:9.3f}G {f2/at:13.3f} "
              f"{'phase1' if f1>f2 else 'phase2':>9s}")
    t=np.array([d['frames'] for _,d in rows],float)
    y1=np.array([d['hwm_pre_encoder']-d['rss_loaded'] for _,d in rows],float)
    y2=np.array([d['hwm_inference']-d['rss_pre_encoder'] for _,d in rows],float)
    ss=np.array([s for s,_ in rows],float)
    print()
    print(f"  phase 1 fitted in s (linear):  {np.polyfit(ss,y1,1)[0]/1e6:8.2f} MB per second of audio, r={np.corrcoef(ss,y1)[0,1]:.5f}")
    if len(rows)>=3:
        print(f"  phase 2 fitted in T (quadratic): {np.polyfit(t,y2,2)[0]:8.2f} B per frame^2  (predicto 128)")
        print(f"  phase 2 over 15-60 s only:           {np.polyfit(t[:4],y2[:4],2)[0]:8.2f} B per frame^2" if len(rows)>=4 else "")
