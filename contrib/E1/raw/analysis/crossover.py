"""Crossover read from the measured curves, with no model."""
import json
from pathlib import Path
import numpy as np
P=Path('work/E1/measurements-phases'); GiB=1024**3
for enc in ('phoneticxeus','xeus'):
    ss=[];f1=[];f2=[]
    for s in (15,30,45,60,90,120):
        d=json.load(open(P/f'after-{enc}-{s:03d}s.json'))
        ss.append(s); f1.append(d['hwm_pre_encoder']-d['rss_loaded']); f2.append(d['hwm_inference']-d['rss_pre_encoder'])
    ss=np.array(ss,float); f1=np.array(f1,float); f2=np.array(f2,float)
    # crossover by linear interpolation of the difference between the two points that change sign
    dif=f2-f1
    k=[i for i in range(len(ss)-1) if dif[i]<0<=dif[i+1]]
    if k:
        i=k[0]
        x=ss[i]-dif[i]*(ss[i+1]-ss[i])/(dif[i+1]-dif[i])
        print(f"{enc:14s} sign change between {ss[i]:.0f}s e {ss[i+1]:.0f}s, interpolating: {x:.1f}s")
    print(f"{enc:14s} phase 1 slope = {np.polyfit(ss,f1,1)[0]/1e6:.2f} MB/s  r={np.corrcoef(ss,f1)[0,1]:.6f}")
    print(f"{enc:14s} phase 1 intercept = {np.polyfit(ss,f1,1)[1]/1e6:.1f} MB")
