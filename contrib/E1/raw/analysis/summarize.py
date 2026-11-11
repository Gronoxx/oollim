"""Consolidates the 30 E1 measurements: fits, per-layer exactness and controls."""
import json
from pathlib import Path
import numpy as np
M = Path('work/E1/measurements')
GiB = 1024**3

def load(arm, enc, secs):
    out = []
    for s in secs:
        p = M / f'{arm}-{enc}-{s:03d}s.json'
        if p.is_file(): out.append(json.load(open(p)))
    return out

def fit(rows):
    t = np.array([r['frames'] for r in rows], float)
    y = np.array([r['hwm_inference'] - r['rss_loaded'] for r in rows], float)
    if len(rows) < 3: return None
    c = np.polyfit(t, y, 2)
    return dict(quad=c[0], lin=c[1], const=c[2], n=len(rows),
                range=f'{int(rows[0]["seconds"] if "seconds" in rows[0] else 0)}',
                at120=float(np.polyval(c, 5999)))

print("="*78)
print("QUADRATIC FITS OF THE PEAK INCREASE (bytes per frame^2)")
print("="*78)
print(f"{'encoder':14s} {'arm':7s} {'range':10s} {'n':>2s} {'quadratic':>11s} {'linear':>11s} {'const (MB)':>11s}")
print("-"*78)
cases = [
    ('phoneticxeus','before',[15,30,45,60]),
    ('phoneticxeus','after',[15,30,45,60]),
    ('phoneticxeus','after',[15,30,45,60,90,120]),
    ('xeus','before',[15,30,45,60]),
    ('xeus','after',[15,30,45,60]),
    ('xeus','after',[15,30,45,60,90,120]),
    ('xlsr','before',[15,30,45,60]),
    ('xlsr','after',[15,30,45,60]),
    ('xlsr','after',[15,30,45,60,90,120]),
]
for enc, arm, secs in cases:
    rows = load(arm, enc, secs)
    f = fit(rows)
    if not f: continue
    rng = f'{secs[0]}-{secs[-1]}s'
    print(f"{enc:14s} {arm:7s} {rng:10s} {f['n']:2d} {f['quad']:11.2f} {f['lin']:11.0f} {f['const']/1e6:11.1f}")

print()
print("="*78)
print("EXACTNESS OF THE PER-LAYER RETENTION (phoneticxeus, before arm)")
print("="*78)
allok = True
for r in load('before','phoneticxeus',[15,30,45,60]):
    T = r['frames']; per = 8*T*T*4
    bad = [(x['layer'], x['retained_attention_bytes'], (x['layer']+1)*per)
           for x in r['layers'] if x['retained_attention_bytes'] != (x['layer']+1)*per]
    allok &= not bad
    tot = r['layers'][-1]['retained_attention_bytes']
    print(f"T={T:5d}  19*32*T^2 = {19*per:>15,}  observed = {tot:>15,}  divergent layers: {len(bad)}")
print(f"\nALL 19 LAYERS EXACT AT EVERY DURATION: {allok}")

print()
print("="*78)
print("RETENTION IN THE AFTER ARM (must be zero at every layer)")
print("="*78)
for enc in ('phoneticxeus','xeus','xlsr'):
    nz = 0; n = 0
    for r in load('after',enc,[15,30,45,60,90,120]):
        for x in r['layers']:
            n += 1
            if x['retained_attention_bytes'] != 0: nz += 1
    print(f"{enc:14s} layers inspected={n:4d}  with nonzero retention={nz}")

print()
print("="*78)
print("CONTROLS: peak increase before against after (GiB)")
print("="*78)
print(f"{'encoder':14s} {'dur':>5s} {'before':>8s} {'after':>8s} {'delta':>8s} {'delta %':>8s}")
print("-"*78)
for enc in ('xeus','xlsr','phoneticxeus'):
    for s in (15,30,45,60):
        b = load('before',enc,[s]); a = load('after',enc,[s])
        if not b or not a: continue
        bi = (b[0]['hwm_inference']-b[0]['rss_loaded'])/GiB
        ai = (a[0]['hwm_inference']-a[0]['rss_loaded'])/GiB
        print(f"{enc:14s} {s:4d}s {bi:8.3f} {ai:8.3f} {ai-bi:+8.3f} {100*(ai-bi)/bi:+7.1f}%")

print()
print("="*78)
print("TOTAL PEAK AND EXTRAPOLATION (phoneticxeus)")
print("="*78)
for arm in ('before','after'):
    for r in load(arm,'phoneticxeus',[15,30,45,60,90,120]):
        print(f"{arm:7s} {r['seconds'] if 'seconds' in r else '':>4} T={r['frames']:5d} "
              f"rss_loaded={r['rss_loaded']/GiB:6.3f}GiB hwm={r['hwm_inference']/GiB:6.3f}GiB "
              f"inference={r['inference_seconds']:6.1f}s")
fb = fit(load('before','phoneticxeus',[15,30,45,60]))
print(f"\nbefore arm extrapolated to 120 s: peak increase = {fb['at120']/GiB:.2f} GiB")
print(f"  plus the PhoneticXeus weights (2.142 GiB) = {(fb['at120']+2300089432)/GiB:.2f} GiB")
print(f"  the manuscript records 22.52 GiB of peak RSS with 5.51 GiB of swap, on a host of about 24 GiB")
