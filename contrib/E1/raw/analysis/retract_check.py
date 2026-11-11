"""Why the decomposition of inc@block0 by subtraction is not valid."""
import json
from pathlib import Path
M=Path('work/E1/measurements'); GiB=1024**3
print("inc@block0 = layers[0].hwm - rss_loaded, read in the OUTPUT hook of block 0,")
print("so it ALREADY INCLUDES the attention transient of layer 0.")
print()
print(f"{'dur':>5s} {'T':>5s} {'inc@block0':>11s} {'128*T^2':>10s} {'residual':>10s} {'inc step':>10s}")
print("-"*62)
prev=None
for s in (15,30,45,60,90,120):
    d=json.load(open(M/f'after-phoneticxeus-{s:03d}s.json'))
    T=d['frames']; inc=d['layers'][0]['hwm']-d['rss_loaded']; at=128*T*T
    step = f"{(inc-prev)/GiB:+10.3f}" if prev is not None else f"{'-':>10s}"
    print(f"{s:4d}s {T:5d} {inc/GiB:10.3f}G {at/GiB:9.3f}G {(inc-at)/GiB:9.3f}G {step}")
    prev=inc
print()
print("The 'residual' rises and falls (0.27, 0.39, 0.37, 0.22, 0.21, 0.25). It is")
print("neither constant nor linear, which shows the subtraction is not valid: VmHWM")
print("is a MAXIMUM over phases and not a sum of contributions.")
print()
print("What survives as an observation: the steps of inc@block0 are nearly")
print("constant from 15 to 60 s (+0.322, +0.318, +0.322) and jump afterwards")
print("(+1.325, +1.920). That suggests a linear term exceeds 128*T^2 in the short")
print("range, but it does NOT allow that term to be quantified by subtraction.")
