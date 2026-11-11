"""Check 1: is VmHWM a maximum of the front end (linear) and attention (T^2)?

Compares the hwm already reached after block 0 with the final hwm. If the front
end sets the peak, the two nearly coincide.
"""
import json
from pathlib import Path
M = Path('work/E1/measurements'); GiB = 1024**3
print(f"{'run':30s} {'T':>5s} {'hwm@block0':>11s} {'hwm final':>11s} {'final/b0':>9s} {'peak at block':>14s}")
print("-"*95)
for arm in ('after','before'):
    for enc in ('phoneticxeus','xeus'):
        for s in (15,30,45,60,90,120):
            p = M / f'{arm}-{enc}-{s:03d}s.json'
            if not p.is_file(): continue
            d = json.load(open(p))
            L = d['layers']
            h0 = L[0]['hwm']; hf = d['hwm_inference']
            # at which block the hwm stops growing
            peak_block = max(range(len(L)), key=lambda i: L[i]['hwm'])
            first_at_max = next(i for i,x in enumerate(L) if x['hwm'] >= hf)  if any(x['hwm']>=hf for x in L) else -1
            print(f"{arm+'-'+enc+'-'+str(s)+'s':30s} {d['frames']:5d} "
                  f"{h0/GiB:10.3f}G {hf/GiB:10.3f}G {hf/h0:9.3f} "
                  f"{first_at_max:14d}")
print()
print("'peak at block' = first block whose hwm already reaches the final hwm, -1 = only after the blocks")
