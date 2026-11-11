import json, subprocess, sys
from pathlib import Path
S = Path('work/E1/scores'); H = 'work/wt-after/contrib/E1/compare.py'
print(f"{'dur':>5s} {'comparison':>24s} {'blocks':>8s} {'max_abs_bl':>11s} {'scores':>7s} {'max_abs_nt':>11s} {'heads':>6s}")
print("-"*82)
allok = True
for s in (15, 30, 45, 60):
    for a, b in (('before1', 'after'), ('before1', 'before2')):
        pa, pb = S / f'{a}-{s:03d}s', S / f'{b}-{s:03d}s'
        if not (pa.with_suffix('.json').is_file() and pb.with_suffix('.json').is_file()):
            print(f"{s:4d}s {a + ' vs ' + b:>24s}   MISSING"); allok = False; continue
        r = subprocess.run([sys.executable, H, 'identity', str(pa), str(pb)],
                           capture_output=True, text=True)
        d = json.loads(r.stdout)
        bw = all(v['bitwise'] for v in d['blocks'].values())
        mb = max(v['max_abs'] for v in d['blocks'].values())
        allok &= bw and d['scores_bitwise'] and mb == 0.0 and d['scores_max_abs'] == 0.0
        print(f"{s:4d}s {a + ' vs ' + b:>24s} {str(bw):>8s} {mb:>11.1f} "
              f"{str(d['scores_bitwise']):>7s} {d['scores_max_abs']:>11.1f} {d['heads']:>6d}")
print()
print("BITWISE IDENTITY AT EVERY DURATION AND COMPARISON:", allok)
