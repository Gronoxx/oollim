"""Identity check between score.py outputs, and quadratic fit of measure.py outputs."""
import argparse,json
from pathlib import Path
import numpy as np

HEADS,BYTES=8,4  # PhoneticXeus and XEUS attention: 8 heads, float32

def identity(a,b):
 fa,fb=np.load(a.with_suffix('.npz')),np.load(b.with_suffix('.npz'));sa,sb=[json.loads(p.with_suffix('.json').read_text())['scores'] for p in (a,b)]
 blocks={k:dict(bitwise=bool(np.array_equal(fa[k],fb[k])),max_abs=float(np.abs(fa[k]-fb[k]).max())) for k in fa.files}
 return dict(blocks=blocks,scores_bitwise=all(sa[k]==sb[k] for k in sa),scores_max_abs=max(abs(sa[k]-sb[k]) for k in sa),heads=len(sa))

def fit(paths):
 rows=[json.loads(p.read_text()) for p in paths];t=np.array([r['frames'] for r in rows],float)
 y=np.array([r['hwm_inference']-r['rss_loaded'] for r in rows],float)
 coef=np.polyfit(t,y,2) if len(rows)>=3 else [np.nan]*3
 kept=[r['layers'][-1]['retained_attention_bytes'] for r in rows];layers=len(rows[0]['layers'])
 return dict(frames=t.tolist(),peak_increase_bytes=y.tolist(),quadratic_bytes_per_frame2=float(coef[0]),linear=float(coef[1]),constant=float(coef[2]),
  retained_after_last_layer=kept,retained_formula=[layers*HEADS*BYTES*int(f)**2 for f in t],
  extrapolated_peak_increase_120s=float(np.polyval(coef,5999)) if len(rows)>=3 else None)

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
 i=sub.add_parser('identity');i.add_argument('a',type=Path);i.add_argument('b',type=Path)
 f=sub.add_parser('fit');f.add_argument('measurements',type=Path,nargs='+')
 a=p.parse_args();print(json.dumps(identity(a.a,a.b) if a.command=='identity' else fit(a.measurements),indent=1))

if __name__=='__main__':main()
