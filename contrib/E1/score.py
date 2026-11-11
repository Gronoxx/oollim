"""Release-path features and scores of every bundled head, for before/after identity checks."""
import argparse,json,sys,tempfile
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo',type=Path,required=True);p.add_argument('--audio',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True,help='Prefix: writes <prefix>.npz and <prefix>.json');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
 repo=a.repo.resolve();sys.path[:0]=[str(repo)]
 from oollim_app.common import ROOT,RESOURCES,imports_path,read_reference
 if ROOT!=repo:raise ValueError('Imported oollim_app from another checkout: '+str(ROOT))
 imports_path()
 import numpy as np,torch
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(a.threads)
 from oollim_app.engine import decode,load_bundle
 from oollim_app.scoring import FAMILIES,SELECTIONS,bundle_path
 from pronunciation.inference_memory import install
 from pronunciation.fusion import UnifiedScorer,DIMENSIONS
 cleanup=install();meta=json.loads((ROOT/'configs/extractor.json').read_text());reference=read_reference(a.reference)
 extractor=UnifiedScorer.__new__(UnifiedScorer);extractor.metadata=meta;extractor.blocks=['xeus','xlsr','phoneticxeus','ipa']
 extractor.widths=meta['block_widths'];extractor.width=sum(extractor.widths[k] for k in extractor.blocks)
 with tempfile.TemporaryDirectory() as temp:
  # Same FFmpeg decoding as the application before feature extraction.
  decoded=Path(temp)/'decoded.wav';duration=decode(a.audio,decoded)
  values,status=extractor.extract(decoded,reference,RESOURCES)
 cleanup.remove()
 blocks,offset={},0
 for k in extractor.blocks:blocks[k]=values[:,offset:offset+extractor.widths[k]];offset+=extractor.widths[k]
 scores={}
 for family in FAMILIES:
  for selection in SELECTIONS:
   path=bundle_path(family,selection);cfg=load_bundle(path)
   for j,d in enumerate(DIMENSIONS):
    head=UnifiedScorer(path.parent/cfg['heads'][d]);x=np.concatenate([blocks[n] for n in head.blocks],axis=1)
    scores[f'{family}/{selection}/{d}']=float(head.predict_features(x)[0,j])
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez(a.output.with_suffix('.npz'),**{k:v[0] for k,v in blocks.items()})
 a.output.with_suffix('.json').write_text(json.dumps(dict(repo=str(repo),audio=str(a.audio),duration_seconds=duration,threads=a.threads,status=status,scores=scores),indent=1,default=str)+'\n')

if __name__=='__main__':main()
