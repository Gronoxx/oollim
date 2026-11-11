"""Per-layer retained attention bytes and peak memory for one frozen encoder on one recording.

Imports the encoders of the worktree given by --repo, so the same script measures
the code before and after a change. Linux only: reads and resets /proc/self/status peaks.
"""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path

def proc(key):
 for line in open('/proc/self/status'):
  if line.startswith(key+':'):return int(line.split()[1])*1024
 raise KeyError(key)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo',type=Path,required=True);p.add_argument('--encoder',choices=['xeus','xlsr','phoneticxeus'],required=True)
 p.add_argument('--audio',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4)
 p.add_argument('--device',default='cpu');p.add_argument('--dtype',choices=['float32','float16'],default='float32')
 p.add_argument('--phase-peaks',action='store_true',help='Separate the two phases of the peak. Reads VmHWM and VmRSS in a forward pre-hook of the first encoder block, before any attention has run, then resets VmHWM so that hwm_inference becomes the peak of the encoder blocks alone. Default behaviour is unchanged.')
 a=p.parse_args()
 repo=a.repo.resolve();sys.path[:0]=[str(repo/'speech'),str(repo/'speech/resources/source/wanchichen-espnet-5b52d57')]
 import torch
 torch.set_num_threads(a.threads);torch.set_num_interop_threads(a.threads)
 from pronunciation.inference_memory import install
 from pronunciation.audio import read_audio
 from pronunciation.encoders import EncoderConfig,build_encoder,FrameBatch
 from pronunciation.scoring import masked_pool
 cleanup=install();resources=repo/'speech/resources';dtype=getattr(torch,a.dtype)
 wave=read_audio(a.audio);samples=len(wave)
 started=time.perf_counter()
 if a.encoder=='phoneticxeus':
  from pronunciation.phonetic import PhoneticXeus
  model=PhoneticXeus(resources/'phoneticxeus').eval();core=model.model;blocks=core.encoder.encoders
 else:
  backbone=json.loads((repo/'configs/extractor.json').read_text())['experiment_provenance']['backbones'][a.encoder]['backbone']
  cfg=EncoderConfig(**backbone);cfg.directory=str(resources/Path(cfg.directory).name)
  model=build_encoder(cfg).eval();core=model;blocks=model.encoder.encoders if a.encoder=='xeus' else model.model.encoder.layers
 model.to(a.device,dtype);load_seconds=time.perf_counter()-started
 gc.collect();rss_loaded=proc('VmRSS')
 # Reset VmHWM to the current RSS so the peak below belongs to inference, not weight loading.
 with open('/proc/self/clear_refs','w') as f:f.write('5')
 cuda=a.device.startswith('cuda')
 if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();cuda_loaded=torch.cuda.memory_allocated()
 layers=[]
 def retained():
  return sum(m.attn.numel()*m.attn.element_size() for m in model.modules() if isinstance(getattr(m,'attn',None),torch.Tensor))
 def record(index):
  def hook(module,inputs,output):
   row=dict(layer=index,rss=proc('VmRSS'),hwm=proc('VmHWM'),retained_attention_bytes=retained())
   if cuda:torch.cuda.synchronize();row.update(cuda_allocated=torch.cuda.memory_allocated(),cuda_peak=torch.cuda.max_memory_allocated())
   layers.append(row)
  return hook
 phase={}
 if a.phase_peaks:
  def phase_split(module,inputs):
   # Before any attention: the peak so far belongs to the front end,
   # the pre-encoder and the positional embedding.
   phase['hwm_pre_encoder']=proc('VmHWM');phase['rss_pre_encoder']=proc('VmRSS')
   with open('/proc/self/clear_refs','w') as f:f.write('5')
  handles_pre=[blocks[0].register_forward_pre_hook(phase_split)]
 else:handles_pre=[]
 handles=[block.register_forward_hook(record(i)) for i,block in enumerate(blocks)]
 x=wave.to(a.device,dtype);started=time.perf_counter()
 with torch.inference_mode():
  if a.encoder=='phoneticxeus':
   frames,lengths=core.encode(x[None],torch.tensor([samples],device=a.device))
   if isinstance(frames,tuple):frames=frames[0]
   frames=frames[0,:int(lengths[0])];logits=core.ctc.ctc_lo(frames).float().cpu().numpy()
  else:
   frames=model.encode_one(x);logits=None
  frames=frames.float().cpu()
 if cuda:torch.cuda.synchronize()
 seconds=time.perf_counter()-started
 for h in handles+handles_pre:h.remove()
 cleanup.remove()
 fb=FrameBatch(frames[None],torch.tensor([len(frames)]),torch.ones(1,len(frames),dtype=torch.bool),16000,320,400)
 pooled=masked_pool(fb)[0].numpy()
 result=dict(encoder=a.encoder,repo=str(repo),audio=str(a.audio),samples=samples,frames=len(frames),threads=a.threads,device=a.device,dtype=a.dtype,
  load_seconds=load_seconds,inference_seconds=seconds,rss_loaded=rss_loaded,hwm_inference=proc('VmHWM'),layers=layers,
  phase_peaks=a.phase_peaks,**phase,
  pooled_sha256=hashlib.sha256(pooled.tobytes()).hexdigest(),logits_sha256=hashlib.sha256(logits.tobytes()).hexdigest() if logits is not None else None)
 if cuda:result.update(cuda_loaded=cuda_loaded,cuda_peak=torch.cuda.max_memory_allocated())
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=1)+'\n')
 print(json.dumps({k:result[k] for k in ('encoder','frames','rss_loaded','hwm_inference','inference_seconds')}))

if __name__=='__main__':main()
