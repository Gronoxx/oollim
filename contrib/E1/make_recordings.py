"""Fixed-length public recordings from SpeechOcean762 train utterances, for memory tests only."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'speech'))
from pronunciation.text_reference import load_dictionary,text_to_phones

def table(p):return dict(line.split(maxsplit=1) for line in p.read_text().splitlines() if line.strip())

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--speechocean',type=Path,required=True,help='Unpacked OpenSLR 101 directory with train/, resource/ and WAVE/')
 p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,nargs='+',default=[15,30,45,60,90,120]);a=p.parse_args()
 wavs=table(a.speechocean/'train/wav.scp');scores=json.loads((a.speechocean/'resource/scores.json').read_text())
 dictionary=load_dictionary(ROOT/'speech/resources/text-reference-v1/cmudict.dict')
 # Utterance-ID order; skip references the release dictionary cannot cover, as the app rejects them.
 ids=[u for u in sorted(wavs) if text_to_phones(scores[u]['text'],dictionary)[1]['complete']]
 a.output.mkdir(parents=True,exist_ok=True);rows=[]
 for seconds in a.seconds:
  target=seconds*16000;parts,texts,used=[],[],[]
  for u in ids:
   audio,rate=sf.read(a.speechocean/wavs[u],dtype='int16')
   if rate!=16000 or audio.ndim!=1:raise ValueError('Expected 16 kHz mono SpeechOcean audio: '+u)
   if sum(map(len,parts))+len(audio)>target:break
   parts.append(audio);texts.append(scores[u]['text'].strip());used.append(u)
  audio=np.concatenate(parts);audio=np.concatenate([audio,np.zeros(target-len(audio),dtype=np.int16)])
  wav=a.output/f'so-{seconds:03d}s.wav';sf.write(wav,audio,16000,subtype='PCM_16');(a.output/f'so-{seconds:03d}s.txt').write_text(' '.join(texts)+'\n')
  rows.append(dict(seconds=seconds,samples=target,frames=(target-400)//320+1,utterances=used,padding_samples=int(target-sum(map(len,parts))),wav_sha256=hashlib.sha256(wav.read_bytes()).hexdigest()))
 (a.output/'recordings.json').write_text(json.dumps(dict(source='SpeechOcean762 train/wav.scp, OpenSLR 101, CC BY 4.0',rule='utterance-ID order, whole utterances, CMUdict-covered references, zero padding to the target length',recordings=rows),indent=2)+'\n')

if __name__=='__main__':main()
