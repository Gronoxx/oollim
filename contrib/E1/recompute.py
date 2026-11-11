"""Recomputes every figure this module publishes, from the raw files in raw/.

Reads nothing but `raw/`, writes `results/recomputed.json`, and prints a report.
The point is that no number in README.md, FINDINGS.md or PHASES.md should exist
only as a table cell: each one is produced here from the measurement files the
harness wrote.

Units. Memory is in bytes internally. GiB means 1024**3 and MB means 10**6, and
both appear only where labelled. Quadratic coefficients are bytes per frame
squared, linear coefficients bytes per frame, constants bytes.
"""
import argparse,itertools,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
GiB=1024**3;MB=10**6
DURATIONS=(15,30,45,60,90,120)
SHORT=(15,30,45,60)
ENCODERS=('xeus','xlsr','phoneticxeus')
# PhoneticXeus weights as loaded, from the safetensors file size.
PHONETIC_WEIGHT_BYTES=2300089432

def load(raw,sub,arm,enc,seconds):
 rows=[]
 for s in seconds:
  p=raw/sub/f'{arm}-{enc}-{s:03d}s.json'
  if p.is_file():
   d=json.load(open(p));d['seconds']=s;rows.append(d)
 return rows

def quadratic(rows):
 """Peak increase above the loaded resident set, against frame count."""
 if len(rows)<3:return None
 t=np.array([r['frames'] for r in rows],float)
 y=np.array([r['hwm_inference']-r['rss_loaded'] for r in rows],float)
 c=np.polyfit(t,y,2)
 return {'quadratic_bytes_per_frame2':float(c[0]),'linear_bytes_per_frame':float(c[1]),
  'constant_bytes':float(c[2]),'constant_mb':float(c[2]/MB),'points':len(rows),
  'seconds':[r['seconds'] for r in rows],'frames':t.tolist(),
  'increase_bytes':y.tolist(),'increase_gib':(y/GiB).tolist(),
  'at_5999_frames_bytes':float(np.polyval(c,5999)),'at_5999_frames_gib':float(np.polyval(c,5999)/GiB)}

def sensitivity(frames):
 """d(quadratic coefficient)/d(y_i) for a least squares quadratic."""
 t=np.asarray(frames,float);V=np.vander(t,3)
 return np.linalg.pinv(V)[0]

def main():
 p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
 p.add_argument('--raw',type=Path,default=HERE/'raw')
 p.add_argument('--output',type=Path,default=HERE/'results/recomputed.json')
 a=p.parse_args();raw=a.raw
 out={'units':{'GiB':1024**3,'MB':10**6,'quadratic':'bytes per frame squared',
   'linear':'bytes per frame','constant':'bytes'}}

 # ---- recordings, so the frame counts every fit uses are traceable
 rec=json.loads((raw/'recordings/recordings.json').read_text())
 out['recordings']=[{k:r[k] for k in ('seconds','samples','frames','wav_sha256')} for r in rec['recordings']]
 out['frame_rule']='T = (samples - 400) // 320 + 1'
 out['frame_rule_holds']=all(r['frames']==(r['samples']-400)//320+1 for r in rec['recordings'])

 # ---- quadratic fits
 fits={}
 for enc in ENCODERS:
  for arm,seconds,label in (('before',SHORT,'15-60'),('after',SHORT,'15-60'),('after',DURATIONS,'15-120')):
   rows=load(raw,'measurements',arm,enc,seconds)
   f=quadratic(rows)
   if f:fits[f'{enc}/{arm}/{label}']=f
 out['quadratic_fits']=fits

 # ---- per-layer retention, byte exact
 retention=[];exact=True
 for r in load(raw,'measurements','before','phoneticxeus',SHORT):
  T=r['frames'];per_layer=8*T*T*4
  divergent=[x['layer'] for x in r['layers'] if x['retained_attention_bytes']!=(x['layer']+1)*per_layer]
  exact&=not divergent
  retention.append({'seconds':r['seconds'],'frames':T,'bytes_per_layer':per_layer,
   'cumulative_after_last_layer':r['layers'][-1]['retained_attention_bytes'],
   'predicted_19_times_32_T2':19*per_layer,'divergent_layers':divergent})
 out['retention']={'rule':'layer i holds 32*T^2 bytes; the cumulative total after layer i is (i+1)*32*T^2',
  'per_duration':retention,'byte_exact_everywhere':bool(exact)}
 zero={}
 for enc in ENCODERS:
  n=nz=0
  for r in load(raw,'measurements','after',enc,DURATIONS):
   for x in r['layers']:
    n+=1;nz+=int(x['retained_attention_bytes']!=0)
  zero[enc]={'layers_inspected':n,'with_retention':nz}
 out['retention']['after_arm']=zero

 # ---- controls and peak reduction
 controls=[];reduction=[]
 for enc in ENCODERS:
  for s in SHORT:
   b=load(raw,'measurements','before',enc,[s]);c=load(raw,'measurements','after',enc,[s])
   if not b or not c:continue
   bi=b[0]['hwm_inference']-b[0]['rss_loaded'];ai=c[0]['hwm_inference']-c[0]['rss_loaded']
   row={'encoder':enc,'seconds':s,'before_gib':bi/GiB,'after_gib':ai/GiB,
    'delta_gib':(ai-bi)/GiB,'delta_percent':100*(ai-bi)/bi}
   (reduction if enc=='phoneticxeus' else controls).append(row)
 out['controls']=controls;out['peak_reduction_phoneticxeus']=reduction

 # ---- total peaks, extrapolation and its uncertainty
 peaks=[]
 for arm in ('before','after'):
  for r in load(raw,'measurements',arm,'phoneticxeus',DURATIONS):
   peaks.append({'arm':arm,'seconds':r['seconds'],'frames':r['frames'],
    'rss_loaded_gib':r['rss_loaded']/GiB,'hwm_inference_gib':r['hwm_inference']/GiB,
    'inference_seconds':r['inference_seconds']})
 out['total_peaks']=peaks
 before=load(raw,'measurements','before','phoneticxeus',SHORT)
 fb=quadratic(before);increase=fb['at_5999_frames_bytes']
 loaded=np.array([r['rss_loaded'] for r in before],float)
 # XEUS is untouched by the change, so its two arms are replicates of the same
 # code. Their spread is the only replicate evidence these runs contain.
 replicate={}
 for s in SHORT:
  b=load(raw,'measurements','before','xeus',[s]);c=load(raw,'measurements','after','xeus',[s])
  if b and c:
   bi=b[0]['hwm_inference']-b[0]['rss_loaded'];ai=c[0]['hwm_inference']-c[0]['rss_loaded']
   replicate[s]={'before_gib':bi/GiB,'after_gib':ai/GiB,'abs_gap_gib':abs(ai-bi)/GiB,'abs_gap_bytes':abs(ai-bi)}
 out['xeus_replicates']=replicate
 def refit_at(index,delta):
  """Moves one observation and refits, rather than scaling a coefficient.

  Moving a point changes the linear and the constant term as well as the
  quadratic one, so propagating only the quadratic shift by T squared
  overstates the effect. The full refit is the honest figure.
  """
  t=np.array(fb['frames'],float);y=np.array(fb['increase_bytes'],float).copy()
  y[index]+=delta
  return float(np.polyval(np.polyfit(t,y,2),5999))
 sens=sensitivity(fb['frames'])
 gap60=replicate.get(60,{}).get('abs_gap_bytes',0.0)
 coefficient_shift=abs(sens[0])*gap60
 refit_up=refit_at(3,gap60);refit_down=refit_at(3,-gap60)
 full_refit_uncertainty=max(abs(refit_up-increase),abs(refit_down-increase))
 # The 15 s loaded resident set runs 2.852 GiB against 2.844 to 2.846 for the
 # other three. The excess enters the fit as a reduction of the increase there.
 loaded_others=float(np.median(loaded[1:]))
 excess=float(loaded[0]-loaded_others)
 outlier_refit=refit_at(0,excess)
 out['extrapolation']={
  'full_refit_uncertainty_gib':full_refit_uncertainty/GiB,
  'full_refit_high_gib':refit_up/GiB,'full_refit_low_gib':refit_down/GiB,
  'outlier_excess_bytes':excess,'outlier_excess_mb':excess/MB,
  'outlier_quadratic_shift_bytes_per_frame2':abs(sens[0])*excess,
  'outlier_full_refit_shift_gib':abs(outlier_refit-increase)/GiB,
  'fit':'before arm, 15 to 60 s, four points',
  'increase_at_5999_frames_gib':increase/GiB,
  'plus_weights_only_gib':(increase+PHONETIC_WEIGHT_BYTES)/GiB,
  'plus_loaded_rss_gib':(increase+float(loaded.mean()))/GiB,
  'loaded_rss_mean_gib':float(loaded.mean())/GiB,
  'loaded_rss_values_gib':(loaded/GiB).tolist(),
  'weights_gib':PHONETIC_WEIGHT_BYTES/GiB,
  'sensitivity_of_quadratic_to_each_point':sens.tolist(),
  'replicate_gap_at_60s_bytes':gap60,
  'quadratic_uncertainty_bytes_per_frame2':coefficient_shift,
  'uncertainty_at_5999_frames_gib_coefficient_only':coefficient_shift*5999**2/GiB,
  'uncertainty_at_5999_frames_gib_coefficient_only_method':'the shift of the quadratic coefficient alone, scaled by T squared. It overstates the uncertainty by a factor of about 1.75, because moving one observation also changes the linear and the constant term. Reported only to show what the shortcut gives. The figure to quote is full_refit_uncertainty_gib.',
  'note':'the comparable figure is the increase plus the loaded resident set, because the measured 120 s peak is a total resident high water mark'}

 # ---- the replicate swap, and which direction it moves the coefficient
 swaps={}
 for arm,other in (('before','after'),('after','before')):
  rows=load(raw,'measurements',arm,'xeus',SHORT)
  base=quadratic(rows)
  swapped=[dict(r) for r in rows]
  donor=load(raw,'measurements',other,'xeus',[60])
  if not donor or not base:continue
  swapped[-1]['hwm_inference']=donor[0]['hwm_inference']
  swapped[-1]['rss_loaded']=donor[0]['rss_loaded']
  s=quadratic(swapped)
  swaps[f'{arm}_with_60s_from_{other}']={'original':base['quadratic_bytes_per_frame2'],
   'after_swap':s['quadratic_bytes_per_frame2'],
   'movement':s['quadratic_bytes_per_frame2']-base['quadratic_bytes_per_frame2']}
 out['replicate_swap']=swaps

 # ---- phases, with the baseline made common
 phases={}
 for enc in ('phoneticxeus','xeus'):
  rows=load(raw,'measurements-phases','after',enc,DURATIONS)
  if len(rows)<3:continue
  s=np.array([r['seconds'] for r in rows],float)
  t=np.array([r['frames'] for r in rows],float)
  loaded=np.array([r['rss_loaded'] for r in rows],float)
  pre_peak=np.array([r['hwm_pre_encoder'] for r in rows],float)
  pre_rss=np.array([r['rss_pre_encoder'] for r in rows],float)
  block_peak=np.array([r['hwm_inference'] for r in rows],float)
  # As published: phase one above rss_loaded, phase two above rss_pre_encoder.
  published_one=pre_peak-loaded;published_two=block_peak-pre_rss
  # Corrected: both above rss_loaded, the state before any audio work. Each is
  # then the peak that phase reaches, and the overall peak is their maximum.
  common_one=pre_peak-loaded;common_two=block_peak-loaded
  def cross(x,y):
   diff=y-x
   k=[i for i in range(len(s)-1) if diff[i]<0<=diff[i+1]]
   if not k:return None
   i=k[0]
   return float(s[i]-diff[i]*(s[i+1]-s[i])/(diff[i+1]-diff[i]))
  slope,intercept=np.polyfit(s,common_one,1)
  q=np.polyfit(t,common_two,2)
  qn=np.polyfit(t[:4],common_two[:4],2)
  phases[enc]={'seconds':s.tolist(),'frames':t.tolist(),
   'baseline_gap_rss_pre_encoder_minus_rss_loaded_gib':((pre_rss-loaded)/GiB).tolist(),
   'phase_one_above_loaded_gib':(common_one/GiB).tolist(),
   'phase_two_above_loaded_gib':(common_two/GiB).tolist(),
   'phase_two_as_published_gib':(published_two/GiB).tolist(),
   'phase_one_slope_mb_per_second':float(slope/MB),
   'phase_one_slope_mib_per_second':float(slope/1024**2),
   'phase_one_intercept_mb':float(intercept/MB),
   'phase_one_r':float(np.corrcoef(s,common_one)[0,1]),
   'phase_two_quadratic_wide':float(q[0]),'phase_two_quadratic_narrow':float(qn[0]),
   'crossover_as_published_s':cross(published_one,published_two),
   'crossover_common_baseline_s':cross(common_one,common_two),
   'phase_two_over_phase_one_ratio':(common_two/common_one).tolist()}
 out['phases']=phases

 # ---- identity of the released outputs
 def blocks(path):
  with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
 identity=[]
 for s in SHORT:
  files={arm:raw/'scores'/f'{arm}-{s:03d}s.npz' for arm in ('before1','before2','after')}
  if not all(q.is_file() for q in files.values()):continue
  data={k:blocks(v) for k,v in files.items()}
  for left,right in (('before1','after'),('before1','before2')):
   gaps={k:float(np.abs(data[left][k]-data[right][k]).max()) for k in data[left]}
   identity.append({'seconds':s,'comparison':f'{left} vs {right}','max_abs':max(gaps.values()),
    'per_block':gaps,'bitwise_identical':all(np.array_equal(data[left][k],data[right][k]) for k in data[left])})
 out['identity']={'durations_checked':sorted({r['seconds'] for r in identity}),
  'comparisons':identity,'all_bitwise_identical':all(r['bitwise_identical'] for r in identity),
  'note':'the before arm was not run at 90 or 120 s, so identity is established at 15, 30, 45 and 60 s only'}
 # pooled_sha256 agreement between the main runs and the phase runs
 agree=[]
 for enc in ('phoneticxeus','xeus'):
  for s in DURATIONS:
   m=load(raw,'measurements','after',enc,[s]);f=load(raw,'measurements-phases','after',enc,[s])
   if m and f:agree.append({'encoder':enc,'seconds':s,'equal':m[0]['pooled_sha256']==f[0]['pooled_sha256']})
 out['identity']['phase_runs_reproduce_pooled_sha256']={'checked':len(agree),
  'all_equal':all(r['equal'] for r in agree),'rows':agree}

 # ---- the loaded resident set across the before arm, for the 15 s outlier
 out['loaded_rss_before_arm_gib']={enc:{r['seconds']:r['rss_loaded']/GiB
   for r in load(raw,'measurements','before',enc,SHORT)} for enc in ENCODERS}

 a.output.parent.mkdir(parents=True,exist_ok=True)
 a.output.write_text(json.dumps(out,indent=2)+'\n')

 # ---- readable report
 print('UNITS: GiB = 1024^3, MB = 10^6. Quadratic in bytes per frame squared.')
 print(f"frame rule T = (samples-400)//320+1 holds for every recording: {out['frame_rule_holds']}")
 print()
 print('QUADRATIC FITS')
 for k,f in fits.items():
  print(f"  {k:34s} n={f['points']}  quad={f['quadratic_bytes_per_frame2']:9.2f}  "
        f"lin={f['linear_bytes_per_frame']:11.0f}  const={f['constant_mb']:7.1f} MB")
 print()
 r=out['retention']
 print(f"RETENTION byte exact at every layer and duration: {r['byte_exact_everywhere']}")
 for d in r['per_duration']:
  print(f"  T={d['frames']:5d}  19*32*T^2={d['predicted_19_times_32_T2']:>15,}  "
        f"observed={d['cumulative_after_last_layer']:>15,}  divergent={len(d['divergent_layers'])}")
 for enc,v in r['after_arm'].items():
  print(f"  after arm {enc:14s} layers={v['layers_inspected']:4d} with retention={v['with_retention']}")
 print()
 e=out['extrapolation']
 print('EXTRAPOLATION of the before arm to 5999 frames')
 print(f"  increase                      {e['increase_at_5999_frames_gib']:7.3f} GiB")
 print(f"  plus weights only             {e['plus_weights_only_gib']:7.3f} GiB   (secondary definition)")
 print(f"  plus loaded resident set      {e['plus_loaded_rss_gib']:7.3f} GiB   (comparable to a measured peak)")
 print(f"  replicate gap at 60 s         {e['replicate_gap_at_60s_bytes']/GiB:7.3f} GiB")
 print(f"  quadratic uncertainty         {e['quadratic_uncertainty_bytes_per_frame2']:7.1f} bytes per frame^2")
 print(f"  uncertainty, coefficient only {e['uncertainty_at_5999_frames_gib_coefficient_only']:7.3f} GiB  (the shortcut, overstates it)")
 print(f"  uncertainty, full refit       {e['full_refit_uncertainty_gib']:7.3f} GiB  (the honest figure)")
 print(f"  outlier excess at 15 s        {e['outlier_excess_mb']:7.2f} MB")
 print(f"    quadratic shift             {e['outlier_quadratic_shift_bytes_per_frame2']:7.2f} bytes per frame^2")
 print(f"    extrapolation shift         {e['outlier_full_refit_shift_gib']:7.3f} GiB  (full refit)")
 print()
 print('REPLICATE SWAP of the XEUS 60 s point')
 for k,v in out['replicate_swap'].items():
  print(f"  {k:36s} {v['original']:8.2f} -> {v['after_swap']:8.2f}   movement {v['movement']:+8.2f}")
 print()
 print('PHASES')
 for enc,v in phases.items():
  print(f"  {enc}")
  print(f"    phase one slope        {v['phase_one_slope_mb_per_second']:7.2f} MB/s  "
        f"({v['phase_one_slope_mib_per_second']:.2f} MiB/s)  r={v['phase_one_r']:.6f}  "
        f"intercept {v['phase_one_intercept_mb']:.1f} MB")
  print(f"    phase two quadratic    wide {v['phase_two_quadratic_wide']:7.2f}  narrow {v['phase_two_quadratic_narrow']:7.2f}")
  print(f"    crossover as published {v['crossover_as_published_s']}")
  print(f"    crossover same baseline{v['crossover_common_baseline_s']}")
  print(f"    baseline gap (GiB)     {[round(x,4) for x in v['baseline_gap_rss_pre_encoder_minus_rss_loaded_gib']]}")
  print(f"    phase two / phase one  {[round(x,3) for x in v['phase_two_over_phase_one_ratio']]}")
 print()
 i=out['identity']
 print(f"IDENTITY durations checked {i['durations_checked']}, all bitwise identical: {i['all_bitwise_identical']}")
 for c in i['comparisons']:
  print(f"  {c['seconds']:3d}s {c['comparison']:22s} max_abs={c['max_abs']:.1f}  bitwise={c['bitwise_identical']}")
 pa=i['phase_runs_reproduce_pooled_sha256']
 print(f"  phase runs reproduce pooled_sha256: {pa['all_equal']} over {pa['checked']} runs")
 print()
 print('LOADED RESIDENT SET, before arm (GiB)')
 for enc,v in out['loaded_rss_before_arm_gib'].items():
  print(f"  {enc:14s} "+'  '.join(f'{s}s={x:.3f}' for s,x in v.items()))
 print()
 print(f'written: {a.output}')
if __name__=='__main__':main()
