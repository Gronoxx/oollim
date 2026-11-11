"""Checks that the diff of each essential branch against main touches only what is declared."""
import json,subprocess,sys

# Files whose hashes are recorded in the checkpoint provenance. If any of them
# changes, the released heads stop loading.
PROVENANCE=('encoders.py','scoring.py','phonetic.py','audio.py','phone_features.py','text_reference.py')
EXPECTED={
 'essential/memory-leak':{
  # The snapshot entry is the recorded hash of the file the fix edits, so it
  # goes stale with it. The card declares the update and it has its own commit.
  'release_modified':{'speech/pronunciation/inference_memory.py','docs/source-snapshot.json'},
  'prefixes':('contrib/E1/','tests/test_inference_memory.py')},
 'essential/dataset-audit':{
  'release_modified':set(),
  'prefixes':('contrib/E2/',)}}

def run(*args):
 return subprocess.run(['git',*args],capture_output=True,text=True,check=True).stdout

report={}
ok=True
for branch,spec in EXPECTED.items():
 lines=[l for l in run('diff','--name-status',f'main..{branch}').splitlines() if l.strip()]
 entries=[tuple(l.split('\t',1)) for l in lines]
 added={p for s,p in entries if s=='A'}
 modified={p for s,p in entries if s=='M'}
 deleted={p for s,p in entries if s=='D'}
 undeclared=[p for p in added|modified if not p.startswith(spec['prefixes']) and p not in spec['release_modified']]
 provenance_touched=[p for p in modified|added|deleted
                     if p.startswith('speech/pronunciation/') and p.rsplit('/',1)[1] in PROVENANCE]
 verdict=(modified==spec['release_modified'] and not undeclared and not deleted and not provenance_touched)
 ok&=verdict
 report[branch]={'added':sorted(added),'modified':sorted(modified),'deleted':sorted(deleted),
  'undeclared':sorted(undeclared),'provenance_files_touched':sorted(provenance_touched),
  'release_files_modified_expected':sorted(spec['release_modified']),
  'verdict':'only declared files' if verdict else 'UNDECLARED CHANGES'}
report['all_branches_clean']=bool(ok)
report['provenance_files_checked']=list(PROVENANCE)
print(json.dumps(report,indent=2))
sys.exit(0 if ok else 1)
