# E1. Release PhoneticXeus attention maps during inference

Version: v1.4 (draft). Branch: `essential/memory-leak`. Base: `cburst/oollim` at `31318b5`.

## What changes

`speech/pronunciation/inference_memory.py` registers a global forward hook that clears the attention probability tensor that ESPnet attention layers keep in `self.attn` after each forward pass. The hook matched only the module path `espnet.nets.pytorch_backend.transformer.attention`, which is the path XEUS uses. PhoneticXeus loads a vendored copy of the same classes from `pxeus.espnet_import.attention`. Its `forward_attention` also stores `self.attn`, so the hook skipped it and every layer kept its map until the forward pass of the whole recording ended.

The change adds `pxeus.espnet_import.attention` to the list of accepted module paths. The class name check and the tensor check stay as they were.

The tensor check is defence in depth rather than load bearing. A PhoneticXeus encoder layer does own a submodule called `attn`, but that layer is `pxeus.model.powsm.e_branchformer.EBranchformerEncoderLayer`, whose module path is not in the accepted list, so the path filter already excludes it before the tensor check is reached. The check is kept because it costs nothing and because it is the only guard if a future layer with a submodule named `attn` ever appears inside the accepted path, and a unit test now exercises exactly that shape so it does not become dead code.

The exact match on `__module__` is a known limitation. A copy of these classes reached by a different import path, for instance through `trust_remote_code` under another package name, would not be matched and would retain its maps. The check is a string comparison against a fixed list, not a structural test.

Nothing else changes. Attention arithmetic, layer outputs, audio context, pooling and scoring heads are untouched. None of the files whose hashes are recorded in the checkpoint provenance (`encoders.py`, `scoring.py`, `phonetic.py`, `audio.py`, `phone_features.py`, `text_reference.py`) is modified, so every bundled head still loads.

## Expected effect

For a recording with T encoder frames (T = (samples − 400) // 320 + 1 at 16 kHz), each PhoneticXeus layer keeps 8 heads × T² float32 values, that is 32·T² bytes. Before the change, 19 layers keep 608·T² bytes at the end of the pass.

| Duration | T | Kept before (19 layers) | Kept after |
|---|---|---|---|
| 15 s | 749 | 0.32 GiB | 0 |
| 30 s | 1499 | 1.27 GiB | 0 |
| 60 s | 2999 | 5.09 GiB | 0 |
| 120 s | 5999 | 20.38 GiB | 0 |

The temporary tensors of one attention call (scores, masked scores, softmax and masked probabilities, about 128·T² bytes) are still allocated and freed inside each layer. The predicted increase of peak memory during PhoneticXeus inference is therefore about 704·T² bytes before the change and about 128·T² bytes after it, that is about 23.6 GiB and 4.3 GiB at 120 s, plus 2.14 GiB of weights and the runtime. XEUS already used the matched module path, so its memory should not change. It serves as a control.

## Files

- `speech/pronunciation/inference_memory.py`: the fix.
- `tests/test_inference_memory.py`: six unit tests. Five need no downloaded assets. `test_installed_phoneticxeus_attention_is_covered` runs only when the PhoneticXeus source files have been downloaded, and is skipped otherwise.
- `contrib/E1/make_recordings.py`: builds fixed-length recordings (15, 30, 45, 60, 90 and 120 s) from public SpeechOcean762 train utterances.
- `contrib/E1/measure.py`: memory per encoder layer for one encoder and one recording.
- `contrib/E1/score.py`: release-path features and the scores of all 35 bundled configurations.
- `contrib/E1/compare.py`: identity check and quadratic fit.
- `contrib/E1/PR-README.md`: a short card that stands on its own, for a review
  that takes the fix and its tests without the evidence tree.
- `contrib/E1/recompute.py`: regenerates every published figure from the committed raw files, into `contrib/E1/results/recomputed.json`.
- `contrib/E1/raw/`: the raw measurement and identity outputs, the recording list and the drivers that produced them. `raw/analysis/` keeps, as provenance, the analysis scripts that ran on the measuring host.
- `contrib/E1/results/`: findings, protocol and suite results, and `VERIFICATION.md` with the checks repeated in a clean clone on 10 November 2026 and their full logs.

The scripts under `contrib/E1/` are not imported by the application.

## What changes for users

Features and scores are bitwise identical, so no result changes. Two
operational consequences are worth stating.

Existing feature caches raise `Changed extraction inputs` and must be rebuilt.
`reproduce/extract.py` records the sha256 of every file in
`speech/pronunciation/` in its extraction contract, and this change edits one
of them, so a cache written before the change no longer matches. Rebuilding it
produces the same features, bit for bit.

`docs/source-snapshot.json` records a `release_sha256` for
`speech/pronunciation/inference_memory.py`, which this change makes stale: 46
of the 47 recorded hashes still match and that one does not. The update is in a
separate commit, so it can be taken or left on its own.

## Dependencies

None in code. `contrib/E1/results/PROTOCOL.md` points to
`contrib/E2/checksums.txt` for the Hugging Face records of the three weight
files, which is a documentary cross-reference, not a dependency of the fix.

## How to verify

1. Unit tests: `python -m pytest -q tests/test_inference_memory.py`.
2. Identity: run `score.py` on the same recordings with the code before and after the change, then `compare.py identity`.
3. Memory: run `measure.py` for each encoder and duration, before and after, under a fixed memory limit, then `compare.py fit`.

The exact commands, machine and limits used for the reported results are in `contrib/E1/results/PROTOCOL.md`.

## Acceptance criteria

- Features: the four pooled blocks (XEUS, XLS-R, PhoneticXeus, phone diagnostics) are bitwise identical before and after the change on every test recording. If two runs of the unchanged code already differ, the difference after the change must not exceed that run-to-run difference, and no score may differ by more than 1e-6 on the 0 to 10 scale.
- Scores: the same rule for all 105 head outputs (35 configurations × 3 constructs).
- Kept attention: after the change, zero bytes are kept after every PhoneticXeus layer at every duration. Before the change, the bytes kept after layer i equal (i + 1) × 32·T².
- Peak memory: the fitted quadratic coefficient of the PhoneticXeus peak increase is close to 128 bytes per frame² after the change and close to 704 before it. The 90 s and 120 s runs complete after the change within the stated memory limit.
- Tests: the new tests pass, and the rest of the suite gives the same results as the unchanged code.

These criteria were fixed before the runs. Note added after the runs: as first written they name a memory limit without giving its value. The 10 GiB cap actually used is recorded in `contrib/E1/results/PROTOCOL.md`, which was written after the runs.

## Results

Measured on 10 November 2026. Full numbers, conditions and adaptations are in
`contrib/E1/results/`.

- Retained attention is exact. Before the change, each of the 19 PhoneticXeus
  layers holds `32 T^2` bytes and the cumulative total after layer `i` is
  `(i + 1) x 32 T^2` bytes, to the byte, at 15, 30, 45 and 60 s. After the
  change, 372 layer observations across the three encoders hold zero bytes.
- Features and scores are bitwise identical before and after the change at 15,
  30, 45 and 60 s, for the four pooled blocks and all 105 head outputs. The arm
  before the change does not run at 90 or 120 s, so those durations have
  nothing to compare against. Two runs of the unchanged code are also bitwise
  identical, so there is no run to run slack for the comparison to hide in.
- The fitted quadratic coefficient of the peak increase is 716.25 bytes per
  frame squared before the change, against 704 predicted, and 131.19 after it,
  against 128 predicted.
- Peak resident memory for a 120 s recording is 7.62 GiB after the change,
  measured. The comparable quantity before the change is an extrapolation of
  27.1 GiB, give or take about 0.9 GiB, stated as an extrapolation, because
  120 s does not fit on a 15 GiB host without the change. Both are total
  resident high water marks. A figure of 26.4 GiB, the weights alone added to
  the extrapolated increase, is a different quantity. The archived capacity
  test in the manuscript records 22.52 GiB of peak resident memory with
  5.51 GiB of swap on a host with about 24 GiB available, and the
  extrapolation is consistent with that rather than a reproduction of it.
- The peak increase falls by 28 percent at 15 s and by 77 percent at 60 s.
- XEUS and XLS-R are controls and are unchanged.
- The test suite gives the same three failures in both arms, so the change
  introduces no regression. See `results/TEST_SUITE.md`.

The predicted coefficients, 704 and 128 bytes per frame squared, and the exact
retention formula were stated in this card before the runs. The range quoted
for the coefficient of the corrected code, 15 to 120 s, was chosen after the
measurements, and `results/FINDINGS.md` gives both
fits and the reason.

## Changelog

- v1: fix, unit tests and measurement harness.
- v1.1: checks repeated in a clean clone, with the logs.
- v1.2: raw measurements committed, every figure recomputed from them, two
  tests that pin the fix without the downloaded assets, the comparable
  extrapolation with its uncertainty, and the phase crossover recomputed from
  one baseline.
- v1.3: the uncertainty of the extrapolation from a full refit, relative
  paths, the analysis scripts in English, the log of the final suite, the state
  files brought up to the current branch, and the change to the acceptance
  criteria marked.
- v1.4: the provenance statements restated without references to the working
  history of this tree, keeping every limitation and uncertainty they carried.
