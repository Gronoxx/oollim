# Raw outputs

What the measurement harness wrote, committed after the runs so the published
figures can be checked against the files that produced them rather than against
a table.

- `measurements/`: 30 runs, three encoders at 15, 30, 45 and 60 s in both arms
  and at 90 and 120 s in the after arm, each with its captured output. One run,
  `before-phoneticxeus-015s`, has no captured output.
- `measurements-phases/`: 12 runs with the three phase readings.
- `scores/`: 12 release-path runs, with the pooled feature arrays as `.npz`,
  which are the arrays the identity check compared.
- `recordings/`: the list of recordings with the sha256 and the source
  utterances of each, and the reference text of each. The WAV files are derived
  audio and stay out.
- `drivers/`: the three shell drivers, as they ran.
- `analysis/`: the analysis scripts, as provenance. See the README there.
- `xlsr/attn_implementation.json`: the attention implementation of the
  installed `Wav2Vec2Model`, read at run time by a one-off check rather than by
  a committed script.

## What changed after the runs

The outputs are as written, except in two respects that no value depends on.
Absolute paths were replaced by `<repo>` so the files do not carry the home
directory of the account that ran them, and line endings were normalised. The
`.npz` arrays are untouched. The scripts changed only in comments, docstrings
and printed messages, which were translated from Portuguese, so every one of
them computes what it computed during the runs.

`contrib/E1/recompute.py` reads this tree and regenerates every published
figure into `contrib/E1/results/recomputed.json`.
