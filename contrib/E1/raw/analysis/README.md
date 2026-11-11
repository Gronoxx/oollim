# The analysis scripts, as provenance

These seven scripts produced the figures that `FINDINGS.md` and `PHASES.md`
report. They are committed so that the path from the measurement files to the
published numbers is on the record, not because they are the tool to run.

They read `work/E1/measurements`, the directory the driver wrote on the
measuring machine, not the `raw/` tree beside them. Only
`contrib/E1/recompute.py` reads `raw/`, and it regenerates every published
figure from the committed files. A reader who wants to check a number should
run that, and read these to see how the number was first obtained.

## What each one computes

`summarize.py` consolidates the 30 measurements of the main grid: the quadratic
fits of the peak increase, the per-layer retention against
`(i + 1) x 32 T^2`, and the two control encoders.

`ident_table.py` runs `contrib/E1/compare.py identity` over the four durations
and both comparisons, and tabulates `max_abs` for the four pooled blocks and
for the 105 head outputs.

`check_leverage.py` measures how much the four-point fit depends on its longest
duration, by substituting the 60 s point of one arm into the other and
refitting.

`check_regimes.py` compares the peak already reached after block 0 with the
peak at the end of the pass, which is the test of whether `VmHWM` is a maximum
over a term linear in the duration and a term in `T^2` rather than a sum of
them.

`analyse_phases.py` separates the peak of the phase runs into the phase before
the first encoder block and the phase of the blocks themselves, reading the
instrumented values directly and fitting nothing to the maximum.

`crossover.py` interpolates the point where the two phase curves cross,
measuring the first phase above `rss_loaded` and the second above
`rss_pre_encoder`. Those are two different constants. `PHASES.md` quotes the
53.67 s and 46.85 s that one common baseline gives, and the 58.66 s and
54.72 s of this pair as the size of the error the choice of baseline
introduces.

`retract_check.py` shows that `inc@block0` cannot be decomposed by
subtraction, because the value read in the output hook of block 0 already
includes the attention transient of layer 0. Its conclusion is that `VmHWM` is
a maximum over phases and not a sum of contributions. It is not used for any
figure quoted in the text.

## What changed after the runs

These scripts changed only in comments, docstrings and printed messages, which
were translated from Portuguese. No expression and no literal that enters a
computation was touched, so each one computes what it computed during the runs.
The outputs they read changed only in the path prefix, which became `<repo>`,
and in line endings.
