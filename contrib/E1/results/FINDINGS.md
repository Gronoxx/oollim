# E1 findings

Measured on 10 November 2026. Conditions, versions and adaptations are in
`PROTOCOL.md`. The suite comparison is in `TEST_SUITE.md`. Thirty measurement
runs and twelve release-path runs completed, and none failed.

Units: GiB means 1024 cubed bytes and MB means 10 to the sixth, and both appear
only where labelled. Quadratic coefficients are bytes per frame squared, linear
coefficients bytes per frame, constants bytes.

## Headline

- Retained attention is exact. Before the change, each PhoneticXeus layer
  holds `32 T^2` bytes and the cumulative total after layer `i` is
  `(i + 1) x 32 T^2` bytes, to the byte, at 15, 30, 45 and 60 s. After the
  change, 372 layer observations across the three encoders hold zero bytes.
- The fitted quadratic coefficient of the peak increase is 716.25 bytes per
  frame squared before the change, against 704 predicted, and 131.19 after it,
  against 128 predicted.
- Features and scores are bitwise identical before and after the change, at
  15, 30, 45 and 60 s, for all four pooled blocks and all 105 head outputs.
  The arm before the change does not run at 90 or 120 s, so those two
  durations have nothing to compare against.
- Peak resident memory for a 120 s recording after the change is 7.62 GiB,
  measured. The comparable quantity before the change is an extrapolation of
  27.1 GiB, give or take about 0.9 GiB, labelled as an extrapolation
  throughout, because 120 s does not fit on this host without the change. Both
  figures are total resident high water marks. A figure of 26.4 GiB, the
  extrapolated increase plus the weights alone, is a different quantity and is
  named as such below.
- The peak increase falls by 28 percent at 15 s and by 77 percent at 60 s, the
  effect growing with duration as a dependence on T squared implies.
- XEUS and XLS-R serve as controls and are unchanged.

## Retained attention per layer

Encoder frames are `T = (samples - 400) // 320 + 1`. One layer holds
`32 T^2` bytes, which is `(1, 8, T, T)` in float32.

| Duration | T | `19 x 32 T^2` | Observed after the last layer | Divergent layers |
|---|---|---|---|---|
| 15 s | 749 | 341,088,608 | 341,088,608 | 0 |
| 30 s | 1499 | 1,366,176,608 | 1,366,176,608 | 0 |
| 45 s | 2249 | 3,075,264,608 | 3,075,264,608 | 0 |
| 60 s | 2999 | 5,468,352,608 | 5,468,352,608 | 0 |

Every one of the 19 layers matches `(i + 1) x 32 T^2` exactly at all four
durations, not only the total after the last layer.

After the change, no layer holds a non-zero amount at any duration: 114
observations for PhoneticXeus, 114 for XEUS and 144 for XLS-R, which has 24
layers, for 372 in all.

## Identity of features and scores

`score.py` runs the release path, decoding with the application's own FFmpeg
call, and writes the four pooled blocks and the output of all 105 heads, which
is 35 configurations by 3 constructs. Two runs of the unchanged code are
included so that any run to run variation would be visible.

| Duration | Comparison | Blocks bitwise | Scores bitwise | Largest absolute difference |
|---|---|---|---|---|
| 15 s | before run 1 against after | yes, all four | yes, all 105 | 0.0 |
| 15 s | before run 1 against before run 2 | yes, all four | yes, all 105 | 0.0 |
| 30 s | before run 1 against after | yes, all four | yes, all 105 | 0.0 |
| 30 s | before run 1 against before run 2 | yes, all four | yes, all 105 | 0.0 |
| 45 s | before run 1 against after | yes, all four | yes, all 105 | 0.0 |
| 45 s | before run 1 against before run 2 | yes, all four | yes, all 105 | 0.0 |
| 60 s | before run 1 against after | yes, all four | yes, all 105 | 0.0 |
| 60 s | before run 1 against before run 2 | yes, all four | yes, all 105 | 0.0 |

The rows comparing the unchanged code with itself matter. The acceptance
criterion allowed for the unchanged code differing between runs, in which case
the change had to stay within that difference. It does not differ: two runs of
the unchanged code are bitwise identical, so there is no run to run slack, and
the identity between the arms is exact rather than within a tolerance.

This is the expected outcome. In the vendored PhoneticXeus attention,
`forward_attention` computes its return value before assigning `self.attn`,
the global forward hook runs after that method returns, and no other method
reads `self.attn`. Clearing it cannot affect any output.

## Quadratic fit of the peak increase

The peak increase is `VmHWM` during inference minus `VmRSS` once the weights
are loaded, with `VmHWM` reset through `/proc/self/clear_refs` after loading.

| Encoder | Arm | Range | n | Quadratic | Linear | Constant |
|---|---|---|---|---|---|---|
| PhoneticXeus | before | 15 to 60 s | 4 | **716.25** | 41,415 | 56.6 MB |
| PhoneticXeus | after | 15 to 120 s | 6 | **131.19** | 9,348 | 327.7 MB |
| PhoneticXeus | after | 15 to 60 s | 4 | 47.47 | 323,516 | 94.2 MB |
| XEUS | before | 15 to 60 s | 4 | 84.44 | 218,668 | 277.8 MB |
| XEUS | after | 15 to 120 s | 6 | 136.97 | 33,900 | 415.1 MB |
| XEUS | after | 15 to 60 s | 4 | 128.11 | 95,991 | 348.3 MB |
| XLS-R | before | 15 to 60 s | 4 | -25.06 | 361,700 | 1157.5 MB |
| XLS-R | after | 15 to 120 s | 6 | 4.35 | 221,767 | 1309.0 MB |
| XLS-R | after | 15 to 60 s | 4 | 2.39 | 234,214 | 1296.1 MB |

The two figures in bold are the ones the acceptance criterion names: 716.25
against 704 predicted, a difference of 1.7 percent, and 131.19 against 128
predicted, a difference of 2.5 percent.

### Which range each fit uses

The predicted coefficients, 704 and 128 bytes per frame squared, and the exact
retention formula were stated before the runs. With four points between 15 and
60 s the quadratic coefficient of the corrected code is not resolved, because
replicate variation at 60 s alone moves it by 44.69 bytes per frame squared. The coefficient is therefore quoted over 15 to 120 s, a range chosen
after the measurements.

The replicate figure is measured rather than estimated. The change does not
affect XEUS, whose attention the unchanged code already released, so the XEUS
rows of the two arms are replicate measurements of one code path. Their narrow
fits give 84.44 for the arm before the change and 128.11 for the arm after it.
Substituting the 60 s point of the other arm moves the first to 129.13 and the
second to 83.41, a movement of 44.69 in both directions. A single point, which
differs by 0.094 GiB between replicates, therefore accounts for the whole
spread. In a four point fit the longest duration carries almost all of the
quadratic leverage.

For the arm before the change there is no choice of range. Durations of 90 and
120 s do not fit on a 15 GiB host without the change, so 15 to 60 s is the
whole of the available range. There the coefficient near 704 is large enough
to be recovered to 1.7 percent despite the same leverage.

Two observations needed an explanation and now have one. First, XLS-R returns
a negative quadratic coefficient over the narrow range. The quantity it
estimates, the growth of the peak with the square of the frame count, cannot be
negative, so the fit is recovering noise around a coefficient near zero rather
than a negative effect. Two measurements support that it is near zero. The
effective attention implementation of the installed `Wav2Vec2Model` is `sdpa`,
read from `config._attn_implementation` at run time and not set by the adapter.
And its 24 layers have 16 heads each, so if a layer retained its `T` by `T`
scores the retained bytes would grow by about 64 per frame squared per layer,
against the 4.35 per frame squared the wide fit returns for the whole peak
increase of all 24 layers together. The value is recorded in
`raw/xlsr/attn_implementation.json`, which a one-off check produced rather than
a committed script. Second, over the wide
range XEUS returns 136.97 and PhoneticXeus after the change returns 131.19,
both near the predicted 128, so after the change PhoneticXeus behaves like the
encoder that never retained. Why the narrow range fails to resolve a
coefficient near 128 is examined separately in `PHASES.md`.

## Controls

| Encoder | 15 s | 30 s | 45 s | 60 s |
|---|---|---|---|---|
| XEUS | -0.4 % | +0.0 % | -0.0 % | +5.9 % |
| XLS-R | +4.4 % | -0.0 % | +0.0 % | +0.0 % |
| PhoneticXeus | -28.3 % | -58.7 % | -72.5 % | -77.4 % |

Six of the eight control points agree within 0.4 percent. Two do not: XEUS at
60 s is 0.094 GiB higher after the change and XLS-R at 15 s is 0.057 GiB
higher. Both are run to run variation in peak resident memory rather than an
effect of the change, because a systematic effect would appear at all four
durations of that encoder, and in each case the other three agree to within
0.4 percent. Both are recorded here rather than omitted. The XEUS figure is
the same 0.094 GiB that drives the spread of the narrow fits above.

## Peak memory and the published capacity test

PhoneticXeus, resident memory once the weights are loaded and peak during
inference, in GiB:

| Duration | Before, loaded | Before, peak | After, loaded | After, peak |
|---|---|---|---|---|
| 15 s | 2.852 | 3.316 | 2.843 | 3.176 |
| 30 s | 2.844 | 4.428 | 2.844 | 3.499 |
| 45 s | 2.845 | 6.384 | 2.845 | 3.818 |
| 60 s | 2.846 | 9.006 | 2.846 | 4.240 |
| 90 s | not measurable here | | 2.848 | 5.641 |
| 120 s | not measurable here | | 2.849 | **7.619** |

The loaded resident set of the arm before the change is 2.852 GiB at 15 s
against 2.844 to 2.846 GiB at the other three durations. The 15 s run is the
first of that arm, and the 7 MB is not otherwise explained. It is also the one
run of the thirty whose stdout was not captured, so it was probably made
outside the driver, which is itself a difference from the other three. Its effect is
bounded. The excess over the median of the other three is 7.32 MB, and it
enters the fit as a reduction of the increase at 15 s. Refitting with that
point moved shifts the quadratic by 3.25 bytes per frame squared and the
extrapolation at 120 s by 0.040 GiB, which is one twentieth of the replicate
uncertainty. All four points carry the same sensitivity of 4.444e-07 per byte,
so this point carries no less leverage than the other three.

Before the change, 60 s reaches 9.006 GiB, close to the 10 GiB cap the runs
were given, and it is the longest duration this host can measure in that arm.
After the change, every duration through 120 s completes within the cap, which
is the acceptance criterion for 90 and 120 s.

Extrapolating the fit of the arm before the change to `T = 5999` gives a peak
increase of 24.290 GiB. Two figures are built from it, and they are not the
same quantity.

The comparable figure is 27.137 GiB: the increase plus the loaded resident set
of that arm, 2.847 GiB on average over its four runs. It is comparable with
the measured 7.619 GiB because both are total resident high water marks. This
is the figure to quote against the measurement.

The secondary figure is 26.433 GiB: the increase plus the 2.142 GiB of
resident PhoneticXeus parameters alone. It omits the interpreter, the
framework and the front-end buffers, which the loaded resident set includes.
It is not comparable with the measurement and should not be quoted against it.

The uncertainty is about 0.9 GiB. XEUS is untouched by the change, so the XEUS
rows of the two arms are replicates of one code path, and they differ by 0.094
GiB at 60 s. Moving the 60 s observation by that gap and refitting the
quadratic moves the extrapolation at `T = 5999` by 0.857 GiB.

The refit is what gives that figure. Multiplying the shift of the quadratic
coefficient alone, 44.7 bytes per frame squared, by `T` squared would give
1.498 GiB instead, but moving one observation also changes the linear and the
constant term, which absorb part of the shift, so scaling the quadratic term
alone overstates the effect by a factor of 1.75. `recompute.py` refits.

This is an extrapolation, not a measurement, and no 120 s measurement of that
arm exists on this host.

The archived 120 s capacity test in the manuscript records 22.52 GiB of peak
process resident memory alongside 5.51 GiB of peak system swap, on a host with
about 24 GiB of available memory. The extrapolated 27.1 GiB lies between that
resident figure and the sum of resident and swap use. A resident figure
recorded that close to the physical memory of its host is a lower bound on
demand, so the extrapolation is consistent with the archived test rather than a
reproduction of it.

## What was fixed before the measurements, and what was not

Fixed in the module card before the measurements were run. The predicted
coefficients 704 and 128 bytes per frame squared. The per-layer formula, which
the card states as the bytes kept after layer `i` equalling
`(i + 1) x 32 T^2` before the change and zero after it. The requirement of
bitwise identity in the features and the 105 scores. XEUS as a control,
because the unchanged code already matched its module path. And the
expectation that 90 and 120 s complete after the change.

Not in the module card: that 90 and 120 s run only in the arm after the
change. The card says instead to run `measure.py` "for each encoder and
duration, before and after", which does not carry that restriction. The
restriction came from a memory budget worked out before the runs, which put
15 to 60 s within reach under the published code, 90 s over the budget and
120 s out of reach. Because the card does not state it, it is counted below as
decided during execution.

The card refers to "a fixed memory limit" without giving a value, so it does
not state the limit that was used. The 10 GiB cap actually used is recorded in
`PROTOCOL.md`, written after the runs.

Decided during execution: the 10 GiB cap and, on this branch, the omission of
90 and 120 s from the arm before the change.

Chosen after the measurements: the range quoted for the coefficient of the
corrected code, with both fits in the table above. Also `PHASES.md` in full,
including the `--phase-peaks` code, which was written after the runs it
analyses, the replicate swap, the attribution of the two control outliers to
run to run variance, the comparison with the archived capacity test, and the
explanation offered for the XLS-R coefficient.

The raw measurement files carry no timestamp of execution, and nothing in this
tree records the order in which they and the analysis were written. The order
given in this section is stated here, not demonstrated.
