# E1 measurement protocol, as executed

Date: 10 November 2026. Base: `cburst/oollim` at `31318b5`. This file records
the machine, the software versions, the exact commands and every adaptation,
so the numbers in this directory can be read against the conditions that
produced them.

## Host

| | |
|---|---|
| Machine | Linux, EndeavourOS, kernel 6.18.44-1-lts, x86_64 |
| CPU | Intel Core i5-10400F at 2.90 GHz, 6 cores and 12 threads, AVX2, no AVX-512 and no VNNI |
| RAM | 15 GiB |
| Swap | zram 3.9 GiB at priority 100, plus an 8 GiB swapfile at priority -2 on the rotational disk. `vm.swappiness` 180 |
| `/home` | `/dev/sda1`, ST1000DM010, 7200 rpm rotational disk. The clone, the environment, the encoders and the corpus all live here |
| GPU | NVIDIA GeForce GTX 1660 SUPER, 6 GiB, compute capability sm_75, driver 610.57.04 |

The GPU was not used. Every measurement in this directory is CPU float32, as
the module specifies. For the record, `torch.cuda.is_bf16_supported()` returns
True on this device while `torch.cuda.is_bf16_supported(including_emulation=False)`
returns False, because sm_75 has no native bfloat16. A later module that uses
the GPU should read the second form, or compare the compute capability against
8.0.

## Software

| | |
|---|---|
| Python | 3.11.15 |
| torch | 2.11.0+cu130 |
| torchaudio | 2.11.0+cu130 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.1 |
| transformers | 5.19.0 |

The environment installs the `encoders` and `evaluation` dependency sets and
omits the `gui` set, so PySide6, nltk, pypdf and WeasyPrint are absent.

## Layout

Two worktrees of one clone share a single environment and a single copy of the
encoder assets.

    work/wt-before    detached at 31318b5, the unchanged code
    work/wt-after     branch essential/memory-leak, the change under test

In each worktree, `speech/resources` and `.venv` are symbolic links to the
ones in the main checkout, so neither the 5.5 GiB of encoder assets nor the
environment is duplicated. Both are listed in each worktree's
`.git/info/exclude`, so neither appears as a change.

## Recordings

    python contrib/E1/make_recordings.py \
      --speechocean <unpacked OpenSLR 101> \
      --output work/E1/recordings

Six recordings at 15, 30, 45, 60, 90 and 120 s, built from SpeechOcean762
train utterances in utterance-ID order, keeping only references the pinned
CMUdict covers, and zero padded to the target length. The script wrote each
file's SHA-256 and the utterances it contains to
`work/E1/recordings/recordings.json`, which is committed as
`contrib/E1/raw/recordings/recordings.json`, with the reference text of each
recording beside it (`so-<NNN>s.txt`). The WAV files are not committed: they
are derived from a CC BY 4.0 corpus and are reproducible from that command.

## Measurements

One process per run, each under a memory cap, so one run cannot press on the
rest of the machine and each `VmHWM` describes exactly one run:

    systemd-run --user --scope --quiet \
      -p MemoryMax=10G -p MemorySwapMax=0 \
      env OMP_NUM_THREADS=4 \
      .venv/bin/python work/wt-after/contrib/E1/measure.py \
        --repo <work/wt-before | work/wt-after> \
        --encoder <phoneticxeus | xeus | xlsr> \
        --audio work/E1/recordings/so-<NNN>s.wav \
        --output work/E1/measurements/<arm>-<encoder>-<NNN>s.json \
        --threads 4

`MemorySwapMax=0` matters on this host: without it a run that exceeded
expectations would reach the swapfile on the rotational disk, and the figures
would describe the disk rather than the algorithm.

Grid: the three encoders, each at 15, 30, 45 and 60 s in both arms, and at
90 and 120 s in the `after` arm only. Under the unchanged code the predicted
totals at 90 and 120 s do not fit on a 15 GiB host. That restriction came from
a memory budget worked out before the runs and is not in the module card, which
says instead to run `measure.py` "for each encoder and duration, before and
after".

The driver ran as `work/E1/run_measurements.sh` and is committed as
`contrib/E1/raw/drivers/run_measurements.sh`, beside `run_identity.sh` and
`run_phase_peaks.sh`, the drivers of the identity check and of the phase runs.
It skips a run whose output file already exists, and it records a failure
without aborting the batch, so an exhausted cap is visible rather than silent.

## Adaptations, and why

1. **torch 2.11.0+cu130 rather than a cu128 build.** The module's draft
   suggested matching the WSL build of the archived capacity test. Both
   worktrees import the same environment, so the before and after comparison
   is unaffected. The version is recorded above rather than matched.
2. **Encoder assets fetched with `curl` and verified with `sha256sum`,** not
   with `speech/scripts/acquire.py`. Repository code was not executed at that
   stage. Each of the 49 resources was checked against
   its `sha256` in `speech/resources-manifest.json` and all 49 matched on the
   first attempt. The three weight files were additionally checked against the
   LFS records the Hugging Face API publishes for their pinned revisions, and
   those agree too, and the values are in `contrib/E2/checksums.txt` on
   `essential/dataset-audit`.
3. **`python -m oollim_app.assets` was then run.** `speech/pronunciation/phonetic.py`
   verifies every `.py`, `.json` and `.yaml` file of the PhoneticXeus snapshot
   against a `<name>.receipt.json` written beside it, and the `curl` path wrote
   no receipts, so the first PhoneticXeus run raised `FileNotFoundError`. The
   repository's own acquisition step found all 49 files already present with
   the expected digest, downloaded nothing, and wrote the 49 receipts. A
   reproduction that uses the documented acquisition path from the start does
   not meet this.
4. **Recordings and measurements written under `work/`,** which each worktree
   ignores, so no audio enters the history. The measurement outputs, including
   the phase runs, the identity outputs, the recording list and the drivers were
   committed afterwards under `contrib/E1/raw/`, and `contrib/E1/recompute.py`
   regenerates every published figure from them. The WAV files stay out of the
   history.

## Results

See `FINDINGS.md` in this directory for the per-layer retention, the quadratic
fit and the identity check, and `TEST_SUITE.md` for the suite comparison.
