# Checks repeated in a clean clone, 10 November 2026

The checks below were repeated in a clean clone of the repository, not in the
worktrees that produced the measurements. The clone was made with `git clone`
into a new directory and holds no untracked state beyond two symbolic links,
`.venv` and `speech/resources`, which point at the installed environment and
the verified asset tree. Every branch was checked out from the remote, so what
was tested is what a reader would fetch.

Host, environment and caps as in `PROTOCOL.md`, except that `OMP_NUM_THREADS`
and `MKL_NUM_THREADS` were 2 rather than 4, because other work occupied the
remaining cores. Each run was capped with
`systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0`.

Sections (a) to (c) describe version 1.1 of the branch. The last section gives
version 1.3, which is the last version that changed anything the checks
execute.

## (a) The upstream suite on the three branches

`python -m pytest -q --ignore=tests/test_gui.py tests/`, as in
`TEST_SUITE.md`. `tests/test_gui.py` stays excluded because the desktop
dependencies are not installed on this host.

| Branch | Result |
|---|---|
| `main`, the upstream base `31318b5` | 3 failed, 48 passed |
| `essential/memory-leak` | 3 failed, 52 passed |
| `essential/dataset-audit` | 3 failed, 48 passed |

The three failures are the same three tests on all three branches, by name:

- `tests/test_granular.py::test_nltk_sentence_character_spans`
- `tests/test_granular.py::test_pdf_reports_segments_and_escapes_reference`
- `tests/test_reproduction_release.py::test_leaders_include_all_ties_and_match_manuscript`

The first two are the omitted `nltk` and `PySide6` dependencies. The third is
the upstream label-notation drift described in `TEST_SUITE.md`, where every
number agrees and only the notation of four labels differs. None of the three
is introduced by either module.

`essential/dataset-audit` gives exactly the counts of `main`, which is the
expected result for a module that adds only documents and modifies no code.

`essential/memory-leak` at version 1.1 gives four passes more than `main`, and
they are the four tests the module had at that version. Run on their own with
`-v -rs` in the same clean clone:

| Test | Result |
|---|---|
| `test_inference_releases_attention_maps[espnet.nets.pytorch_backend.transformer.attention]` | passed |
| `test_inference_releases_attention_maps[pxeus.espnet_import.attention]` | passed |
| `test_other_modules_and_attn_submodules_untouched` | passed |
| `test_installed_phoneticxeus_attention_is_covered` | passed |

4 passed, 0 skipped. The fourth test did not skip, so it constructed the
`MultiHeadedAttention` class of the downloaded PhoneticXeus snapshot and
confirmed that the hook clears its `attn`. It is the test that exercises the
defect directly.

Full logs in `verification/`.

## (b) The bitwise identity at 60 s

Not repeated in the clean clone. At 60 s the arm before the change peaks at
9.006 GiB, which was more than the memory free on the host while these checks
ran. The identity therefore rests on the original runs, which compared four
durations and found `max_abs = 0.0` in all eight comparisons, over the four
pooled blocks and the 105 head outputs (`FINDINGS.md`). A repetition in a fresh
process remains open.

## (c) Each branch against `main`

`verification/diff-check.json`, produced by `verification/check_diffs.py`,
which asserts rather than reports. Both branches passed at the versions
checked.

| Branch | Release files modified | Undeclared paths | Deletions |
|---|---|---|---|
| `essential/memory-leak` | `speech/pronunciation/inference_memory.py` | none | none |
| `essential/dataset-audit` | none | none | none |

At version 1.1, `essential/memory-leak` added nine files under `contrib/E1/`
and `tests/test_inference_memory.py`, and modified one release file, the file
the card declares as the fix. `essential/dataset-audit` added four files under
`contrib/E2/` and modified nothing. After this file and its logs were added,
the check ran again and passed unchanged, with 17 added paths on
`essential/memory-leak` (`verification/diff-check-after-commit.json`).

The check also asserts that none of the six files whose hashes are recorded in
the checkpoint provenance is touched on either branch: `encoders.py`,
`scoring.py`, `phonetic.py`, `audio.py`, `phone_features.py` and
`text_reference.py`. None is. That is what keeps every bundled head loading.

## The branch at version 1.3

At version 1.3 the branch differs from `31318b5` in 162 paths.

- 159 are added under `contrib/E1/`. They are the scripts, the results, the raw
  measurement and identity outputs under `raw/`, and the logs.
- `tests/test_inference_memory.py` is added, with six tests.
- Two release files are modified. One is `speech/pronunciation/inference_memory.py`,
  the fix. The other is `docs/source-snapshot.json`, whose recorded hash for
  that file is updated in a separate commit. `check_diffs.py` declares both.

None of the six provenance files is modified. The full suite at this version
gives 3 failed and 54 passed, with the same three failures as `main`
(`verification/suite-final-memory-leak.log`).

That last run is the one exception to the opening sentence of this file. It ran
in the worktree, not in the clean clone, because the clone had been removed by
then. Its log says so in its header. The three clean-clone runs of section (a)
stand as written.
