# Test suite results

Run on 10 November 2026 on the host described in `PROTOCOL.md`, with
`python -m pytest -q --ignore=tests/test_gui.py tests/`. `tests/test_gui.py` is
excluded because the desktop dependencies are not installed on this host.

| Code | Result | Log |
|---|---|---|
| `31318b5`, unchanged | 3 failed, 48 passed | `verification/suite-main.log` |
| `31318b5` plus this module, version 1.3 | 3 failed, 54 passed | `verification/suite-final-memory-leak.log` |

Version 1.4 changes documents only, apart from one renamed field in
`recompute.py`, which the suite does not cover. `speech/` and `tests/` are
untouched since that run, so it stands for version 1.4 as well.

The six additional passes are the six tests of `tests/test_inference_memory.py`,
which this module adds. The run shows no skipped test. The same three tests fail
with and without the change, so the change introduces no regression. This is
the evidence for the acceptance criterion that the rest of the suite gives the
same results as the unchanged code.

Versions 1 and 1.1 of the module had four tests. At version 1.1 the suite gave
3 failed and 52 passed (`verification/suite-essential-memory-leak.log`), and the
four tests passed with none skipped (`verification/suite-e1-tests.log`).
Version 1.2 added the two tests marked below.

## tests/test_inference_memory.py

| Test | What it pins | Downloaded assets |
|---|---|---|
| `test_inference_releases_attention_maps[espnet.nets.pytorch_backend.transformer.attention]` | the hook clears `attn` in inference mode and keeps it in training mode, on the ESPnet path | not needed |
| `test_inference_releases_attention_maps[pxeus.espnet_import.attention]` | the same on the vendored PhoneticXeus path | not needed |
| `test_other_modules_and_attn_submodules_untouched` | a PhoneticXeus encoder layer with an `attn` submodule, and an attention class from another package, are left alone | not needed |
| `test_vendored_module_path_is_covered_by_its_literal_name` (version 1.2) | the vendored path is written out, so removing it from the accepted list fails | not needed |
| `test_tensor_check_spares_an_attn_submodule_on_a_matching_path` (version 1.2) | a module on an accepted path whose `attn` is a submodule is left alone | not needed |
| `test_installed_phoneticxeus_attention_is_covered` | the hook clears `attn` on the `MultiHeadedAttention` class of the downloaded PhoneticXeus snapshot | needed, skipped without them |

All six passed. `test_installed_phoneticxeus_attention_is_covered` did not skip
on this host, so it constructed the class of the downloaded snapshot and
confirmed that the hook clears its `attn` after a forward pass in inference
mode.

## The three failures, identical with and without the change

| Test | Cause |
|---|---|
| `tests/test_granular.py::test_nltk_sentence_character_spans` | `ModuleNotFoundError: No module named 'nltk'` |
| `tests/test_granular.py::test_pdf_reports_segments_and_escapes_reference` | `ModuleNotFoundError: No module named 'PySide6'` |
| `tests/test_reproduction_release.py::test_leaders_include_all_ties_and_match_manuscript` | Label notation, described below |

The first two follow from the installed environment. This host installs the
`encoders` and `evaluation` dependency sets and omits the `gui` set, so
`nltk`, `PySide6`, `pypdf` and WeasyPrint are absent. Both tests import them
directly.

The third does not depend on the environment. It regenerates the leader table
with `reproduce.leader_tables.render` and asserts that the result appears in
`paper/main.tex`. Comparing the two line by line: every statistic and every
interval is identical, and the difference is the notation of the selection
rule.

| `paper/main.tex` | `reproduce/leader_tables.py` output |
|---|---|
| `1G-S` | `S` |
| `5G-AMAE` | `H` |
| `3RB-C` | `B-S` |
| `3RB-A` | `B-D` |

For example, the manuscript row

    & aesopocean & 1G-S & \textbf{0.872} & [0.842, 0.894] & 0.598 & [0.563, 0.631]\\

is generated as

    & aesopocean & S & \textbf{0.872} & [0.842, 0.894] & 0.598 & [0.563, 0.631]\\

The two label vocabularies describe the same five selection rules. No number
differs. The test is left as it is, because which vocabulary the release
should use is the author's decision.
