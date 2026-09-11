# Stopping point: v8 confirmed, review pipeline committed (2026-09-10)

**Temporary stopping point, not a session close.** The 2026-09-08 hand-off
(`2026-09-08-v8-review-pool-training.md`) is still the substantive record — read
that first. This note covers only what changed on 2026-09-10 and where to resume.

## What happened on 2026-09-10

### 1. A `git reset --hard` + clean destroyed most of the 2026-09-08 work

Recovered in full, except one item. The reflog shows the sequence:

```
e2e49c9 HEAD@{0}: pull --ff --recurse-submodules origin: Merge made by 'ort'
6d4b805 HEAD@{1}: reset: moving to HEAD
6d4b805 HEAD@{2}: commit: Hand-off: v8 training...
```

The clean ran while the `.gitignore` rules were still active, which is the only
reason `Labeling/review/crops/` and `decisions.sqlite` survived. Everything else
untracked was removed and every modified tracked file was reverted.

**Lost and restored:** all four review-tool scripts, `train_vehicle_v8.py`,
`find_duplicate_boxes.py`, `docs/v4-integration-plan.md`, three eval-script bug
fixes, the `_paths` additions, and the schema/README edits.

**Lost and rebuilt:** the v8 run directory (see §2).

**Never at risk:** `Labeling/review/decisions.sqlite` — 1,409 rows, 1,201 crops,
Luke's ~2 hours of labeling. Verified intact after restore: `export_labels.py`
still reports 1,201/1,313 at 0.028 prior agreement and the same 6,494-box export
against a regenerated manifest.

**This cannot happen again**: `decisions.backup-2026-09-10.sqlite` is now
committed, and `.gitignore` ignores the live db while explicitly un-ignoring
`decisions.backup-*.sqlite`.

### 2. v8 was re-run and reproduced EXACTLY

| | original (09-08) | rerun (09-10) |
|---|---|---|
| epochs / best epoch | 37 / **22** | 37 / **22** |
| best mAP50-95 | **0.3959** | **0.3959** |
| final epoch | 0.3578 | 0.3578 |
| class-agnostic recall | 0.770 | 0.770 |
| class-agnostic precision | 0.821 | 0.821 |
| predicted boxes | 4,787 | 4,787 |

Identical to four decimals including every per-class AP50. Ultralytics defaults
to `seed=0, deterministic=True`, so these runs are reproducible on demand. **The
v8 conclusion now rests on two independent runs.**

Against v4: mAP50-95 **0.4322 → 0.3966 (-0.036)**, class-agnostic recall
**0.802 → 0.770 (-0.032)**, every per-class AP50 down. Merging the reviewed pool
into the detector does not work, and no hyper-parameter change reaches the cause.

### 3. Four commits, pushed

| sha | what |
|---|---|
| `f74860f` | three latent evaluation-script bugs |
| `1a389dc` | `Vehicle` is the umbrella class + `find_duplicate_boxes.py` |
| `fb202d8` | the review tool (4 files) + the 1,201 decisions as a tracked backup |
| `9ef48b9` | `train_vehicle_v8.py` + integration plan + README run history |

Merged cleanly with the collaborator's `manual_label.py` work (`e384f7f`,
`f0a11cc`) which arrived in the same pull. **The two labelers are complementary,
not duplicates** — `manual_label.py` draws and assigns boxes one image at a time
on the augmented `train` split; `label_app.py` classifies existing boxes 60 to a
page on the clean `valid`/`test` splits. Same schema, same 1-7 keys, separate
output dirs. Both documented side by side in `scripts/README.md`.

## State

- Branch **`ModelTraining`**, HEAD **`<see git log>`**, **pushed to `origin`**.
  This is the first time anything in this repo has been pushed.
- Working tree clean apart from deliberate untracked items:
  `Vehicle_type_detection/runs/Vehicle_type_detection_v8/` (104MB, reproducible),
  `YoloTraining/`, `yolo11n.pt` (ultralytics AMP-check download, deletable), and
  three 0-byte placeholders (`scripts/check_labels.py`,
  `scripts/label_distribution.py`, `Labeling/auto_label_yolo.py`).
- `v4` is still champion at **0.4322**.

### Verified

- All eight touched scripts compile; `find_duplicate_boxes.py` reproduces its
  numbers (SUV 115/370, Standard Car 81/403); the regenerated manifest matches
  every one of the 1,201 decision crop ids.
- v8 reproduced end to end, twice.

### NOT verified

- The classifier (Path A) — no code written.
- The duplicate-box fix — `--fix-dir` has still never been run.
- The 112 undecided crops — still exported as generic `Vehicle`.

## Gotchas

Everything in the 2026-09-08 hand-off still applies. New or reinforced:

1. **scipy in `myenv` is broken** (`cannot import name '_ccallback_c'`).
   Training completes and saves weights, then dies drawing `results.png`, so the
   process **exits code 1 despite succeeding** and `results.png` is missing from
   the v8 run. Fix:
   `.\myenv\Scripts\python.exe -m pip install --force-reinstall scipy`
2. **`grep "review" .gitignore` matches the existing `Labeling/preview/` line.**
   A naive existence check silently skips adding the review rules. Use `grep -qx`.
3. **Restart `label_app.py` after any `build_crops.py` run** — the manifest is
   read once at startup. This corrupted 53 decisions on 2026-09-08.
4. Ultralytics training IS deterministic here (`seed=0`), so a rerun is a valid
   way to recover a lost run directory.

## Next steps

1. **Fix scipy** (one command, above) so runs stop reporting false failures.
2. **Build the subtype classifier — Path A.** The recommendation v8 did not
   displace. v4 stays the detector untouched; a small classifier types the crops
   so `Vehicle` and `SUV` never compete for one slot. Combined training set:
   Truck 951, SUV 818, Bus 789, Standard Car 771, Motorcycle 234, Van 186. Hold
   out 20% of the Kaggle batch (not the base close-ups) for validation.
   **Gate: if SUV vs Standard Car lands near chance, merge them into one `Car`
   class** — a finding, not a failure. `docs/v4-integration-plan.md` §3.
3. **Resolve the duplicate-box defect** before trusting any SUV/Standard Car
   conclusion: `find_duplicate_boxes.py --fix-dir <scratch>`, then re-measure v4.
4. Optional: finish the last 112 crops (`label_app.py` resumes automatically).
5. Consider a PR from `ModelTraining` into `main` — not opened, since v8
   regressed and this branch is shared active work, not a finished feature.

## References

- `docs/luke-hand-offs/2026-09-08-v8-review-pool-training.md` — **the substantive record**
- `docs/v4-integration-plan.md` — §2 the conflict, §3 Path A, §5b the v8 result
- `docs/luke-hand-offs/2026-09-01-manual-labeler.md` — the collaborator's labeler
- `scripts/README.md` — both labeling flows, run history including v8
