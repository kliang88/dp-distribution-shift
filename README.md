# Readmission prevalence-shift experiment

Study whether source pretraining helps predict readmission within 30 days, and
whether lowering source readmission prevalence affects ordinary SGD and
private DP-SGD differently.

Each repeat compares six models:

| Initialization | SGD | DP-SGD |
|---|---|---|
| Scratch | A | C |
| S0: randomly sampled source prevalence | B-S0 | D-S0 |
| S1: lower, configured source prevalence | B-S1 | D-S1 |

## Install and run

Use Python 3.12. Run all commands from the repository root.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/dp-shift all --config configs/experiment.json --output results/my_experiment
```

Inputs are fixed in code: `data/diabetic_data.csv` and `data/IDS_mapping.csv`.
`--output` selects the run directory; its default is `results/prevalence_shift_v3`.
Use a fresh directory when changing settings or inputs. Existing outputs from
older protocol versions are not adopted or modified.

To run the stages separately, use the same config and output directory:

```sh
.venv/bin/dp-shift prepare --config configs/experiment.json --output results/my_experiment
.venv/bin/dp-shift train --config configs/experiment.json --output results/my_experiment
.venv/bin/dp-shift evaluate --config configs/experiment.json --output results/my_experiment
```

| Stage | Purpose |
|---|---|
| `prepare` | Select patients, allocate partitions, construct source pairs, and preprocess features |
| `train` | Tune on public proxy data, freeze settings, and train all model conditions |
| `evaluate` | Score the fixed test set, publish tables and figures, verify outputs, and clean up |

## Configuration

Edit [configs/experiment.json](configs/experiment.json). This is the only experiment
config. Every field is required; unknown fields are rejected. Fixed defaults
are defined in [config.py](src/dp_shift/config.py).

| Parameter | Meaning |
|---|---|
| `seed` | Master seed for splits, source samples, initialization, and ordinary training |
| `repeats` | Number of paired source-sampling/training repeats; at least two |
| `data.source_pool_size` | Patients available for source pretraining samples |
| `data.proxy_train_size` | Patients used to train target-tuning candidates |
| `data.proxy_validation_size` | Patients used to score tuning candidates |
| `data.target_train_size` | Patients used for final target training |
| `data.target_test_size` | Patients held out for final evaluation |
| `source_conditions.pretrain_size` | Patients in each S0 and S1 pretraining sample |
| `source_conditions.shifted_positive_rate` | Required positive fraction in S1 |
| `privacy.epsilon` | Privacy cap per private model |
| `privacy.clipping_norm` | Maximum norm of each patient's gradient |
| `tuning.source_epochs` | Fixed source training epochs, used in tuning and final pretraining |
| `tuning.source_learning_rates` | Candidate source learning rates |
| `tuning.target_learning_rates` | Candidate target learning rates |
| `tuning.target_epoch_candidates` | Candidate target epoch counts |

### Patient sampling

Keep the encounter with the smallest encounter ID per patient, then exclude
death/hospice discharges (codes 11, 13, 14, 19, 20, 21). The supplied dataset
leaves 69,973 eligible patients.

One seeded shuffle allocates the exact configured counts to source pool, proxy
training, proxy validation, target training, and target test, in that order.
These groups are patient-disjoint, sampled without replacement, and not
stratified by label. Remaining patients are recorded as unused. All partitions
stay fixed across repeats.

For each repeat, randomly sample S0 from the source pool. Construct S1 by
replacing randomly chosen S0 positives with unused source negatives until S1
reaches the configured prevalence. Both samples have `pretrain_size` patients
and share most patients. For example, 16,000 patients at 5% prevalence means
800 positives; the actual rate is set by your config.

Preparation checks partition capacity, class presence, and feasibility of every
source pair before training. The shifted positive count must be integral and
below S0's positive count. The source pool must have enough replacement
negatives. Proxy and target training sizes must be divisible by 12. Invalid
draws fail explicitly rather than retrying with another seed.

### Tuning and training

Source tuning trains each learning-rate candidate on a source sample for
`source_epochs`, then selects the highest proxy-validation average precision
(AP). Target tuning trains each learning-rate/epoch candidate on proxy training
data and averages proxy-validation AP across SGD/DP-SGD × scratch/pretrained.
One shared target setting is selected. Tuning uses baseline source pretraining
and one seed setup. AP ties within `1e-6` favor lower Brier score, then fewer
target epochs, then lower learning rate.

Selected settings are frozen across all final models and repeats. Final target
training and test patients do not participate in tuning.

All models use an MLP with hidden widths 64 and 32, ReLU, and binary
cross-entropy. SGD uses momentum 0.9, no weight decay, and a fresh optimizer for
target training. Source training shuffles patients each epoch into batches of
256, retaining the final partial batch. Both target methods use Poisson
sampling: independently include each patient with probability `1/12` per step,
with 12 steps per epoch. Expected batch size is training size divided by 12.
Execution uses CPU, one thread, and deterministic algorithms.

### Privacy

Opacus PRV accounting uses fixed delta `1e-5`. Noise is calibrated for each
candidate schedule to `epsilon * (7.99 / 8)`; achieved epsilon is checked against
the configured cap. Noise standard deviation is the noise multiplier times the
clipping norm. The guarantee is **per private model**, not jointly across all
released models. This public-data simulation uses `secure_mode=False`.

Final private sampling and noise use independent randomness, with replay states
kept only in the temporary workspace. The master seed reproduces data membership
and ordinary training, but does not reproduce final private noise.

### Features

Numeric predictors: `time_in_hospital`, `num_lab_procedures`, `num_procedures`,
`num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`,
and `number_diagnoses`.

Categorical predictors: `age`, `gender`, `race`, `admission_type_id`,
`admission_source_id`, `discharge_disposition_id`, `diabetesMed`, and `change`.

Preprocessing fits only on the source pool: median numeric imputation, `log1p`
on the three prior-visit counts, standard scaling, and categorical imputation
with `Missing` followed by one-hot encoding that ignores unknown categories.
The encoded feature count depends on categories observed in the source pool.

## Results

The two research tables and privacy ledger are saved under `<output>/evaluation/`:

| File | Contents |
|---|---|
| `metrics.csv` | Six model conditions; separate mean and sample-SD columns for AP, AUROC, Brier, and calibration bias. `n` is repeats. |
| `effects.csv` | Seven paired AP comparisons: four pretraining gains, two source-shift effects, and one DP interaction. Includes mean, SD, 95% interval, and interpretation where defined. |
| `privacy.csv` | Privacy settings, achieved epsilon, and accountant history for each private model/repeat |

**AP** is the primary metric: a non-interpolated summary of the precision–recall
curve. Higher AP and AUROC are better. Lower **Brier score** means better overall
probability accuracy. **Calibration bias** is mean prediction minus observed
prevalence; positive means overprediction, negative means underprediction, and
zero means no average bias. A zero bias alone does not establish calibration
across risk groups.

Intervals describe variability across repeats on the **same fixed test set**;
they do not measure uncertainty for a new hospital. Effects use paired
within-repeat differences. The practical AP threshold is 0.005, equivalent to
0.5 AP percentage points.

### Figures

Four PNGs are saved under `<output>/evaluation/figures/`:

| Figure | How to read it |
|---|---|
| `performance.png` | AP, AUROC, and Brier for all six models. Small dots are repeats; large points and bars are means and 95% intervals. |
| `paired_effects.png` | Three panels answer whether pretraining helps, whether S1 hurts, and whether DP responds differently. Points and bars show paired means and 95% intervals; panels use separate horizontal scales. |
| `precision_recall.png` | S0 and S1 panels each compare SGD/DP-SGD × scratch/pretrained. Bold lines show mean step precision on a common recall grid; faint lines show repeats. The dotted line is test prevalence. |
| `calibration.png` | SGD/DP-SGD rows and Scratch/S0/S1 columns. Each curve is a repeat; histograms below show prediction distributions averaged across repeats. |

In `paired_effects.png`, the calculations are:

1. **Pretraining benefit:** pretrained AP − scratch AP. Positive means improvement.
2. **Source-shift effect:** S1 AP − S0 AP. Negative means S1 performs worse.
3. **DP interaction:** DP-SGD shift − SGD shift. Negative means DP responds less
   favorably to the source shift. A value near zero means similar shift effects,
   **not a model score of zero**. The shaded band is ±0.5 AP percentage points.

In `calibration.png`, patients are grouped by predicted risk, not sampled
randomly. For each model/repeat, sort all test patients into up to ten quantile
bins, keeping tied predictions together. Each point compares a bin's mean
prediction (x) with its actual readmission fraction (y). Above the diagonal
means underprediction; below means overprediction. All panels share a relevant
risk range; histogram mass outside that range is labeled.

Performance reference lines show constant-score AP, chance AUROC, and Brier for
a constant probability equal to test prevalence. That Brier reference is
descriptive. PR panels reuse the same scratch models; predictions are not
ensembled, and AP is calculated from exact per-run curves.

### Supporting files and cleanup

- `evaluation/supporting/runs.csv` retains each model/repeat's scalar scores.
- `evaluation/supporting/curves.npz` retains PR points, calibration counts/rates,
  and 50-bin prediction histograms. Keys use `repeat_NN/path_id/field`; arrays
  contain no patient IDs or aligned per-patient predictions.
- `data/*.npz` contains processed features, labels, and patient/encounter IDs.
  `source_pairs.npz` stores S0/S1 indices into the source pool. Test arrays are
  created only during evaluation.
- `protocol.json` records choices, resolved defaults, seeds, input hashes,
  partition summaries, tuning results, preprocessing, metric definitions, and
  published-file hashes. This README is the referenced protocol document.

After successful verification, `.work/` is deleted, including weights and aligned
predictions. Retained scores, curves, and protocol metadata can redraw all four
figures. New inference or metrics requiring individual predictions need a new run.

## Resume, verify, and test

```sh
.venv/bin/dp-shift evaluate --config configs/experiment.json --output results/my_experiment --resume
.venv/bin/dp-shift verify all --output results/my_experiment
.venv/bin/python -m pytest -q
```

Completed stages are verified and reused. `--resume` recovers evaluation; partial
or failed training jobs stop for investigation. Tests cover config validation,
real-data preparation, numerical calculations, and a synthetic end-to-end run.
They do not train or evaluate the final real-data experiment.
