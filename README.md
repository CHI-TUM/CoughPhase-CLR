<div align="center">
  <h1>CoughPhase-CLR</h1>
  <p><em>An acoustics-informed foundation model for coughing sound classification</em></p>
</div>

-----------------------------------------
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/Marius-Moldo/CoughPhase-CLR)
[![Language: Python](https://img.shields.io/badge/language-Python%203.10%2B-green?logo=python&logoColor=green)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built on: OPERA](https://img.shields.io/badge/built%20on-OPERA-blue.svg)](https://github.com/evelyn0414/OPERA)

This repository contains the code for **CoughPhase-CLR**, a self-supervised learning
framework that leverages the *physiological phases of a cough* for robust representation
learning. Unlike generic contrastive frameworks that build positive pairs through random
cropping, CoughPhase-CLR constructs positive pairs from the distinct acoustic phases of a
single cough event. The model is pretrained on ~40 hours of public cough audio and
evaluated on five downstream tasks spanning COVID-19 detection, COPD state classification,
gender, and smoker-status prediction.

This is the code release for the paper *"CoughPhase-CLR: Designing an acoustics-informed
foundation model for coughing sound classification"* (Moldovan et al., Chair of Health
Informatics, TUM). Key findings: cough-specific phase-aware pretraining consistently
outperforms standard random cropping when pretraining on coughs, and is more
**data-efficient**; however, COPD-state classification from coughs remains hard (best 57 %
UAR vs. 84 % from speech).

> **Built on OPERA.** This work builds on and reuses the pretraining and benchmarking
> scaffolding of [OPERA](https://github.com/evelyn0414/OPERA) (Zhang et al., 2024)

## Method

CoughPhase-CLR shares OPERA-CE's architecture (an **EfficientNet-B0** encoder trained with
a contrastive objective) but differs in how contrastive views are constructed:

- Each cough is split into **two segments**: the **explosive phase** (first segment) and
  the combined **intermediate + voiced phases** (second segment).
- The two segments of the *same* cough form a **positive pair**; segments from other coughs
  in the batch are negatives.
- Mel spectrograms are computed per segment, with
  SpecAugment applied during pretraining.

For comparison, **OPERA-CE-Cough** is OPERA-CE re-pretrained on the *same* cough data but
using the original random-crop view construction. This isolates the effect of the
phase-aware contrastive task.

<div align="center">
<img width=80% alt="CoughPhase-CLR" src="https://github.com/user-attachments/assets/13f98560-2736-46e1-8e8d-7470747e5b43" />
</div>

## Installation

The environment with all the needed dependencies can be created on a Linux machine by
running:

```
git clone <your-repo-url>/CoughPhase-CLR.git
cd ./CoughPhase-CLR

conda env create --file environment.yml
sh ./prepare_env.sh
source ~/.bashrc

conda init
conda activate audio
sh ./prepare_code.sh
```

This sets up a Python 3.10 environment. `prepare_env.sh` adds the project to `PYTHONPATH`; `prepare_code.sh`
patches the custom `swin_transformer.py` into the timm install. Afterwards you only need
`conda activate audio` to run the code.

## Datasets

CoughPhase-CLR is **pretrained** on two openly available cough datasets and **evaluated**
on four datasets (three public + one private).

| Dataset | Used for | Source | Access | License |
| ------- | -------- | ------ | ------ | ------- |
| UK COVID-19 | pretrain + eval | UKHSA / Alan Turing Inst. | [zenodo.org/records/10043978](https://zenodo.org/records/10043978) | OGL 3.0 |
| COUGHVID | pretrain + eval | EPFL | [zenodo.org/records/4048312](https://zenodo.org/records/4048312) | CC BY 4.0 |
| Coswara | eval | IISc | [github.com/iiscleap/Coswara-Data](https://github.com/iiscleap/Coswara-Data) | CC BY 4.0 |
| COPD-DE | eval | Univ. Hospital Augsburg | private — not publicly available | — |

*COVID-19 Sounds (used by OPERA for pretraining) was not available for this work due to
licensing. COPD-DE is a private clinical dataset of forced-cough recordings from COPD
patients (exacerbation vs. stable state) and is not redistributed here.*

## Data preparation

Pretraining uses a **segmented cough** pipeline: long recordings are resampled to 16 kHz,
individual coughs are extracted by energy-based onset/offset detection, each cough is split
into its two acoustic phases, and per-segment Mel spectrograms are saved.

Relevant scripts:

- `src/segment_cough.py`, `src/split_cough_final.py` — cough extraction and two-phase split.
- `src/pretrain/prepare_data/coughvid_pressl_segmented.py` — prepare segmented COUGHVID.
- `src/pretrain/prepare_data/covidUK_pressl_segmented.py` — prepare segmented UK COVID-19.
- `datasets/dataset_filename_utils.py` — shared helpers for building/transforming the
  `.npy` filename lists each dataset uses.

## Pretraining CoughPhase-CLR

The phase-aware model is trained with `src/pretrain/cola_training.py`:

```
python -u src/pretrain/cola_training.py \
  --title "CoughPhase-CLR-100pct" \
  --covidUKcough True --coughvid True \
  --encoder efficientnet \
  --preprocessing segmented \
  --strategy phase \
  --data_percentage 1.0 \
  --specaugment True \
  --target_steps 20000
```

Key flags:

- `--strategy phase` — phase-aware positive pairs (CoughPhase-CLR). The default
  `--strategy crop` reproduces the random-crop **OPERA-CE-Cough** baseline.
- `--preprocessing segmented` — use the segmented per-cough spectrograms.
- `--data_percentage` — fraction of pretraining data (for the data-efficiency study).
- `--target_steps`, `--specaugment`, `--encoder`, `--batch_size` — training controls.

To reproduce the CoughPhase-CLR experiments (data leakage: COUGHVID / covidUK / both with and without the
benchmark test recordings; data size: 100/80/60/40/20 %), including benchmarks and audit logs:

```
bash scripts/clr_experiments.sh
```

Trained checkpoints are written under `cks/model/combined/coughvid_covidUKcough/`.

## Running the benchmark

CoughPhase-CLR is evaluated against baselines on five downstream tasks. The benchmark scripts share
`scripts/lib/benchmark_common.sh`, which runs a feature-extraction phase followed by
`src/benchmark/linear_eval.py`.

| Task | Dataset | `linear_eval` task | Script |
| ---- | ------- | ------------------ | ------ |
| T1 | COUGHVID — COVID / non-COVID | `coughvidcovid` | `linear_eval --task coughvidcovid` (see `scripts/eval_all.sh`) |
| T2 | COUGHVID — female / male | `coughvidsex` | `scripts/coughvid_benchmark.sh` |
| T3 | COPD-DE — exacerbation / stable | `customCoughCOPD` | `scripts/custom_copd_benchmark.sh` |
| T4 | Coswara — smoker / non-smoker | `coswarasmoker` | `scripts/coswara_benchmark.sh` |
| T5 | Coswara — female / male | `coswarasex` | `scripts/data_efficiency_benchmark.sh` |

`scripts/data_efficiency_benchmark.sh` evaluates the 20–100 % checkpoints (eval-only; assumes
features are already extracted). For the public COPD dataset and full fine-tuning baselines,
see `scripts/copd_eval.sh` and `scripts/finetune_eval.sh`.


## Repository layout

- `src/model/` — model definitions (EfficientNet/COLA, HT-SAT, MAE/ViT).
- `src/pretrain/` — pretraining entry points (`cola_training.py`, `mae_training.py`) and
  `prepare_data/` preprocessing.
- `src/benchmark/` — feature extraction (`model_util.py`), linear evaluation
  (`linear_eval.py`), per-dataset `processing/`, and baselines.
- `scripts/` — pretraining and benchmark scripts (+ shared `lib/benchmark_common.sh`).
- `datasets/` — datasets and dataset-specific preprocessing utilities.
- `cks/` — model checkpoints and logs.
- `res_analysis/` — result analysis and visualization.

## Citation

If you use this code, please cite:

```
@misc{moldovan2025coughphaseclr,
      title={CoughPhase-CLR: Designing an acoustics-informed foundation model for coughing sound classification},
      author={Marius Moldovan and Anton Batliner and Thomas M. Berghaus and Bj\"orn W. Schuller and Andreas Triantafyllopoulos},
      year={2026},
}
```

## License

Released under the MIT License
