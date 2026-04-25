# toss-next-challenge-solution

5th place solution for the **toss Ad CTR (Click-Through Rate) Prediction Challenge**.

This repository contains:
- reproducible training / inference pipelines,
- multiple model families (boosting + deep CTR models),
- and the final ensemble recipe used for submission.

---

## 1) Environment Setup

We manage dependencies with [Poetry](https://python-poetry.org/).

### Requirements
- Python **3.11.x**
- Poetry **2.1.1**

```bash
python --version
poetry --version
```

If your Python version is lower than 3.11, install and switch with `pyenv` (or your preferred version manager), then run:

```bash
poetry env use python3.11
poetry install
```

Optional: inspect the created virtual environment.

```bash
poetry env info
```

---

## 2) Git Hooks (Recommended)

Enable automatic linting/format checks on commit:

```bash
poetry run pre-commit install
```

---

## 3) Dataset Layout

Place competition files under `input/toss-next-challenge/`:

```text
input/
└── toss-next-challenge/
    ├── train.parquet
    ├── test.parquet
    └── sample_submission.csv
```

---

## 4) Quick Start

### Train all key models

```bash
sh scripts/train.sh
```

What this script runs (high level):
1. `src/preprocessor.py`
2. 5-fold training for `dcn` and `dcn_v2` (sequence-aware variants)
3. LightGBM (DART) training with multiple seeds
4. XGBoost training

### Run inference + ensemble

```bash
sh scripts/inference.sh
```

What this script runs (high level):
1. `src/preprocessor.py`
2. 5-fold inference for `dcn` and `dcn_v2`
3. per-model CV submission generation
4. LightGBM / XGBoost inference
5. final blending via `src/ensemble.py`

Final outputs are generated in `output/`.

---

## 5) Solution Architecture

### Ensemble

![Ensemble](https://github.com/user-attachments/assets/6bba8d01-c5e1-4744-a7d0-0ebd6d38ffcf)

The final submission is a sigmoid-based blend of:
- LightGBM (DART),
- XGBoost,
- sequence-aware DCN,
- sequence-aware DCN V2.

### Why LightGBM DART?

[DART](https://arxiv.org/abs/1505.01866) introduces tree dropout during boosting, which helped stabilize generalization in our experiments.

Observed strengths for this task:
- robust behavior on sparse / high-cardinality features,
- stable CV trends across folds,
- good resilience under shift / imbalance.

### Seq-aware DCN family

- [DCN](https://arxiv.org/abs/1708.05123) + sequence encoding (MHA)
- [DCN V2](https://arxiv.org/abs/2008.13535) + sequence encoding (MHA)

![Seq-aware DCN](https://github.com/user-attachments/assets/44bfb186-313c-401c-80f5-d1a0eb6f9c37)

![Seq-aware DCN V2](https://github.com/user-attachments/assets/4ae3802d-1e89-4763-892f-830a9634e8be)

---

## 6) Implemented Models and Results

| Model | CV Score | Public LB | Private LB | Used in Final Ensemble |
|---|---:|---:|---:|:---:|
| Sigmoid Ensemble | - | **0.35126** | **0.35073** | ✅ |
| LightGBM | 0.35501 | 0.35024 | 0.34960 | ✅ |
| XGBoost | 0.35489 | 0.34788 | 0.34757 | ✅ |
| dcn_v2_seq | 0.35375 | 0.34471 | 0.34452 | ✅ |
| dcn_seq | 0.35345 | 0.34645 | 0.34602 | ✅ |
| CatBoost | 0.34348 | 0.34804 | 0.34790 | ❌ |
| dcn_v2 | 0.35395 | 0.34512 | NA | ❌ |
| dcn | - | 0.34709 | 0.346708 | ❌ |
| ffm_seq | - | NA | NA | ❌ |
| ffm | - | 0.34579 | 0.34565 | ❌ |
| xdeepfm_seq | - | NA | NA | ❌ |
| xdeepfm | 0.34861 | 0.34321 | NA | ❌ |
| deepfm_seq | 0.35219 | 0.34497 | NA | ❌ |
| deepfm | - | NA | NA | ❌ |
| fm_seq | - | NA | NA | ❌ |
| fm | - | NA | NA | ❌ |
| fibinet | - | NA | NA | ❌ |

---

## 7) Main Config Files

| Model Family | Config |
|---|---|
| LightGBM | `config/models/lightgbm.yaml` |
| XGBoost | `config/models/xgboost.yaml` |
| CatBoost | `config/models/catboost.yaml` |
| FM-based models | `config/models/fm.yaml` |
| Final blend | `config/blends/total.yaml` |

---

## 8) Reproducibility Notes

- Keep `poetry.lock` committed for deterministic environments.
- When adding a dependency:

```bash
poetry add "package==x.y.z"
poetry lock
```

- Prefer running provided scripts (`train.sh`, `inference.sh`) for end-to-end consistency.

---

## 9) License

This project is licensed under the terms in [LICENSE](./LICENSE).
