# defector-api

Automated visual quality control for machine parts using unsupervised anomaly
detection (PatchCore, MVTec AD).

Backend for a bachelor's thesis project at the Faculty of Information
Technologies, University "Džemal Bijedić" in Mostar.

> **Status:** in development. See [Roadmap](#roadmap) for what works today.

## What it does

You upload a photograph of a machine part. The service returns:

- a **verdict** (pass / fail),
- a **numeric anomaly score** and the decision threshold it was compared against,
- a **heatmap** showing where the defect is,
- the **processing time** for that image.

## Why unsupervised

In real manufacturing, defective parts are rare and the set of possible defects
cannot be enumerated in advance. Training a supervised pass/fail classifier
would need a balanced, exhaustively labelled defect set that does not exist.

**PatchCore** sidesteps this: it is trained on defect-free images only. It
extracts patch features with an ImageNet-pretrained CNN, keeps a coreset of them
in a memory bank, and flags anything whose nearest-neighbour distance is too
large. No network weights are trained.

Three part categories from MVTec AD are supported, each with its own model:
`screw`, `metal_nut`, `cable`.

## Requirements

| | |
|---|---|
| Python | 3.12 or newer |
| Hardware | CPU only. No CUDA needed |
| Disk | ~6 GB (dataset ~4.9 GiB + models) |

Note on Python versions: `anomalib` 2.6.0 declares support for 3.10 to 3.12 and
its CI tests 3.10. This project runs on **3.14.7**, outside that range. Every
dependency resolved to a `cp314` wheel except `FrEIA`, which has no wheel and
builds cleanly from source. Imports and a PatchCore forward pass are verified.

One known consequence of being off the tested path: `torch` emits
`FutureWarning: torch.jit.script is not supported in Python 3.14+ and may break`
on import. It is a warning today. If something fails in an unexplained way
later, the interpreter version is the first thing to rule out.

`requirements.lock.txt` pins the exact working set, so the environment is
reproducible regardless of what upstream claims to support.

## Setup

```bash
# 1. Virtual environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2. CPU-only PyTorch FIRST.
#    Skipping this step pulls the CUDA build and several GB of unused
#    nvidia-* packages, because torch reaches this project transitively
#    through lightning and would resolve from PyPI by default.
pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.0 torchvision==0.29.0

# 3. Everything else (torch is already satisfied and will not be refetched)
pip install -e ".[dev]"
```

Verify:

```bash
python -c "import anomalib, torch; print(anomalib.__version__, torch.__version__, torch.cuda.is_available())"
# expected: 2.6.0 2.14.0+cpu False
```

## Dataset

MVTec AD is **not** included in this repository and is not redistributable here.
Download it yourself from
[mvtec.com](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
(a short form is required) and extract it so that the layout is:

```
datasets/MVTecAD/<category>/train/good/*.png
datasets/MVTecAD/<category>/test/good/*.png
datasets/MVTecAD/<category>/test/<defect>/*.png
datasets/MVTecAD/<category>/ground_truth/<defect>/*_mask.png
```

Then confirm it is intact:

```bash
python scripts/01_check_data.py
```

Expected counts:

| category | train/good | test/good | test defective | defect types | resolution |
|---|---|---|---|---|---|
| screw | 320 | 41 | 119 | 5 | 1024×1024 |
| metal_nut | 220 | 22 | 93 | 4 | 700×700 |
| cable | 224 | 58 | 92 | 8 | 1024×1024 |

## Usage

Training and evaluation run as standalone scripts, deliberately outside the web
application. The API only loads finished models at startup.

```bash
python scripts/01_check_data.py         # validate the dataset
python scripts/02_train_detectors.py    # train one PatchCore model per category
python scripts/03_train_classifier.py   # train the category classifier
python scripts/04_evaluate.py           # threshold analysis, false positives vs negatives
python scripts/05_benchmark_latency.py  # per-image processing time
python scripts/06_ablation_routing.py   # category-routing ablation
```

Run the API:

```bash
uvicorn defector.api.app:app --reload
# interactive docs at http://localhost:8000/docs
```

Run the tests. They use a throwaway database and storage directory, so they
never touch analyses made by hand:

```bash
pytest
```

## Configuration

Every path, hyperparameter and limit lives in `src/defector/settings.py` and can
be overridden with `DEFECTOR_`-prefixed environment variables or a `.env` file:

```bash
DEFECTOR_BACKBONE=wide_resnet50_2      # what the committed results use
DEFECTOR_CORESET_SAMPLING_RATIO=0.01
DEFECTOR_TORCH_NUM_THREADS=12
```

## Results

Image-level AUROC with `wide_resnet50_2` and a 1% coreset, on CPU, against the
figures anomalib publishes for the same backbone:

| category | measured | published | delta |
|---|---|---|---|
| screw | 0.9738 | 0.960 | +0.0138 |
| metal_nut | 0.9990 | 0.994 | +0.0050 |
| cable | 0.9908 | 0.990 | +0.0008 |

`artifacts/` also carries the results for `resnet18` under `*_resnet18.json`,
so the backbone comparison is reproducible without retraining.

A separability metric is not an operating point. At the threshold anomalib
calibrates, screw reaches recall 1.0000 while rejecting 37 of 41 good screws,
and improving the backbone did not change that count. See
`artifacts/evaluation.json` and the reasoning in `scripts/04_evaluate.py`.

## Project structure

```
src/defector/
├── settings.py        configuration (paths, categories, hyperparameters, limits)
├── threads.py         PyTorch thread caps; must be imported before torch work
├── ml/                preprocessing, model registry, detector, classifier, overlay
├── data/              SQLite engine, tables, repository
├── api/               FastAPI app, response contracts, routes
└── security/          upload validation
scripts/               training, evaluation and benchmarking entry points
```

## Roadmap

- [x] Project skeleton and configuration
- [x] Environment and dependency lock
- [x] Dataset validation
- [x] PatchCore training and evaluation
- [x] Category classifier
- [x] Routing ablation
- [x] Inference layer and heatmap rendering
- [x] FastAPI endpoints and analysis history
- [x] Threshold analysis
- [x] Latency benchmark

## Licensing

The **code** in this repository is licensed under the GNU General Public License
v2.0; see [LICENSE](LICENSE).

The **MVTec AD dataset** is a separate work under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and is
**not** distributed here. It permits non-commercial use only. This project uses
it for academic research.

If you use the dataset, cite:

> Bergmann, P., Fauser, M., Sattlegger, D., Steger, C. (2019).
> *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly
> Detection.* CVPR, pp. 9584–9592. doi:10.1109/CVPR.2019.00982

The method implemented here is:

> Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., Gehler, P. (2022).
> *Towards Total Recall in Industrial Anomaly Detection.* CVPR, pp. 14298–14308.
> doi:10.1109/CVPR52688.2022.01392
