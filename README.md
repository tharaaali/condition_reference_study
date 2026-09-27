# Operating conditions and healthy-reference fault detection

This repository contains the code used for the study **“Effect of Operating
Conditions on Healthy-Reference Spectral Fault Detection in Induction Motors.”**

The idea is simple. A healthy motor does not have one fixed current spectrum:
its spectrum changes with load. If every load is pooled into one broad
“healthy” model, that model may tolerate changes that should instead look
anomalous. We compare that global model with an otherwise identical model that
uses healthy recordings from the query's load.

This is a one-class fault detector, not a fault-type classifier. It learns only
from healthy recordings and gives every query an anomaly score. Fault labels
are used afterward to measure ROC-AUC, false-positive rate, and sensitivity.

## What the repository runs

The complete experiment answers three questions:

1. How strongly does load change healthy three-phase current spectra?
2. Does a load-matched healthy reference improve fault detection over one
   reference pooled across all loads?
3. How does performance change when only a few healthy recordings are
   available at each load?

Two public datasets are used:

- **LIMAN-C**, available from [Mendeley
  Data](https://data.mendeley.com/datasets/kccmrf3864/1).
- The **AMPERE rotor-fault branch**, available from
  [dataUBFC](https://search-data.ubfc.fr/FR-13002091000019-2023-03-06-03_AMPERE-Detection-and-diagnostics-of-rotor-and.html).

The code intentionally uses a small model: mean removal, a Hann window, a
one-second FFT, averaged three-phase power, a base-10 logarithm, and a robust
distance from healthy spectra. Frequencies from 5 to 1,000 Hz are retained.
There are no neural-network weights and no fault-label tuning.

## Expected results

The full run should reproduce the following main values, allowing only tiny
floating-point differences:

| Dataset | Global ROC-AUC | Load-matched ROC-AUC |
|---|---:|---:|
| LIMAN-C | 0.953 | 1.000 |
| AMPERE, experiment-block level | 0.740 | 1.000 |

Healthy-only load identification reaches 0.710 on LIMAN-C and 1.000 on AMPERE.
With two healthy references per load, the median matched ROC-AUC is 0.996 and
0.980, respectively.

These numbers need context. LIMAN-C fault state is confounded with source motor
group, so it is treated as a stress test rather than clean causal fault
evidence. AMPERE contains one motor and one test bench. Its 16 numbered files
per experiment are not assumed independent; their scores are aggregated into
25 condition-by-load experiment blocks.

## Installation

Python 3.10 or newer is required. Create an isolated environment and install
the two runtime dependencies.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare the datasets

The datasets are not included in this repository.

### LIMAN-C

Download LIMAN-C from Mendeley Data and extract it. The path passed to the code
may be either the dataset directory or a parent directory containing exactly
one LIMAN-C copy. The runner looks for this structure:

```text
LIMAN-C-or-engine_2/
└── experiment_1/
    └── current/
        ├── 1st_load_0/
        ├── 2nd_load_0/
        └── ...
```

Each condition directory must contain phase folders `1`, `2`, and `3` with the
source CSV files.

### AMPERE

Download the original AMPERE ZIP and leave it compressed. The runner reads only
the 400 rotor measurement pairs needed for the study, skips the stator branch,
and never extracts the 5.73 GB archive. MATLAB-v4 measurements are read
directly; for 81 MATLAB-v5 members, the paired CSV representation is streamed
instead.

## Run the complete study

From the repository root:

```powershell
python run_study.py `
  --limanc-root "D:\datasets\LIMAN-C" `
  --ampere-archive "D:\datasets\AMPERE-Detection_and_Diagnostics_of_Rotor_And_Stator_Faults_In_Rotating_Machines.zip" `
  --output-dir output
```

On Linux or macOS, use the same arguments with your paths and normal shell line
continuation:

```bash
python run_study.py \
  --limanc-root /data/LIMAN-C \
  --ampere-archive /data/AMPERE-Detection_and_Diagnostics_of_Rotor_And_Stator_Faults_In_Rotating_Machines.zip \
  --output-dir output
```

The first run reads the raw datasets and creates compressed FFT feature caches.
Later runs reuse those caches. For a fast pipeline check with fewer bootstrap,
permutation, and training-subset repetitions, append `--quick`. Do not use
`--quick` numbers in the paper.

## Run each stage separately

The full runner calls three transparent stages. They can also be run directly.

```powershell
python run_engine2.py `
  --data-root "D:\datasets\LIMAN-C" `
  --output-dir output\limanc

python run_ampere.py `
  --archive "D:\datasets\AMPERE-Detection_and_Diagnostics_of_Rotor_And_Stator_Faults_In_Rotating_Machines.zip" `
  --output-dir output\ampere

python run_two_dataset_questions.py `
  --limanc-cache output\limanc\engine2_fft_features.npz `
  --ampere-cache output\ampere\ampere_rotor_fft_features.npz `
  --output-dir output\two_dataset
```

## Output files

The important files are:

```text
output/
├── limanc/
│   ├── engine2_fft_features.npz
│   ├── parent_scores.csv
│   └── results.json
├── ampere/
│   ├── ampere_rotor_fft_features.npz
│   ├── block_scores.csv
│   ├── capture_scores_optimistic.csv
│   └── results.json
└── two_dataset/
    ├── two_dataset_results.json
    ├── training_size_summary.csv
    ├── training_size_repeats.csv
    ├── healthy_load_structure.png
    └── training_size_auc.png
```

`results.json` files contain the reported metrics and uncertainty summaries.
The CSV files retain auditable scores. The AMPERE capture-level table is marked
optimistic and is not used for inference; the paper uses block-level results.

## Tests

Run the deterministic unit tests without installing any extra test framework:

```bash
python -m unittest discover -s tests -v
```

The tests check ROC-AUC, the expected benefit of matching on synthetic shifted
loads, dataset-path discovery, and command-line input validation.

## Repository map

- `run_study.py` — one-command end-to-end runner.
- `condition_reference.py` — LIMAN-C reader, FFT feature extraction, reusable
  `HealthyReferenceDetector`, evaluation, and statistics.
- `run_engine2.py` — LIMAN-C experiment entry point (`engine2` is the internal
  dataset name retained from the source project).
- `run_ampere.py` — memory-bounded AMPERE ZIP reader and block-level analysis.
- `run_two_dataset_questions.py` — healthy-load and training-size analyses,
  plus paper figures.
- `tests/` — deterministic unit tests.

## Reproducibility notes

- Default seeds are fixed in every runner.
- Fault labels never enter reference fitting or feature selection.
- Global and load-matched models use the same selected healthy recordings in
  the training-size experiment.
- One-second FFT windows are summarized within their source acquisition; they
  are never counted as independent test samples.
- Dataset files, ZIP archives, generated caches, and outputs are excluded by
  `.gitignore`.

If you use this repository, cite the accompanying paper and the two original
dataset records linked above.

Suggested dataset citations:

- A. Khizhik, S. Ali, A. Ryzhikov, and D. Derkach, “LIMAN-C: Three-Phase
  Induction-Motor Current Measurements under Multiple Fault and Load
  Conditions,” Mendeley Data, version 1, 2026,
  [doi:10.17632/kccmrf3864.1](https://doi.org/10.17632/kccmrf3864.1).
- M. Soualhi, A. Soualhi, K. T. P. Nguyen, K. Medjaher, G. Clerc, and H. Razik,
  “Open Heterogeneous Data for Condition Monitoring of Multi Faults in Rotating
  Machines Used in Different Operating Conditions,” *International Journal of
  Prognostics and Health Management*, vol. 14, no. 2, 2023,
  [doi:10.36001/IJPHM.2023.v14i2.3497](https://doi.org/10.36001/IJPHM.2023.v14i2.3497).
