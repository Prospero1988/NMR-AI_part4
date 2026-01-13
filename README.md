# NMR-AI_part4

This repository contains a fully reproducible machine-learning pipeline for training, optimizing, and evaluating one-dimensional convolutional neural networks (1D CNNs) on molecular descriptor data.  
The workflow was developed as part of the *NMR-AI* project and accompanies a scientific publication submitted to *Angewandte Chemie*.

---

## Overview

The code implements a complete regression framework based on PyTorch, Optuna, and MLflow, designed for fixed-length molecular representations such as NMR-derived descriptors, fingerprints, or hybrid feature vectors.

The pipeline is intended for research-grade model development and reproducible benchmarking.

---

## Main features

- PyTorch-based 1D CNN regression models  
- Bayesian hyperparameter optimization with Optuna  
- k-fold cross-validation and held-out test evaluation  
- Early stopping with rollback to best validation epoch  
- Latent-space applicability domain analysis  
- Full experiment tracking using MLflow  

---

## Repository contents

```text
.
├── CNN_1D_pytorch.py        # Main training, optimization, and evaluation script
├── tags_config_CNN_1D.py    # Required MLflow tag configuration (mandatory)
└── README.md                # This file
```

---

## Required configuration

The script requires the presence of the file `tags_config_CNN_1D.py` in the same directory.

This file defines MLflow tag dictionaries used during different stages of the workflow
(hyperparameter optimization, evaluation, and final training).
The script imports this file unconditionally; therefore, it must be present for successful execution.

An example structure of the file is shown below:

```python
mlflow_tags1 = {
    "architecture": "Pytorch",
    "model": "CNN",
    "stage": "Optuna HP"
}

mlflow_tags2 = {
    "architecture": "Pytorch",
    "model": "CNN",
    "stage": "evaluation"
}

mlflow_tags3 = {
    "architecture": "Pytorch",
    "model": "CNN",
    "stage": "training"
}
```

The exact tag content may be adapted to local experiment-tracking conventions,
but the file itself is mandatory.

---

## Input data format

The script expects input data in CSV format, where:

- the first column is a sample identifier (ignored during training)
- one column named `LABEL` contains the regression target
- all remaining columns are numerical descriptors (floats or integers)

Example structure:

```csv
ID,f1,f2,f3,...,fN,LABEL
mol_001,0.12,1.0,0.0,...,0.45
mol_002,0.08,0.0,1.0,...,0.62
```

Missing values are handled deterministically using median imputation,
fitted exclusively on training data to prevent information leakage.

---

## Usage

Basic execution:

```bash
python CNN_1D_pytorch.py \
    --csv_path path/to/csv_directory \
    --experiment_name MyExperiment \
    --n_trials 1000
```

Arguments:

- `--csv_path` – directory containing one or more CSV files  
- `--experiment_name` – MLflow experiment name  
- `--n_trials` – number of Optuna optimization trials  

Each CSV file is processed independently.

---

## Output and artifacts

For each dataset, the pipeline produces:

- optimized model weights (`*.pth`)
- cross-validation and test-set metrics
- prediction CSV files
- Optuna parameter importance plots
- Williams plots in latent feature space
- applicability domain diagnostics (Mahalanobis distance)
- complete MLflow experiment logs and artifacts

---

## Reproducibility

Reproducibility is enforced through:

- fixed global random seed  
- deterministic PyTorch and CUDA settings  
- persisted train/test split indices  
- explicit logging of all hyperparameters and model states  

Given identical input data and software versions,
the pipeline yields identical results.

---

## Software requirements

Developed and tested with:

- Python ≥ 3.12  
- PyTorch ≥ 2.4  
- scikit-learn ≥ 1.5  
- Optuna ≥ 3.6  
- MLflow ≥ 2.13  
- NumPy, SciPy, pandas, matplotlib  

CUDA-enabled GPUs are supported but not required.

---

## Scope and limitations

This code is intended for:

- scientific benchmarking of molecular representations  
- methodological studies on neural networks  
- applicability domain analysis in latent feature spaces  

It is not designed as a production inference service or end-user application.

---

## Citation

If you use this code in academic work, please cite the associated publication:

From NMR to AI: reference to be added upon publication

---

## Author

Arkadiusz Leniak  
Computational chemistry, NMR spectroscopy, and machine learning

---

## License

Released for academic and research use.
