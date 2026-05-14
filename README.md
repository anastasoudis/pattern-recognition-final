# Pattern Recognition — Final Project

Two-part final project from the **Pattern Recognition** course at Democritus University of Thrace (Fall 2023 / early 2024).

The project combines a computer-vision classification task with anomaly detection (Part 1) and a high-dimensional drug-discovery / cheminformatics task (Part 2). The headline result:

> **AUC 0.9509** on the drug-discovery binary classifier — predicting whether a chemical molecule binds to a target biological receptor across 3,473 mixed-type features.

## At a glance

| Part | Task | Approach | Result |
|---|---|---|---|
| 1 — [Face-Mask Classifier with Anomaly Detection](part1-mask-classifier/) | Binary image classification on faces (with-mask vs without-mask) plus identification of an unseen "incorrect-use" class that is never shown during training | CNN built in Keras; hyperparameters tuned with **Keras-Tuner**; anomaly-detection head that flags unseen classes using a calibrated **threshold-score of 0.0005** on the softmax output | Solid validation accuracy; the anomaly head correctly flagged a significant share of the held-out "mask-incorrect-use" images as out-of-distribution |
| 2 — [Drug-Discovery Binary Classifier](part2-drug-discovery/) | Predict whether a chemical molecule binds to a target biological receptor — a screening problem used in early-stage pharmacology to surface candidate active substances before wet-lab work | **PCA** for dimensionality reduction on the mixed continuous + binary feature space, followed by a **CNN** classifier that outputs binding probability per molecule | **AUC 0.9509** on the held-out evaluation; calibrated prediction scores delivered as `test_predictions.csv` per the assignment specification |

The 8-page report ([`report/Report_Final_Project.pdf`](report/Report_Final_Project.pdf)) and 7.5 MB slide deck ([`report/slides.pptx`](report/slides.pptx)) cover both parts end-to-end — methodology, code walk-through, results, and figures.

## Part 1 — Face-Mask Classifier with Anomaly Detection

The training set was [`Mask_DB.zip`](https://drive.google.com/) (provided with the assignment): 1,044 `with_mask` images, 1,044 `without_mask` images, plus 56 `mask_incorrect_use` images held out as the **unseen class**.

**Pipeline:**

1. **Data split** — 60 / 20 / 20 train / validation / test, stratified across `with_mask` and `without_mask` only.
2. **Model selection** — chose CNN over classical (SVM, Random Forest) because the task is image classification with non-trivial spatial structure. Trade-off acknowledged: more compute and overfitting risk, mitigated by tuning + validation monitoring.
3. **Hyperparameter search** — Keras-Tuner sweep over filter counts, dense-layer widths, and dropout rates, with an `EarlyStopping` callback (`patience = 3`).
4. **Held-out evaluation** — final accuracy, AUC, and precision-recall curves on the test set.
5. **Anomaly detection** — feed the unseen `mask_incorrect_use` images through the trained model. The CNN produces low-confidence softmax outputs on these. A threshold-score of **0.0005** on the maximum softmax probability flags a sample as "neither with_mask nor without_mask". This is the cleanest of the four mitigation strategies considered (data augmentation, cost-sensitive loss, transfer learning, anomaly detection).

Code: [`part1-mask-classifier/notebook.ipynb`](part1-mask-classifier/notebook.ipynb).

## Part 2 — Drug-Discovery Binary Classifier

**Problem framing:** virtual screening — given features of a chemical molecule, predict whether it binds effectively to a target biological receptor (label 1) or not (label 0).

**Dataset (provided as `Data_Receptors.zip`):**

| Field | Value |
|---|---|
| Training molecules (`Train_features.csv` + `Train_labels.csv`) | 1,115 |
| Test molecules (`test_features.csv`) | 124 |
| Features per molecule | **3,473** |
| Continuous physicochemical descriptors (columns 1 – 1,425) | 1,425 |
| Binary molecular fingerprints (columns 1,426 – 3,473) | 2,048 |

**Pipeline:**

1. **Preprocessing** — separate continuous and binary blocks; scale the continuous block (centring + variance normalisation); leave the binary fingerprints untouched.
2. **Dimensionality reduction** — **PCA** on the scaled continuous block to suppress the curse of dimensionality and decorrelate inputs before the CNN.
3. **Classifier** — CNN that consumes the PCA-projected continuous features alongside the raw binary fingerprints; binary cross-entropy loss; the network outputs `p(binding | features) ∈ [0, 1]`.
4. **Optimisation protocol** — k-fold cross-validation on the training set to choose model hyperparameters; held-out estimate of test-set error reported.
5. **Result** — **AUC 0.9509** on the cross-validated evaluation.
6. **Output** — `test_predictions.csv` with one `predicted_label, prediction_score` row per test molecule, in the order of `test_features.csv`. The `prediction_score` is the raw classifier output (binding probability).

Code: [`part2-drug-discovery/notebook.ipynb`](part2-drug-discovery/notebook.ipynb).
Predictions file: [`part2-drug-discovery/test_predictions.csv`](part2-drug-discovery/test_predictions.csv) — 123 rows.

## Repository structure

```
.
├── report/
│   ├── Report_Final_Project.pdf       # 8-page report covering both parts (Greek)
│   └── slides.pptx                    # Presentation slides (Greek)
├── part1-mask-classifier/
│   └── notebook.ipynb                 # CNN + anomaly head on Mask_DB
└── part2-drug-discovery/
    ├── notebook.ipynb                 # PCA + CNN on Data_Receptors
    └── test_predictions.csv           # Deliverable: 123 rows of predicted_label,prediction_score
```

The report is in Greek and embeds the original problem statement for each part, so the assignment context is preserved without needing a separate brief. The slides are in Greek; the README and notebook prose are mixed Greek / English.

## Running the code

Python 3.10+ recommended.

```bash
python -m venv .venv && source .venv/bin/activate
pip install jupyter numpy pandas matplotlib scikit-learn tensorflow keras-tuner
jupyter notebook
```

**Datasets are not redistributed in this repository.** Both notebooks were originally run on Google Colab against the assignment-provided archives:

- `Mask_DB.zip` — uploaded into the Colab `content/` folder before running Part 1.
- `Data_Receptors.zip` — uploaded into the Colab `content/` folder before running Part 2; it contains `Train_features.csv`, `Train_labels.csv`, and `test_features.csv`.

Equivalent datasets:

- Face-mask classification: the [`andrewmvd/face-mask-detection`](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) dataset on Kaggle is a close substitute for the assignment's `Mask_DB`.
- Molecular binding: the assignment's `Data_Receptors` was a curated subset for the course; comparable public benchmarks include the **MoleculeNet** suite (`BACE`, `HIV`).

## Course

Pattern Recognition (Αναγνώριση Προτύπων) — Department of Electrical & Computer Engineering, Democritus University of Thrace. Fall 2023 / early 2024.

## License

[MIT](LICENSE) — shared as-is for educational reference.

## Author

[Dimitrios Anastasoudis](https://github.com/anastasoudis) · [LinkedIn](https://linkedin.com/in/anastasoudis)
