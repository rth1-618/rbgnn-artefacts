# Recurrent Bipartite Graph Neural Network (R-BGNN) for QC-MDPC Decoding

This repository contains the complete PyTorch implementation, locked evaluation datasets, trained model weights, and raw experimental results for the Recurrent Bipartite Graph Neural Network (R-BGNN) decoder designed for Quasi-Cyclic Moderate-Density Parity-Check (QC-MDPC) codes in McEliece-style post-quantum cryptosystems.

The R-BGNN architecture integrates Gated Recurrent Units (GRUs) directly into the check and variable nodes of the Tanner graph. This provides temporal message memory across unrolled iterations, allowing the decoder to identify and damp cycle-driven Log-Likelihood Ratio (LLR) oscillations without modifying the underlying secret key matrix.

This repository includes a pre-executed reference output folder, `QCMDPC_RBGNN_N254/`, allowing journal editors, reviewers, and researchers to verify all datasets, trained network parameters, baseline evaluations, and publication figures directly without needing to re-run the training pipeline.

### [Click here to read the paper on arXiv](https://doi.org/10.48550/arXiv.2610.38479)

---

## 1. Quick Artifact Inspection Guide for Editors and Reviewers

To inspect key experimental claims and raw data without re-running `final-final.ipynb`, refer to the locations within `QCMDPC_RBGNN_N254/` outlined below.

| Target Verification Item           | Relative File Path in `QCMDPC_RBGNN_N254/`             | Description & Key Contents                                                                                                   |
| :--------------------------------- | :----------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------- |
| **Trained GNN Weights**            | `Phase_4_Training/best_inference_model.pth`            | PyTorch checkpoint containing state dict (110,145 parameters) and training configuration.                                    |
| **Locked Validation Datasets**     | `Phase_1_Topology/Locked_Validation/`                  | 8 locked parity-check matrices (`val_graph_000` to `007`) and compressed `.npz` evaluation frames.                           |
| **Locked Test Evaluation Suites**  | `Phase_1_Topology/Locked_Test/`                        | 20 unseen master key matrices (`test_graph_000` to `019`) with 220,000 total evaluation frames.                              |
| **Classical Baseline Outputs**     | `Phase_2_Baselines/classical_baseline_results.json`    | Baseline metrics for Bit-Flipping (BF) and Attenuated Min-Sum (MS) decoders across all test graphs.                          |
| **Raw Neural Evaluation Results**  | `Phase_5_Evaluation/final_evaluation_results.json`     | Frame-by-frame Decoding Failure Rate (DFR), Bit Error Rate (BER), and iteration counts across error weights $t \in [2, 12]$. |
| **Execution Latency Data**         | `Phase_5_Evaluation/latency_benchmark.json`            | Per-frame inference latency (ms) benchmarking R-BGNN against Bit-Flipping and Min-Sum decoders.                              |
| **Publication Performance Tables** | `Phase_5_Evaluation/pooled_performance_table.csv`      | Summary CSV table reporting pooled DFR, BER, iteration speed, and confidence intervals across 20 test graphs.                |
| **Per-Graph Breakdown Table**      | `Phase_5_Evaluation/graph_level_performance_table.csv` | Granular per-graph evaluation breakdown across all 20 unseen parity-check matrices.                                          |
| **Residual Failure Weight Data**   | `Phase_5_Evaluation/failure_topology_association.json` | Residual error bit distributions showing isolation down to 1–3 bits per frame.                                               |
| **Master Research Log**            | `FINAL_RESEARCH_SUMMARY.json`                          | Consolidated JSON summary including exact hyperparameter configurations, seed settings, and system metadata.                 |

---

## 2. Complete Repository Directory Structure

The repository root contains the core script/notebook (`final-final.ipynb`), the python environment specification (`requirements.txt`), and the full executed output directory (`QCMDPC_RBGNN_N254/`).

```text
.
├── README.md
├── requirements.txt
├── final-final.ipynb
└── QCMDPC_RBGNN_N254/
    ├── config_hyperparams.json
    ├── system_metadata.json
    ├── sanity_checks.json
    ├── FINAL_RESEARCH_SUMMARY.json
    ├── artifact_manifest.json
    │
    ├── Phase_1_Topology/
    │   ├── H_master.npy
    │   ├── master_topology_profile.json
    │   ├── master_structural_certificate.json
    │   ├── Phase_1_Topological_Profile.png
    │   ├── Locked_Validation/
    │   │   ├── validation_suite_index.json
    │   │   ├── validation_topology_summary.json
    │   │   └── val_graph_000 ... val_graph_007/
    │   │       ├── H.npy
    │   │       ├── locked_dataset.npz
    │   │       ├── topology.json
    │   │       └── structural_certificate.json
    │   └── Locked_Test/
    │       ├── test_suite_index.json
    │       ├── test_topology_summary.json
    │       └── test_graph_000 ... test_graph_019/
    │           ├── H.npy
    │           ├── locked_dataset.npz
    │           ├── topology.json
    │           └── structural_certificate.json
    │
    ├── Phase_2_Baselines/
    │   ├── bitflip_threshold_tuning.json
    │   ├── bitflip_threshold_tuning.csv
    │   └── classical_baseline_results.json
    │
    ├── Phase_3_Architecture/
    │   ├── model_architecture_summary.json
    │   ├── architecture_verification.json
    │   └── untrained_model_checkpoint.pth
    │
    ├── Phase_4_Training/
    │   ├── best_gnn_checkpoint.pth
    │   ├── latest_gnn_checkpoint.pth
    │   ├── best_inference_model.pth
    │   ├── best_validation_snapshot.json
    │   ├── training_metrics_log.json
    │   └── Phase_4_Training_Profile.png
    │
    └── Phase_5_Evaluation/
        ├── final_evaluation_results.json
        ├── latency_benchmark.json
        ├── failure_topology_association.json
        ├── pooled_performance_table.csv
        ├── graph_level_performance_table.csv
        ├── Phase_5_Master_Publication_Plot.png
        ├── Phase_5_Graph_Generalization.png
        ├── Phase_5_Supplementary_Analysis.png
        └── Master_Statistical_Dashboard_FULLY_ANNOTATED.png
```

---

## 3. Detailed Data & Artifact Descriptions

### Phase 1: Matrix Topology and Locked Evaluation Suites
The code generates a two-block Quasi-Cyclic Moderate-Density Parity-Check (QC-MDPC) matrix parameter set ($N = 254$, $r = 127$, $W = 14$, $d = 7$).
* `H_master.npy`: Binary parity-check matrix for the reference master graph stored as a NumPy array.
* `master_structural_certificate.json`: Topological verification details, including exact 4-cycle counts and graph degree checks.
* `Locked_Validation/`: Contains 8 fixed validation graph directories (`val_graph_000` through `val_graph_007`). Each directory stores its parity matrix (`H.npy`) and pre-generated error vectors (`locked_dataset.npz`) used to evaluate validation loss/DFR at every training epoch.
* `Locked_Test/`: Contains 20 fixed test graph directories (`test_graph_000` through `test_graph_019`). Each directory stores pre-generated test datasets spanning error weights $t \in [2, 12]$ (1,000 frames per error weight per graph, totaling 220,000 frames).

### Phase 2: Classical Baseline Benchmark Data
* `classical_baseline_results.json`: Benchmarks Bit-Flipping and Attenuated Min-Sum algorithms across all 20 locked test graphs.
* `bitflip_threshold_tuning.json` & `.csv`: Grid-search logs used to select optimal bit-flipping thresholds.

### Phase 3 & 4: Neural Architecture and Model Checkpoints
* `best_inference_model.pth`: The primary PyTorch model checkpoint selected based on validation performance during Phase 4 curriculum training. It contains the trained parameters (110,145 parameters) and hyperparameter metadata.
* `training_metrics_log.json`: Epoch-by-epoch loss, syndrome loss, validation DFR, and learning rate progression over 60 epochs.
* `Phase_4_Training_Profile.png`: Visual plot showing convergence profiles across training epochs.

### Phase 5: Final Multi-Graph Evaluation & Figures
* `final_evaluation_results.json`: Complete numerical evaluation output covering R-BGNN, Bit-Flipping, and Attenuated Min-Sum decoders across 20 unseen test matrices.
* `pooled_performance_table.csv`: Aggregated performance metrics (DFR, BER, average iterations, and 95% confidence intervals) across all test graphs.
* `graph_level_performance_table.csv`: Detailed breakdown showing performance on each individual test matrix topology.
* `latency_benchmark.json`: Execution time benchmarks measuring inference latency across all decoders on identical hardware.
* `failure_topology_association.json`: Analysis of residual error weights when decoding fails, showing that R-BGNN residual errors are isolated to small 1–3 bit patterns.
* `Master_Statistical_Dashboard_FULLY_ANNOTATED.png`: Publication dashboard displaying DFR waterfall curves, iteration speedups, and generalization distributions.

---

## 4. Setup & Environment Requirements

### Requirements
The code requires Python 3.9+ and standard scientific packages. Dependencies are listed in `requirements.txt`:
* `torch >= 2.0.0`
* `torch-geometric >= 2.3.0`
* `networkx >= 3.0`
* `numpy >= 1.22.0`
* `scipy >= 1.8.0`
* `pandas >= 1.5.0`
* `scikit-learn >= 1.0.0`
* `matplotlib >= 3.5.0`

### Execution Instructions
To re-run the entire pipeline from scratch:
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the notebook or Python script:
   ```bash
   jupyter notebook final-final.ipynb
   ```
   *Note: Execution creates a new timestamped output directory while leaving `QCMDPC_RBGNN_N254/` intact as the reference benchmark run.*

---
