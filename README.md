# EcoSort Drift Guard
**Zero-Label On-Device Model Degradation Monitoring for Edge Waste Sorting**

---

## Overview

EcoSort Drift Guard is a local, self-contained machine learning monitoring application designed to detect real-world model degradation without relying on cloud logging or ground-truth labels. Built for an edge waste sorting pipeline classifying items into four target recycling streams (**Glass**, **Plastic**, **Metal**, and **Paper**), the system pairs a pretrained ResNet-50 vision backbone (`kendrickfff/my_resnet50_garbage_classificationv1.2`) with a two-stage unsupervised telemetry engine:

1. **Stage 1 — Input Distribution Shift (Covariate Drift):** Intercepts $2048$-dimensional penultimate feature embeddings via a non-invasive PyTorch forward pre-hook on `model.fc` and measures the $L_2$ Euclidean distance against a calibrated Hybrid Baseline.
2. **Stage 2 — Prediction Uncertainty (Concept/Boundary Drift):** Computes the normalized Shannon entropy of the $4$-class softmax distribution to track classification confidence.

When environmental degradation—such as low ambient bin illumination, lens smudges, or sensor noise—pushes both feature representations and prediction uncertainty beyond statistical safety bounds, the application triggers a timed **"It's Getting Worse"** operational alert.

---

## System Architecture

```text
                                [ Camera Input Frame ]
                                          |
                                          v
                               +----------------------+
                               |  ResNet-50 Backbone  |
                               |  (conv1 to layer4)   |
                               +----------+-----------+
                                          |
                        Global Average Pooling (2048-D)
                                          |
                  +-----------------------+-----------------------+
                  |                                               |
                  v                                               v
    [ PyTorch Forward Pre-Hook ]                    [ FourClassLogSumExpHead ]
                  |                                               |
         L2 Normalization                                12 Pretrained Logits
        ||e||_2 = 1.0 (2048-D)                                    |
                  |                                         LogSumExp Merge
                  v                                               |
     [ Hybrid Baseline Centroid ]                       4 Target Class Logits
       (Factory + User Golden)                                    |
                  |                                            Softmax
                  v                                               v
      L2 Distance Calculation                           Normalized Shannon Entropy
        D_L2 = ||e - mu_hyb||_2                           H_norm in [0.0, 1.0]
                  |                                               |
                  +-----------------------+-----------------------+
                                          |
                                          v
                           [ Rolling Buffer (W = 5 Frames) ]
                                          |
                                          v
                         [ 4-State Health Decision Matrix ]
                                          |
            +-----------------------------+-----------------------------+
            |                             |                             |
            v                             v                             v
       [ Nominal ]               [ Covariate Warning ]          [ Critical Alert ]
     System Healthy                 Inspect Camera            "It's Getting Worse"
```

---

## Core Technical Methodology

### 1. Non-Invasive Feature Extraction via Forward Pre-Hooks
Instead of modifying the ResNet-50 `forward()` pass or decomposing the network into separate sub-modules, the monitoring engine registers a PyTorch forward pre-hook directly on the final classification module (`model.fc`):

```python
model.fc.register_forward_pre_hook(capture_fc_input_embedding)
```

This intercepts the flattened $(B, 2048)$ feature vector exiting `model.avgpool` immediately before linear projection, detaches it from the autograd graph to prevent memory accumulation across Streamlit reruns, and normalizes it onto the $2048$-dimensional unit hypersphere ($\|\mathbf{e}\|_2 = 1.0$).

### 2. Exact 12-to-4 Target Consolidation via LogSumExp
The source checkpoint (`kendrickfff/my_resnet50_garbage_classificationv1.2`) is trained on $12$ fine-grained garbage classes, where target materials are partitioned into sub-classes (e.g., `brown-glass`, `green-glass`, and `white-glass` for **Glass**; `paper` and `cardboard` for **Paper**). Averaging linear weight rows causes negative activations from inactive sub-classes to suppress active detections.

By the law of total probability under a softmax model, the exact probability of a merged target class $C_k$ is the sum of its constituent sub-class probabilities, which corresponds to the $\text{LogSumExp}$ of the raw logits:

$$z_{C_k}(\mathbf{x}) = \ln \sum_{j \in C_k} \exp\left(z_j(\mathbf{x})\right) = \text{LogSumExp}_{j \in C_k}\left(z_j(\mathbf{x})\right)$$

### 3. Dynamic Hybrid Baseline and Anti-Poisoning Gatekeeper
To account for domain shift between factory training images and a user's local camera setup, the reference centroid $\boldsymbol{\mu}_{\text{hyb}}$ dynamically blends an immutable Factory Centroid ($\boldsymbol{\mu}_{\text{fact}}$) with the running centroid of the user's first $K = 5$ high-confidence uploads ($\boldsymbol{\mu}_{\text{user}}$):

$$\boldsymbol{\mu}_{\text{hyb}} = \text{Normalize}\left((1 - \alpha)\boldsymbol{\mu}_{\text{fact}} + \alpha \boldsymbol{\mu}_{\text{user}}\right), \qquad \alpha = \min\left(0.40, \, 0.08 \times N_{\text{accepted}}\right)$$

To prevent degraded initial frames from poisoning the local baseline during calibration, a candidate frame is admitted into $\boldsymbol{\mu}_{\text{user}}$ only if it satisfies both gatekeeper conditions:

$$H_{\text{norm}}(\mathbf{p}) \le \tau_{H,\text{golden}} \quad (0.425, \text{ equivalent to } p_{\max} \ge 0.85) \qquad \text{and} \qquad D_{L2}(\mathbf{e}, \boldsymbol{\mu}_{\text{fact}}) \le \tau_{D,\text{warn}}$$

### 4. Normalized Shannon Entropy
Shannon entropy over the $C = 4$ target classes is normalized by $\ln(4)$ so that uncertainty always scales strictly within $[0.0, 1.0]$:

$$H_{\text{norm}}(\mathbf{p}) = -\frac{1}{\ln(4)} \sum_{c=1}^{4} p_c \ln(p_c + \epsilon)$$

---

## Repository Structure

```text
ecosort_resnet50_drift/
├── requirements.txt              # Python package dependencies
├── resnet_model.py               # ResNet-50 12-to-4 LogSumExp head & preprocessing transforms
├── setup_model_and_baseline.py   # Checkpoint downloader, target verifier, and baseline generator
├── drift_engine.py               # Forward pre-hook, Hybrid Baseline, and rolling telemetry engine
├── app.py                        # Streamlit monitoring interface and drift perturbation laboratory
├── artifacts/                    # Generated by setup_model_and_baseline.py
│   └── resnet50_4class_bundle.pt # Serialized 4-class weights, centroids, and calibrated thresholds
└── sample_images/                # Clean reference samples (Glass, Plastic, Metal, Paper)
```

---

## Installation and Setup

### Prerequisites
* Python $3.10$ or higher
* `pip` and `venv`

### 1. Create and Activate Virtual Environment
```bash
mkdir ecosort_resnet50_drift
cd ecosort_resnet50_drift
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Initialize Model Bundle and Factory Baseline
Run the setup script once before launching the application. This downloads `resnet50_garbage_classificationv1.2.pth` from Hugging Face, validates the $12$-to-$4$ class mapping against reference images, extracts $2048$-dimensional unit embeddings, and computes the empirical $L_2$ dispersion thresholds ($\tau_{D,\text{warn}} = \mu_d + 2\sigma_d$ and $\tau_{D,\text{crit}} = \mu_d + 3\sigma_d$):

```bash
python setup_model_and_baseline.py
```

---

## Running the Application

Start the local Streamlit server:

```bash
streamlit run app.py
```

The interface will open in your default browser at `http://localhost:8501`.

---

## Telemetry Decision Matrix

To prevent single-frame motion blur from causing false alarms, the engine evaluates rolling averages ($\bar{D}_{L2}$ and $\bar{H}_{\text{norm}}$) across a sliding window of $W = 5$ frames:

| System State | Threshold Condition | Diagnostic Meaning | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Nominal** | $\bar{D}_{L2} \le \tau_{D,\text{warn}}$ and $\bar{H}_{\text{norm}} \le \tau_{H,\text{crit}}$ | Input features lie within the calibrated manifold and class predictions are decisive. | None. Sorting operates normally. |
| **Covariate Warning** | $\bar{D}_{L2} > \tau_{D,\text{warn}}$ and $\bar{H}_{\text{norm}} \le \tau_{H,\text{crit}}$ | Environmental conditions (lighting or optics) have shifted, but classification remains confident. | Inspect sorting bay illumination and check camera lens for early dust buildup. |
| **Ambiguous Object** | $\bar{D}_{L2} \le \tau_{D,\text{warn}}$ and $\bar{H}_{\text{norm}} > \tau_{H,\text{crit}}$ | Camera conditions are normal, but the waste item sits between decision boundaries. | Flag item for manual inspection (e.g., multi-material packaging). |
| **Critical Drift** | $\bar{D}_{L2} > \tau_{D,\text{crit}}$ or ($\bar{D}_{L2} > \tau_{D,\text{warn}}$ and $\bar{H}_{\text{norm}} > \tau_{H,\text{crit}}$) | Severe feature manifold shift and elevated decision uncertainty. Sorting reliability is compromised. | Triggers the timed **"It's Getting Worse"** alert. Clean camera optics or restore lighting. |

---

## Drift Perturbation Testing Procedure

The sidebar includes a **Drift Perturbation Lab** designed to test monitoring responsiveness during live demonstrations:

1. **Baseline Calibration:** Select clean sample items (`Glass`, `Plastic`, `Metal`, `Paper`) with sliders at nominal defaults (`Ambient Lighting = 1.00`, `Lens Blur = 0.0`, `Sensor Noise = 0.00`). Observe the **Golden Uploads** counter increment from `0 / 5` to `5 / 5` as the Hybrid Baseline adapts up to $\alpha = 0.40$.
2. **Covariate Shift Test:** Reduce **Ambient Lighting Level** to `0.55`. Observe Stage 1 Rolling $L_2$ Shift cross $\tau_{D,\text{warn}}$ prior to classification failure.
3. **Critical Degradation Test:** Click **Induce Drift** (or set `Lens Blur = 4.5` and `Ambient Lighting = 0.25`). Observe both Rolling $L_2$ Distance and Normalized Entropy breach critical thresholds, activating the timed alert banner.
4. **System Recovery:** Click **Clean Lens** to restore nominal optical inputs and observe the rolling telemetry return to the **Nominal** state within $5$ frames.

---

## Technical Specifications

| Component | Specification |
| :--- | :--- |
| **Vision Backbone** | Torchvision ResNet-50 (`resnet50`) |
| **Pretrained Checkpoint** | `kendrickfff/my_resnet50_garbage_classificationv1.2` ($94.5\text{ MB}$) |
| **Total Parameters** | $23,526,476$ |
| **Intercepted Layer** | `model.fc` via `register_forward_pre_hook` |
| **Feature Embedding Dimension** | $2048$ (unit-hypersphere normalized, $\|\mathbf{e}\|_2 = 1.0$) |
| **Target Classification Classes** | $4$ (`glass`, `plastic`, `metal`, `paper`) |
| **Entropy Alert Threshold ($\tau_{H,\text{crit}}$)** | $0.750$ (normalized Shannon entropy) |
| **Golden Calibration Gate ($\tau_{H,\text{golden}}$)** | $0.425$ (normalized Shannon entropy) |
| **Telemetry Window Size ($W$)** | $5$ frames |

---

## License

Released under the Apache 2.0 License.
