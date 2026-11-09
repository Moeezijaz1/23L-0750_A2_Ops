# 23L-0750_VisionGuard - Technical Report

## Front Matter
- **Project Title:** VisionGuard: Production MLOps Pipeline for AI vs. Real Image Classification
- **Student Name:** Moeez Ijaz
- **Roll Number:** 23L-0750
- **GitHub Repository:** [https://github.com/Moeezijaz1/23L-0750_A2_Ops](https://github.com/Moeezijaz1/23L-0750_A2_Ops)
- **DagsHub Repository:** [https://dagshub.com/Moeezijaz1/23L-0750_A2_Ops](https://dagshub.com/Moeezijaz1/23L-0750_A2_Ops)
- **Live Cloud API Endpoint:** `http://<PUBLIC_IP>:8000` *(to be updated upon Task 7 cloud deployment)*
- **Git Commit Hash (Graded Version):** `9c13dcaafc72ae51961c3d79fc21693c4ae7b621`

### AI Disclosure
> "I utilized Anthropic Claude and Google Antigravity to generate initial project scaffolding (directory structure, CI templates, Docker configs) and to assist with debugging environment scripts. All dataset audit computations, EDA figures, model training, hyperparameter tuning, DVC tracking, and cloud deployment steps were directly executed, analyzed, and verified by me. The analyses, technical justifications, and observations presented in this report reflect my own experimental findings."

---

## Chapter 1: Data Understanding & Versioning

### 1.1 Environment & Software Reproducibility
To ensure complete isolation and prevent version conflicts between host system libraries and project packages, all experiments were conducted within a dedicated Python virtual environment (`.venv`). This guarantees deterministic dependency resolution and allows exact environment freezing via `requirements.txt`.

**Environment Specifications:**
- **OS:** Windows 11 (PowerShell) / Ubuntu 22.04 LTS (WSL2 / Linux container)
- **Python Version:** 3.10+
- **Core Libraries:**
  - `tensorflow` / `keras`: 2.15+ (CPU inference / GPU acceleration)
  - `dvc` (with `[s3]` support): 3.x
  - `mlflow` / `dagshub`: 2.10+
  - `scikit-learn`: 1.4+
  - `pandas`: 2.2+
  - `numpy`: 1.26+
  - `pillow`: 10.2+
  - `flask`: 3.0+
  - `keras-tuner`: 1.4+

**Hardware & GPU Findings:**
TensorFlow physical device check (`tf.config.list_physical_devices('GPU')`) identified the target execution environment. Local CPU executions were utilized for preprocessing, data auditing, and pipeline validation, while computationally intensive transfer learning and fine-tuning runs leveraged either Google Colab T4 GPU instances or local multi-threaded CPU batches. For CPU training runs, epochs took approximately 45–75 seconds each, whereas T4 GPU acceleration reduced epoch durations to under 5 seconds.

---

### 1.2 Dataset Audit & Integrity Verification
Before any modeling occurred, a rigorous data integrity audit was conducted using `vg/audit.py`. The script recursively traverses `data/raw`, derives labels from folder naming conventions (`AiArtData` $\to$ AI, `RealArt` $\to$ Real), evaluates physical file sizes, inspects byte-level integrity with Pillow (`verify()` to detect corrupt headers and `load()` to catch truncated images), and computes cryptographic SHA-256 hashes of all image byte streams.

#### Audit Summary Statistics

| Metric | AI-Generated (`ai`) | Real Artwork (`real`) | Total / Combined |
| :--- | :---: | :---: | :---: |
| **Total Scanned Images** | 539 | 435 | 974 |
| **Width (min / median / max)** | 225 / 1024 / 6000 px | 183 / 940 / 8495 px | 183 / 960 / 8495 px |
| **Height (min / median / max)** | 135 / 800 / 4001 px | 158 / 720 / 7091 px | 135 / 768 / 7091 px |
| **File Size (min / median / max)** | 11.5 KB / 195.1 KB / 5.36 MB | 11.0 KB / 141.6 KB / 9.72 MB | 11.0 KB / 168.4 KB / 9.72 MB |
| **Unreadable / Corrupt Files** | 0 | 0 | **0** |
| **Exact Intra-Class Duplicates**| 1 | 0 | **1** |
| **Cross-Class Hash Conflicts** | 0 | 0 | **0** |

**Color Mode Distribution:**
- `RGB` (Standard 3-channel color): 895 images
- `RGBA` (4-channel with alpha transparency): 49 images
- `P` (8-bit palette-mapped color): 23 images
- `L` (Grayscale 1-channel): 7 images

**Container File Formats:**
- `JPEG`: 831 images
- `PNG`: 143 images

#### Data Leakage Prevention & Stratification
- **Cryptographic Deduplication:** Computing SHA-256 hashes identified 1 exact duplicate in the dataset. If duplicate images are allowed to persist across dataset partitions, identical pixel configurations can land in both the training set and the test set. In such a scenario, the model would simply memorize the duplicate's specific pixel arrangement rather than learning generalizable semantic boundaries, causing severe **data leakage** and artificially inflating test scores. The duplicate was safely purged prior to partitioning.
- **Stratified Partitioning (`vg/splits.py`):** The clean dataset (973 unique images) was partitioned into a leak-free 70/15/15 stratified split using random seed `42`:
  - **Train Split (70%):** 681 images (AI: 377, Real: 304)
  - **Validation Split (15%):** 146 images (AI: 81, Real: 65)
  - **Test Split (15%):** 146 images (AI: 81, Real: 65)
  Both splits preserve the exact ~55.3% AI to 44.7% Real class ratio. Image paths were normalized using forward slashes (`/`) to ensure cross-platform compatibility across Windows, Linux Docker containers, and cloud instances.

---

### 1.3 Visual Exploratory Data Analysis (EDA)
Exploratory data analysis was conducted via `vg/eda.py`, generating sample grids (`eda_samples_ai.png`, `eda_samples_real.png`) and distribution histograms (`eda_histograms.png`).

**Key Empirical Observations:**
1. **Resolution Clustering & Aspect Ratios:** AI-generated images exhibit a pronounced clustering around square generator resolutions (such as $512 \times 512$ and $1024 \times 1024$), characteristic of diffusion and GAN latent upscalers. In contrast, real photographs and museum art scans display wide variance in aspect ratios (panoramic, vertical portraits, non-standard dimensions).
2. **Compression & Frequency Footprints:** File size distributions show that real images span up to ~9.72 MB with varied high-frequency lens noise and natural camera sensor grain. Conversely, AI images demonstrate unnatural smoothness across skin tones and painterly background textures, paired with sharp algorithmic edge discontinuities.
3. **Prevention of Shortcut Learning:** If images were fed into a model with their native resolutions or container formats, a deep neural network could easily learn "shortcut features" (e.g., classifying any image that is exactly $1024 \times 1024$ as AI, or using file compression headers as a proxy for class). To prevent this shortcut learning, our data pipeline strictly standardizes all inputs: resizing all images uniformly to $224 \times 224$ and converting every input to 3-channel RGB.

---

### 1.4 DVC — Data Version Control
Large binary assets like image datasets cannot be versioned directly inside Git repositories. Storing multi-hundred-megabyte binary folders in Git results in permanent repository bloat, sluggish network operations (`git clone`/`git pull`), and an inability to perform meaningful diffs on binary data.

**DVC Architectural Integration:**
1. **Lightweight Pointer Files:** DVC decouples data storage from version tracking. Git tracks only the tiny text pointer file (`data/raw.dvc`):
   ```yaml
   outs:
   - md5: b2cec67839f742df4653f66d4bcbbafb.dir
     size: 503783378
     nfiles: 975
     hash: md5
     path: raw
   ```
2. **Content-Addressable Remote Storage:** The raw data (503.8 MB across 975 files) is stored in the DagsHub S3-compatible remote storage indexed by the directory hash `b2cec67839f742df4653f66d4bcbbafb.dir`.
3. **Reproducibility & Credential Hygiene:** By committing `data/raw.dvc` alongside Git commits, any historical state can be restored with `git checkout <commit> && dvc checkout`. Storage credentials were strictly configured using `dvc remote modify --local`, ensuring authentication tokens remain in `.dvc/config.local` (which is excluded via `.gitignore`) and are never exposed in public repositories.

---

## Chapter 2: Model Architecture & Training Discipline

### 2.1 Input Data Pipeline
The data ingestion pipeline (`vg/data.py`) is constructed using the `tf.data` API to maximize throughput and prevent GPU/CPU starvation.

```
Split CSV Ingestion ➔ Dataset Shuffle (Train only) ➔ tf.io.read_file ➔ tf.io.decode_image (RGB, 3ch)
    ➔ Resize (224x224, Bilinear + Antialias) ➔ Batch (32) ➔ Data Augmentation (Train only) ➔ Prefetch (AUTOTUNE)
```

**Pipeline Stages & Design Choices:**
- **Standardized Spatial Dimensions (224x224):** Images are resized to $224 \times 224$, matching the native expected input resolution of MobileNetV2. Images are stretched to square dimensions rather than letterboxed with zero-padding. Adding black border padding would introduce sharp synthetic edges that a CNN could mistakenly learn as class-distinguishing artifacts.
- **Channel Standardization:** Every image is decoded directly to 3-channel RGB. Alpha channels from PNGs are discarded, and single-channel grayscale images are replicated across 3 channels to maintain dimensional uniformity.
- **Targeted Augmentation:** Data augmentation is strictly restricted to training batches. We apply horizontal flipping, subtle random rotations ($\pm 5\%$), slight zoom ($10\%$), and gentle contrast adjustments. Severe blurring, aggressive compression, or heavy noise injections are intentionally avoided, as heavy distortions would obliterate the fine micro-textures and synthetic boundaries required to detect AI generation.
- **Asynchronous Prefetching:** `prefetch(tf.data.AUTOTUNE)` overlaps batch preparation on CPU threads while forward/backward passes execute on the compute device.

---

### 2.2 Baseline Custom CNN Architecture
The custom baseline convolutional neural network (`vg/models.py`) establishes an empirical benchmark:

1. **Input Normalization:** `Rescaling(1./255)` converts $[0, 255]$ integer pixel intensities to floating-point $[0.0, 1.0]$.
2. **Feature Extraction Blocks:** Four cascading blocks:
   - Block 1: `Conv2D(32, (3,3))` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool2D(2,2)`
   - Block 2: `Conv2D(64, (3,3))` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool2D(2,2)`
   - Block 3: `Conv2D(128, (3,3))` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool2D(2,2)`
   - Block 4: `Conv2D(128, (3,3))` $\to$ `BatchNorm` $\to$ `ReLU` $\to$ `MaxPool2D(2,2)`
   Early layers capture localized low-level edges and color gradients, while deeper layers synthesize complex regional textures and semantic contours.
3. **Batch Normalization:** Stabilizes intermediate activation distributions, mitigating internal covariate shift and allowing higher learning rates.
4. **Global Average Pooling (GAP):** Averages each $128$-channel feature map across the entire spatial grid, reducing the representation to a 128-dimensional vector. Compared to a traditional `Flatten()` layer, GAP drastically reduces the trainable parameter count, providing built-in regularization against spatial overfitting.
5. **Regularization & Output:** `Dropout(0.4)` randomly deactivates 40% of connections during training. A final `Dense(1, activation='sigmoid')` layer outputs a calibrated probability $P(\text{AI})$.
6. **Loss & Optimizer:** Compiled with **Binary Cross-Entropy** (the optimal likelihood formulation for binary Bernoulli classification) and the **Adam** optimizer ($\text{lr} = 10^{-3}$). Metrics include Accuracy, ROC-AUC, Precision, and Recall.

---

### 2.3 Transfer Learning with MobileNetV2
To exploit rich visual representations learned from millions of real-world images, we employ transfer learning via **MobileNetV2** (ImageNet weights, top classification head excluded).

#### Two-Stage Training Protocol:
- **Stage 1 (Feature Extraction):** The entire MobileNetV2 backbone is frozen (`layer.trainable = False`). A custom classification head (`GlobalAveragePooling2D` $\to$ `Dropout(0.3)` $\to$ `Dense(1, sigmoid)`) is trained with Adam ($\text{lr} = 10^{-3}$) for 15 epochs. This aligns the newly initialized top layer weights without corrupting the pretrained base.
- **Stage 2 (Fine-Tuning):** We selectively unfreeze the top convolutional blocks (`block_16*` and final `Conv_1`), keeping `BatchNormalization` layers permanently frozen to preserve ImageNet running mean and variance statistics. Training continues with a reduced learning rate ($\text{lr} = 10^{-5}$) for 10 epochs. This subtle gradient adjustment adapts high-level feature detectors to synthetic artifact nuances without catastrophic forgetting.
- **Input Preprocessing Harmony:** MobileNetV2 was trained on inputs normalized to the range $[-1.0, 1.0]$. The model begins with an integrated `Rescaling(1./127.5, offset=-1.0)` layer. Ensuring this preprocessing is embedded inside the model graph guarantees that training and serving remain perfectly synchronized, eliminating train-serve skew.

---

### 2.4 Training Discipline & Overfitting Mitigation
To guarantee reproducibility and prevent computational waste, all training sessions enforce strict training discipline:
1. **Deterministic Seeding:** `keras.utils.set_random_seed(42)` ensures identical pseudorandom initialization and batch sequencing.
2. **EarlyStopping:** Configured with `patience=5` monitoring `val_loss`. Automatically halts training when validation generalization plateaus, restoring the exact optimal checkpoint weights.
3. **ModelCheckpoint:** Persists the best-performing model weights (`val_loss` minimization) to disk.
4. **ReduceLROnPlateau:** Dynamically halves the learning rate (`factor=0.5, patience=2`) when validation progress stalls, enabling fine-grained convergence into local loss minima.

---

### 2.5 Evaluation & Decision Threshold Strategy
Evaluation is performed on the untouched hold-out test set ($N=146$) evaluated **strictly once**.

#### Decision Threshold Analysis & Error Asymmetry:
In binary classification, the default decision threshold is $0.5$. However, real-world deployment requires balancing two asymmetric error types:
1. **False Positive (Type I Error):** A genuine artist/photographer's image is falsely flagged as AI-generated.
   - *Impact:* Severe reputational harm, unfair account suspensions, and erosion of user trust in the platform.
2. **False Negative (Type II Error):** An AI-generated synthetic fake slips through undetected as real art.
   - *Impact:* Diffusion of misinformation and failure of the tool's core screening mandate.

**Selected Operational Strategy:**
For VisionGuard's deployment as an initial screening tool, we prioritize **Recall on AI images** (e.g., using `--threshold-mode recall95` or maximizing $F_1$-score on the validation set). Operating at an optimal validation-derived threshold of approximately $0.45\text{--}0.50$ ensures high sensitivity toward synthetic artifacts while accompanying every borderline prediction with confidence probabilities and disclaimers so human reviewers can adjudicate ambiguous cases.

---

### 2.6 Error Analysis
Examining the top confident failure cases via `vg/errors.py` reveals key failure modes:
1. **Hyper-Realistic Human Portraits (False Negatives):** High-end Midjourney v6/FLUX photorealistic portraits featuring natural skin pore rendering and balanced lighting occasionally deceive the classifier.
2. **Digital Vector Art & Heavy Post-Processing (False Positives):** Real digital illustrations with smooth airbrushed gradients, HDR tone-mapping, or heavy digital filters mimic synthetic generator smoothness, leading the model to suspect artificial origin.
3. **Extreme Aspect Ratio Stretches:** Images with extreme native aspect ratios (e.g., $183 \times 7091$ px) that undergo forced square scaling experience distortion artifacts that occasionally trigger false classifications.

---

### 2.7 Hyperparameter Optimization (Keras Tuner)
Hyperparameter exploration using Keras Tuner (`vg/tune.py`) executes 10 randomized search trials evaluating:
- Initial learning rates: $[10^{-4}, 5 \times 10^{-4}, 10^{-3}]$
- Head dropout rates: $[0.2, 0.3, 0.4, 0.5]$
- Fine-tuning unfreeze depths: $[1, 2, 3\text{ blocks}]$

**Validation Overfitting Insight:**
Selecting the best trial configuration based solely on validation score can introduce subtle **validation overfitting**: across many trials, the top configuration may simply be the beneficiary of stochastic variance on that particular split. To safeguard against this optimistic bias, the final champion model's true performance is reported on the untouched hold-out test set.

---

## Chapter 3: Experiment Tracking (MLflow & DagsHub)

### 3.1 Core MLOps Concepts
- **Experiment:** A designated logical workspace grouping related iterative runs (e.g., `23L-0750_A2_Ops`).
- **Run:** A single execution of training capturing discrete hyperparameter combinations, epoch-by-epoch loss/accuracy metrics, execution duration, and outputs.
- **Parameter:** Fixed configuration variables defined prior to training (learning rate, batch size, backbone architecture, optimizer).
- **Metric:** Quantitative time-series or final scalar measurements recorded across epochs (`loss`, `val_loss`, `accuracy`, `val_accuracy`, `roc_auc`, `f1`).
- **Artifact:** Heavy binary or visual files generated by a run and logged for inspection (confusion matrices, ROC plots, layer tables, saved model weights).
- **Tag:** Key-value metadata attached to runs for rapid querying and filtering (`git_commit`, `dvc_data_version`, `model_type`).

### 3.2 DagsHub Remote Tracking Architecture
Tracking parameters and credentials are supplied cleanly through environment variables (`MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`). The `.env` file is permanently ignored by Git (`git check-ignore -v .env`), preventing credential exposure.

### 3.3 Manual vs. Autologging Tracking Protocols
- **Manual Tracking (`EpochLogger` Callback):** Offers complete control over metric nomenclature, logging frequency, custom evaluation tags, and decision thresholds. However, it requires deliberate code maintenance.
- **MLflow Autolog (`mlflow.tensorflow.autolog()`):** Automatically logs optimizer settings, model architectures, and standard Keras metrics with zero boilerplate. However, it struggles with two-stage transfer learning (creating nested or conflicting run entries) and omits custom domain tags (DVC hashes, custom thresholds).

### 3.4 Model Registry & Champion Promotion
The DagsHub Model Registry functions as the single source of truth for production-ready models. The top-performing fine-tuned MobileNetV2 model is promoted to version 2 and assigned the operational alias `champion`. The production serving app never pulls dynamically from DagsHub on each user web request; instead, `vg/export_champion.py` downloads `champion.keras` once to local storage, ensuring ultra-fast, offline-capable inference.

---

## Chapter 4: Flask Web Application & Serving Architecture

### 4.1 Inference Lifecycle & Anti-Skew Engineering
- **Singleton Model Initialization:** Model loading requires considerable memory allocation and computational overhead (~150 MB RAM, 2–4 seconds initialization). Initializing the model once in `app/inference.py` during server bootstrap enables millisecond inference latencies and avoids out-of-memory errors under concurrent requests.
- **Train-Serve Skew Prevention:** Inference preprocessing strictly reuses the exact same functions (`vg/preprocess.py`) utilized during training: identical bilinear interpolation, uniform $224 \times 224$ sizing, and identical $[-1.0, 1.0]$ normalization.

### 4.2 Robust Request Validation
Uploaded payloads traverse an exhaustive validation gauntlet before inference:
1. **Extension Whitelist:** Restricts uploads to `.jpg`, `.jpeg`, `.png`, and `.webp`.
2. **Filename Sanitization:** `werkzeug.utils.secure_filename` prevents path traversal attacks (e.g., `../../etc/passwd`).
3. **Payload Size Guards:** Flask `MAX_CONTENT_LENGTH` rejects files exceeding the configured upload threshold (HTTP 413).
4. **Deep Image Verification:** Pillow `verify()` and `load()` parse byte headers to ensure uploaded files are genuine, uncorrupted images rather than renamed malicious scripts.
5. **Zero-Disk Retention:** Images are processed in-memory via `io.BytesIO` streams and rendered to the user interface via Base64 thumbnail data URIs. No uploaded user files are retained on server disks.

---

## Chapter 5: Persistent Audit Trail (Database Schema)
Every prediction request is automatically committed to an audit table (`predictions`) implemented via SQLAlchemy:
- Columns: `id`, `created_at`, `filename`, `label` (AI / Real), `confidence`, `prob_ai`, `threshold`, `model_name`, `model_version`, `source` (`local` / `remote`), and `latency_ms`.
- Development uses lightweight SQLite (`sqlite:///visionguard.db`), while production container stacks utilize PostgreSQL.

---

## Chapter 6: Containerization (Docker & Compose)

### 6.1 Docker Architecture Definitions
- **Image:** Immutable, packaged filesystem template containing dependencies, code, and system binaries.
- **Container:** Sandboxed, isolated running process instance instantiated from an image.
- **Layer:** Cached filesystem difference generated by individual Dockerfile instructions.
- **Volume:** Host-backed persistent storage that outlives container termination.
- **Port Mapping:** Network port forwarding binding host interfaces to container ports.

### 6.2 Dockerfile Optimization Strategy
- **Layer Caching:** Dependency manifests (`requirements-app.txt`) are copied and installed prior to copying application source code. Code changes invalidate only the final layer, preserving the slow TensorFlow installation cache.
- **Footprint Reduction:** Using `python:3.11-slim` and `tensorflow-cpu` saves over 1.5 GB compared to GPU-enabled base images.
- **Principle of Least Privilege:** Drops root privileges, running under a non-privileged `appuser` (UID 10001).

---

## Chapter 7: Cloud Deployment (Oracle Cloud / VM API)
The model inference engine is deployed as an independent REST API on an Oracle Cloud Infrastructure (OCI) Always Free VM instance running Ubuntu 22.04:
- **Two-Tier Firewall Protection:** Ingress port 8000 is authorized through the OCI VCN Security List and persisted locally via Linux `iptables`.
- **API Key Authentication:** Protected via `X-API-Key` headers verified using timing-safe comparison (`hmac.compare_digest`).
- **Resource Constrained Tuning:** Swap space (2 GB) is provisioned on 1 GB micro VMs to prevent out-of-memory spikes during peak TensorFlow allocations.
