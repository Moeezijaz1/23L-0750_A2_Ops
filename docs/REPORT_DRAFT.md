# <RollNumber>_VisionGuard - report draft

> Replace every `<...>`. Numbers, screenshots and "what went wrong" must come from YOUR runs. The explanatory text
> below is a starting point: rewrite it in your own words (the brief asks for this), keep the facts.

## Front matter
Name / Roll no / GitHub / DagsHub / Live API URL / Commit hash (`git rev-parse HEAD`) of graded version.
**AI disclosure:** "I used Claude (Anthropic) to generate the project scaffolding (training, Flask, Docker, CI scripts) and a report outline, and
reviewed, ran and modified it myself. Dataset audit numbers, training results, screenshots and cloud deployment are from my own runs."
(Write this truthfully - adjust to what you actually did.)

## Chapter 1 - Data understanding and versioning
**1.1** versions screenshot; GPU: `<found / none>`, consequence: `<training took X min>`.
**1.2** Script `vg/audit.py` walks the folders, derives the class from the folder name, hashes bytes with SHA-256, opens each file with Pillow
(`verify()` then `load()` to catch truncated files). Tables: counts/class `<>`, min/median/max width/height/size `<>`, corrupt `<>`, duplicates `<>`.
Why hash: identical bytes in train and test would inflate test scores (leakage), so duplicates are dropped before splitting (`vg/splits.py`, stratified 70/15/15, seed 42).
**1.3** Figures: samples per class, histograms. Interpretation prompts: are sizes/format systematically different between classes? If AI images are
mostly one resolution/PNG and real ones vary, the model could learn *file properties instead of content* - which is why every image is resized the same way.
**1.4** Pointer file (`data/raw.dvc`) is a tiny YAML holding the content hash (md5), size and file count of the data and the path. Git stores only this pointer;
the real bytes live in the DVC cache/remote. Large binary data does not belong in Git because every version is stored forever in history,
clones become huge, diffs/merges are meaningless for binaries, and hosts cap file/repo size. Checking out an older commit gives the older pointer, and `dvc checkout/pull` fetches the matching data.

## Chapter 2 - Models
**2.1 Pipeline stages:** read split CSV -> shuffle (train only) -> read file -> decode (RGB, 3 channels) -> resize 224x224 (bilinear, antialiased; aspect ratio
is *not* preserved, images are stretched - 224 is the pretrained network's native size and keeps the same shape for batching) -> batch(32) -> augment (train only: flip, +-5% rotation, 10% zoom, contrast) -> prefetch (overlap data loading with GPU compute).
Augmentation is mild on purpose: blur/JPEG noise could destroy the generation artefacts the model must find.
**2.2 Baseline CNN:** Rescaling(1/255) normalises inputs; 4x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool] with 32-64-128-128 filters learn edges -> textures -> parts, pooling halves resolution and adds translation tolerance, BN stabilises/speeds training, GlobalAveragePooling gives a small vector (far fewer parameters than Flatten, less overfitting), Dropout(0.4) regularises, Dense(1, sigmoid) outputs P(AI).
Loss binary cross-entropy (matches a sigmoid/Bernoulli output), Adam lr 1e-3 (robust default), metrics accuracy/AUC/precision/recall. Table from `model.summary()`.
**2.3 Transfer:** MobileNetV2 (ImageNet, `include_top=False`), frozen -> GAP -> Dropout -> Dense(sigmoid); stage 2 unfreezes only the last block (`block_16*`, `Conv_1`; BatchNorm kept frozen) with lr 1e-5 so pretrained features are nudged, not destroyed. MobileNetV2 was trained with inputs in [-1, 1], so the model begins with `Rescaling(1/127.5, offset=-1)` (identical to `preprocess_input`); feeding 0-1 or 0-255 values would shift every activation away from what the pretrained weights expect. List trainable layers from `reports/*_trainable_layers.json`.
**2.4** `keras.utils.set_random_seed(42)`; EarlyStopping(val_loss, patience 5, restore best), ModelCheckpoint(val_loss), ReduceLROnPlateau(x0.5, patience 2). Best epoch: `<n>`.
**2.5** Threshold is picked on *validation* predictions (default: max F1; options youden/recall95), then the test set is evaluated once. Which error is worse? Argue it: calling a real photo fake (false accusation, harms honest users, erodes trust) vs calling a fake real (misinformation slips through, which is VisionGuard's reason to exist). A defensible choice for a *screening* tool is to favour recall on AI images (`--threshold-mode recall95`) and show the caveat/confidence so humans review borderlines - state your choice and the cost it has in the confusion matrix.
**2.6** 10 most confident mistakes, hypothesis each (blur, painting/illustration style, heavy filters, watermark, tiny size, stylised faces).
**2.7** Toy table (Fashion-MNIST) + VisionGuard tuner table. Choosing the best validation score overfits the validation set because with many trials you are selecting the lucky noise as well as the signal - the winning score is optimistically biased; the untouched test set (or CV) gives the honest number.

## Chapter 3 - MLflow / DagsHub
Definitions (own words): **experiment** = named group of related runs; **run** = one execution of training with its own record; **parameter** = fixed input setting (lr, epochs); **metric** = numeric result, can be logged per step; **artifact** = a file attached to a run (plots, model, split lists); **tag** = free label for filtering (git commit, data version).
**3.2** `.env` holds `MLFLOW_TRACKING_URI/USERNAME/PASSWORD`; MLflow reads them from the environment; `git check-ignore -v .env` proves it is ignored.
**3.3** Manual (`EpochLogger` callback): I choose exactly which names, steps, tags; consistent across runs; but I may forget things. Autolog: captures optimizer, epochs, batch size, per-epoch metrics, model summary automatically; misses custom things (git/DVC tags, threshold, test metrics - I still log them), and with two `fit` calls (frozen + fine-tune) it re-logs params, so stage 2 is logged in a nested run.
**3.5** Runs r1-r8 each change ONE thing vs a neighbour (see `scripts/run_experiments.sh`). Choose by *validation* ROC-AUC/F1 (threshold-independent AUC first, then F1/recall at the chosen threshold), not by test, to avoid test leakage; report test only for the winner.
**3.6** Registry: two versions, alias `champion`; `vg/load_champion.py` loads `models:/<name>@champion`. Serving plan: `vg/export_champion.py` downloads the champion once to `models/champion.keras`, which is mounted into the server, so requests never hit DagsHub.

## Chapter 4 - Flask app
**4.1** Loading a TF model takes seconds and hundreds of MB of RAM; per request it would make every call slow and could exhaust memory under concurrency - so `LocalPredictor` loads once at start-up. Preprocessing must match training exactly (same decode, same resize function `vg/preprocess.py`, same in-model scaling) because the network learned on that distribution; a mismatch (e.g. different resize or 0-1 vs 0-255) silently degrades accuracy without any error ("train/serve skew").
**4.2** Validation order: extension allow-list -> `secure_filename` -> size limit (`MAX_CONTENT_LENGTH`, 413) -> Pillow `verify()` + `load()` (rejects fake images) -> format allow-list. Image stays in memory (preview is a base64 thumbnail), never on disk. Error pages are friendly templates.
**4.4** Settings from env vars (`MODEL_PATH, THRESHOLD, DATABASE_URL, MAX_UPLOAD_MB, SECRET_KEY, ...`) so the same image runs anywhere and secrets stay out of code. Debug off because Flask's debugger exposes an interactive Python console (remote code execution) and leaks source/stack traces. (Flask's built-in server is not hardened for heavy traffic; the model API uses gunicorn.)

## Chapter 5 - Database
Table `predictions(id, created_at, filename, label, confidence, prob_ai, threshold, model_name, model_version, source, latency_ms)`.
Written inside the `/predict` view after the model answers. `/history` queries it (newest first). SQLite for local runs, PostgreSQL in Compose via `DATABASE_URL`.

## Chapter 6 - Docker
**6.1 definitions:** image = read-only template; container = running instance; layer = cached filesystem diff from one Dockerfile instruction; volume = Docker-managed storage that outlives containers; port mapping = forwarding host port -> container port.
**6.2** Order: base -> requirements -> `pip install` -> copy source: a code edit only invalidates layers after it, so the heavy TensorFlow layer stays cached. `python:3.11-slim`, user `appuser` (uid 10001), HEALTHCHECK hits `/health`, `.dockerignore` excludes `.env`, data, models. Package: `tensorflow-cpu` (no CUDA libraries; inference on a small VM is CPU anyway) - saves ~1-2 GB.
**6.3** Compose: `db` has named volume `pgdata`, `pg_isready` health check, no `ports`; `web` has `depends_on: condition: service_healthy`; they talk via host name `db`; passwords from `.env` (untracked) with `.env.example` committed. Model is **mounted read-only** rather than built in: the image stays small and a new champion is deployed by replacing a file, no rebuild (trade-off: the file must exist on the host - which is exactly the 6.5 failure case).
**6.4** `<before> -> <after>` after switching to tensorflow-cpu / `--no-cache-dir` / `.dockerignore`; biggest layer = pip install of TensorFlow.
**6.5** Suggested failure: wrong DB password -> `docker compose logs web` shows `password authentication failed for user`; or missing model -> `FileNotFoundError: Model file not found at /models/champion.keras`. Fix and restart.
**6.6** Secrets are stored encrypted in repo Settings -> Secrets and variables -> Actions, injected as `${{ secrets.X }}` and masked in logs; writing them in the YAML would publish them in Git history for anyone with repo access.

## Chapter 7 - API deployment
Pathway `<A/B/C>`; step-by-step with screenshots (VM creation, security list + iptables rule, docker install, clone, copy model, `.env`, compose up, health check). API URL: `<http://ip:8000>`. Postman: success (200 JSON), wrong key (401), wrong file (400). Protected by `X-API-Key` compared with `hmac.compare_digest`. Tested from another network: `<mobile hotspot screenshot>`. Mention: Oracle idle-instance reclamation, free-shape availability, two firewalls.

## Chapter 8 - Integration
Flask on localhost with `INFERENCE_MODE=remote`; each request is posted to the cloud API; the row is saved with `source=remote` and round-trip latency. Screenshots: >=5 inputs with results + `/history`.
