# VisionGuard - complete step-by-step guide (every sub-task)

Legend: **DO** = what to run/click - **SHOT** = screenshot to take - **WRITE** = what the report must say.
Replace `<ROLL>` with your roll number, `<user>` with your GitHub/DagsHub username.
Naming rule from the brief: **everything is called `<ROLL>_A2_Ops`** (GitHub repo, DagsHub repo, MLflow experiment, cloud instance/project).

Every screenshot must show: terminal prompt with your username, the date/time, and the project name. In a terminal run
`date` right before the command (or use a prompt that shows time). Browser screenshots must include the address bar.
Caption each one ("Figure 3.2 - ...") and refer to it in the text. Blur tokens/passwords.

---------------------------------------------------------------------------------------------------
## PART 0 - Setup (1-2 hours, do it first)

### 0.1 Tools to install
Git, Python 3.10-3.12, Docker Desktop (or Docker Engine), Postman, an SSH client, a PDF-capable editor (Word/Google Docs).
Windows: install **WSL2 + Ubuntu** and work inside it (TensorFlow GPU is not supported on native Windows; Docker Desktop integrates with WSL2).

### 0.2 Accounts
| Account | Why | Notes |
|---|---|---|
| GitHub | code, CI | repo `<ROLL>_A2_Ops` |
| DagsHub | MLflow + DVC storage + registry | sign in with GitHub |
| Kaggle | dataset download | Account -> Settings -> Create API token (or download zip manually) |
| Oracle Cloud (or AWS/GCP/Azure) | Task 7 | needs a card for verification; sign up EARLY, approval can take a day; free A1 capacity may be unavailable - try another region/shape or use pathway C |
| Google Colab (optional) | free GPU for training | Runtime -> Change runtime type -> T4 GPU |

### 0.3 Create the repo and unpack the project
```bash
unzip visionguard.zip && cd visionguard
git init -b main
git add . && git commit -m "Initial scaffold"
git remote add origin https://github.com/<user>/<ROLL>_A2_Ops.git     # create the EMPTY repo on github.com first
git push -u origin main
```
Search-and-replace the placeholders `<user>`, `<ROLLNO_A2_Ops>`, `ROLLNO` in `README.md`, `.env.example`, `docs/`.

### 0.4 Connect DagsHub
1. dagshub.com -> **Create -> New Repo -> Connect a repo -> GitHub** -> choose `<ROLL>_A2_Ops`. Make it **Public** (or invite your instructor).
2. Repo page -> **Remote** button -> **Experiments** tab: copy the MLflow URI `https://dagshub.com/<user>/<ROLL>_A2_Ops.mlflow`.
3. Avatar -> **Settings -> Tokens -> New token** -> copy it once.
4. Locally:
```bash
cp .env.example .env
# edit .env: MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME=<user>, MLFLOW_TRACKING_PASSWORD=<token>,
#            MLFLOW_EXPERIMENT_NAME=<ROLL>_A2_Ops, REGISTERED_MODEL_NAME=<ROLL>_A2_Ops_VisionGuard
git check-ignore -v .env          # SHOT (proof .env is ignored; needed for 3.2)
git status                        # .env must NOT be listed
```

### 0.5 Python environment (needed for 1.1)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

### 0.6 Training on Colab (if you have no GPU)
In a Colab notebook:
```python
!git clone https://github.com/<user>/<ROLL>_A2_Ops.git && %cd <ROLL>_A2_Ops
!pip install -q mlflow dagshub "keras-tuner" "dvc[s3]" python-dotenv scikit-learn pandas    # Colab already has TensorFlow
import os; os.environ.update(MLFLOW_TRACKING_URI="...", MLFLOW_TRACKING_USERNAME="...", MLFLOW_TRACKING_PASSWORD="...", MLFLOW_EXPERIMENT_NAME="<ROLL>_A2_Ops")
!dvc remote modify --local origin access_key_id <TOKEN> && dvc remote modify --local origin secret_access_key <TOKEN>
!dvc pull
```
Then run the same `python -m vg.train ...` commands with `!`. Download `models/champion.keras` afterwards. Never commit tokens in a notebook.

---------------------------------------------------------------------------------------------------
## TASK 1 - Data understanding and versioning (7 marks)

### 1.1 Environment (1)
```bash
date; whoami
python --version
pip freeze | grep -Ei "^(tensorflow|keras|mlflow|dvc|scikit-learn|pandas|numpy|matplotlib|pillow|flask|keras-tuner)=="
python -c "import tensorflow as tf; print(tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```
**SHOT:** this terminal. **WRITE:** why a venv (isolation, reproducible versions), exact versions, whether a GPU was found and what that means (training speed; "no GPU -> used Colab T4 / CPU, epochs took X min").

### 1.2 Dataset audit (3)
Download from Kaggle and arrange:
```bash
pip install kaggle   # put kaggle.json in ~/.kaggle (chmod 600)
kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images -p data && unzip -q data/*.zip -d data/raw
find data/raw -maxdepth 2 -type d      # expect AiArtData and RealArt (nested folders are fine)
```
If your folder names differ, edit `class_from_path()` in `vg/config.py`.
```bash
python -m vg.audit          # prints tables; saves reports/audit.csv + reports/audit_summary.txt
```
**SHOT:** the printed summary. **WRITE:** how the script works (walks tree -> class from folder name -> `os.stat` size -> Pillow `verify()` then `load()` -> SHA-256 of file bytes ->
`duplicated()` on hashes); the table (count/class; min/median/max width, height, file size); number of corrupt files; number of duplicates; any hash in both classes (label conflict).
"What went wrong": e.g. nested folder names, CMYK/grayscale images, duplicates you found.

### 1.3 Visual EDA (1.5)
```bash
python -m vg.eda      # reports/eda_samples_ai.png, eda_samples_real.png, eda_histograms.png
```
**SHOT/FIG:** the 3 figures with captions. **WRITE (2-3 sentences each):** e.g. "AI samples are mostly square 512x512/1024x1024 whereas real photos vary widely"
(class-correlated size = potential shortcut learning -> why we resize uniformly); file-size histogram differences (compression); visible content differences.
Use the real numbers you see; do not copy this example.

### 1.4 DVC (1.5)
**Toy part** (in a NEW EMPTY folder OUTSIDE the project):
1. Create an empty DagsHub repo `dvc-toy-<ROLL>` (New Repo -> Empty). Copy its git URL.
2. Run `labs/dvc_toy.sh` line by line (edit URLs/token). Key points: `git init`, `dvc init`, `dvc add data.csv`, commit v1, edit CSV, `dvc add`, commit v2,
   `git log --oneline`, go back (`git checkout HEAD~1 -- data.csv.dvc && dvc checkout`), add remote, `dvc push`, delete `data.csv` and `.dvc/cache`, `dvc pull`.
3. `git remote add origin <dagshub git url> && git push -u origin main` so the commits are visible on DagsHub.
**SHOT:** initial commits (`git log --oneline`), `dvc add` output, `cat data.csv` at v1 and v2, push output, `ls` after deletion, `dvc pull`, file restored, DagsHub page showing the commits.

**Apply part** (inside the project):
```bash
sed -i '/data\/raw/d' .gitignore                 # DVC refuses paths ignored by git
dvc init
dvc add data/raw                                  # creates data/raw.dvc + data/.gitignore
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<user>/<ROLL>_A2_Ops.s3
dvc remote modify --local origin access_key_id <DAGSHUB_TOKEN>
dvc remote modify --local origin secret_access_key <DAGSHUB_TOKEN>
python -m vg.splits                               # create split CSVs (small -> commit them to git)
git add .dvc data/raw.dvc data/.gitignore data/splits .gitignore
git commit -m "Track dataset with DVC, add splits"
dvc push && git push
cat data/raw.dvc                                  # SHOT: the pointer file
```
**WRITE:** the pointer holds `md5` hash, `size`, `nfiles`, `path`; Git versions that tiny text, while the bytes sit in cache/remote; large binaries do not belong in Git (history bloat forever,
slow clones, no meaningful diffs, hosting limits); changing a Git commit changes which data version `dvc checkout` restores = reproducibility. `--local` keeps the token out of Git.

---------------------------------------------------------------------------------------------------
## TASK 2 - Models (14 marks)

All commands from the repo root. First run with `--tracking none` while debugging; real runs go in Task 3.

### 2.1 Input pipeline (2)
```bash
python scripts/augmented_batch.py      # reports/augmented_batch.png
```
**WRITE** each stage of `vg/data.py`: read split CSV -> (train) shuffle -> `read_file` -> `decode_image` (RGB) -> `resize` to 224x224 (bilinear + antialias)
-> `batch(32)` -> (train only) random flip/rotation/zoom/contrast -> `prefetch(AUTOTUNE)`.
Image size: 224 = MobileNetV2/EfficientNet native resolution, small enough for CPU/T4. Aspect ratio: **not preserved** (stretched to a square) - simple, consistent batching;
alternative is resize-with-pad (adds black borders the model might learn) - state which you chose and why. Augmentation only on train so val/test measure real performance.

### 2.2 Baseline CNN (3)
```bash
python scripts/model_table.py          # SHOT the summary + table (also reports/baseline_layer_table.csv)
```
**WRITE** a justification per layer type: Rescaling (inputs 0-255 -> 0-1), Conv (learn local filters), BatchNorm (stabilise/speed up), ReLU (non-linearity),
MaxPool (downsample, translation tolerance), GlobalAveragePooling (few params vs Flatten), Dropout (regularise), Dense-sigmoid (P(AI)).
Loss = binary cross-entropy (probabilistic output for 2 classes), optimizer Adam lr 1e-3, metrics accuracy/AUC/precision/recall (accuracy alone hides class imbalance).

### 2.3 Transfer learning (3)
```bash
python -m vg.train --model transfer --run-name r3_mnv2_finetune --tracking none
cat reports/r3_mnv2_finetune_trainable_layers.json     # SHOT: trainable layers per stage
```
Output: `reports/r3_mnv2_finetune_curves.png` (stage 1 and stage 2 on one plot, fine-tune start marked).
**WRITE:** why MobileNetV2 (small, fast on CPU, good ImageNet features); stage 1 frozen backbone + new head (lr 1e-3); stage 2 unfreeze only last block (`block_16*`, `Conv_1`) with lr 1e-5
(tiny steps so pretrained weights are adjusted, not destroyed); BatchNorm kept frozen; re-compile needed after changing `trainable`; preprocessing: MobileNetV2 expects pixels in [-1,1],
so `Rescaling(1/127.5, -1)` is inside the model - wrong scaling shifts every activation and wastes the pretrained weights.

### 2.4 Training discipline (1)
Already in `vg/train.py`: `set_random_seed(42)`, EarlyStopping (val_loss, patience 5, restore best), ModelCheckpoint (val_loss), ReduceLROnPlateau.
**SHOT:** the curves figure (best epoch is the red dashed line; the script prints `best epoch (lowest val_loss): N`). **WRITE:** the best epoch and what each callback prevents
(overfitting / losing the best weights / stalling at a too-high learning rate). Mention seeds make runs repeatable but GPU kernels can still differ slightly.

### 2.5 Evaluation (2)
`vg/train.py` prints validation and test metric dicts, saves `*_confusion.png` and `*_roc.png`.
**SHOT:** metric table for BOTH models (r1 baseline and r3 transfer): accuracy, precision, recall, F1, ROC-AUC (positive class = AI-generated).
**WRITE:** threshold chosen on validation only (`--threshold-mode f1|youden|recall95`) and why (test must stay untouched; otherwise you tune to the test set);
test evaluated once per model. Decide the worse error: real-called-fake (false positive: wrongly accuses a genuine photographer/news photo, damages trust) vs fake-called-real
(false negative: misinformation passes). Pick one with reasoning (e.g. for a screening tool favour catching fakes -> higher recall, accept some false alarms; show confidence + caveat so a human decides).
Show the cost in your confusion matrix numbers.

### 2.6 Error analysis (1)
```bash
python -m vg.errors --run r3_mnv2_finetune --threshold <t printed by the training run>
```
**SHOT:** `reports/r3_mnv2_finetune_errors.png`. **WRITE:** one hypothesis per image (blur, painting/illustration style, heavy filter, watermark/text, tiny original size, unusual subject, photoreal AI portrait).
Look at original sizes printed in titles - tie to the EDA from 1.3.

### 2.7 Keras Tuner (2)
Toy: `python labs/keras_tuner_toy.py` -> **SHOT** results summary; build a small table (trial, units, lr, val_accuracy).
Apply: `python -m vg.tune --trials 10 --epochs 6` (random search over dropout, learning rate, number of unfrozen layers; >= 8 trials). **SHOT:** `results_summary` output; table of all trials.
**WRITE:** best settings; which hyperparameter mattered most (compare trials where only one changed - typically learning rate / #unfrozen layers); why picking the best validation score can overfit
the validation set (many trials -> you select noise too, so the winning score is optimistic -> confirm on untouched test set / cross-validation). Then copy the best values into a normal tracked run.

### Bonus Grad-CAM (+1)
```bash
python -m vg.gradcam --model models/r3_mnv2_finetune_final.keras --run r3_mnv2_finetune --threshold <t>
```
**FIG:** `reports/gradcam.png` (4 correct + 4 wrong). **WRITE:** where the heat sits (faces, edges, background textures, watermarks?) and what it implies (real content vs shortcut).

---------------------------------------------------------------------------------------------------
## TASK 3 - MLflow + DagsHub (12 marks)

### 3.1 MLflow locally (2)
```bash
unset MLFLOW_TRACKING_URI                  # make sure it logs locally (or run from a shell without .env)
python labs/mlflow_toy.py                  # two runs: lr 1e-2 and 1e-4
mlflow ui --port 5001                      # open http://127.0.0.1:5001   (5000 is used by Flask)
```
In the UI: open each run (**SHOT** params, metrics, artifacts incl. plot and model), tick both runs -> **Compare** (**SHOT**).
**WRITE definitions** in your own words: experiment, run, parameter, metric, artifact, tag (see `docs/REPORT_DRAFT.md` for ideas) and what differed between the two runs.

### 3.2 DagsHub setup (2)
Already done in 0.4. Re-run the toy against DagsHub:
```bash
python labs/mlflow_toy.py      # with .env loaded (load_dotenv is called in the script)
```
**SHOT:** (a) DagsHub project page, (b) toy run in its Experiments tab, (c) `git check-ignore -v .env`, (d) `git ls-files | grep env` showing only `.env.example`.
**WRITE:** credentials flow: environment variables `MLFLOW_TRACKING_URI/USERNAME/PASSWORD` are read by MLflow; code never contains them.

### 3.3 Instrument training (3)
```bash
python -m vg.train --model transfer --run-name r3_mnv2_finetune --tracking manual
python -m vg.train --model transfer --run-name r8_mnv2_ft_autolog --tracking autolog
```
Both log params, per-epoch curves, final test metrics and tags `git_commit`, `dvc_data_version`, `seed`. Commit your code first so the git tag is meaningful (`git add -A && git commit`).
**SHOT:** run page for each (Parameters, Metrics charts, Tags). **WRITE:** manual (callback) = I choose names/steps/tags, consistent, but only what I remember to log;
autolog = captures optimizer/epochs/batch/metrics automatically with no code, but misses custom items (threshold, test metrics, tags) and has quirks with two `fit()` calls (stage 2 goes to a nested run).
If autologging errors on your Keras/MLflow version, say so under "what went wrong", pin versions that work, and describe the manual approach vs what autolog captured before failing.

### 3.4 Artifacts (1.5)
Logged automatically: confusion matrix, ROC, curves, trainable-layer JSON, split lists (`splits/`), `pip_freeze.txt`, the model (`model/`), `model_meta.json`.
**SHOT:** the Artifacts tab of the chosen run showing all of them.

### 3.5 Experiment comparison (2)
```bash
bash scripts/run_experiments.sh        # r1..r8, each differs by ONE change (see comments in the script)
python -m vg.compare_runs              # reports/run_comparison.csv/.png; also use the DagsHub compare view
```
**SHOT:** MLflow compare view or the chart/table. **WRITE:** state what single change each run made, pick the best by **validation** ROC-AUC then F1 (and recall if you decided FN is worse),
justify, and only then quote its test metrics. Mention that comparing on test would leak.

### 3.6 Registry (1.5)
In DagsHub: Experiments -> open run -> note **Run ID**.
```bash
python -m vg.register --run-id <RUN_ID_OF_A_WEAKER_RUN> --no-alias      # version 1
python -m vg.register --run-id <RUN_ID_OF_BEST_RUN>                      # version 2 + alias champion
python -m vg.load_champion path/to/image.jpg                              # SHOT terminal JSON output
python -m vg.export_champion                                              # models/champion.keras + champion_meta.json
```
**SHOT:** registry page showing two versions and the `champion` alias. **WRITE:** the server never calls DagsHub per request: `export_champion` downloads the file once, the file is mounted/copied to the server (Task 6/7).
(If alias UI differs on DagsHub, the scripts set it through the MLflow API - show that output.)

---------------------------------------------------------------------------------------------------
## TASK 4 - Flask app (12 marks)
```bash
export MODEL_PATH=models/champion.keras DATABASE_URL=sqlite:///visionguard.db
python -m app       # http://127.0.0.1:5000
```

### 4.1 Inference module (2)
```bash
python scripts/predict_two.py data/raw/RealArt/<a real image> data/raw/AiArtData/<an AI image>     # SHOT
```
**WRITE:** `app/inference.py` loads the model once in `__init__` (loading takes seconds + hundreds of MB; per request would make every call slow and risk out-of-memory);
preprocessing uses the same `vg/preprocess.py` function as training (same decode, resize, scaling in model). A mismatch (different resize filter, 0-1 vs 0-255, BGR vs RGB) silently reduces accuracy = train/serve skew.

### 4.2 Routes and validation (3)
Routes: `/` upload, `/predict` result, `/history`, `/health`. Test and **SHOT** the three failures:
1. upload `notes.txt` -> "Unsupported file type"
2. upload an image bigger than `MAX_UPLOAD_MB` (e.g. set `MAX_UPLOAD_MB=1` and upload 3 MB) -> "File too large" (413)
3. `echo "not an image" > fake.png` and upload -> "could not be decoded"
Also show: filename sanitising (`secure_filename`, test `../../x.png`), image never saved (check no new files: `git status`, `ls`), friendly 404 page (`/nothing`).
**WRITE:** the validation order and why decoding is needed (extension/MIME can be faked).

### 4.3 HTML/CSS (2.5)
**SHOT:** upload page, result page (label, confidence, preview, caveat), history page via the menu. Explain the template inheritance (`base.html`) and CSS.

### 4.4 Configuration and server (1)
```bash
curl -s http://127.0.0.1:5000/health | python -m json.tool        # SHOT
env | grep -E "MODEL_PATH|THRESHOLD|DATABASE_URL"
```
**WRITE:** all settings come from env vars (model path, threshold, DB URL, max upload, secret key); `app.run(debug=False)` because debug mode ships the Werkzeug interactive debugger
(arbitrary Python execution from the browser) and detailed stack traces, and auto-reloader; the built-in server is single-purpose, so the model API (Task 7) uses gunicorn.

---------------------------------------------------------------------------------------------------
## TASK 5 - Database trail (8 marks)
Implemented in `app/db.py` (table `predictions`) and written in `/predict`; read by `/history`.
1. Upload 4-5 images with the app running. **SHOT:** `/history` showing time, filename, decision, confidence, P(AI), model + version, source.
2. Prove persistence in the DB itself:
```bash
sqlite3 instance/visionguard.db "select id,created_at,label,confidence,model_name,model_version from predictions;"   # SQLite path may be instance/ or project root
# in Compose: docker compose exec db psql -U $POSTGRES_USER -d $POSTGRES_DB -c "select * from predictions order by id desc limit 5;"
```
3. Restart the app and show rows remain (permanent record). **WRITE:** table schema, why each column (what/when/which model/how confident), SQLite locally vs PostgreSQL in Compose, rows written after the model answers.

---------------------------------------------------------------------------------------------------
## TASK 6 - Docker and Compose (14 marks)

### 6.1 Warm-up (2)
```bash
bash labs/docker_warmup.sh     # or run line by line, one SHOT each: pull, run -p, curl, -v mount, logs, exec sh, ps -a, images, stop/rm/rmi
```
**WRITE** definitions: image, container, layer, volume, port mapping (in your words).

### 6.2 Dockerfile (3)
```bash
docker build -t visionguard-web .
docker run --rm visionguard-web whoami          # SHOT: appuser (not root)
docker inspect --format='{{json .Config.Healthcheck}}' visionguard-web
```
**SHOT/LINK:** Dockerfile on GitHub. **WRITE:** order = base -> `requirements-app.txt` -> `pip install` -> source; Docker caches layers top-down, so code edits don't re-install TensorFlow.
`python:3.11-slim` (small), `tensorflow-cpu` (no CUDA wheels; inference on a CPU VM; saves GBs), `--no-cache-dir`/`PIP_NO_CACHE_DIR`, non-root `appuser`, HEALTHCHECK on `/health`, `.dockerignore` excludes `.env`, data, git, models; secrets passed at runtime via env, never `COPY`/`ENV` in the image (`docker history` shows no secrets).

### 6.3 Compose stack (4)
```bash
cp .env.example .env     # set POSTGRES_PASSWORD etc. (no quotes or '@' ':' '/' in the password, it goes into a URL)
cp /path/to/champion.keras models/ && cp /path/to/champion_meta.json models/
docker compose up -d --build
docker compose ps                                  # SHOT: db (healthy), web (healthy)
docker compose config | head -40                   # shows services/network; SHOT optional
# proof DB port is closed from outside:
docker compose ps --format json | grep -c 5432     # or: just show PORTS column has no 0.0.0.0:5432
nc -zv localhost 5432  ;  nc -zv <your-LAN-ip> 5432   # "refused" (from host and from another device/phone on Wi-Fi)
# proof web CAN reach db by service name:
docker compose exec web python -c "import socket; s=socket.create_connection(('db',5432),3); print('db reachable by name')"
docker compose exec db psql -U $POSTGRES_USER -d $POSTGRES_DB -c "\dt"
```
**LINK:** compose file. **WRITE:** internal network by service name (`db`), named volume `pgdata` (survive `docker compose down`; show `docker volume ls`, and that history survives `down` + `up`),
`depends_on: service_healthy` + `pg_isready` healthcheck (web waits until Postgres accepts connections), no `ports:` on db, passwords from untracked `.env` (+ `.env.example` committed),
model **mounted read-only** (small image, swap champion without rebuild; trade-off: file must exist on host) vs built-in (self-contained but huge/rebuild per model).

### 6.4 Image analysis (1.5)
```bash
docker build -f docs/Dockerfile.naive -t visionguard-naive .       # "before"
docker images | grep visionguard                                   # compare SIZE of visionguard-naive vs visionguard-web ("after")
docker history visionguard-web | head -15                          # biggest layer = TensorFlow install
```
**WRITE:** the two sizes (e.g. X GB -> Y GB), which layers are largest, which optimisations gave the gain (slim base, tensorflow-cpu, `.dockerignore`, no pip cache). Optionally try multi-stage and report.

### 6.5 Operations and troubleshooting (1.5)
```bash
docker compose stop; docker compose start; docker compose restart web
docker compose logs --tail=50 web ; docker compose logs -f db
docker compose exec web sh ; docker compose exec db sh              # SHOT each; exit
```
Break on purpose (pick one, SHOT the log):
- Missing model: `mv models/champion.keras /tmp/ && docker compose restart web` -> `docker compose logs web` shows `FileNotFoundError: Model file not found at /models/champion.keras`.
- Wrong password: edit `.env` DB password for web only (or `docker compose run -e DATABASE_URL=... web`) -> log shows `password authentication failed for user`.
**WRITE:** symptom -> log line -> diagnosis (only from logs) -> fix -> verification (`ps` healthy again).

### 6.6 GitHub Actions (2)
Toy: create a new throwaway GitHub repo with the files from `labs/ci_toy/` (`calc.py`, `test_calc.py`, `.github/workflows/test.yml`). Push -> red run (assert 2+2==5) on Actions tab -> **SHOT**.
Fix the test to `== 4`, push -> green with both matrix legs (Python 3.10 and 3.12) -> **SHOT**.
Project: `.github/workflows/ci.yml` runs on every push: install `requirements-ci.txt`, `ruff check .`, `pytest`, `docker build`. Push, **SHOT** the green run. Add badge to `README.md`:
`![CI](https://github.com/<user>/<ROLL>_A2_Ops/actions/workflows/ci.yml/badge.svg)`.
**WRITE:** secrets live in repo Settings -> Secrets and variables -> Actions (encrypted, masked in logs, not in Git history); the workflow YAML is committed and public, so a token there would leak. CI doesn't install TensorFlow because tests use a fake predictor (fast).

---------------------------------------------------------------------------------------------------
## TASK 7 - API deployment (15 marks) - Pathway A: Oracle Cloud Always Free

Research items to cover in the report: current free limits shown in YOUR console (Compute -> shapes; Always Free labels), Arm (A1.Flex, up to 4 OCPU/24 GB) vs x86 (E2.1.Micro, 1 GB RAM),
capacity availability ("Out of capacity" -> retry/other availability domain), idle-instance reclamation policy (Always Free instances with very low usage may be reclaimed; check Oracle's current docs),
two firewalls (VCN security list/NSG + the VM's iptables).

### 7.1 Prepare the model + API locally
```bash
python -m vg.export_champion                                   # models/champion.keras, champion_meta.json
python -c "import secrets; print(secrets.token_urlsafe(32))"    # your REMOTE_API_KEY / API_KEY
```
Test locally first: `API_KEY=test MODEL_PATH=models/champion.keras gunicorn -w1 -b 0.0.0.0:8000 model_api.app:app` then `curl -F image=@x.jpg -H "X-API-Key: test" localhost:8000/predict`.

### 7.2 Create the VM (SHOT every step, with console URL)
1. cloud.oracle.com -> Compute -> Instances -> **Create instance**; name `<ROLL>_A2_Ops`.
2. Image: Ubuntu 22.04/24.04; Shape: `VM.Standard.A1.Flex` (2 OCPU/12 GB) if available else `VM.Standard.E2.1.Micro`.
3. Networking: public IPv4 assigned. Add your SSH public key (`ssh-keygen -t ed25519` locally) or download the generated key.
4. Create; note the **public IP**.

### 7.3 Open the first firewall (cloud network)
Instance -> Subnet -> **Security List** -> Add Ingress Rule: Source CIDR `0.0.0.0/0`, TCP, Destination port `8000`. (Also keep 22 open.) **SHOT.**

### 7.4 Connect, install Docker, open the second firewall (VM)
```bash
chmod 600 key.pem && ssh -i key.pem ubuntu@<IP>
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER && exit      # reconnect so the group applies
ssh -i key.pem ubuntu@<IP>
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo apt install -y iptables-persistent && sudo netfilter-persistent save     # keep rule after reboot
# 1 GB micro VM only: add swap
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
```

### 7.5 Deploy
```bash
git clone https://github.com/<user>/<ROLL>_A2_Ops.git && cd <ROLL>_A2_Ops
mkdir -p models       # exit, then from your laptop:
scp -i key.pem models/champion.keras models/champion_meta.json ubuntu@<IP>:~/<ROLL>_A2_Ops/models/
# back on the VM:
printf "REMOTE_API_KEY=<your key>\nTHRESHOLD=\n" > .env && chmod 600 .env
docker compose -f docker-compose.api.yml up -d --build        # first build takes several minutes
docker compose -f docker-compose.api.yml ps
curl http://localhost:8000/health
```
On Arm the same Dockerfile works (TensorFlow-CPU has aarch64 wheels); if the build fails on a wheel, say so and fix by pinning versions.

### 7.6 Test with Postman (from a DIFFERENT network - phone hotspot)
- GET `http://<IP>:8000/health` -> 200.
- POST `http://<IP>:8000/predict`, Headers: `X-API-Key: <key>`, Body -> form-data -> key `image` type **File** -> choose an image -> **200** JSON with label/confidence/prob_ai.
- Same without header -> **401**; with a `.txt` file -> **400**.
**SHOT:** each request + response, with your hotspot/other-network context visible. **WRITE** the logged result (copy the JSON), and **put the API URL in the report front matter and Task 7 chapter.**
**WRITE** also: why API key (open endpoint = anyone can burn your CPU), why one gunicorn worker (each worker loads its own model copy), and (if relevant) Vercel alternative reasoning: TensorFlow won't fit the function limit, no persistent disk, so the model API must live elsewhere.

Pathway C instead: any VM (AWS EC2 t3.micro, GCP e2-micro, Azure B1s) works with the same 7.3-7.6 steps, using that provider's security group + OS firewall. Check the free-tier terms on the day.

---------------------------------------------------------------------------------------------------
## TASK 8 - Integrate Flask with the cloud API (15 marks)
```bash
unset MODEL_PATH
export INFERENCE_MODE=remote REMOTE_API_URL=http://<IP>:8000 REMOTE_API_KEY=<key> DATABASE_URL=sqlite:///visionguard.db
python -m app            # runs on localhost only, but predictions come from the cloud
```
1. Upload **at least 5** different images (mix of real and AI, include one bad file). **SHOT** each result page (address bar shows `127.0.0.1:5000`).
2. **SHOT** `/history` (shows `Via = remote`, latency = round-trip) and a DB query (`sqlite3 ... select ...`).
3. Prove the cloud is used: on the VM `docker compose -f docker-compose.api.yml logs --tail=20 api` shows the requests; or stop the local model (not loaded in remote mode).
4. Failure demo: stop the API container -> upload -> friendly "Model unavailable" page (502) and no history row; restart.
**WRITE:** the flow browser -> local Flask -> HTTPS/HTTP request with API key -> cloud model -> JSON -> row in DB -> result. Note HTTP is unencrypted: say how you'd add HTTPS (reverse proxy/Caddy + domain) as a limitation.

---------------------------------------------------------------------------------------------------
## FINAL - Report and submission

1. Assemble the report from `docs/REPORT_DRAFT.md`: front matter (name, roll no., GitHub, DagsHub, live URL, commit hash, AI disclosure), one chapter per task, each sub-task ID with
   what/why/how/what went wrong, every screenshot captioned + referenced. Don't paste code without commentary.
2. Freeze:
```bash
git status                     # clean
dvc status -c                  # data in sync with remote
ruff check . && python -m pytest -q
git add -A && git commit -m "Final submission" && git push
git tag submission && git push --tags
git rev-parse HEAD             # paste into report front matter; make NO commits after this
```
3. Check GitHub Actions is green on that commit; the VM + API still answer from another network; DagsHub project is public/shared with Experiments, Registry (2 versions, `champion`), and DVC data visible.
4. Export PDF -> `<ROLL>_VisionGuard.pdf` (check page limit). Submit: PDF + GitHub link + DagsHub link + API URL.

### Common problems
| Problem | Fix |
|---|---|
| `dvc add` says path is git-ignored | remove `data/raw/` from `.gitignore` |
| DagsHub 401 for MLflow/DVC | token wrong/expired; username must be your DagsHub username; re-export env vars |
| `tensorflow` install fails | Python 3.13 unsupported - use 3.10-3.12 |
| `mlflow.tensorflow.autolog` error with Keras 3 | pin versions that work (`pip freeze` shows yours), or note limitation in the report |
| Out of memory in training | lower `--batch 16` and/or `--img-size 160` |
| Port 5000 busy | MLflow UI on 5001; or `PORT=5050 python -m app` |
| Compose web exits at start | `docker compose logs web` (missing model / DB password) |
| Postgres password with `@` or `/` breaks URL | use letters+digits only |
| Oracle "Out of host capacity" | try later, another availability domain, or pathway C |
| Can't reach `<IP>:8000` | check BOTH firewalls (security list + iptables) and that the container is running |
| Colab loses files | commit code, `dvc pull` data, download `champion.keras` before runtime resets |
