# Runbook - what to run, in order, and what to screenshot

Rename everything to `<RollNumber>_A2_Ops`. Every screenshot must show your prompt/username, date/time, project name.
Windows users: run TensorFlow in WSL2 (native Windows TF has no GPU support after 2.10).

## 0. Accounts you must create yourself (I cannot do these for you)
GitHub repo `<RollNumber>_A2_Ops` -> DagsHub (connect repo) -> DagsHub access token -> Kaggle (download data) -> a cloud account (Oracle Always Free recommended).

## Task 1
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
pip freeze | grep -Ei "tensorflow|keras|mlflow|dvc|scikit|pandas"          # 1.1 screenshot
# download Kaggle dataset, unzip so that data/raw/AiArtData and data/raw/RealArt exist
python -m vg.audit            # 1.2 -> reports/audit_summary.txt (copy tables into report)
python -m vg.eda              # 1.3 -> reports/eda_*.png
python -m vg.splits           # leak-free train/val/test lists
bash labs/dvc_toy.sh          # 1.4 toy (in a throwaway folder, outside the repo!)
# 1.4 apply: inside the repo
dvc init
dvc add data/raw
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<user>/<repo>.s3
dvc remote modify --local origin access_key_id <TOKEN>
dvc remote modify --local origin secret_access_key <TOKEN>
git add data/raw.dvc data/.gitignore .dvc && git commit -m "track dataset with DVC" && dvc push && git push
```
Note: `data/splits/*.csv` are small - commit them to Git (or `dvc add` them) so the exact split is versioned.

## Task 2 (needs the data; GPU or Colab recommended)
```bash
python -m vg.train --model baseline --run-name r1_baseline_cnn --epochs 25 --ft-epochs 0 --tracking none
python -m vg.train --model transfer --run-name r3_mnv2_finetune --tracking none
python -c "from vg.models import *; build_baseline().summary()"                 # 2.2 summary screenshot
python -m vg.errors --run r3_mnv2_finetune --threshold <T from train output>     # 2.6
python -m vg.tune --trials 10 ; python labs/keras_tuner_toy.py                    # 2.7
python -m vg.gradcam --model models/r3_mnv2_finetune_final.keras --run r3_mnv2_finetune   # bonus
```
Outputs land in `reports/` (curves with best epoch, confusion matrix, ROC, trainable-layer list).
2.1 augmented-batch figure: take one batch from `make_ds("train", True)` and plot 9 images (5-line snippet in a notebook).

## Task 3
```bash
cp .env.example .env     # fill DagsHub URI/user/token. Prove ignored:  git check-ignore -v .env
python labs/mlflow_toy.py && mlflow ui          # 3.1 local; then set DagsHub vars and rerun for 3.2
bash scripts/run_experiments.sh                 # 3.3 + 3.4 + 3.5 (8 runs; r3 = manual, r8 = autolog)
python -m vg.compare_runs                       # comparison table/chart
python -m vg.register --run-id <run_id_A> --no-alias
python -m vg.register --run-id <run_id_B>       # version 2 gets alias champion
python -m vg.load_champion some_image.jpg       # 3.6 terminal screenshot
python -m vg.export_champion                    # models/champion.keras for serving (no DagsHub call per request)
```

## Task 4/5 (local app)
```bash
export MODEL_PATH=models/champion.keras
python -m app            # http://127.0.0.1:5000  - upload, result, /history, /health
python -m pytest -q
```
Screenshots for 4.2: wrong type (.txt), oversize (> MAX_UPLOAD_MB), fake `.png` containing text.

## Task 6
```bash
bash labs/docker_warmup.sh                                   # 6.1 (screenshot each step)
cp .env.example .env && docker compose up -d --build
docker compose ps                                            # shows (healthy)
docker compose exec web whoami                               # appuser (non-root)
nmap -p 5432 <your-host-ip>   # or: nc -zv <host-ip> 5432  from ANOTHER machine -> closed
docker images visionguard-web ; docker history visionguard-web   # 6.4 sizes / biggest layers
# 6.4 optimisation example: tensorflow -> tensorflow-cpu, --no-cache-dir, .dockerignore: report before/after
docker compose stop && docker compose start && docker compose restart web
docker compose logs -f web ; docker compose exec db sh
# 6.5 break: set POSTGRES_PASSWORD wrong in .env for web only, OR remove models/champion.keras -> read `docker compose logs web`
```
6.6: push `labs/ci_toy` as its own throwaway repo (red run with the wrong assert, then green); then push the project, add the badge.

## Task 7 (pathway A - Oracle Always Free VM)
1. Create Ubuntu VM (Ampere A1 Arm or AMD E2.1.Micro). Open ingress TCP 22 and 8000 in the VCN security list AND on the VM: `sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT` (Oracle images ship restrictive iptables).
2. `sudo apt update && sudo apt install -y docker.io docker-compose-v2 git && sudo usermod -aG docker $USER` (re-login).
3. `git clone <repo> && cd repo`; copy `models/champion.keras` + `champion_meta.json` to the VM (`scp`); create `.env` with `REMOTE_API_KEY` (long random) and `THRESHOLD`.
4. `docker compose -f docker-compose.api.yml up -d --build`; check `curl http://<ip>:8000/health`.
5. Arm VM: `tensorflow-cpu` publishes aarch64 wheels, so the same Dockerfile builds; x86 micro VM has 1 GB RAM - add swap or use the A1 shape.
6. Postman: POST `http://<ip>:8000/predict`, Headers `X-API-Key`, Body form-data key `image` (type File). Screenshot request + JSON response; also a call with wrong key (401).
7. Test from another network (mobile hotspot). Put the API URL in the report.

## Task 8
```bash
export INFERENCE_MODE=remote REMOTE_API_URL=http://<ip>:8000 REMOTE_API_KEY=<key> DATABASE_URL=sqlite:///visionguard.db
python -m app
```
Upload 5+ images, screenshot each result and `/history` (shows `Via = remote`).
