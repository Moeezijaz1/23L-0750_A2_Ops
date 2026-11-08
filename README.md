# VisionGuard (ROLLNO_A2_Ops)

![CI](https://github.com/<user>/<ROLLNO_A2_Ops>/actions/workflows/ci.yml/badge.svg)

AI-generated vs real image classifier: TensorFlow/Keras, MLflow on DagsHub, Flask + database, Docker Compose, cloud-hosted model API.

* DagsHub: https://dagshub.com/<user>/<ROLLNO_A2_Ops>
* Live model API: http://<cloud-ip>:8000  (POST /predict, header `X-API-Key`)

| Folder | Purpose |
|---|---|
| `vg/` | audit, EDA, splits, tf.data, models, training, evaluation, tuner, registry scripts |
| `app/` | Flask web app (upload, result, history, health) + SQLAlchemy history table |
| `model_api/` | the cloud-deployed inference API (Task 7) |
| `labs/` | toy labs: DVC, MLflow, Keras Tuner, Docker warm-up, CI toy |
| `docs/RUNBOOK.md` | exact command order + screenshot checklist |
| `docs/REPORT_DRAFT.md` | report skeleton with draft explanations |

Quick start: see `docs/RUNBOOK.md`.  Run the app stack: `cp .env.example .env && docker compose up -d --build`.
