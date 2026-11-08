"""Tasks 2.3 / 2.4 / 3.3 / 3.4 - train one experiment end-to-end and track it.

Examples
  python -m vg.train --model baseline --run-name baseline_lr1e-3
  python -m vg.train --model transfer --backbone mobilenetv2 --tracking manual --run-name mnv2_manual
  python -m vg.train --model transfer --backbone mobilenetv2 --tracking autolog --run-name mnv2_autolog
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorflow import keras  # noqa: E402

from vg import evaluate as ev  # noqa: E402
from vg import models, tracking  # noqa: E402
from vg.config import BATCH, IMG_SIZE, MODEL_DIR, REPORT_DIR, SEED, SPLIT_DIR  # noqa: E402
from vg.data import make_ds  # noqa: E402


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--model", choices=["baseline", "transfer"], default="transfer")
    a.add_argument("--backbone", default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0"])
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--lr-ft", type=float, default=1e-5, help="fine-tune learning rate (much smaller)")
    a.add_argument("--epochs", type=int, default=15)
    a.add_argument("--ft-epochs", type=int, default=10)
    a.add_argument("--dropout", type=float, default=0.3)
    a.add_argument("--img-size", type=int, default=IMG_SIZE)
    a.add_argument("--batch", type=int, default=BATCH)
    a.add_argument("--no-augment", action="store_true")
    a.add_argument("--no-finetune", action="store_true")
    a.add_argument("--threshold-mode", default="f1", choices=["f1", "youden", "recall95"])
    a.add_argument("--tracking", choices=["manual", "autolog", "none"], default="manual")
    a.add_argument("--run-name", default=None)
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--register", action="store_true", help="register model in the registry after logging")
    return a.parse_args()


def callbacks(ckpt, offset, manual):
    cbs = [
        keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]
    if manual:
        cbs.append(tracking.EpochLogger(offset))
    return cbs


def plot_history(histories, path):
    keys = {k: sum((h.history[k] for h in histories), []) for k in histories[0].history}
    best = int(np.argmin(keys["val_loss"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (m, t) in zip(ax, [("loss", "Loss"), ("accuracy", "Accuracy")]):
        a.plot(keys[m], label="train")
        a.plot(keys[f"val_{m}"], label="val")
        a.axvline(best, color="red", ls="--", label=f"best epoch = {best + 1}")
        if len(histories) > 1:
            a.axvline(len(histories[0].history["loss"]) - 0.5, color="grey", ls=":", label="fine-tune starts")
        a.set(title=t, xlabel="epoch")
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return best + 1


def main():
    args = parse()
    keras.utils.set_random_seed(args.seed)  # python + numpy + TF seeds
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    name = args.run_name or f"{args.model}_{args.backbone}"

    train_ds = make_ds("train", True, args.batch, args.img_size, augment=not args.no_augment)
    val_ds = make_ds("val", False, args.batch, args.img_size)
    test_ds = make_ds("test", False, args.batch, args.img_size)

    use_mlflow = args.tracking != "none"
    manual = args.tracking == "manual"
    if use_mlflow:
        tracking.setup()
        if args.tracking == "autolog":
            import mlflow.tensorflow
            mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

    run_ctx = mlflow.start_run(run_name=name) if use_mlflow else None
    try:
        if use_mlflow:
            mlflow.set_tags({"git_commit": tracking.git_commit(), "dvc_data_version": tracking.dvc_data_version(),
                             "seed": str(args.seed), "tracking_mode": args.tracking, "model_family": args.model})
            mlflow.log_params({k: v for k, v in vars(args).items() if k != "run_name"})

        histories, trainable = [], {}
        ckpt = MODEL_DIR / f"{name}.keras"
        if args.model == "baseline":
            model = models.build_baseline(args.img_size, args.dropout)
            models.compile_model(model, args.lr)
            trainable["stage1"] = [lyr.name for lyr in model.layers if lyr.trainable]
            histories.append(model.fit(train_ds, validation_data=val_ds, epochs=args.epochs + args.ft_epochs,
                                       callbacks=callbacks(ckpt, 0, manual), verbose=2))
        else:
            model, base = models.build_transfer(args.backbone, args.img_size, args.dropout)
            models.compile_model(model, args.lr)
            trainable["stage1_head_only"] = [lyr.name for lyr in model.layers if lyr.trainable]
            histories.append(model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                                       callbacks=callbacks(ckpt, 0, manual), verbose=2))
            if not args.no_finetune:
                trainable["stage2_finetune"] = models.unfreeze_last_block(base, args.backbone)
                models.compile_model(model, args.lr_ft)  # must re-compile after changing trainable flags
                off = len(histories[0].history["loss"])
                if args.tracking == "autolog":  # autolog would re-log params -> use a nested run
                    with mlflow.start_run(run_name=f"{name}_finetune", nested=True):
                        histories.append(model.fit(train_ds, validation_data=val_ds, epochs=args.ft_epochs,
                                                   callbacks=callbacks(ckpt, off, False), verbose=2))
                else:
                    histories.append(model.fit(train_ds, validation_data=val_ds, epochs=args.ft_epochs,
                                               callbacks=callbacks(ckpt, off, manual), verbose=2))
        (REPORT_DIR / f"{name}_trainable_layers.json").write_text(json.dumps(trainable, indent=2))
        best_epoch = plot_history(histories, REPORT_DIR / f"{name}_curves.png")
        print("best epoch (lowest val_loss):", best_epoch)

        # ---- evaluation: threshold from VAL, then test set evaluated once ----
        yv, pv = ev.predict(model, val_ds)
        thr = ev.choose_threshold(yv, pv, args.threshold_mode)
        val_m = ev.metric_table(yv, pv, thr)
        yt, pt = ev.predict(model, test_ds)
        test_m = ev.metric_table(yt, pt, thr)
        print("val:", val_m, "\ntest:", test_m)
        ev.plot_confusion(yt, pt, thr, REPORT_DIR / f"{name}_confusion.png")
        ev.plot_roc(yt, pt, REPORT_DIR / f"{name}_roc.png")
        np.savez(REPORT_DIR / f"{name}_test_preds.npz", y=yt, p=pt)
        model.save(MODEL_DIR / f"{name}_final.keras")

        if use_mlflow:
            mlflow.log_metric("best_epoch", best_epoch)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
            for f in [f"{name}_curves.png", f"{name}_confusion.png", f"{name}_roc.png",
                      f"{name}_trainable_layers.json", f"{name}_test_preds.npz"]:
                mlflow.log_artifact(str(REPORT_DIR / f))
            for s in ["train", "val", "test"]:  # split lists = exact data used
                mlflow.log_artifact(str(SPLIT_DIR / f"{s}.csv"), artifact_path="splits")
            tracking.log_pip_freeze(REPORT_DIR / "pip_freeze.txt")
            import mlflow.keras
            mlflow.keras.log_model(model, "model",
                                   registered_model_name=os.getenv("REGISTERED_MODEL_NAME", "VisionGuard")
                                   if args.register else None)
            mlflow.log_dict({"threshold": thr, "mode": args.threshold_mode, "img_size": args.img_size},
                            "model_meta.json")
    finally:
        if run_ctx:
            mlflow.end_run()


if __name__ == "__main__":
    main()
