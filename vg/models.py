"""Task 2.2 / 2.3 - model builders. Pre-processing is INSIDE the model so serving only resizes."""
from tensorflow import keras
from tensorflow.keras import layers

from vg.config import IMG_SIZE

METRICS = lambda: [  # noqa: E731
    "accuracy",
    keras.metrics.AUC(name="auc"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]


def build_baseline(size=IMG_SIZE, dropout=0.4):
    inp = keras.Input((size, size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="rescale")(inp)  # 0..255 -> 0..1
    for i, f in enumerate([32, 64, 128, 128], 1):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inp, out, name="baseline_cnn")


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=METRICS())


def build_transfer(backbone="mobilenetv2", size=IMG_SIZE, dropout=0.3):
    inp = keras.Input((size, size, 3), name="image")
    if backbone == "mobilenetv2":
        base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(size, size, 3))
        # MobileNetV2 expects [-1, 1]; this Rescaling is identical to mobilenet_v2.preprocess_input
        x = layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocess")(inp)
    elif backbone == "efficientnetb0":
        base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(size, size, 3))
        x = inp  # EfficientNet has rescaling built in and wants raw 0..255
    else:
        raise ValueError(f"unknown backbone {backbone}")
    base.trainable = False
    x = base(x, training=False)  # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inp, out, name=f"vg_{backbone}"), base


def get_base(model):
    return next(lyr for lyr in model.layers if isinstance(lyr, keras.Model))


def last_block_names(backbone):
    return {"mobilenetv2": ("block_16", "Conv_1", "out_relu"),
            "efficientnetb0": ("block7", "top_")}[backbone]


def unfreeze_last_block(base, backbone, n_layers=None):
    """Unfreeze the last block (or last n_layers). BatchNorm layers stay frozen. Returns trainable layer names."""
    base.trainable = True
    prefixes = last_block_names(backbone)
    for i, layer in enumerate(base.layers):
        if n_layers is not None:
            want = i >= len(base.layers) - n_layers
        else:
            want = layer.name.startswith(prefixes)
        layer.trainable = bool(want) and not isinstance(layer, layers.BatchNormalization)
    return [lyr.name for lyr in base.layers if lyr.trainable]
