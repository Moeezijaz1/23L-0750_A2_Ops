"""The ONE decode+resize function used by training AND serving (prevents train/serve skew).
Output is float32 in [0, 255]; model-specific scaling lives inside the model graph."""
import tensorflow as tf


def decode_resize(image_bytes, size: int):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    # Aspect ratio is NOT preserved (image is stretched to size x size). Antialiasing avoids
    # aliasing artefacts when shrinking large photos, which could otherwise fake "AI texture".
    img = tf.image.resize(img, (size, size), method="bilinear", antialias=True)
    img = tf.cast(img, tf.float32)
    img.set_shape((size, size, 3))
    return img
