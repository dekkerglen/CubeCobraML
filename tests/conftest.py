"""Force CPU + quiet TF before any test imports tensorflow.

pytest imports conftest before collecting tests, so setting these here
guarantees they land before the first `import tensorflow` in the suite.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")
