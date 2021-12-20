import tensorflow as tf
import time
import numpy as np
from tqdm.notebook import tqdm


def __get_inf_time(model, n_iterations, image_size):
    """
    Determine inference speed.
    Args:
        model: TensorFlow model that shal be used.
        n_iterations: Number of predictions for measuring inference speed.
        image_size: Input image size.
    Returns:
    Average inference speed.
    """
    # create random image for inference
    image = np.random.uniform(-1.0, 1.0, image_size)
    ts = []
    for _ in tqdm(range(n_iterations)):
        # predict image and measure inference time
        t0 = time.time()
        model.predict(image)
        t1 = time.time()
        ts.append(t1 - t0)
    # omit the first 100 measures since they might be slower than the rest
    ts = np.asarray(ts[100::])
    ts_mean = np.mean(ts)
    # return mean inference time
    return ts_mean


def get_inference_time(model, n_iterations, image_size=(64, 64, 1), gpu=True):
    """
    Get inference speed using CPU or GPU.
    Args:
        model: TensorFlow model that shal be used.
        n_iterations: Number of predictions for measuring inference speed.
        image_size: Input image size.
        gpu: 'True' if GPU shall be used for inference instead of CPU.
    Returns:
    Average inference speed.
    """
    if gpu is False:
        with tf.device('/cpu:0'):
            ts_mean = __get_inf_time(model, n_iterations, image_size)
    else:
        ts_mean = __get_inf_time(model, n_iterations, image_size)
    return ts_mean
