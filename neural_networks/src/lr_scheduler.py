import numpy as np


def exp_scheduler(epoch, lr):
    """
    Exponential learning rate scheduler.
    Args:
        epoch: Current epoch during TensorFlow training.
        lr: Current learning rate after each epoch.
    Returns:
    Learning rate shrinking exponentially.
    """
    if epoch < 3:
        return 0.001
    else:
        return lr * np.exp(-0.1)
