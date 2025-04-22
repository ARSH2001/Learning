# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def calculate_mse(v):
    """Calculate the mse for vector v."""
    return np.mean(v**2)


def calculate_mae(v):
    """Calculate the mae for vector v."""
    return np.mean(np.abs(v))

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    error = (y - tx.dot(w))
    return (1/(2*tx.shape[0]))*calculate_mse(error)
    #return (1/(2*tx.shape[0]))*calculate_mae(error)
    # ***************************************************
    raise NotImplementedError