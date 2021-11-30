import numpy as np

def not_zero(x: float, eps: float = 1e-3) -> float:
    if abs(x) > eps:
        return x
    elif x >= 0:
        return eps
    else:
        return -eps

def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi

