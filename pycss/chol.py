import numpy as np

def cholupdate(R, x):
    p = len(x)
    x = x.copy()
    for i in range(p):
        r = np.sqrt(R[i, i]**2 + x[i]**2)
        c = r / R[i, i]
        s = x[i] / R[i, i]
        R[i, i] = r
        if i < p - 1:
            R[i, i+1:] = (R[i, i+1:] + s * x[i+1:]) / c
            x[i+1:] = c * x[i+1:] - s * R[i, i+1:]
    return R
