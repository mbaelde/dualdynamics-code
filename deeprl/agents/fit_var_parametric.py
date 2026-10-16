"""Least-squares step matrix M of x_{t+1} = M x_t + noise, x = (a_mean, lam) deviations."""

import sys

import numpy as np

for path in sys.argv[1:]:
    d = np.load(path)
    n = d["lam"].size
    x = np.stack([d["a_mean"], d["lam"]])[:, n // 3 :]
    x = x - x.mean(axis=1, keepdims=True)
    X, Y = x[:, :-1], x[:, 1:]
    M = Y @ X.T @ np.linalg.inv(X @ X.T)
    ev = np.linalg.eigvals(M)
    k = np.argmax(np.abs(ev.imag)) if np.any(ev.imag != 0) else np.argmax(np.abs(ev))
    sigma, w = -np.log(np.abs(ev[k])), abs(np.angle(ev[k]))
    eta_th, eta_lam = -M[0, 0], M[1, 0]  # M = [[1 - eta_th a, -eta_th c], [eta_lam c, 1]] with a = c = 1
    print(
        f"{path.split('/')[-1]}: M={np.round(M, 5).tolist()} eig={np.round(ev, 5)} "
        f"w_ex={w:.3e} sigma={sigma:.3e} zeta={sigma / max(w, 1e-12):.2f}"
    )
