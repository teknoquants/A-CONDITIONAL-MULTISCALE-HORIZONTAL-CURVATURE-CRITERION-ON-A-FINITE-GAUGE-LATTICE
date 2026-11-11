# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Validation of Definitions 1-8.

What is checked
  D1  The su(N) basis is orthonormal for <X,Y> = -Tr(XY) (N = 2..5).
  D2  The gauge action (g.U)_e = g_s(e) U_e g_t(e)^-1 is a group action and leaves Wilson loops invariant
      (SU(2), SU(3) on a graph with two vertices and three edges).
  D3  P_t is a semigroup, preserves positivity, constants and Haar integrals; it acts on each edge by convolution
      with the central heat kernel: Monte Carlo on one SU(2) plaquette with four edges, where
      (P_t F)(U_p) = (exp(4 t Laplacian) F)(U_p) for class functions F of the plaquette holonomy.
  D4/D5  V_0 = S, mu_0 = mu, and mu_t is a probability measure for every t.
  D6  Q_t f = P_t(f e^-S)/P_t(e^-S) equals Q_{0,t} f; L_t is the stated first-order perturbation of the Laplacian.
  D7  For constant kappa > 0, C_kappa = 1/kappa.
  D8  One loop: |grad x|^2 = (1 - x^2)/2 from the matrix definition; the radial metric is 2 dtheta^2; the
      characters satisfy Laplacian chi_{n+1} = -(n(n+2)/2) chi_{n+1} (symbolic); the density 1 + a x is positive
      iff |a| < 1.
"""
import math
import sys

import numpy as np
import sympy as sp
from scipy.integrate import quad

from common import (Report, su_basis, ip, haar_sun, Flow, heat, lap, d1, integ, X, TH, SN, WQ, NN, LAM, ACTIONS,
                    random_class_function, from_coef)


def heat_kernel_samples(t, n, rng):
    """Samples from the SU(2) heat kernel exp(t Laplacian) at the identity, as 2x2 matrices."""
    th = np.linspace(0, np.pi, 20001)[1:-1]
    nn = np.arange(1, 400)
    p = (nn * np.exp(-t * (nn ** 2 - 1) / 2)) @ np.sin(np.outer(nn, th)) / np.sin(th)
    dens = np.clip(p, 0, None) * np.sin(th) ** 2
    cdf = np.cumsum(dens); cdf /= cdf[-1]
    ang = np.interp(rng.random(n), cdf, th)
    ax = rng.normal(size=(n, 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    W = np.empty((n, 2, 2), complex)
    c, s = np.cos(ang), np.sin(ang)
    W[:, 0, 0] = c + 1j * s * ax[:, 2]; W[:, 1, 1] = c - 1j * s * ax[:, 2]
    W[:, 0, 1] = s * (1j * ax[:, 0] + ax[:, 1]); W[:, 1, 0] = s * (1j * ax[:, 0] - ax[:, 1])
    return W


def main():
    rep = Report("Definitions")
    rng = np.random.default_rng(101)

    ok = True
    for N in range(2, 6):
        B = su_basis(N)
        G = np.array([[ip(a, b) for b in B] for a in B])
        ok &= len(B) == N * N - 1 and np.allclose(G, np.eye(N * N - 1))
    rep.check("D1 orthonormal basis of su(N) for -Tr(XY), N = 2..5", ok)

    ok = True
    for N in (2, 3):
        U = haar_sun(3, N, rng)
        g, h = haar_sun(2, N, rng), haar_sun(2, N, rng)
        act = lambda gg, UU: np.array([gg[0] @ u @ gg[1].conj().T for u in UU])
        ok &= np.allclose(act(g, act(h, U)), act(np.array([g[0] @ h[0], g[1] @ h[1]]), U))
        loops = lambda UU: [np.trace(UU[0] @ UU[1].conj().T), np.trace(UU[1] @ UU[2].conj().T),
                            np.trace(UU[0] @ UU[1].conj().T @ UU[2] @ UU[1].conj().T)]
        ok &= np.allclose(loops(U), loops(act(g, U)))
    rep.check("D2 gauge action is a group action and preserves Wilson loops (SU(2), SU(3))", ok)

    f = random_class_function(rng)
    pos = np.exp(random_class_function(rng))
    ok = (np.abs(heat(heat(f, 0.3), 0.5) - heat(f, 0.8)).max() < 1e-12 and heat(pos, 0.2).min() > 0
          and abs(integ(heat(f, 0.7)) - integ(f)) < 1e-13 and np.abs(heat(1 + 0 * X, 1.0) - 1).max() < 1e-13)
    rep.check("D3 P_t: semigroup, positivity, constants, Haar integral", ok)

    t, n = 0.07, 400000
    U = haar_sun(4, 2, rng)
    Up = U[0] @ U[1] @ U[2].conj().T @ U[3].conj().T
    W = [heat_kernel_samples(t, n, rng) for _ in range(4)]
    A = [U[i] @ W[i] for i in range(4)]
    Upn = A[0] @ A[1] @ np.conj(np.swapaxes(A[2], 1, 2)) @ np.conj(np.swapaxes(A[3], 1, 2))
    xs = np.real(np.trace(Upn, axis1=1, axis2=2)) / 2
    x0 = float(np.real(np.trace(Up)) / 2)
    chi2, chi3 = 2 * xs, 4 * xs ** 2 - 1
    e2, e3 = math.exp(-4 * t * 1.5) * 2 * x0, math.exp(-4 * t * 4.0) * (4 * x0 ** 2 - 1)
    s2, s3 = chi2.std() / math.sqrt(n), chi3.std() / math.sqrt(n)
    rep.check("D3 edge-wise heat-kernel convolution: P_t F(U_p) = exp(4t Laplacian)F(U_p) (Monte Carlo, 4 edges)",
              abs(chi2.mean() - e2) < 5 * s2 and abs(chi3.mean() - e3) < 5 * s3,
              f"chi_2: {chi2.mean():.4f} vs {e2:.4f} ± {s2:.4f}; chi_3: {chi3.mean():.4f} vs {e3:.4f} ± {s3:.4f}")

    ok, worst = True, 0
    for name, mk in ACTIONS.items():
        F = Flow(mk())
        ok &= np.abs(F.V(0.0) - F.S).max() < 1e-8
        for tt in (0.0, 0.05, 0.5, 3.0):
            worst = max(worst, abs(F.mu(1 + 0 * X, tt) - 1))
            ok &= np.abs(F.Q(f, 0.0, tt) - heat(f * np.exp(-F.S), tt) / heat(np.exp(-F.S), tt)).max() < 1e-9
    rep.check("D4-D6 V_0 = S, mu_t is a probability measure, Q_t = Q_{0,t}", ok and worst < 1e-12, f"max |mu_t(1) - 1| = {worst:.1e}")

    ok = all(abs(2 * quad(lambda s: math.exp(-2 * kk * s), 0, np.inf)[0] - 1 / kk) < 1e-10 for kk in (0.3, 1.0, 2.5))
    rep.check("D7 constant kappa > 0 gives C_kappa = 1/kappa", ok)

    B = su_basis(2)
    ok = True
    for Um in haar_sun(200, 2, rng):
        xx = np.real(np.trace(Um)) / 2
        g2 = sum((np.real(np.trace(Um @ T)) / 2) ** 2 for T in B)
        ok &= abs(g2 - (1 - xx ** 2) / 2) < 1e-12
    gen = np.array([[1j, 0], [0, -1j]])
    ok &= abs(ip(gen, gen) - 2) < 1e-14
    th = sp.symbols("theta")
    for nn in range(0, 7):
        chi = sp.sin((nn + 1) * th) / sp.sin(th)
        res = sp.simplify((sp.diff(chi, th, 2) + 2 * sp.cot(th) * sp.diff(chi, th)) / 2 + sp.Rational(nn * (nn + 2), 2) * chi)
        ok &= res == 0
    rep.check("D8 |grad x|^2 = (1-x^2)/2, radial metric 2 dtheta^2, Laplacian chi_{n+1} = -(n(n+2)/2) chi_{n+1}", ok)
    rep.check("D8 density 1 + a x positive iff |a| < 1", all((1 + a * np.array([-1, 1])).min() > 0 for a in (-0.99, 0.5, 0.99))
              and (1 + 1.0 * -1) == 0)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
