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
"""Validation of Lemma 1 (geometry and gauge invariance) and every step of its proof.

What is checked
  (i)   Ric = -B/4 = (N/2)<.,.> on su(N), N = 2..6: Killing form B(X,X) = Tr(ad_X^2) = 2N Tr(X^2), and
        independently Ric(X,X) = (1/4) sum_a |[X, T_a]|^2.
  (ii)  Left and right translations preserve the metric; the heat kernel is central, so P_t commutes with the gauge
        action; hence V_t and Q_{s,t} f are gauge invariant whenever S and f are.
  (iii) The gradient of a gauge-invariant function is orthogonal to the gauge orbit: finite differences on a graph
        with two vertices and three edges, SU(2) and SU(3), at random configurations AND at the identity
        configuration, whose stabilizer is the whole diagonal group.
"""
import sys

import numpy as np

from common import Report, su_basis, ip, haar_sun, expm_skew


def f_inv(U):
    a = np.trace(U[0] @ U[1].conj().T)
    b = np.trace(U[1] @ U[2].conj().T)
    c = np.trace(U[0] @ U[1].conj().T @ U[2] @ U[1].conj().T)
    return float(np.real(a) + 0.7 * np.real(b) ** 2 + 0.3 * np.imag(c) + 0.5 * np.real(a) * np.real(c))


def gradient(U, B, h=1e-5):
    g = []
    for e in range(len(U)):
        comp = np.zeros_like(U[0])
        for T in B:
            Up, Um = U.copy(), U.copy()
            Up[e] = expm_skew(h * T) @ U[e]; Um[e] = expm_skew(-h * T) @ U[e]
            comp = comp + (f_inv(Up) - f_inv(Um)) / (2 * h) * T
        g.append(comp)
    return g


def torus(N, rng):
    ph = rng.uniform(0, 2 * np.pi, N)
    ph[-1] = -ph[:-1].sum()
    return np.diag(np.exp(1j * ph))


def main():
    rep = Report("Lemma 1")
    rng = np.random.default_rng(111)

    ok, det = True, []
    for N in range(2, 7):
        B = su_basis(N)
        worst = 0
        for _ in range(8):
            Xm = sum(c * T for c, T in zip(rng.normal(size=len(B)), B))
            ad = np.array([[ip(Xm @ Tb - Tb @ Xm, Ta) for Tb in B] for Ta in B])
            kil = np.trace(ad @ ad)
            ric = 0.25 * sum(ip(Xm @ T - T @ Xm, Xm @ T - T @ Xm) for T in B)
            worst = max(worst, abs(kil - 2 * N * np.real(np.trace(Xm @ Xm))), abs(ric - N / 2 * ip(Xm, Xm)), abs(-kil / 4 - ric))
        ok &= worst < 1e-9
        det.append(f"N={N}: {worst:.0e}")
    rep.check("(i) Ric = -B/4 = (N/2)<.,.>", ok, "; ".join(det))

    ok = True
    for N in (2, 3):
        B = su_basis(N)
        for _ in range(20):
            g, h = haar_sun(2, N, rng)
            Xm = sum(c * T for c, T in zip(rng.normal(size=len(B)), B))
            Ym = sum(c * T for c, T in zip(rng.normal(size=len(B)), B))
            ok &= abs(ip(g @ Xm @ g.conj().T, g @ Ym @ g.conj().T) - ip(Xm, Ym)) < 1e-12        # Ad-invariance
            ok &= abs(ip(h.conj().T @ Xm @ h, h.conj().T @ Ym @ h) - ip(Xm, Ym)) < 1e-12
    rep.check("(ii) the metric is bi-invariant, so the gauge action is by isometries", ok)

    # heat kernel central: p_t(g W g^-1) = p_t(W) because p_t is a function of the class angle only
    nn = np.arange(1, 200)
    def p_t(W, t=0.2):
        th = np.arccos(np.clip(np.real(np.trace(W)) / 2, -1, 1))
        return float((nn * np.exp(-t * (nn ** 2 - 1) / 2)) @ np.sin(nn * th) / np.sin(th))
    ok = True
    for _ in range(50):
        W, g = haar_sun(2, 2, rng)
        ok &= abs(p_t(g @ W @ g.conj().T) - p_t(W)) < 1e-9 * max(1, abs(p_t(W)))
    rep.check("(ii) the heat kernel is central, so P_t commutes with left and right translations", ok)

    ok, det = True, []
    for N in (2, 3):
        B = su_basis(N)
        configs = [("random", haar_sun(3, N, rng)), ("random", haar_sun(3, N, rng)),
                   ("identity (maximal stabilizer)", np.array([np.eye(N, dtype=complex)] * 3)),
                   ("U_1 = U_2 = U_3 (stabilizer SU(N))", np.array([haar_sun(1, N, rng)[0]] * 3)),
                   ("diagonal (stabilizer contains the maximal torus)", np.array([torus(N, rng) for _ in range(3)]))]
        for name, U in configs:
            g = gradient(U, B)
            worst = 0
            for T in B:
                # orbit tangents: g_0 = exp(sT): A_e = T ;  g_1 = exp(sT): A_e = -U_e T U_e^-1
                worst = max(worst, abs(sum(ip(g[e], T) for e in range(3))),
                            abs(sum(ip(g[e], -U[e] @ T @ U[e].conj().T) for e in range(3))))
            gn = sum(ip(x, x) for x in g) ** 0.5
            ok &= worst < 1e-6 * max(1, gn)
            det.append(f"SU({N}) {name}: {worst:.0e} (|grad| = {gn:.2f})")
    rep.check("(iii) grad f is orthogonal to the gauge orbit, including points with nontrivial stabilizer", ok, "; ".join(det[i] for i in (0, 2, 4, 5, 9)))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
