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
"""Validation of Theorem 5 (the criterion), Corollary 6 (free action), and every step of their proofs.

What is checked
  Theorem 5, for seven actions on one SU(2) loop (free, two linear densities, Wilson at beta = 1 and 2, a double
  well, a cubic), with the optimal horizontal kappa(t) of each action:
    (a) convergence used in the proof: Q_T f -> int f dmu uniformly, mu_T -> Haar, int (Q_T f)^2 dmu_T -> (int f dmu)^2;
    (b) the exact decomposition  Var_mu(f) = 2 int_0^infty int |grad Q_t f|^2 dmu_t dt;
    (c) Var_mu(f) <= C_kappa int |grad f|^2 dmu for 200 random class functions per action;
    (d) the same for ALL class functions in a 120-dimensional space: the optimal Poincare constant (generalized
        eigenvalue problem) is <= C_kappa;
    (e) kappa(t) is negative on an interval of scales for the non-convex actions, and C_kappa is finite anyway;
    (f) horizontal versus full Hessian: kappa_hor(t) >= kappa_full(t), hence C_hor <= C_full (on one loop the two
        coincide for the tested actions; reported, not asserted in the paper).
  Corollary 6:
    (g) S = 0: V_t = 0, kappa = N/2, C = 2 int_0^infty exp(-N t) dt = 2/N (N = 2..8), numerically 1 for SU(2);
    (h) the sharp constant N/(N^2-1) is smaller than 2/N for every N >= 2; for SU(2) the optimal constant on class
        functions is 2/3.
"""
import sys

import numpy as np
from scipy.integrate import quad

from common import Report, Flow, ACTIONS, random_class_function, optimal_constant, d1, X


def main():
    rep = Report("Theorem 5 / Corollary 6")
    rng = np.random.default_rng(505)
    rows, ok_a, ok_b, ok_c, ok_d, ok_f = [], True, True, True, True, True
    neg = []
    for name, mk in ACTIONS.items():
        F = Flow(mk())
        C, ts, kap, Lam = F.constant()
        Cfull = F.constant(full=True)[0]
        Copt = optimal_constant(F.S)
        f = random_class_function(rng)
        mean = F.mu(f)
        T = 25.0
        ok_a &= np.abs(F.Q(f, 0.0, T) - mean).max() < 1e-8 and np.abs(F.z(T) / F.Z - 1).max() < 1e-8
        ok_a &= abs(F.mu(F.Q(f, 0.0, T) ** 2, T) - mean ** 2) < 1e-8
        tt = np.concatenate([[0.0], np.geomspace(1e-6, T, 900)])
        diss = np.array([2 * F.mu(d1(F.Q(f, 0.0, s)) ** 2 / 2, s) for s in tt])
        var = F.mu(f ** 2) - mean ** 2
        ok_b &= abs(np.sum(0.5 * (diss[1:] + diss[:-1]) * np.diff(tt)) - var) / var < 3e-4
        worst = 0.0
        for _ in range(200):
            g = random_class_function(rng, deg=int(rng.integers(1, 9)))
            v = F.mu(g ** 2) - F.mu(g) ** 2
            e = F.mu(d1(g) ** 2 / 2)
            worst = max(worst, v / (C * e))
        ok_c &= worst <= 1 + 1e-9
        ok_d &= Copt <= C * (1 + 1e-6)
        ok_f &= C <= Cfull * (1 + 1e-9)
        if kap.min() < 0:
            neg.append(name)
        rows.append(f"{name}: C_kappa={C:.4g}, optimal={Copt:.4f}, max Var/(C E)={worst:.3f}, min kappa={kap.min():.2f}")
    for r in rows:
        print("      " + r)
    rep.check("(a) Q_T f -> mean, mu_T -> Haar, second moment -> mean^2", ok_a)
    rep.check("(b) Var_mu(f) = 2 int_0^infty int |grad Q_t f|^2 dmu_t dt", ok_b)
    rep.check("(c) Var <= C_kappa * Dirichlet energy for 200 random class functions per action", ok_c)
    rep.check("(d) optimal Poincare constant on class functions <= C_kappa for every action", ok_d)
    rep.check("(e) kappa(t) < 0 on some scales, yet C_kappa finite", len(neg) >= 4, "actions: " + ", ".join(neg))
    rep.check("(f) C_horizontal <= C_full-Hessian", ok_f)

    Ff = Flow(0 * X)
    Cf = Ff.constant()[0]
    ok = np.abs(Ff.V(0.7)).max() < 1e-12 and abs(Ff.kappa(0.3) - 1) < 1e-6 and abs(Cf - 1) < 1e-6
    ok &= all(abs(2 * quad(lambda t: np.exp(-N * t), 0, np.inf)[0] - 2 / N) < 1e-10 for N in range(2, 9))
    rep.check("(g) free action: V_t = 0, kappa = N/2, C = 2/N", ok, f"SU(2) numerical C = {Cf:.8f}")
    ok = all(N / (N * N - 1) < 2 / N for N in range(2, 50)) and abs(optimal_constant(0 * X) - 2 / 3) < 1e-9
    rep.check("(h) sharp constant N/(N^2-1) < 2/N; SU(2) optimal constant on class functions = 2/3", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
