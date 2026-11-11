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
"""Validation of Lemma 4 (dissipation identity) and every step of its proof.

What is checked (seven actions, random class functions, 8 scales each)
  (a) the three integrations by parts used in the proof:
        int u^2 Lap z = int z Lap(u^2),   Lap(u^2) = 2 u Lap u + 2|grad u|^2,   int <grad z, grad u^2> = -int z Lap(u^2);
  (b) the identity  -(d/dt) int (Q_t f)^2 dmu_t = 2 int |grad Q_t f|^2 dmu_t  (central differences in t);
  (c) its integrated form  int f^2 dmu - int (Q_T f)^2 dmu_T = 2 int_0^T int |grad Q_t f|^2 dmu_t dt.
"""
import sys

import numpy as np

from common import Report, Flow, ACTIONS, random_class_function, d1, lap, integ


def main():
    rep = Report("Lemma 4")
    rng = np.random.default_rng(404)
    wa = wb = wc = 0.0
    h = 2e-3
    for name, mk in ACTIONS.items():
        F = Flow(mk())
        for _ in range(2):
            f = random_class_function(rng)
            diss0 = 2 * F.mu(d1(f) ** 2 / 2)
            for t in np.geomspace(0.01, 3, 8):
                u, z = F.Q(f, 0.0, t), F.z(t)
                sc = max(1.0, np.abs(lap(u ** 2)).max()) * max(1.0, z.max())
                a1 = abs(integ(u ** 2 * lap(z)) - integ(z * lap(u ** 2)))
                a2 = np.abs(lap(u ** 2) - (2 * u * lap(u) + d1(u) ** 2)).max()          # 2|grad u|^2 = u'^2
                a3 = abs(integ(d1(z) * d1(u ** 2) / 2) + integ(z * lap(u ** 2)))
                wa = max(wa, a1 / sc, a2 / sc, a3 / sc)
                m = lambda s: F.mu(F.Q(f, 0.0, s) ** 2, s)
                lhs = -(8 * (m(t + h) - m(t - h)) - (m(t + 2 * h) - m(t - 2 * h))) / (12 * h)   # 4th-order difference
                rhs = 2 * F.mu(d1(u) ** 2 / 2, t)
                wb = max(wb, abs(lhs - rhs) / max(abs(rhs), 1e-3 * diss0))
            T = 2.0
            ts = np.concatenate([[0.0], np.geomspace(1e-5, T, 600)])
            diss = np.array([2 * F.mu(d1(F.Q(f, 0.0, s)) ** 2 / 2, s) for s in ts])
            integral = np.sum(0.5 * (diss[1:] + diss[:-1]) * np.diff(ts))
            lhs = F.mu(f ** 2) - F.mu(F.Q(f, 0.0, T) ** 2, T)
            wc = max(wc, abs(lhs - integral) / abs(lhs))
    rep.check("(a) integrations by parts on the compact manifold", wa < 1e-6, f"max error {wa:.1e}")
    rep.check("(b) -(d/dt) int (Q_t f)^2 dmu_t = 2 int |grad Q_t f|^2 dmu_t", wb < 1e-6, f"max rel residual {wb:.1e}")
    rep.check("(c) integrated form on [0, T]", wc < 2e-4, f"max rel error {wc:.1e}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
