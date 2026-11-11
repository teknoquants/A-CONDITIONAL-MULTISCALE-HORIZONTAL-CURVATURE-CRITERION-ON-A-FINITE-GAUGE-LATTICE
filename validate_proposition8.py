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
"""Validation of Proposition 8 (comparison of bounds on one loop), its proof, and the table that follows it.

What is checked
  (i)   Single-scale Bakry-Emery: inf over horizontal vectors of Ric + Hess S equals 1 - |a|/(2(1-|a|)), attained at
        x = -sgn(a); it is positive iff |a| < 2/3; where positive, the optimal constant is <= its reciprocal.
        At a = 0.6 the single-scale test does NOT fail (constant 4) although kappa(0) = -0.5.
  (ii)  Holley-Stroock: the Haar constant on class functions is 2/3, the density lies in [1-|a|, 1+|a|], and the
        optimal constant is <= (2/3)(1+|a|)/(1-|a|).
  (iii) Multiscale: C(a) finite for all |a| < 1, C(a) >= optimal constant, growth (1-|a|)^(-4/3).
  Table: every printed entry (multiscale, single-scale, Holley-Stroock, optimal) to its printed precision.
  Observations in the text: multiscale < single-scale at every tabulated |a| > 0 where both exist; near |a| = 1
  the Holley-Stroock bound is smaller than the multiscale one; the optimal constant decreases with |a|.
"""
import math
import sys

import mpmath as mp
import numpy as np

from common import Report, optimal_constant, X, Flow

TABLE = [  # |a|, multiscale, single-scale, Holley-Stroock, optimal (as printed in the paper)
    (0.0, 1.0000, 1.0000, 0.6667, 0.6667), (0.3, 1.2548, 1.2727, 1.2381, 0.6526), (0.5, 1.6217, 2.0000, 2.0000, 0.6269),
    (0.6, 1.9648, 4.0000, 2.6667, 0.6088), (2 / 3, 2.3259, None, 3.3333, 0.5947), (0.8, 3.9214, None, 6.0000, 0.5615),
    (0.9, 8.6962, None, 12.667, 0.5323), (0.95, 20.509, None, 26.000, 0.5165), (0.99, 166.21, None, 132.67, 0.5033)]


def C_ms(a):
    a = abs(a)
    return float((1 - a) ** (-mp.mpf(4) / 3) * mp.hyp2f1(-mp.mpf(4) / 3, mp.mpf(4) / 3, mp.mpf(7) / 3, a))


def be_kappa(a):
    return 1 - abs(a) / (2 * (1 - abs(a)))


def main():
    rep = Report("Proposition 8")
    xs = np.linspace(-1, 1, 200001)

    ok, worst = True, 0
    for a in np.linspace(-0.98, 0.98, 197):
        if a == 0:
            continue
        hess = 0.5 * a * (xs + a) / (1 + a * xs) ** 2          # Hess S on the unit radial vector = S''/2
        worst = max(worst, abs(1 + hess.min() - be_kappa(a)))
        ok &= xs[hess.argmin()] == -np.sign(a)
        ok &= (be_kappa(a) > 0) == (abs(a) < 2 / 3)
    for a in (0.2, 0.4, 0.6, 0.65):
        ok &= optimal_constant(-np.log(1 + a * X)) <= 1 / be_kappa(a)
    k0 = 1 - 0.6 / 0.4
    rep.check("(i) single-scale: inf(Ric + Hess S) = 1 - |a|/(2(1-|a|)), positive iff |a| < 2/3, and it bounds the optimal constant",
              ok and worst < 1e-9, f"at a = 0.6: single-scale constant {1 / be_kappa(0.6):.4f}, kappa(0) = {k0:.2f}")

    ok = abs(optimal_constant(0 * X) - 2 / 3) < 1e-9
    for a in (0.1, 0.3, 0.6, 0.9, 0.99, -0.7):
        w = 1 + a * xs
        ok &= abs(w.min() - (1 - abs(a))) < 1e-12 and abs(w.max() - (1 + abs(a))) < 1e-12
        ok &= optimal_constant(-np.log(1 + a * X)) <= (2 / 3) * (1 + abs(a)) / (1 - abs(a))
    rep.check("(ii) Holley-Stroock: optimal <= (2/3)(1+|a|)/(1-|a|)", ok)

    ok = True
    for a in (0.1, 0.5, 0.9, 0.99, 0.999):
        ok &= math.isfinite(C_ms(a)) and (a > 0.99 or optimal_constant(-np.log(1 + a * X)) <= C_ms(a))
    lim = float(mp.gamma(mp.mpf(7) / 3) ** 2 / mp.gamma(mp.mpf(11) / 3))
    ok &= abs((1 - 0.9999) ** (4 / 3) * C_ms(0.9999) - lim) < 1e-3
    rep.check("(iii) C(a) finite, >= optimal constant, and ~ 0.3533 (1-|a|)^(-4/3)", ok)

    ok, det = True, []
    opt_prev = 1.0
    for a, ms, ss, hs, opt in TABLE:
        o = optimal_constant(-np.log(1 + a * X))
        e_ms = abs(C_ms(a) - ms) / ms
        e_hs = abs((2 / 3) * (1 + a) / (1 - a) - hs) / hs
        e_opt = abs(o - opt)
        ok &= e_ms < 6e-5 and e_hs < 6e-5 and e_opt < 6e-5
        if ss is None:
            ok &= be_kappa(a) <= 1e-12
        else:
            ok &= abs(1 / be_kappa(a) - ss) < 6e-5
            ok &= a == 0 or C_ms(a) < 1 / be_kappa(a)
        ok &= o < opt_prev + 1e-12
        opt_prev = o
        det.append(f"{a:.3g}: {C_ms(a):.5g}/{o:.4f}")
    ok &= (2 / 3) * 1.99 / 0.01 < C_ms(0.99) and (2 / 3) * 1.95 / 0.05 > C_ms(0.95)
    rep.check("table: all 36 printed entries; multiscale < single-scale; optimal decreasing; Holley-Stroock wins at 0.99", ok,
              "multiscale/optimal " + "; ".join(det))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
