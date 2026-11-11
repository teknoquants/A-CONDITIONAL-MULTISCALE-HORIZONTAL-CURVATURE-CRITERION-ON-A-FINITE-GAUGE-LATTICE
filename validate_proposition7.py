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
"""Validation of Proposition 7 (one SU(2) loop, exact) and every step of its proof.

What is checked
  (i)   P_t(1 + a x) = 1 + a exp(-3t/2) x and V_t = -log(1 + b(t) x), against the character-basis heat flow.
  (ii)  Symbolic: for V = -log(1 + b cos(theta)), V'' = b(x + b)/(1 + b x)^2, and
        d/dx [b(x+b)/(1+bx)^2] = b(1 - b x - 2 b^2)/(1 + b x)^3.
        Numerical: inf over [-1, 1] equals -|b|/(1 - |b|), attained at x = -sgn(b), for 400 values of b in (-1, 1),
        including b >= 1/2 where the function is not monotone; the endpoint value at x = sgn(b) is |b|/(1 + |b|).
        Isotropy at the central elements: the tangential eigenvalue cot(theta) V' has the same limit as V''.
  (iii) kappa(t) = 1 - |b(t)|/(1 - |b(t)|) agrees with the numerically optimal horizontal bound; kappa(0) < 0 iff
        |a| > 1/2; kappa(t*) = 0 at t* = (2/3) ln(2|a|).
  (iv)  Symbolic: int_0^t |b|/(1-|b|) ds = (2/3) ln[(1 - |b(t)|)/(1 - |a|)]. Numerical: the closed form of
        exp(-2 Lambda(t)); Euler's integral; C(a) = (1-|a|)^(-4/3) 2F1(-4/3, 4/3; 7/3; |a|) against direct
        quadrature (mpmath, agreement to 12 digits) and against the generic flow code.
  (v)   C(0) = 1, C(0.6) = 1.964815..., (1-|a|)^(4/3) C(a) -> Gamma(7/3)^2/Gamma(11/3) = 0.353328...; C(-a) = C(a).
"""
import math
import sys

import mpmath as mp
import numpy as np
import sympy as sp

from common import Report, Flow, heat, X, TH, SN


def C_closed(a):
    a = abs(a)
    return (1 - a) ** (-mp.mpf(4) / 3) * mp.hyp2f1(-mp.mpf(4) / 3, mp.mpf(4) / 3, mp.mpf(7) / 3, a)


def main():
    rep = Report("Proposition 7")
    mp.mp.dps = 30

    ok = True
    for a in (0.6, -0.9, 0.3):
        F = Flow(-np.log(1 + a * X))
        for t in (0.0, 0.1, 0.7, 2.0):
            b = a * math.exp(-1.5 * t)
            ok &= np.abs(F.z(t) - (1 + b * X)).max() < 1e-11 and np.abs(F.V(t) + np.log(1 + b * X)).max() < 1e-9
    rep.check("(i) P_t w = 1 + a exp(-3t/2) x and V_t = -log(1 + b(t) x)", ok)

    th, b, x = sp.symbols("theta b x", real=True)
    V = -sp.log(1 + b * sp.cos(th))
    e1 = sp.simplify(sp.diff(V, th, 2) - (b * (sp.cos(th) + b) / (1 + b * sp.cos(th)) ** 2))
    e2 = sp.simplify(sp.diff(b * (x + b) / (1 + b * x) ** 2, x) - b * (1 - b * x - 2 * b ** 2) / (1 + b * x) ** 3)
    rep.check("(ii) symbolic: V'' = b(x+b)/(1+bx)^2 and its x-derivative", e1 == 0 and e2 == 0)

    xs = np.linspace(-1, 1, 20001)
    ok, worst = True, 0
    for bb in np.concatenate([np.linspace(-0.995, 0.995, 399), [0.5]]):
        if bb == 0:
            continue
        vals = bb * (xs + bb) / (1 + bb * xs) ** 2
        worst = max(worst, abs(vals.min() + abs(bb) / (1 - abs(bb))))
        ok &= xs[vals.argmin()] == -np.sign(bb)
        ok &= abs(vals[0 if bb < 0 else -1] - abs(bb) / (1 + abs(bb))) < 1e-12
    rep.check("(ii) inf of 2 Hess V on the radial vector = -|b|/(1-|b|), attained at x = -sgn(b), all |b| < 1", ok and worst < 1e-10,
              f"max error {worst:.1e}")
    tb = sp.symbols("b", positive=True)
    Vr = -sp.log(1 + tb * sp.cos(th))
    lim_tan = sp.limit(sp.cot(th) * sp.diff(Vr, th), th, sp.pi)
    lim_rad = sp.limit(sp.diff(Vr, th, 2), th, sp.pi)
    lim0 = sp.simplify(sp.limit(sp.cot(th) * sp.diff(Vr, th), th, 0) - sp.limit(sp.diff(Vr, th, 2), th, 0))
    rep.check("(ii) isotropy at U = +-1: tangential and radial Hessian eigenvalues coincide",
              sp.simplify(lim_tan - lim_rad) == 0 and lim0 == 0 and sp.simplify(lim_rad + tb / (1 - tb)) == 0)

    ok, worst = True, 0
    for a in (0.3, 0.6, -0.8, 0.95):
        F = Flow(-np.log(1 + a * X))
        for t in (0.0, 0.05, 0.3, 1.0, 3.0):
            bb = abs(a) * math.exp(-1.5 * t)
            worst = max(worst, abs(F.kappa(t) - (1 - bb / (1 - bb))))
            ok &= F.kappa(t) <= F.kappa(t, full=False) + 1e-12
        ok &= ((1 - abs(a) / (1 - abs(a))) < 0) == (abs(a) > 0.5)
        if abs(a) > 0.5:
            ts = (2 / 3) * math.log(2 * abs(a))
            bb = abs(a) * math.exp(-1.5 * ts)
            ok &= abs(1 - bb / (1 - bb)) < 1e-12
    rep.check("(iii) kappa(t) = 1 - |b|/(1-|b|) is the optimal horizontal bound; sign at 0; zero at t* = (2/3)ln(2|a|)",
              ok and worst < 2e-3, f"max |numerical - formula| = {worst:.1e}; t*(0.6) = {(2 / 3) * math.log(1.2):.6f}")

    s_, a_, t_ = sp.symbols("s a t", positive=True)
    bs = a_ * sp.exp(-sp.Rational(3, 2) * s_)
    anti = sp.Rational(2, 3) * sp.log(1 - bs)
    e3 = sp.simplify(sp.diff(anti, s_) - bs / (1 - bs))
    rep.check("(iv) symbolic: d/ds (2/3) ln(1 - b(s)) = b/(1-b), so int_0^t b/(1-b) = (2/3) ln[(1-b(t))/(1-a)]", e3 == 0)

    ok, worst = True, mp.mpf(0)
    for a in (0.0, 0.3, 0.6, 0.9, 0.99):
        fac = lambda t, a=a: mp.e ** (-2 * t) * ((1 - a * mp.e ** (-1.5 * t)) / (1 - a)) ** (mp.mpf(4) / 3)
        direct = 2 * mp.quad(fac, [0, 0.5, 2, 10, mp.inf])
        euler = mp.quad(lambda u, a=a: u ** (mp.mpf(1) / 3) * (1 - a * u) ** (mp.mpf(4) / 3), [0, 1])
        hyp = mp.mpf(3) / 4 * mp.hyp2f1(-mp.mpf(4) / 3, mp.mpf(4) / 3, mp.mpf(7) / 3, a)
        worst = max(worst, abs(direct - C_closed(a)) / C_closed(a), abs(euler - hyp))
        # exp(-2 Lambda) against a direct integral of kappa
        for t in (0.2, 1.0):
            lam = mp.quad(lambda s, a=a: 1 - a * mp.e ** (-1.5 * s) / (1 - a * mp.e ** (-1.5 * s)), [0, t])
            worst = max(worst, abs(mp.e ** (-2 * lam) - fac(t)))
    rep.check("(iv) exp(-2 Lambda) closed form, Euler integral, and C(a) = (1-|a|)^(-4/3) 2F1(-4/3,4/3;7/3;|a|)", worst < mp.mpf(10) ** -12,
              f"max error {mp.nstr(worst, 3)}")
    ok, det = True, []
    for a in (0.3, 0.6, -0.9):
        Cn = Flow(-np.log(1 + a * X)).constant()[0]
        ok &= abs(Cn - float(C_closed(a))) / float(C_closed(a)) < 1e-5
        det.append(f"a={a}: {Cn:.6f} vs {float(C_closed(a)):.6f}")
    rep.check("(iv) closed form agrees with the generic numerical flow", ok, "; ".join(det))

    lim = mp.gamma(mp.mpf(7) / 3) ** 2 / mp.gamma(mp.mpf(11) / 3)
    ok = (abs(C_closed(0) - 1) < mp.mpf(10) ** -25 and abs(C_closed(0.6) - mp.mpf("1.964815")) < mp.mpf("5e-7")
          and abs(lim - mp.mpf("0.353328")) < mp.mpf("5e-7")
          and abs((1 - mp.mpf("0.999999")) ** (mp.mpf(4) / 3) * C_closed(mp.mpf("0.999999")) - lim) < mp.mpf("1e-5")
          and C_closed(-0.6) == C_closed(0.6))
    rep.check("(v) C(0) = 1, C(0.6) = 1.964815, limit constant Gamma(7/3)^2/Gamma(11/3) = 0.353328, C(-a) = C(a)", ok,
              f"C(0.6) = {mp.nstr(C_closed(0.6), 12)}, limit = {mp.nstr(lim, 10)}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
