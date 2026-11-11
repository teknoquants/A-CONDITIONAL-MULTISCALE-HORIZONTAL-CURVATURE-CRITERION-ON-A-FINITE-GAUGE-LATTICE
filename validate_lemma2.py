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
"""Validation of Lemma 2 (flow identities) and every step of its proof.

What is checked
  Symbolic (SymPy, radial calculus on one SU(2) loop, arbitrary smooth functions):
    (a) Laplacian e^{-V} = e^{-V}(|grad V|^2 - Laplacian V), which gives the Polchinski equation;
    (b) z^-1 Laplacian(z u) - (Laplacian z / z) u = Laplacian u + 2 <grad log z, grad u> = L u.
  Numerical (character basis, seven actions, random class functions):
    (i)   dV_t/dt = Laplacian V_t - |grad V_t|^2, and z_t > 0;
    (ii)  Q_{s,t} 1 = 1, positivity, Q_{t,u} Q_{s,t} = Q_{s,u}, and int Q_{s,t} a dmu_t = int a dmu_s;
    (iii) d/dt Q_{s,t} f = L_t Q_{s,t} f and d/ds Q_{s,t} f = -Q_{s,t} L_s f.
"""
import sys

import numpy as np
import sympy as sp

from common import Report, Flow, ACTIONS, random_class_function, lap, d1, X


def radial():
    th = sp.symbols("theta")
    lapl = lambda f: (sp.diff(f, th, 2) + 2 * sp.cot(th) * sp.diff(f, th)) / 2
    dot = lambda f, g: sp.diff(f, th) * sp.diff(g, th) / 2
    return th, lapl, dot


def main():
    rep = Report("Lemma 2")
    rng = np.random.default_rng(202)

    th, lapl, dot = radial()
    V, z, u = sp.Function("V")(th), sp.Function("z")(th), sp.Function("u")(th)
    a = sp.simplify(lapl(sp.exp(-V)) - sp.exp(-V) * (dot(V, V) - lapl(V)))
    b = sp.simplify(lapl(z * u) / z - lapl(z) / z * u - (lapl(u) + 2 * dot(sp.log(z), u)))
    rep.check("(a) symbolic: Laplacian e^-V = e^-V (|grad V|^2 - Laplacian V)", a == 0)
    rep.check("(b) symbolic: z^-1 Laplacian(zu) - (Laplacian z/z) u = Laplacian u + 2<grad log z, grad u>", b == 0)

    h = 1e-4
    w1 = w2 = w3 = w4 = w5 = 0.0
    pos = True
    for name, mk in ACTIONS.items():
        F = Flow(mk())
        for t in (0.02, 0.1, 0.5, 2.0):
            Vt = F.V(t)
            lhs = (F.V(t + h) - F.V(t - h)) / (2 * h)
            rhs = lap(Vt) - d1(Vt) ** 2 / 2
            w1 = max(w1, np.abs(lhs - rhs).max() / max(1, np.abs(rhs).max()))
            pos &= F.z(t).min() > 0
        f, a_ = random_class_function(rng), random_class_function(rng)
        for s, t, uu in ((0.0, 0.1, 0.4), (0.05, 0.3, 1.0), (0.2, 0.25, 2.0)):
            w2 = max(w2, np.abs(F.Q(1 + 0 * X, s, t) - 1).max(), np.abs(F.Q(F.Q(f, s, t), t, uu) - F.Q(f, s, uu)).max(),
                     abs(F.mu(F.Q(a_, s, t), t) - F.mu(a_, s)))
            pos &= F.Q(np.exp(f), s, t).min() > 0
            dt = (F.Q(f, s, t + h) - F.Q(f, s, t - h)) / (2 * h)
            w3 = max(w3, np.abs(dt - F.L(F.Q(f, s, t), t)).max() / max(1, np.abs(dt).max()))
            if s > 0:
                ds = (F.Q(f, s + h, t) - F.Q(f, s - h, t)) / (2 * h)
                w4 = max(w4, np.abs(ds + F.Q(F.L(f, s), s, t)).max() / max(1, np.abs(ds).max()))
    rep.check("(i) dV_t/dt = Laplacian V_t - |grad V_t|^2 and z_t > 0", w1 < 1e-5 and pos, f"max rel residual {w1:.1e}")
    rep.check("(ii) Q_{s,t}1 = 1, positivity, composition, int Q_{s,t} a dmu_t = int a dmu_s", w2 < 1e-9 and pos, f"max error {w2:.1e}")
    rep.check("(iii) d/dt Q_{s,t} f = L_t Q_{s,t} f", w3 < 1e-5, f"max rel residual {w3:.1e}")
    rep.check("(iii) d/ds Q_{s,t} f = -Q_{s,t} L_s f", w4 < 1e-5, f"max rel residual {w4:.1e}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
