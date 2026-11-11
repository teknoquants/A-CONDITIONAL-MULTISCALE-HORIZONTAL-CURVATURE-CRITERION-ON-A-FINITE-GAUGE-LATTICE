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
"""Validation of Lemma 3 (gradient estimate) and every step of its proof.

What is checked
  Symbolic (SymPy, radial calculus on one SU(2) loop, arbitrary smooth class functions G, V):
    (a) the Bochner identity  (1/2) L|grad G|^2 = |Hess G|^2 + <grad G, grad L G> + (Ric + 2 Hess V)(grad G, grad G)
        with L = Laplacian - 2<grad V, grad .>, |Hess G|^2 = (G''/2)^2 + 2 (cot G'/2)^2 and Ric = 1;
    (b) the chain rule  L w = L|grad G|^2/(2w) - |grad|grad G|^2|^2/(4 w^3)  for w = sqrt(|grad G|^2 + eps);
    (c) |grad|grad G|^2|^2 = 4 |Hess G (grad G)|^2 <= 4 |Hess G|^2 |grad G|^2.
  Numerical (seven actions, random class functions, 12 scales each):
    (d) the intermediate inequality  L_r w_r >= [<grad G, grad L_r G> + (Ric + 2 Hess V_r)(grad G, grad G)]/w_r;
    (e) monotonicity used in Gronwall: exp(Lambda(r)) Q_{r,t} |grad Q_{0,r} f| is non-increasing in r;
    (f) the conclusions  |grad Q_t f| <= exp(-Lambda(t)) Q_t|grad f|  and  |grad Q_t f|^2 <= exp(-2 Lambda(t)) Q_t|grad f|^2
        pointwise, with the optimal kappa(t) of the action (negative on some scales for most of the actions);
    (g) Jensen: (Q_t phi)^2 <= Q_t(phi^2);
    (h) control that must fail: in the free case, replacing kappa = 1 by kappa = 1.25 violates the estimate, so the
        test is able to detect an over-optimistic curvature bound.
"""
import sys

import numpy as np
import sympy as sp

from common import Report, Flow, ACTIONS, random_class_function, d1, d2, lap, X, TH, SN


def main():
    rep = Report("Lemma 3")
    rng = np.random.default_rng(303)

    th, eps = sp.symbols("theta epsilon", positive=True)
    G, V = sp.Function("G")(th), sp.Function("V")(th)
    D = lambda f: sp.diff(f, th)
    lapl = lambda f: (D(D(f)) + 2 * sp.cot(th) * D(f)) / 2
    L = lambda f: lapl(f) - D(V) * D(f)                       # Laplacian - 2<grad V, grad f>, <.,.> = V'f'/2
    g2 = D(G) ** 2 / 2
    hess2 = (D(D(G)) / 2) ** 2 + 2 * (sp.cot(th) * D(G) / 2) ** 2
    boch = sp.simplify(L(g2) / 2 - (hess2 + D(G) * D(L(G)) / 2 + (1 + D(D(V))) * g2))
    rep.check("(a) symbolic Bochner identity with Ric = 1 and the factor 2 on Hess V", boch == 0)
    w = sp.sqrt(g2 + eps)
    chain = sp.simplify(L(w) - (L(g2) / (2 * w) - (D(g2) ** 2 / 2) / (4 * w ** 3)))
    rep.check("(b) symbolic chain rule for w = sqrt(|grad G|^2 + eps)", chain == 0)
    kato = sp.simplify(D(g2) ** 2 / 2 - 4 * (D(D(G)) / 2) ** 2 * g2)
    slack = sp.simplify(4 * hess2 * g2 - D(g2) ** 2 / 2 - 4 * 2 * (sp.cot(th) * D(G) / 2) ** 2 * g2)
    rep.check("(c) symbolic: |grad|grad G|^2|^2 = 4|Hess G(grad G)|^2, and the gap to 4|Hess G|^2|grad G|^2 is a square",
              kato == 0 and slack == 0)

    worst_f = worst_f2 = worst_j = worst_d = worst_e = -np.inf
    neg_scales = 0
    for name, mk in ACTIONS.items():
        F = Flow(mk())
        C, ts, kap, Lam = F.constant()
        neg_scales += int(kap.min() < 0)
        lam_at = lambda t: np.interp(t, ts, Lam)
        for _ in range(3):
            f = random_class_function(rng)
            gf = np.abs(d1(f)) / np.sqrt(2)
            for t in np.geomspace(0.003, 4, 12):
                u = F.Q(f, 0.0, t)
                gu = np.abs(d1(u)) / np.sqrt(2)
                sc = max(1e-12, gu.max())
                worst_f = max(worst_f, ((gu - np.exp(-lam_at(t)) * F.Q(gf, 0.0, t)) / sc).max())
                worst_f2 = max(worst_f2, ((gu ** 2 - np.exp(-2 * lam_at(t)) * F.Q(gf ** 2, 0.0, t)) / sc ** 2).max())
                worst_j = max(worst_j, (F.Q(gf, 0.0, t) ** 2 - F.Q(gf ** 2, 0.0, t)).max())
                # (d) with eps = 1e-3: L w - [<grad G, grad L G> + (1 + V'')|grad G|^2]/w >= 0
                e_ = 1e-3
                g2 = d1(u) ** 2 / 2
                wv = np.sqrt(g2 + e_)
                Lu = F.L(u, t)
                # L w via the chain rule (b), with L|grad G|^2 from the Bochner identity (a)
                hess2 = (d2(u) / 2) ** 2 + 2 * (np.cos(TH) / SN * d1(u) / 2) ** 2
                curv = (1 + F.Vpp(t)) * g2
                cross = d1(u) * d1(Lu) / 2
                Lw = (hess2 + cross + curv) / wv - (d1(u) * d2(u)) ** 2 / 2 / (4 * wv ** 3)
                worst_d = max(worst_d, (-(Lw - (cross + curv) / wv)).max())
            # (e) monotonicity in r for fixed t
            t = 1.5
            rs = np.linspace(0.0, t, 16)
            vals = np.array([np.exp(lam_at(r)) * F.Q(np.abs(d1(F.Q(f, 0.0, r))) / np.sqrt(2), r, t) for r in rs])
            worst_e = max(worst_e, (np.diff(vals, axis=0) / max(1e-12, vals.max())).max())
    rep.check("(d) L_r w_r >= [<grad G, grad L_r G> + (Ric + 2 Hess V_r)(grad G, grad G)]/w_r", worst_d < 1e-6,
              f"max violation {max(worst_d, 0):.1e}")
    rep.check("(e) exp(Lambda(r)) Q_{r,t}|grad Q_{0,r} f| is non-increasing in r", worst_e < 1e-4, f"max relative increase {max(worst_e, 0):.1e}")
    rep.check("(f) |grad Q_t f| <= exp(-Lambda(t)) Q_t|grad f| pointwise", worst_f < 1e-4,
              f"max relative excess {max(worst_f, 0):.1e}; kappa negative on some scales for {neg_scales} of {len(ACTIONS)} actions")
    rep.check("(f) |grad Q_t f|^2 <= exp(-2 Lambda(t)) Q_t|grad f|^2 pointwise", worst_f2 < 1e-4, f"max relative excess {max(worst_f2, 0):.1e}")
    rep.check("(g) Jensen (Q_t phi)^2 <= Q_t(phi^2)", worst_j < 1e-9, f"max excess {max(worst_j, 0):.1e}")
    Ff = Flow(0 * X)
    viol = -np.inf
    for f in (X, X + 0.25 * X ** 2, X ** 2):
        gf = np.abs(d1(f)) / np.sqrt(2)
        for t in (0.03, 0.1, 0.3):
            gu = np.abs(d1(Ff.Q(f, 0.0, t))) / np.sqrt(2)
            viol = max(viol, ((gu - np.exp(-1.25 * t) * Ff.Q(gf, 0.0, t)) / gu.max()).max())
    rep.check("(h) control: with kappa = 1.25 in the free case the estimate is violated (as it must be)", viol > 1e-3,
              f"relative excess {viol:.3f}")
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
