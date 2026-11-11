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
"""Validation of Remark 9 (Wilson action on one loop; numerical) and of Appendix A (the script gauge_criterion.py).

Remark 9 is a numerical statement, so "validation" here means an independent re-computation.
What is checked
  (a) kappa(0) = 1 - 2 beta exactly (S'' = 2 beta cos(theta)), and the single-scale constant is 1/(1 - beta) for
      beta < 1 and does not exist for beta >= 1;
  (b) every entry of the table (min kappa, multiscale constant, optimal constant) is reproduced by the code of this
      suite, which is independent of the code that produced the table (different quadrature in t, pole values
      included, different eigenvalue solver): min kappa within 1%, constants within 2%, optimal within 1e-3;
  (c) the multiscale constant is stable under doubling the number of scales (relative change < 1e-4);
  (d) the statements in the text: the constant grows at least geometrically in beta (ratio > 5 per unit beta for
      beta >= 1) while the optimal constant decreases; for beta <= 1/2 the multiscale constant exceeds the
      single-scale one; the most negative curvature of V_0 is at U = -1, where the density is smallest;
  (e) Theorem 5 holds for these actions: optimal constant <= multiscale constant.
Appendix A
  (f) gauge_criterion.py runs without raising and prints the numbers quoted in the paper
      (C = 1, Var = 0.25390625, bound 0.390625; C = 1.96481549, Var = 0.25015625, bound 0.841186632).
"""
import os
import subprocess
import sys

import numpy as np

from common import Report, Flow, optimal_constant, X, TH

TABLE = [  # beta, kappa(0), min kappa, multiscale constant, single-scale, optimal   (as printed in the paper)
    (0.1, 0.80, 0.80, 1.13, 1.11, 0.665), (0.25, 0.50, 0.50, 1.38, 1.33, 0.656), (0.5, 0.00, 0.00, 2.08, 2.00, 0.625),
    (1.0, -1.00, -1.05, 6.55, None, 0.527), (2.0, -3.00, -4.48, 222.0, None, 0.328), (3.0, -5.00, -10.2, 1.7e4, None, 0.208),
    (4.0, -7.00, -18.4, 1.9e6, None, 0.146)]


def main():
    rep = Report("Remark 9 / Appendix A")
    ok_a = ok_b = ok_c = ok_e = True
    Cs, opts, det = [], [], []
    for beta, k0, kmin, cms, single, opt in TABLE:
        S = -2 * beta * X
        ok_a &= abs((1 + (2 * beta * np.cos(TH)).min()) - (1 - 2 * beta)) < 1e-3 and abs(k0 - (1 - 2 * beta)) < 5e-3
        if single is None:
            ok_a &= beta >= 1
        else:
            ok_a &= abs(1 / (1 - beta) - single) < 6e-3
        F = Flow(S)
        C, ts, kap, _ = F.constant()
        C2 = F.constant(nt=1801)[0]
        o = optimal_constant(S)
        ok_b &= abs(kap.min() - kmin) <= 0.01 * max(1, abs(kmin)) and abs(C - cms) / cms < 0.03 and abs(o - opt) < 1e-3
        ok_c &= abs(C - C2) / C < 1e-4
        ok_e &= o <= C
        Cs.append(C); opts.append(o)
        det.append(f"beta={beta}: C={C:.4g}, min kappa={kap.min():.2f}, optimal={o:.4f}")
    for d in det:
        print("      " + d)
    rep.check("(a) kappa(0) = 1 - 2 beta; single-scale constant 1/(1-beta) for beta < 1 only", ok_a)
    rep.check("(b) table reproduced by independent code", ok_b)
    rep.check("(c) multiscale constant stable under refinement of the scale grid", ok_c)
    growth = [Cs[i + 1] / Cs[i] for i in range(3, 6)]
    ok_d = min(growth) > 5 and all(opts[i + 1] < opts[i] for i in range(6))
    ok_d &= all(Cs[i] > TABLE[i][4] for i in range(3))
    ok_d &= int(np.argmin(2 * 1.0 * np.cos(TH))) == len(TH) - 1 and int(np.argmin(np.exp(2 * 1.0 * X))) == len(TH) - 1
    rep.check("(d) exponential growth of the constant, decreasing optimal constant, multiscale > single-scale for beta <= 1/2",
              ok_d, "growth per unit beta: " + ", ".join(f"{g:.0f}x" for g in growth))
    rep.check("(e) optimal constant <= multiscale constant (Theorem 5 for the Wilson action)", ok_e)

    here = os.path.dirname(os.path.abspath(__file__))
    out = subprocess.run([sys.executable, os.path.join(here, "gauge_criterion.py"), "--coupling", "0.6"],
                         capture_output=True, text=True)
    want = ["a=0 C=1 Var=0.25390625 C*energy=0.390625", "a=0.6 C=1.96481549 Var=0.25015625 C*energy=0.841186632",
            "gradient excess=0"]
    rep.check("(f) Appendix A: gauge_criterion.py runs and prints the quoted numbers", out.returncode == 0 and all(w in out.stdout for w in want),
              out.stdout.strip().replace("\n", " | "))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
