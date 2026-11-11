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
"""Finite gauge lattice exercise: one vertex, one SU(2) loop, conjugation gauge action.

x = Tr(U)/2 = cos(theta). The radial metric is 2 dtheta^2 and Haar
measure is (2/pi) sin(theta)^2 dtheta. Use a positive density w=1+a*x
(action S=-log(w)), and polynomial class observables. Requires NumPy.
"""
import argparse

import numpy as np
from numpy.polynomial import Polynomial as P
from numpy.polynomial.legendre import leggauss

X = P([0, 1])
G = (1 - X**2) / 2  # |grad x|^2


def heat(poly, t):
    """Apply exp(t Delta) in the SU(2) character basis, without spatial gridding."""
    chars = [P([1]), 2 * X]
    while len(chars) <= poly.degree():
        chars.append(2 * X * chars[-1] - chars[-2])
    remainder = poly
    result = P([0])
    for n in range(poly.degree(), -1, -1):
        char = chars[n]
        coeff = remainder.coef[n] / char.coef[n] if n <= remainder.degree() else 0
        result += coeff * np.exp(-t * n * (n + 2) / 2) * char
        remainder = remainder - coeff * char
    return result


def run(a, f):
    if not abs(a) < 1:
        raise ValueError("The density requires |a| < 1.")
    theta = np.arange(1, 513) * np.pi / 513
    x = np.cos(theta)
    weights = 2 * np.sin(theta)**2 / 513
    w = 1 + a * X
    grad2 = G * f.deriv()**2
    mean = weights @ (w(x) * f(x))
    variance = weights @ (w(x) * (f(x) - mean)**2)
    energy = weights @ (w(x) * grad2(x))

    # For b=a exp(-3t/2), twice the horizontal Hessian of V=-log(1+b*x)
    # is b*x/(1+b*x) + b^2*(1-x^2)/(1+b*x)^2. Its minimum occurs
    # at the adverse endpoint. At either central element isotropy makes
    # the Hessian scalar. Adding Ric=1 gives kappa=1-|b|/(1-|b|).
    def factor(s):
        # exp(-2 integral_0^t kappa), with s=exp(-t)
        return s**2 * ((1 - abs(a)*s**1.5)/(1 - abs(a)))**(4/3)

    nodes, quadrature = leggauss(96)
    s = (nodes + 1)/2
    constant = quadrature @ (factor(s)/s)  # 2 dt = -2 ds/s

    def moment(t):
        d = heat(w, t)(x)
        n = heat(w*f, t)(x)
        return weights @ (n*n/d)

    worst_excess = 0.
    worst_residual = 0.
    for t in np.geomspace(0.001, 8, 45):
        d, n = heat(w, t), heat(w*f, t)
        u_prime = (n.deriv()(x)*d(x) - n(x)*d.deriv()(x))/d(x)**2
        lhs = G(x)*u_prime**2
        rhs = factor(np.exp(-t))*heat(w*grad2, t)(x)/d(x)
        worst_excess = max(worst_excess, np.max(lhs-rhs))
        h = 1e-5
        derivative = (moment(t+h)-moment(t-h))/(2*h)
        worst_residual = max(worst_residual, abs(derivative + 2*(weights @ (d(x)*lhs))))
    print(f"a={a:g} C={constant:.9g} Var={variance:.9g} C*energy={constant*energy:.9g}")
    print(f" gradient excess={worst_excess:.3g} dissipation residual={worst_residual:.3g}")
    if max(variance-constant*energy, worst_excess, worst_residual) > 1e-6:
        raise ArithmeticError("Numerical criterion check exceeded tolerance.")
    if a == 0 and abs(constant-1) > 1e-12:
        raise ArithmeticError("Free normalization failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=0.6)
    args = parser.parse_args()
    observable = X + 0.25 * X**2
    run(0, observable)
    if args.coupling:
        run(args.coupling, observable)
