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
"""Run every validation script for the white paper
"A Conditional Multiscale Horizontal Curvature Criterion on a Finite Gauge Lattice".

Usage:  python run_all.py      (exit code 0 if every check passes)
"""
import sys
import time

import validate_definitions
import validate_lemma1
import validate_lemma2
import validate_lemma3
import validate_lemma4
import validate_theorem5
import validate_proposition7
import validate_proposition8
import validate_remark9_appendix

SUITES = [
    ("Definitions 1-8", validate_definitions),
    ("Lemma 1 - geometry and gauge invariance", validate_lemma1),
    ("Lemma 2 - flow identities", validate_lemma2),
    ("Lemma 3 - gradient estimate", validate_lemma3),
    ("Lemma 4 - dissipation identity", validate_lemma4),
    ("Theorem 5 and Corollary 6 - the criterion and the free action", validate_theorem5),
    ("Proposition 7 - one SU(2) loop, exact", validate_proposition7),
    ("Proposition 8 - comparison of bounds", validate_proposition8),
    ("Remark 9 and Appendix A - Wilson action; the script", validate_remark9_appendix),
]


def main():
    results = []
    for title, mod in SUITES:
        print(f"\n=== {title} ===")
        t = time.time()
        ok = mod.main()
        results.append((title, ok, time.time() - t))
    print("\n=== Summary ===")
    for title, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {title}  ({dt:.1f} s)")
    print("\nNot validated: hypothesis (H) for any interacting lattice action on more than one loop; any bound uniform in\n"
          "the lattice (Remark 10); the novelty of Theorem 5. Theorem 5 and Lemmas 2-4 are stated for SU(N)^E on any\n"
          "finite graph; their numerical checks are on one SU(2) loop, where gauge-invariant functions are class\n"
          "functions, plus symbolic identities and matrix-level checks for N up to 6. Passing these checks supports the\n"
          "proofs; it does not replace them.")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
