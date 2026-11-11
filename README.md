# Validation scripts: A Conditional Multiscale Horizontal Curvature Criterion on a Finite Gauge Lattice

Companion code to the white paper by R. Cooperman and A.E. Petrunina V.V. (XISOSIS, Vienna, 11 November 2026).
Released under the MIT License (see `LICENSE`; the licence is also reproduced at the top of every script).

GitHub: [GITHUB REPOSITORY URL — placeholder] · DOI: [DOI — placeholder]

## Run

```
pip install numpy scipy sympy mpmath
python run_all.py
```

Python 3.10+, NumPy, SciPy ≥ 1.12 (for `cumulative_simpson`), SymPy, mpmath. Runtime is about 40 seconds.
`run_log.txt` is the output of a full run: 57 checks, all passing.

## Coverage

| Script | Paper item | Checks |
|---|---|---|
| `validate_definitions.py` | Definitions 1–8 | Orthonormal su(N) basis; gauge action and Wilson loops; heat semigroup properties; **edge-wise heat-kernel convolution by Monte Carlo on a four-edge plaquette**; μ_t is a probability measure; Q_t = Q_{0,t}; C_κ for constant κ; one-loop geometry (\|∇x\|², radial metric, character eigenvalues, symbolic) |
| `validate_lemma1.py` | Lemma 1, all proof steps | Ric = −B/4 = N/2 for N = 2–6 (two independent computations); bi-invariance; central heat kernel; **gradient of a gauge-invariant function is orthogonal to the orbit**, by finite differences on a three-edge graph for SU(2) and SU(3), including points with nontrivial stabilizer |
| `validate_lemma2.py` | Lemma 2, all proof steps | Symbolic: the two identities behind the Polchinski equation and the generator. Numerical: ∂_t V_t, the Markov and composition properties of Q_{s,t}, both derivative formulas |
| `validate_lemma3.py` | Lemma 3, all proof steps | **Symbolic Bochner identity** (with Ric = 1 and the factor 2), chain rule and Kato step; the differential inequality; Gronwall monotonicity; both pointwise gradient estimates for seven actions; Jensen; **a control that must fail** (κ = 1.25 in the free case) |
| `validate_lemma4.py` | Lemma 4, all proof steps | The three integrations by parts; the dissipation identity (4th-order differences); its integrated form |
| `validate_theorem5.py` | Theorem 5, Corollary 6 | Convergence at large scale; the exact variance decomposition; the inequality for 200 random observables per action **and for the whole 120-dimensional class-function space via the optimal constant**; negative κ on some scales with finite C; horizontal ≤ full; free case 2/N and the sharp constant |
| `validate_proposition7.py` | Proposition 7, all proof steps | Closed flow; symbolic Hessian and its derivative; infimum and its location for 400 values of b; isotropy at ±𝟙 (symbolic limits); optimal κ(t), its sign and zero; symbolic antiderivative; e^(−2Λ), Euler's integral and **the hypergeometric closed form to 12 digits**; quoted values and the limit constant |
| `validate_proposition8.py` | Proposition 8 and its table | Single-scale infimum and the threshold 2/3; Holley–Stroock; growth exponent; **all 36 printed table entries**; the three observations in the text |
| `validate_remark9_appendix.py` | Remark 9, Appendix A | κ(0) = 1 − 2β; the Wilson table reproduced by independent code; stability under refinement; growth factors; Theorem 5 for the Wilson action; `gauge_criterion.py` runs and prints the quoted numbers |

`gauge_criterion.py` is the script of Appendix A. `common.py` holds the character-basis tools and the flow.

## What is not validated

Hypothesis (H) for an interacting lattice action on more than one loop; any bound uniform in the lattice (Remark 10);
the novelty of Theorem 5. Theorem 5 and Lemmas 2–4 are stated for SU(N)^E on an arbitrary finite graph. Their numerical
checks are carried out on one SU(2) loop, where gauge-invariant functions are class functions and everything can be
computed to high accuracy, together with symbolic identities and matrix-level checks for N up to 6. On one loop the
horizontal and full-Hessian bounds coincide for the tested actions, so these checks do not exhibit a case where the
restriction to horizontal directions changes the constant. Passing the checks supports the proofs; it does not
replace them, and it says nothing about a continuum limit or a mass gap.
