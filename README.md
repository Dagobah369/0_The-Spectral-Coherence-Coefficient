[![DOI](https://zenodo.org/badge/1094632894.svg)](https://doi.org/10.5281/zenodo.17752494)

November 11, 2025

Abstract

We establish an exact identity for the spectral coherence measure computed on
stationary unfolded gap sequences: for any N ≥ 2, E[CN] = (N − 1)/N. Under
short-range mixing assumptions, we further show that Var(CN) ∼ c/N2, where the
constant c depends solely on the symmetry class (GOE/GUE/GSE). Applied to the
non-trivial zeros of ζ(s), the framework predicts and explains the value C10 = 0.9:
N = 10 is the unique window length aligning the natural window shift contraction
(N − 1)/N with the mean E[CN]. An AR(1) modeling with negative rank-1 correlation
(ϕ ≈ −0.36) accounts for the observed dispersion and the N − 2 regime.
Numerically, we validate these results on the first 105 zeros and describe a reproducible
streaming pipeline extensible to 1010 zeros (simple/refined unfolding, block
bootstrap, ACF). The formalism extends to L-function families: the mean remains
universal, while the variance encodes the class. We position 0.9 as a universal spectral
invariant—a key to analytic stability—without claiming to prove the Riemann
Hypothesis, but providing proven statements on means and falsifiable predictions
on dispersions.


# The-Spectral-Coherence-Coefficient
The Spectral Coherence 9/10

Coherence Pipeline — Guide 
=====================================
Script: coherence_pipeline.py

Exemple:
python coherence_pipeline.py --input /mnt/data/zeros1.txt --outdir /mnt/data/run_real --unfolding refined --N 5 10 20 50 100 --overlap 0.5 --acf-lags 20 --seed 1729

Output: tables CSV (cn_by_height, cn_summary, acf, var_vs_N, ar1_fit.json), figures PNG (fig1..fig4), manifest.json, summary.json.
