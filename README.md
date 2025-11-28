[![DOI](https://zenodo.org/badge/1094632894.svg)](https://doi.org/10.5281/zenodo.17752494)

# The-Spectral-Coherence-Coefficient
The Spectral Coherence 9/10

Coherence Pipeline — Guide 
=====================================
Script: coherence_pipeline.py

Exemple:
python coherence_pipeline.py --input /mnt/data/zeros1.txt --outdir /mnt/data/run_real --unfolding refined --N 5 10 20 50 100 --overlap 0.5 --acf-lags 20 --seed 1729

Output: tables CSV (cn_by_height, cn_summary, acf, var_vs_N, ar1_fit.json), figures PNG (fig1..fig4), manifest.json, summary.json.
