#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, math, os, json, hashlib, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def load_zeros(path):
    nums = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for ch in [",", ";", "\t"]:
                line = line.replace(ch, " ")
            for tok in line.strip().split():
                try:
                    val = float(tok)
                    if math.isfinite(val) and val > 0:
                        nums.append(val)
                except Exception:
                    pass
    arr = np.array(nums, dtype=float)
    if arr.size == 0:
        raise ValueError("No numeric values parsed from file.")
    arr = np.unique(arr)
    arr.sort()
    return arr

def rho_simple(t):
    two_pi = 2.0 * math.pi
    t = np.asarray(t, dtype=float)
    return (1.0/(two_pi)) * np.log(np.maximum(t / two_pi, 1.0000001))

def rho_refined(t):
    two_pi = 2.0 * math.pi
    t = np.asarray(t, dtype=float)
    return (1.0/(two_pi)) * np.log(np.maximum(t / two_pi, 1.0000001)) + (1.0/(two_pi * np.maximum(t, 1.0)))

def unfolded_gaps(zeros, mode="simple"):
    gaps = np.diff(zeros)
    t_mid = zeros[:-1]
    rho = rho_simple(t_mid) if mode=="simple" else rho_refined(t_mid)
    s = gaps * rho
    s = s[np.isfinite(s) & (s > 0)]
    if s.size < 3:
        raise ValueError("Not enough valid unfolded gaps.")
    return s

def compute_CN_series(series, N, overlap=0.5):
    m = len(series)
    if m < N:
        return np.array([])
    step = max(1, int(N*(1.0-overlap)))
    out = []
    for start in range(0, m - N + 1, step):
        seg = series[start:start+N]
        den = float(np.sum(seg))
        if den > 0:
            out.append(float(np.sum(seg[:-1]) / den))
    return np.array(out, dtype=float)

def acf_lag(x, k):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= k:
        return float("nan")
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x[:-k], x[k:]) / denom)

def acf_series(x, max_lag=20):
    return pd.DataFrame({"lag": np.arange(1, max_lag+1),
                         "rho": [acf_lag(x, k) for k in range(1, max_lag+1)]})

def mean_ci_norm(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n, float("nan")
    mu = float(np.mean(v))
    sd = float(np.std(v, ddof=1))
    half = 1.96 * sd / math.sqrt(n)
    return mu, mu - half, mu + half, n, sd

def fisher_ci(r, n, alpha=0.05):
    if not (np.isfinite(r) and n > 3):
        return float("nan"), float("nan")
    z = 0.5 * math.log((1+r)/(1-r))
    se = 1.0 / math.sqrt(n - 3)
    z_lo, z_hi = z - 1.96*se, z + 1.96*se
    def invz(zv):
        e2z = math.exp(2*zv)
        return (e2z - 1) / (e2z + 1)
    return invz(z_lo), invz(z_hi)

def thirds(n):
    b = n // 3
    return (slice(0, b), slice(b, 2*b), slice(2*b, n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--unfolding", choices=["simple","refined"], default="simple")
    ap.add_argument("--N", type=int, nargs="+", default=[5,10,20,50,100])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--acf-lags", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    tab_dir = os.path.join(args.outdir, "tables")
    fig_dir = os.path.join(args.outdir, "figures")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    zeros = load_zeros(args.input)
    sha = sha256_file(args.input)

    meta = {
        "input": args.input,
        "sha256": sha,
        "n_zeros": int(len(zeros)),
        "height_range": [float(zeros[0]), float(zeros[-1])],
        "unfolding": args.unfolding,
        "N_list": [int(n) for n in args.N],
        "overlap": float(args.overlap),
        "acf_lags": int(args.acf_lags),
        "seed": int(args.seed),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)

    s = unfolded_gaps(zeros, mode=args.unfolding)

    CN_dict = {}
    rows_blocks, rows_summary = [], []
    for N in args.N:
        CN = compute_CN_series(s, N, overlap=args.overlap)
        CN = CN[np.isfinite(CN)]
        CN_dict[N] = CN
        m = len(CN)
        if m == 0:
            continue
        low, mid, high = thirds(m)
        for label, sl in zip(["bas","milieu","haut"], [low, mid, high]):
            mu, lo, hi, n, sd = mean_ci_norm(CN[sl])
            rows_blocks.append({"bloc_hauteur": label, "N": N, "n_win": n,
                                "mean_C_N": mu, "IC95_bas": lo, "IC95_haut": hi, "sd": sd})
        mu_all, lo_all, hi_all, n_all, sd_all = mean_ci_norm(CN)
        rows_summary.append({"N": N, "n_win": n_all,
                             "mean_C_N": mu_all, "IC95_bas": lo_all, "IC95_haut": hi_all, "sd": sd_all})

    cn_by_height = pd.DataFrame(rows_blocks)
    cn_summary = pd.DataFrame(rows_summary)
    cn_by_height.to_csv(os.path.join(tab_dir, "cn_by_height.csv"), index=False)
    cn_summary.to_csv(os.path.join(tab_dir, "cn_summary.csv"), index=False)

    acf_df = acf_series(s, max_lag=args.acf_lags)
    acf_df.to_csv(os.path.join(tab_dir, "acf.csv"), index=False)
    phi_hat = float(acf_df.loc[acf_df["lag"]==1, "rho"].values[0]) if not acf_df.empty else float("nan")
    lo_phi, hi_phi = fisher_ci(phi_hat, len(s))
    with open(os.path.join(tab_dir, "ar1_fit.json"), "w") as f:
        json.dump({"phi_hat": phi_hat, "ci95": [lo_phi, hi_phi]}, f, indent=2)

    var_rows, k_vals = [], []
    for N in args.N:
        CN = CN_dict.get(N, np.array([]))
        if CN.size > 1:
            vv = float(np.var(CN, ddof=1))
            var_rows.append({"N": int(N), "var_C_N": vv})
            k_vals.append(vv * (N**2))
    var_vs_N = pd.DataFrame(var_rows)
    var_vs_N.to_csv(os.path.join(tab_dir, "var_vs_N.csv"), index=False)
    k_est = float(np.mean(k_vals)) if len(k_vals) > 0 else float("nan")

    # Figures
    # Fig 1 — Histogram C10
    C10 = CN_dict.get(10, np.array([]))
    plt.figure()
    bins = 60 if C10.size >= 2000 else max(10, int(np.sqrt(max(1, C10.size)))) if C10.size>0 else 10
    plt.hist(C10, bins=bins)
    plt.xlabel("C10")
    plt.ylabel("Fréquence")
    plt.title("Figure 1 : Histogramme de C10 (données réelles)")
    mu, lo, hi, nC10, _ = mean_ci_norm(C10)
    if np.isfinite(lo) and np.isfinite(hi):
        plt.axvline(lo, linestyle="--")
        plt.axvline(hi, linestyle="--")
    plt.savefig(os.path.join(fig_dir, "fig1_hist_C10.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2 — mean vs theory
    plt.figure()
    if not cn_summary.empty:
        yerr_low = cn_summary["mean_C_N"] - cn_summary["IC95_bas"]
        yerr_high = cn_summary["IC95_haut"] - cn_summary["mean_C_N"]
        plt.errorbar(cn_summary["N"], cn_summary["mean_C_N"],
                     yerr=[yerr_low, yerr_high], fmt="o", capsize=3, label="Moyenne observée ± IC95")
    N_dense = np.linspace(min(args.N), max(args.N), 200)
    plt.plot(N_dense, (N_dense-1)/N_dense, label="Théorie (N-1)/N")
    plt.xlabel("Taille de fenêtre N")
    plt.ylabel("Moyenne <C_N>")
    plt.title("Figure 2 : Moyenne <C_N> vs théorie (données réelles)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "fig2_mean_vs_theory.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3 — Var vs N log–log
    plt.figure()
    if not var_vs_N.empty:
        plt.loglog(var_vs_N["N"], var_vs_N["var_C_N"], "o-", label="Var(C_N) observée")
        if np.isfinite(k_est):
            N_dense = np.linspace(min(args.N), max(args.N), 200)
            plt.loglog(N_dense, k_est / (N_dense**2), "--", label=f"Réf. y={k_est:.3f}/N^2")
    plt.xlabel("N (log)")
    plt.ylabel("Var(C_N) (log)")
    plt.title("Figure 3 : Variance Var(C_N) vs N (réel, log–log)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "fig3_var_vs_N.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4 — ACF
    plt.figure()
    markerline, stemlines, baseline = plt.stem(acf_df["lag"], acf_df["rho"], use_line_collection=True)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Décalage (lag)")
    plt.ylabel("ACF")
    plt.title(f"Figure 4 : ACF des gaps unfolded (phi_hat = {phi_hat:.3f}, IC95 [{lo_phi:.3f}, {hi_phi:.3f}])")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "fig4_acf.png"), dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "n_zeros": int(len(zeros)),
        "phi_hat": phi_hat,
        "phi_ci95": [lo_phi, hi_phi],
        "k_est": k_est,
        "tables": {
            "sources_hashes.csv": os.path.join(tab_dir, "sources_hashes.csv"),
            "cn_by_height.csv": os.path.join(tab_dir, "cn_by_height.csv"),
            "cn_summary.csv": os.path.join(tab_dir, "cn_summary.csv"),
            "acf.csv": os.path.join(tab_dir, "acf.csv"),
            "var_vs_N.csv": os.path.join(tab_dir, "var_vs_N.csv"),
            "ar1_fit.json": os.path.join(tab_dir, "ar1_fit.json"),
        },
        "figures": {
            "fig1_hist_C10.png": os.path.join(fig_dir, "fig1_hist_C10.png"),
            "fig2_mean_vs_theory.png": os.path.join(fig_dir, "fig2_mean_vs_theory.png"),
            "fig3_var_vs_N.png": os.path.join(fig_dir, "fig3_var_vs_N.png"),
            "fig4_acf.png": os.path.join(fig_dir, "fig4_acf.png"),
        }
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
