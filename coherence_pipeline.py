#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, math, os, json, hashlib, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

# --- CONSTANTS ---
ODLYZKO_URL = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1"
DEFAULT_FILENAME = "zeros1.txt"

# --- UTILITY FUNCTIONS ---

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def download_dataset_if_needed(url, dest_path):
    """Downloads the dataset automatically if it doesn't exist locally."""
    if os.path.exists(dest_path):
        print(f"Local file found: {dest_path}")
        return
    
    print(f"File '{dest_path}' not found locally.")
    print(f"Downloading automatically from: {url}")
    print("Please wait... (approx 1.8 MB)")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Download successful!")
    except Exception as e:
        print(f"ERROR: Failed to download dataset. {e}")
        print(f"Please manually download it from {url} and save it as '{dest_path}'")
        exit(1)

def load_zeros(path):
    print(f"Processing data file...")
    nums = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Replace common separators
                for ch in [",", ";", "\t"]:
                    line = line.replace(ch, " ")
                parts = line.strip().split()
                # Parse numbers
                for tok in parts:
                    try:
                        val = float(tok)
                        if math.isfinite(val) and val > 0:
                            nums.append(val)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

    arr = np.array(nums, dtype=float)
    if arr.size == 0:
        raise ValueError("No valid numeric data found in file.")
    
    # Sort and unique to be safe
    arr = np.unique(arr)
    arr.sort()
    print(f"Success: {len(arr)} zeros loaded.")
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

# --- MAIN ---

def main():
    ap = argparse.ArgumentParser(description="Spectral Coherence Analysis Pipeline")
    # Defaults set for "One-Click" usage
    ap.add_argument("--input", default=DEFAULT_FILENAME, help=f"Input file path (default: {DEFAULT_FILENAME})")
    ap.add_argument("--outdir", default="results", help="Output directory (default: results)")
    ap.add_argument("--unfolding", choices=["simple","refined"], default="simple", help="Unfolding method")
    ap.add_argument("--N", type=int, nargs="+", default=[5,10,20,50,100], help="Window sizes to analyze")
    ap.add_argument("--overlap", type=float, default=0.5, help="Window overlap (0 to 1)")
    ap.add_argument("--acf-lags", type=int, default=20, help="Max lag for ACF")
    ap.add_argument("--seed", type=int, default=1729, help="Random seed")
    args = ap.parse_args()

    print(f"--- Starting Analysis ---")
    
    # 1. Ensure Output Directory Exists
    os.makedirs(args.outdir, exist_ok=True)
    tab_dir = os.path.join(args.outdir, "tables")
    fig_dir = os.path.join(args.outdir, "figures")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 2. Auto-Download Data if missing
    download_dataset_if_needed(ODLYZKO_URL, args.input)

    # 3. Load Data
    zeros = load_zeros(args.input)
    sha = sha256_file(args.input)

    # Timestamp logic
    try:
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    except AttributeError:
        ts = datetime.datetime.utcnow().isoformat() + "Z"

    # Save Metadata
    meta = {
        "input_file": args.input,
        "sha256": sha,
        "n_zeros": int(len(zeros)),
        "height_range": [float(zeros[0]), float(zeros[-1])],
        "params": vars(args),
        "timestamp": ts,
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 4. Processing
    print("Computing unfolded gaps...")
    s = unfolded_gaps(zeros, mode=args.unfolding)

    print("Computing Coherence (C_N)...")
    CN_dict = {}
    rows_blocks, rows_summary = [], []
    for N in args.N:
        CN = compute_CN_series(s, N, overlap=args.overlap)
        CN = CN[np.isfinite(CN)]
        CN_dict[N] = CN
        m = len(CN)
        if m == 0: continue
        
        # Statistics by height block
        low, mid, high = thirds(m)
        for label, sl in zip(["low", "middle", "high"], [low, mid, high]):
            mu, lo, hi, n, sd = mean_ci_norm(CN[sl])
            rows_blocks.append({
                "height_block": label, "N": N, "n_win": n,
                "mean_C_N": mu, "CI95_low": lo, "CI95_high": hi, "std_dev": sd
            })
        
        # Global Statistics
        mu_all, lo_all, hi_all, n_all, sd_all = mean_ci_norm(CN)
        rows_summary.append({
            "N": N, "n_win": n_all,
            "mean_C_N": mu_all, "CI95_low": lo_all, "CI95_high": hi_all, "std_dev": sd_all
        })

    # Save CSVs
    pd.DataFrame(rows_blocks).to_csv(os.path.join(tab_dir, "cn_by_height.csv"), index=False)
    pd.DataFrame(rows_summary).to_csv(os.path.join(tab_dir, "cn_summary.csv"), index=False)

    # ACF Analysis
    print("Computing ACF...")
    acf_df = acf_series(s, max_lag=args.acf_lags)
    acf_df.to_csv(os.path.join(tab_dir, "acf.csv"), index=False)
    
    phi_hat = float(acf_df.loc[acf_df["lag"]==1, "rho"].values[0]) if not acf_df.empty else float("nan")
    lo_phi, hi_phi = fisher_ci(phi_hat, len(s))
    with open(os.path.join(tab_dir, "ar1_fit.json"), "w") as f:
        json.dump({"phi_hat": phi_hat, "ci95": [lo_phi, hi_phi]}, f, indent=2)

    # Variance Analysis
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

    # 5. Plotting (Matplotlib)
    print("Generating figures...")
    
    # Fig 1: Histogram C10
    C10 = CN_dict.get(10, np.array([]))
    plt.figure()
    bins = 60 if C10.size >= 2000 else 10
    plt.hist(C10, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("C10 Value")
    plt.ylabel("Frequency")
    plt.title("Figure 1: Histogram of C10 (Real Data)")
    mu, lo, hi, _, _ = mean_ci_norm(C10)
    if np.isfinite(lo):
        plt.axvline(lo, color='red', linestyle="--", label="95% CI")
        plt.axvline(hi, color='red', linestyle="--")
        plt.legend()
    plt.savefig(os.path.join(fig_dir, "fig1_hist_C10.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: Mean vs Theory
    plt.figure()
    df_sum = pd.DataFrame(rows_summary)
    if not df_sum.empty:
        yerr = [df_sum["mean_C_N"] - df_sum["CI95_low"], df_sum["CI95_high"] - df_sum["mean_C_N"]]
        plt.errorbar(df_sum["N"], df_sum["mean_C_N"], yerr=yerr, fmt="o", capsize=3, label="Observed")
    
    x_theory = np.linspace(min(args.N), max(args.N), 200)
    plt.plot(x_theory, (x_theory-1)/x_theory, label="Theory (N-1)/N", color='orange')
    plt.xlabel("Window Size N")
    plt.ylabel("Mean <C_N>")
    plt.title("Figure 2: Mean vs Theory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "fig2_mean_vs_theory.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 3: Variance log-log
    plt.figure()
    if not var_vs_N.empty:
        plt.loglog(var_vs_N["N"], var_vs_N["var_C_N"], "o-", label="Observed Var")
        if np.isfinite(k_est):
            plt.loglog(x_theory, k_est / (x_theory**2), "--", label=f"Ref ~ 1/N^2")
    plt.xlabel("N (log)")
    plt.ylabel("Variance (log)")
    plt.title("Figure 3: Variance Scaling")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "fig3_var_vs_N.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 4: ACF
    plt.figure()
    # FIXED: Removed 'use_line_collection' for compatibility with newer Matplotlib
    plt.stem(acf_df["lag"], acf_df["rho"])
    plt.axhline(0, linewidth=1, color='black')
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.title(f"Figure 4: Autocorrelation (phi={phi_hat:.3f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, "fig4_acf.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"--- DONE! Results saved in: {args.outdir} ---")

if __name__ == "__main__":
    main()