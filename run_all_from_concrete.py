import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional but recommended for the OLS p-values
try:
    import statsmodels.api as sm
except Exception as e:
    sm = None
    print("WARNING: statsmodels not available. p-values will be skipped.\n", e)

# ---------------------------
# 1) Load & rename columns
# ---------------------------
def load_concrete(csv_path="Concrete_Data.csv"):
    # Try robust read (some CSVs include BOM or quotes in header)
    df = pd.read_csv(csv_path)
    # Normalize headers (strip, lower, remove extra spaces/quotes)
    df.columns = [c.strip().replace('"', '') for c in df.columns]

    # The UCI dataset header names can vary a bit. Build a safe rename map.
    name_map_candidates = {
        "Cement (component 1)(kg in a m^3 mixture)": "Cement",
        "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "Slag",
        "Fly Ash (component 3)(kg in a m^3 mixture)": "FlyAsh",
        "Water  (component 4)(kg in a m^3 mixture)": "Water",
        "Superplasticizer (component 5)(kg in a m^3 mixture)": "Superplasticizer",
        "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "CoarseAgg",
        "Fine Aggregate (component 7)(kg in a m^3 mixture)": "FineAgg",
        "Age (day)": "Age",
        "Concrete compressive strength(MPa, megapascals) ": "Strength",
        "Concrete compressive strength(MPa, megapascals)": "Strength",
    }

    # Fallbacks: try to match robustly by contains
    rename_map = {}
    for col in df.columns:
        key = None
        for k, v in name_map_candidates.items():
            if col.lower() == k.lower():
                key = k
                break
        if key:
            rename_map[col] = name_map_candidates[key]
        else:
            # heuristic contains:
            cl = col.lower()
            if "cement" in cl and "component 1" in cl:
                rename_map[col] = "Cement"
            elif "blast furnace slag" in cl or ("slag" in cl and "component" in cl):
                rename_map[col] = "Slag"
            elif "fly ash" in cl:
                rename_map[col] = "FlyAsh"
            elif "water" in cl and "component" in cl:
                rename_map[col] = "Water"
            elif "superplasticizer" in cl:
                rename_map[col] = "Superplasticizer"
            elif "coarse" in cl and "aggregate" in cl:
                rename_map[col] = "CoarseAgg"
            elif "fine" in cl and "aggregate" in cl:
                rename_map[col] = "FineAgg"
            elif cl.strip() in ["age", "age (day)", "age(day)"]:
                rename_map[col] = "Age"
            elif "compressive strength" in cl or "strength" in cl:
                rename_map[col] = "Strength"

    df = df.rename(columns=rename_map)

    expected = ["Cement","Slag","FlyAsh","Water","Superplasticizer","CoarseAgg","FineAgg","Age","Strength"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"Could not find columns: {missing}\nFound columns: {list(df.columns)}")

    # Enforce numeric
    for c in expected:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=expected)

    return df[expected]

# ---------------------------
# 2) Split (train/test)
# test = rows 501–630 (inclusive of 501, exclusive of 631)
# train = everything else
# ---------------------------
def split_train_test(df):
    # Index is 0-based; the instruction says "Assuming the columns labels are row 0" — that matches pandas default
    test_idx = range(501, 631)  # 130 rows: 501..630
    test_mask = df.index.isin(test_idx)
    df_train = df.loc[~test_mask].copy()
    df_test  = df.loc[test_mask].copy()
    return df_train, df_test

# ---------------------------
# 3) Preprocessing helpers
# ---------------------------
def standardize_fit(X_train):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=0).replace(0, 1.0)
    return mu, sd

def standardize_transform(X, mu, sd):
    return (X - mu) / sd

def log1p_fit(X_train):
    # No params needed; but return None for API symmetry
    return None

def log1p_transform(X, _):
    return np.log1p(X)

# ---------------------------
# 4) Metrics
# ---------------------------
def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred)**2))

def r2(y_true, y_pred):
    var = float(np.var(y_true, ddof=0))
    if var == 0:
        return 0.0
    return 1.0 - mse(y_true, y_pred) / var

# ---------------------------
# 5) Fit using OLS (for p-values and predictions)
# ---------------------------
def fit_ols(X_train, y_train):
    if sm is None:
        return None
    Xc = sm.add_constant(X_train, has_constant='add')
    model = sm.OLS(y_train, Xc).fit()
    return model

def predict_ols(model, X):
    Xc = sm.add_constant(X, has_constant='add')
    return model.predict(Xc)

# ---------------------------
# 6) Plot helpers
# ---------------------------
def save_performance_bars(metrics_df):
    # bars for MSE
    plt.figure(figsize=(10,5))
    idx = np.arange(len(metrics_df))
    w = 0.35
    plt.bar(idx - w/2, metrics_df["Train MSE"], width=w, label="Train MSE")
    plt.bar(idx + w/2, metrics_df["Test MSE"],  width=w, label="Test MSE")
    plt.xticks(idx, metrics_df["Setting"], rotation=15)
    plt.ylabel("MSE"); plt.title("Model Performance: MSE (Train vs Test)")
    plt.legend(); plt.tight_layout()
    plt.savefig("model_performance_mse_bar.png", dpi=200); plt.close()

    # bars for R2
    plt.figure(figsize=(10,5))
    plt.bar(idx - w/2, metrics_df["Train R²"], width=w, label="Train R²")
    plt.bar(idx + w/2, metrics_df["Test R²"],  width=w, label="Test R²")
    plt.xticks(idx, metrics_df["Setting"], rotation=15)
    plt.ylabel("R²"); plt.ylim(0,1)
    plt.title("Model Performance: R² (Train vs Test)")
    plt.legend(); plt.tight_layout()
    plt.savefig("model_performance_r2_bar.png", dpi=200); plt.close()

def pvalues_plots(pvals_df, tag):
    # -log10(p)
    pv = pvals_df.copy()
    pv = pv[pv["Predictor"] != "const"]  # drop constant for the bar chart
    neglog10 = -np.log10(pv["p-value"].clip(lower=1e-300))
    plt.figure(figsize=(11,5))
    plt.bar(pv["Predictor"], neglog10)
    plt.xticks(rotation=20)
    plt.ylabel("-log10(p-value)")
    plt.title(f"Feature Significance (OLS) — {tag}")
    plt.tight_layout()
    plt.savefig(f"pvalues_{tag}_neglog10_bar.png", dpi=200)
    plt.close()

    # table-as-image
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.35*len(pvals_df)))
    ax.axis('off')
    df_show = pvals_df.copy()
    df_show["p-value"] = df_show["p-value"].map(lambda v: f"{v:.3e}")
    tbl = ax.table(cellText=df_show.values, colLabels=df_show.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)
    plt.title(f"p-values (OLS) — {tag}")
    plt.tight_layout()
    plt.savefig(f"pvalues_{tag}_table.png", dpi=200)
    plt.close()

# ---------------------------
# 7) Main run
# ---------------------------
def main():
    df = load_concrete("Concrete_Data.csv")
    df_train, df_test = split_train_test(df)

    X_train_raw = df_train.drop(columns=["Strength"]).copy()
    y_train = df_train["Strength"].values
    X_test_raw  = df_test.drop(columns=["Strength"]).copy()
    y_test = df_test["Strength"].values

    # Standardized (fit on train predictors only)
    mu, sd = standardize_fit(X_train_raw)
    X_train_std = standardize_transform(X_train_raw, mu, sd)
    X_test_std  = standardize_transform(X_test_raw,  mu, sd)

    # Log1p (predictors only)
    X_train_log = log1p_transform(X_train_raw, None)
    X_test_log  = log1p_transform(X_test_raw,  None)

    # Collect settings
    settings = [
        ("Raw predictors", X_train_raw, X_test_raw),
        ("Standardized predictors", X_train_std, X_test_std),
        ("Log(x+1) predictors", X_train_log, X_test_log),
    ]

    metrics_rows = []
    all_pvals = {}

    for name, Xtr, Xte in settings:
        if sm is None:
            # simple closed-form for predictions (no p-values)
            # normal equation: w = (X'X)^(-1) X'y with constant; use np.linalg.lstsq for stability
            Xtr_c = np.c_[np.ones(len(Xtr)), Xtr.values]
            w, *_ = np.linalg.lstsq(Xtr_c, y_train, rcond=None)
            yhat_tr = Xtr_c @ w
            Xte_c = np.c_[np.ones(len(Xte)), Xte.values]
            yhat_te = Xte_c @ w
            train_mse, test_mse = mse(y_train, yhat_tr), mse(y_test, yhat_te)
            train_r2, test_r2 = r2(y_train, yhat_tr), r2(y_test, yhat_te)
            metrics_rows.append([name, train_mse, test_mse, train_r2, test_r2])
        else:
            # OLS fit for p-values + predictions
            model = fit_ols(Xtr, y_train)
            yhat_tr = predict_ols(model, Xtr)
            yhat_te = predict_ols(model, Xte)
            train_m, test_m = mse(y_train, yhat_tr), mse(y_test, yhat_te)
            train_r, test_r = r2(y_train, yhat_tr), r2(y_test, yhat_te)
            metrics_rows.append([name, train_m, test_m, train_r, test_r])

            # p-values
            p = model.pvalues  # index: const + feature names
            pvals_df = pd.DataFrame({"Predictor": p.index, "p-value": p.values})
            all_pvals[name] = pvals_df

    metrics_df = pd.DataFrame(metrics_rows, columns=["Setting","Train MSE","Test MSE","Train R²","Test R²"])
    metrics_df.to_csv("metrics_summary_from_concrete.csv", index=False)
    print("\n=== Metrics Summary (from Concrete_Data.csv) ===")
    print(metrics_df.to_string(index=False))

    # Plots: performance bars
    save_performance_bars(metrics_df)

    # Plots: p-values (if available)
    if sm is not None:
        for name, pdf in all_pvals.items():
            tag = ("raw" if "Raw" in name
                   else "std" if "Standardized" in name
                   else "log")
            pvalues_plots(pdf, tag)
        print("\nSaved p-value figures for raw/std/log.")
    else:
        print("\nstatsmodels not installed — skipped p-value figures.")

    print("\nSaved figures:")
    print(" - model_performance_mse_bar.png")
    print(" - model_performance_r2_bar.png")
    if sm is not None:
        print(" - pvalues_raw_neglog10_bar.png, pvalues_raw_table.png")
        print(" - pvalues_std_neglog10_bar.png, pvalues_std_table.png")
        print(" - pvalues_log_neglog10_bar.png, pvalues_log_table.png")

if __name__ == "__main__":
    main()
