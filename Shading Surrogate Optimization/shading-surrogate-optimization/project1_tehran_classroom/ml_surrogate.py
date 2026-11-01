
import re, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import permutation_importance

src = sys.argv[1] if len(sys.argv) > 1 else "data/raw_simulations.xlsx"  # raw simulation data is not included in this repository
d = pd.read_excel(src, header=0)
d.columns = ["wwr", "tcp", "pmv", "eui", "cool", "heat", "penergy", "lighting",
             "sdg", "sda", "ase", "udi", "angle", "spacing", "fins", "depth", "type", "ls_angle"]
for c in ["tcp", "pmv", "eui", "cool", "heat", "penergy", "lighting", "sdg", "sda", "ase", "udi", "angle", "ls_angle"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")

def part(s, i):                      # "1.0, 0.2" -> 0.2 ; "1, 4" -> 4
    try: return float(re.split(r"[,\s]+", str(s).strip())[i])
    except Exception: return np.nan

pat = re.compile(r"^\s*([\d.]+)\(([\d.]+)\+([\d.]+)\)\+([\d.]+)")
def depth(s):                        # "1.0(0.5+0.5)+.2" -> light-shelf depth 1.0, louvre depth 0.2
    g = pat.match(str(s))
    return (float(g.group(2)) + float(g.group(3)), float(g.group(4))) if g else (np.nan, np.nan)

d["spacing_l"] = d.spacing.map(lambda s: part(s, 1))
d["fins_l"] = d.fins.map(lambda s: part(s, 1))
d[["ls_depth", "depth_l"]] = d.depth.map(depth).apply(pd.Series)

feats = ["wwr", "angle", "ls_angle", "depth_l", "spacing_l", "fins_l", "ls_depth"]
labels = {"wwr": "WWR", "angle": "Louvre angle", "ls_angle": "Light-shelf angle",
          "depth_l": "Louvre depth", "spacing_l": "Louvre spacing",
          "fins_l": "Fin number", "ls_depth": "Light-shelf depth"}
targets = {"udi": "UDI (%)", "sdg": "sDG (%)", "eui": "EUI (kWh/m²·yr)", "penergy": "Primary energy"}

raw = len(d)
ok = ((d.tcp > 0) & (d.eui > 0) & d.udi.between(0, 100) & d.sdg.between(0, 100)
      & (d.penergy > 100))
d = d[ok].dropna(subset=feats)
# Drop extreme simulation outliers (|z| > 5 on energy, EUI or comfort) and report how many.
z = ((d[["penergy", "eui", "tcp"]] - d[["penergy", "eui", "tcp"]].mean()) / d[["penergy", "eui", "tcp"]].std()).abs()
n_out = int((z > 5).any(axis=1).sum()); d = d[~(z > 5).any(axis=1)]
# Physically implausible rows (UDI below 40 % or sDG above 60 %): simulation errors.
bad = (d.udi < 40) | (d.sdg > 60); n_bad = int(bad.sum()); d = d[~bad]
print(f"rows: raw {raw}, clean {len(d)}, z-outliers removed {n_out}, implausible rows removed {n_bad}")

# Repeated designs (same variables, several runs): average them to avoid leakage.
dd = d.groupby(feats, as_index=False)[list(targets) + ["tcp"]].mean()
print(f"unique designs: {len(dd)}")

metrics, models, tests = {}, {}, {}
for t in targets:
    Xtr, Xte, ytr, yte = train_test_split(dd[feats], dd[t], test_size=0.25, random_state=0)
    rf = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1).fit(Xtr, ytr)
    p = rf.predict(Xte)
    cv = cross_val_score(RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1),
                         dd[feats], dd[t], cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
    pi = permutation_importance(rf, Xte, yte, n_repeats=10, random_state=0)
    imp = pd.Series(pi.importances_mean, index=feats).clip(lower=0)
    metrics[t] = {"r2_test": round(r2_score(yte, p), 3),
                  "rmse_test": round(mean_squared_error(yte, p) ** 0.5, 3),
                  "r2_cv5": round(cv.mean(), 3), "importance": (imp / imp.sum()).round(3).to_dict()}
    models[t], tests[t] = rf, (yte, p)
json.dump(metrics, open("figures/metrics.json", "w"), indent=1)
dd.to_csv("data/clean_designs.csv", index=False)

col = {"udi": "#E0A526", "sdg": "#C0392B", "eui": "#4C78A8", "penergy": "#2F4858"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: permutation importance (share of total) with R2 on every panel
fig, ax = plt.subplots(1, 4, figsize=(18, 5.2))
for a, t in zip(ax, targets):
    imp = pd.Series(metrics[t]["importance"]).sort_values()
    a.barh([labels[i] for i in imp.index], imp.values, color=col[t])
    for y, v in enumerate(imp.values): a.text(v + 0.01, y, f"{v:.2f}", va="center")
    a.set_title(f"{targets[t]}\nR² = {metrics[t]['r2_test']} (test)", fontsize=13)
    a.set_xlim(0, max(0.5, imp.max() * 1.25)); a.set_xlabel("Share of permutation importance")
fig.suptitle("Which design variables drive each outcome", fontsize=15, y=1.02)
fig.text(0.5, -0.03, f"Light shelf + horizontal louvre · {len(dd)} unique designs · random forest, 25% held out",
         ha="center", fontsize=10, style="italic")
fig.tight_layout(); fig.savefig("figures/fig1_feature_importance.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Fig 2: predicted vs simulated
fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
for a, t in zip(ax, targets):
    y, p = tests[t]
    a.scatter(y, p, s=10, alpha=.5, color=col[t])
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max()); a.plot([lo, hi], [lo, hi], "k--", lw=1)
    a.set_title(f"{targets[t]}  R² = {metrics[t]['r2_test']}, RMSE = {metrics[t]['rmse_test']}", fontsize=12)
    a.set_xlabel("Simulated"); a.set_ylabel("Predicted")
fig.tight_layout(); fig.savefig("figures/fig2_predicted_vs_simulated.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# Fig 3: partial dependence on WWR and louvre angle
def pdp(rf, var):
    vals = np.sort(dd[var].unique()); out = []
    for v in vals:
        X = dd[feats].copy(); X[var] = v; out.append(rf.predict(X).mean())
    return vals, np.array(out)
fig, ax = plt.subplots(2, 4, figsize=(18, 7))
for j, t in enumerate(targets):
    for i, (var, xl) in enumerate([("wwr", "WWR"), ("angle", "Louvre angle (°)")]):
        v, y = pdp(models[t], var)
        ax[i, j].plot(v * (100 if var == "wwr" else 1), y, "o-", color=col[t]); ax[i, j].set_xlabel(xl + (" (%)" if var == "wwr" else ""))
        ax[i, j].set_ylabel(targets[t])
fig.suptitle("Effect of window size and louvre angle (partial dependence, other variables averaged)", fontsize=14)
fig.tight_layout(); fig.savefig("figures/fig3_partial_dependence.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print(json.dumps({k: {m: v for m, v in x.items() if m != "importance"} for k, x in metrics.items()}, indent=1))
