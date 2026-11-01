
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error

SEED = 42
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------- 1. data
df = pd.read_csv("data/zahedan_search_space.csv").drop(columns=["img"])
n0 = len(df)
df = df.dropna()
df = df[df["out:EUI"] > 0]          # -999 = failed simulation
df = df[df["out:DG"] < 100]         # 100 = saturated / suspicious value
df = df.drop_duplicates(subset=[c for c in df.columns if c.startswith("in:")])
print(f"Rows: {n0} -> {len(df)} after cleaning")

X_COLS = ["in:angle", "in:depth", "in:wwr-n", "in:wwr-s", "in:orientation"]
Y_COLS = ["out:EUI", "out:DG", "out:UDI"]
X, Y = df[X_COLS].values, df[Y_COLS].values

# Search space = grid actually present in the data
GRID = [np.unique(X[:, j]) for j in range(X.shape[1])]
LB = np.array([g.min() for g in GRID])
UB = np.array([g.max() for g in GRID])
DIM = len(X_COLS)


def snap(pos):
    """Round each variable to the nearest allowed grid value."""
    out = np.empty_like(pos)
    for j, g in enumerate(GRID):
        out[..., j] = g[np.abs(pos[..., j, None] - g).argmin(axis=-1)]
    return out


# -------------------------------------------------------------- 2. surrogate
def make_model():
    return ExtraTreesRegressor(n_estimators=500, min_samples_leaf=1,
                               random_state=SEED, n_jobs=-1)

cv = KFold(5, shuffle=True, random_state=SEED)
models = {}
print("\nSurrogate quality (5-fold CV, random folds; optimistic because neighbouring designs are similar):")
for k, name in enumerate(Y_COLS):
    pred = cross_val_predict(make_model(), X, Y[:, k], cv=cv)
    print(f"  {name:8s} R2 = {r2_score(Y[:, k], pred):.3f}   "
          f"MAE = {mean_absolute_error(Y[:, k], pred):.3f}")
    models[name] = make_model().fit(X, Y[:, k])


def predict(pop):
    """pop (n, 5) -> (n, 3) array of [EUI, DG, UDI]."""
    return np.column_stack([models[c].predict(pop) for c in Y_COLS])


# objective vector to MINIMISE: [EUI, DG, -UDI]
def objectives(pop):
    p = predict(pop)
    p[:, 2] = -p[:, 2]
    return p


# ------------------------------------------------- 3. single-objective GWO
Y_MIN, Y_MAX = Y.min(0), Y.max(0)


def weighted_cost(pop, w=(1 / 3, 1 / 3, 1 / 3)):
    p = predict(pop)
    n = (p - Y_MIN) / (Y_MAX - Y_MIN)          # 0..1
    return w[0] * n[:, 0] + w[1] * n[:, 1] + w[2] * (1 - n[:, 2])


def gwo(cost_fn, n_wolves=30, n_iter=100):
    pos = snap(rng.uniform(LB, UB, (n_wolves, DIM)))
    fit = cost_fn(pos)
    order = np.argsort(fit)
    leaders = pos[order[:3]].copy()            # alpha, beta, delta
    leader_fit = fit[order[:3]].copy()
    history = []
    for t in range(n_iter):
        a = 2 - 2 * t / n_iter                 # linearly 2 -> 0
        new = np.zeros_like(pos)
        for L in leaders:
            A = 2 * a * rng.random((n_wolves, DIM)) - a
            C = 2 * rng.random((n_wolves, DIM))
            D = np.abs(C * L - pos)
            new += (L - A * D) / 3
        pos = snap(np.clip(new, LB, UB))
        fit = cost_fn(pos)
        # update leaders (keep best three seen so far, unique)
        allp = np.vstack([leaders, pos])
        allf = np.concatenate([leader_fit, fit])
        _, uidx = np.unique(allp, axis=0, return_index=True)
        allp, allf = allp[uidx], allf[uidx]
        order = np.argsort(allf)[:3]
        leaders, leader_fit = allp[order], allf[order]
        history.append(leader_fit[0])
    return leaders[0], leader_fit[0], history


print("\n=== Single-objective GWO (equal weights) ===")
best_x, best_f, hist = gwo(weighted_cost)
best_y = predict(best_x[None])[0]
print(pd.Series(best_x, index=X_COLS).to_string())
print(f"Predicted: EUI={best_y[0]:.2f}  DG={best_y[1]:.2f}  UDI={best_y[2]:.2f}  "
      f"(cost={best_f:.4f})")


# ------------------------------------------------------------------ 4. MOGWO
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def update_archive(arch_x, arch_f, cand_x, cand_f, max_size=100):
    X_all = np.vstack([arch_x, cand_x]) if len(arch_x) else cand_x
    F_all = np.vstack([arch_f, cand_f]) if len(arch_f) else cand_f
    _, u = np.unique(X_all, axis=0, return_index=True)       # drop duplicates
    X_all, F_all = X_all[u], F_all[u]
    keep = np.ones(len(F_all), bool)
    for i in range(len(F_all)):
        for j in range(len(F_all)):
            if i != j and dominates(F_all[j], F_all[i]):
                keep[i] = False
                break
    X_all, F_all = X_all[keep], F_all[keep]
    while len(F_all) > max_size:                              # crowding trim
        z = (F_all - F_all.min(0)) / (np.ptp(F_all, 0) + 1e-12)
        d = np.linalg.norm(z[:, None] - z[None], axis=2)
        np.fill_diagonal(d, np.inf)
        drop = d.min(1).argmin()
        X_all, F_all = np.delete(X_all, drop, 0), np.delete(F_all, drop, 0)
    return X_all, F_all


def select_leader(arch_x, arch_f):
    """Roulette selection favouring sparsely populated regions of the front."""
    if len(arch_x) == 1:
        return arch_x[0]
    z = (arch_f - arch_f.min(0)) / (np.ptp(arch_f, 0) + 1e-12)
    d = np.linalg.norm(z[:, None] - z[None], axis=2)
    np.fill_diagonal(d, np.inf)
    sparsity = d.min(1) + 1e-9
    return arch_x[rng.choice(len(arch_x), p=sparsity / sparsity.sum())]


def mogwo(n_wolves=40, n_iter=150, max_archive=100):
    pos = snap(rng.uniform(LB, UB, (n_wolves, DIM)))
    arch_x, arch_f = update_archive(np.empty((0, DIM)), np.empty((0, 3)),
                                    pos, objectives(pos), max_archive)
    for t in range(n_iter):
        a = 2 - 2 * t / n_iter
        new = np.zeros_like(pos)
        for _ in range(3):                                     # alpha,beta,delta
            L = select_leader(arch_x, arch_f)
            A = 2 * a * rng.random((n_wolves, DIM)) - a
            C = 2 * rng.random((n_wolves, DIM))
            new += (L - A * np.abs(C * L - pos)) / 3
        pos = snap(np.clip(new, LB, UB))
        arch_x, arch_f = update_archive(arch_x, arch_f, pos,
                                        objectives(pos), max_archive)
    return arch_x, arch_f


print("\n=== Multi-objective GWO (Pareto front) ===")
px, pf = mogwo()
pareto = pd.DataFrame(px, columns=X_COLS)
pareto["EUI"], pareto["DG"], pareto["UDI"] = pf[:, 0], pf[:, 1], -pf[:, 2]
pareto = pareto.sort_values("EUI").reset_index(drop=True)
pareto.to_csv("data/pareto_front.csv", index=False)
print(f"{len(pareto)} non-dominated solutions saved to pareto_front.csv")
print(pareto.head(8).round(3).to_string())

# ------------------------------------------------------------------- plots
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(hist, lw=2)
ax[0].set(title="GWO convergence (weighted cost)", xlabel="Iteration",
          ylabel="Best cost")
sc = ax[1].scatter(pareto["EUI"], pareto["UDI"], c=pareto["DG"], cmap="viridis",
                   s=40, edgecolor="k", lw=.3)
ax[1].set(title="Pareto front: EUI vs UDI", xlabel="EUI", ylabel="UDI (%)")
plt.colorbar(sc, ax=ax[1], label="DG")
sc2 = ax[2].scatter(pareto["EUI"], pareto["DG"], c=pareto["UDI"], cmap="plasma",
                    s=40, edgecolor="k", lw=.3)
ax[2].set(title="Pareto front: EUI vs DG", xlabel="EUI", ylabel="DG")
plt.colorbar(sc2, ax=ax[2], label="UDI (%)")
plt.tight_layout()
plt.savefig("figures/gwo_results.png", dpi=150)
print("\nSaved gwo_results.png")


# ------------------------------------------------- 5. stricter validation
# Hold out whole window combinations (3 north x 4 south ratios) so the model is tested
# on designs unlike anything it was trained on.
from sklearn.model_selection import GroupKFold
groups = df[["in:wwr-n", "in:wwr-s"]].astype(str).agg("-".join, axis=1)
print("\nSurrogate quality (grouped CV, unseen window combinations):")
for k, name in enumerate(Y_COLS):
    pred = cross_val_predict(make_model(), X, Y[:, k], cv=GroupKFold(6), groups=groups)
    print(f"  {name:8s} R2 = {r2_score(Y[:, k], pred):.3f}   MAE = {mean_absolute_error(Y[:, k], pred):.3f}")

# ------------------------------------------------- 6. sanity check against exhaustive search
nrm = (Y - Y_MIN) / (Y_MAX - Y_MIN)
cost_all = nrm[:, 0] / 3 + nrm[:, 1] / 3 + (1 - nrm[:, 2]) / 3
i = int(np.argmin(cost_all))
print("\nBest simulated design by the same cost (exhaustive):",
      dict(zip(X_COLS, X[i])), f"cost={cost_all[i]:.4f}")
print(f"GWO best cost: {best_f:.4f}")
