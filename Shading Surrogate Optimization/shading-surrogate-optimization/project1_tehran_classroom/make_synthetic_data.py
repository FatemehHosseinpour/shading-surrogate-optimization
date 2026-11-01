
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
grid = {"wwr": [0.15, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5], "angle": [0, 10, 15, 30, 45],
        "ls_angle": [0, 30], "depth_l": [0.1, 0.2, 0.3, 0.4], "spacing_l": [0.2, 0.3, 0.5, 1.0],
        "fins_l": [1, 2, 3, 4], "ls_depth": [1.0, 1.1, 1.2, 1.3, 1.4]}
n = 5000
d = pd.DataFrame({k: rng.choice(v, n) for k, v in grid.items()}).drop_duplicates()
n = len(d)
noise = lambda s: rng.normal(0, s, n)
d["udi"] = (92 - 60 * (d.wwr - 0.3) ** 2 + 8 * d.depth_l - 0.03 * d.angle + noise(1.5)).clip(40, 100)
d["sdg"] = (8 + 60 * d.wwr - 15 * d.depth_l - 0.2 * d.angle + noise(2)).clip(0, 60)
d["eui"] = 262 + 20 * d.depth_l + 0.05 * d.angle - 5 * d.wwr + noise(0.5)
d["penergy"] = 520 + 120 * d.wwr + 0.2 * d.angle + noise(2)
d["tcp"] = 55 + 6 * d.wwr - 0.02 * d.angle + noise(0.5)
d.round(3).to_csv("data/clean_designs.csv", index=False)
print(f"Wrote data/clean_designs.csv with {n} SYNTHETIC designs (not real results).")
