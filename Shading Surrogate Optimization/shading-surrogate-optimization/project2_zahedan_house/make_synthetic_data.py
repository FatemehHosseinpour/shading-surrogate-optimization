
import itertools
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
rows = list(itertools.product([0, 10, 15, 30, 45], np.round(np.arange(0, 1.01, 0.1), 1),
                              [0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4], [135, 150, 165, 180]))
d = pd.DataFrame(rows, columns=["in:angle", "in:depth", "in:wwr-n", "in:wwr-s", "in:orientation"])
n = len(d)
off = (d["in:orientation"] - 165).abs()
d["out:EUI"] = (168 + 2 * d["in:depth"] + 0.02 * d["in:angle"] - 8 * d["in:wwr-s"] + 0.02 * off
                + rng.normal(0, 0.3, n)).round(3)
d["out:DG"] = (12 + 40 * d["in:wwr-s"] - 8 * d["in:depth"] - 0.1 * d["in:angle"] + 0.1 * off
               + rng.normal(0, 1, n)).clip(0, 60).round(3)
d["out:UDI"] = (78 + 20 * d["in:wwr-s"] - 0.05 * d["in:angle"] - 0.04 * off
                + rng.normal(0, 1, n)).clip(40, 100).round(3)
bad = rng.choice(n, 15, replace=False)
d.loc[bad[:8], "out:EUI"] = -999
d.loc[bad[8:], "out:DG"] = 100
d = pd.concat([d, d.sample(40, random_state=0)], ignore_index=True)     # repeated runs
d["img"] = ""
d.to_csv("data/raw_simulations.csv", index=False)
print(f"Wrote data/raw_simulations.csv with {len(d)} SYNTHETIC rows (not real results).")
