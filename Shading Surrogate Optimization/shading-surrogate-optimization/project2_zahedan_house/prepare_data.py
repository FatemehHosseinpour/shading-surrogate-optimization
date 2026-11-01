
import sys
import pandas as pd

raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/raw_simulations.csv").drop(columns=["img"], errors="ignore")
raw.columns = ["angle", "depth", "wn", "ws", "ori", "eui", "dg", "udi"]
n0 = len(raw)
d = raw.dropna()
d = d[(d.eui > 0) & (d.dg < 100) & d.angle.isin([0, 10, 15, 30, 45]) & (d.depth <= 1.0 + 1e-9)]
d = d.groupby(["angle", "depth", "wn", "ws", "ori"], as_index=False).mean()
d.to_csv("data/zahedan_final.csv", index=False)
print(f"{n0} raw rows -> {len(d)} unique designs")

# orientations for which all 12 window combinations (3 north x 4 south) were simulated
full = [o for o, g in d.groupby("ori") if g.groupby(["wn", "ws"]).ngroups == 12]
s = d[d.ori.isin(full)].copy()
s.columns = ["in:angle", "in:depth", "in:wwr-n", "in:wwr-s", "in:orientation", "out:EUI", "out:DG", "out:UDI"]
s["img"] = ""
s.to_csv("data/zahedan_search_space.csv", index=False)
print(f"search space: orientations {full}, {len(s)} designs")
