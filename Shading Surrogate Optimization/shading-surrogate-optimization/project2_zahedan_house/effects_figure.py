
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UDI, SDG, EUI = "#E0A526", "#C0392B", "#4C78A8"   # amber, red, blue: one colour per metric
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
d = pd.read_csv("data/zahedan_final.csv")
shared = [k for k, v in d.groupby(["wn", "ws"]) if v.ori.nunique() == 4 and v.groupby("ori").size().min() >= 40]
o = d[d.set_index(["wn", "ws"]).index.isin(shared) & d.depth.between(0.1, 1.0) & d.angle.isin([0, 15, 30, 45])]
ref = d[(d.ori == 180) & (d.wn == 0.3) & (d.ws == 0.3) & d.angle.isin([0, 15, 30, 45])]
sw = d[(d.ori == 180) & (d.wn == 0.1) & d.depth.between(0.1, 1.0) & d.angle.isin([0, 15, 30, 45])]
panels = [("ori", "Orientation (deg)", o), ("depth", "Overhang depth (m)", ref),
          ("angle", "Overhang angle (deg)", ref), ("ws", "South window ratio", sw)]
rows = [("udi", "UDI (%)", UDI), ("dg", "sDG (%)", SDG), ("eui", "EUI (kWh/m2.yr)", EUI)]
fig, ax = plt.subplots(3, 4, figsize=(14, 7.2))
for i, (m, ml, c) in enumerate(rows):
    for j, (v, vl, sub) in enumerate(panels):
        g = sub.groupby(v)[m]; mean = g.mean(); a = ax[i, j]
        if v == "ori":
            a.bar([str(int(x)) for x in mean.index], mean.values, color=c, width=.55)
        else:
            a.fill_between(mean.index, g.quantile(.25), g.quantile(.75), color=c, alpha=.2, linewidth=0)
            a.plot(mean.index, mean.values, "o-", color=c, lw=2)
        if j == 0: a.set_ylabel(ml)
        if i == 2: a.set_xlabel(vl)
fig.tight_layout(); fig.savefig("figures/effects.png", dpi=200)
print("saved figures/effects.png")
