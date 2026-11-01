"""Extra ML analyses: model comparison, learning curve, surrogate-assisted search, direction plots."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
d = pd.read_csv("data/clean_designs.csv")
feats = ["wwr","angle","ls_angle","depth_l","spacing_l","fins_l","ls_depth"]
T = {"udi":"UDI","sdg":"sDG","eui":"EUI","penergy":"Primary energy"}
mk = {"Linear regression": lambda: LinearRegression(),
      "Random forest": lambda: RandomForestRegressor(300, random_state=0, n_jobs=-1),
      "Gradient boosting": lambda: HistGradientBoostingRegressor(max_iter=400, random_state=0),
      "Neural network": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,64), max_iter=1500, random_state=0))}
res = {}
for m, f in mk.items():
    res[m] = {}
    for t in T:
        Xtr,Xte,ytr,yte = train_test_split(d[feats], d[t], test_size=.25, random_state=0)
        res[m][t] = round(r2_score(yte, f().fit(Xtr,ytr).predict(Xte)), 3)
json.dump(res, open("figures/model_comparison.json","w"), indent=1); print(json.dumps(res, indent=1))
# learning curve (random forest), fixed 25% test set
Xtr_all,Xte,ytr_all,yte_all = train_test_split(d[feats+list(T)+["tcp"]], d[list(T)+["tcp"]], test_size=.25, random_state=0)
Xte_f = Xte[feats]; Yte = Xte[list(T)]
sizes=[100,200,400,800,1600,3200]; lc={t:[] for t in T}
for n in sizes:
    for t in T:
        sc=[]
        for seed in range(3):
            s = Xtr_all.sample(n, random_state=seed)
            rf = RandomForestRegressor(200, random_state=seed, n_jobs=-1).fit(s[feats], s[t]); sc.append(r2_score(Xte[t], rf.predict(Xte_f)))
        lc[t].append((np.mean(sc), np.std(sc)))
col={"udi":"#E0A526","sdg":"#C0392B","eui":"#4C78A8","penergy":"#2F4858"}
plt.rcParams.update({"font.size":12,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"#fbfaf7","axes.facecolor":"#fbfaf7"})
fig,ax=plt.subplots(figsize=(6.2,4.4))
for t in T:
    m=np.array([a for a,b in lc[t]]); s=np.array([b for a,b in lc[t]])
    ax.plot(sizes,m,"o-",color=col[t],label=T[t]); ax.fill_between(sizes,m-s,m+s,color=col[t],alpha=.15,linewidth=0)
ax.set_xscale("log"); ax.set_xticks(sizes); ax.set_xticklabels([str(s) for s in sizes]); ax.set_ylim(0,1.02)
ax.axhline(.9,color="#999",ls="--",lw=.8); ax.set_xlabel("Simulations used for training"); ax.set_ylabel("R² on unseen designs"); ax.legend(frameon=False,loc="lower right")
fig.tight_layout(); fig.savefig("figures/learning_curve.png",dpi=200); plt.close(fig)
print({t:[round(a,2) for a,b in lc[t]] for t in T})
# surrogate-assisted search: train on n designs, rank all others by predicted balanced score
nrm=lambda s:(s-s.min())/(s.max()-s.min())
d["score"]=(nrm(d.udi)+(1-nrm(d.sdg))+nrm(d.tcp)+(1-nrm(d.penergy)))/4*100
out={}
for n in [200,500,1000]:
    rs=[]
    for seed in range(5):
        tr=d.sample(n,random_state=seed); te=d.drop(tr.index)
        pred={t:RandomForestRegressor(200,random_state=seed,n_jobs=-1).fit(tr[feats],tr[t]).predict(te[feats]) for t in ["udi","sdg","penergy","tcp"]}
        lo={t:(d[t].min(),d[t].max()) for t in ["udi","sdg","penergy","tcp"]}
        n2=lambda v,t:(v-lo[t][0])/(lo[t][1]-lo[t][0])
        sc=(n2(pred["udi"],"udi")+(1-n2(pred["sdg"],"sdg"))+n2(pred["tcp"],"tcp")+(1-n2(pred["penergy"],"penergy")))/4*100
        te=te.assign(pred=sc); top_pred=te.nlargest(50,"pred"); top_true=set(te.nlargest(50,"score").index)
        rs.append((len(set(top_pred.index)&top_true), top_pred.score.mean(), te.score.max(), te.score.mean()))
    r=np.mean(rs,axis=0); out[n]={"overlap_top50":round(r[0],1),"mean_true_score_of_picks":round(r[1],1),"best_true_score":round(r[2],1),"mean_score_all":round(r[3],1)}
json.dump(out,open("figures/surrogate_search.json","w"),indent=1); print(json.dumps(out,indent=1))
# direction plots (partial dependence) for the leading driver of each output
rf={t:RandomForestRegressor(300,random_state=0,n_jobs=-1).fit(d[feats],d[t]) for t in T}
pairs=[("udi","wwr","WWR (%)",100),("sdg","wwr","WWR (%)",100),("eui","depth_l","Louvre depth (m)",1),("penergy","wwr","WWR (%)",100)]
fig,ax=plt.subplots(1,4,figsize=(14,3.3))
for a,(t,v,xl,k) in zip(ax,pairs):
    vals=np.sort(d[v].unique()); y=[]
    for x in vals:
        X=d[feats].copy(); X[v]=x; y.append(rf[t].predict(X).mean())
    a.plot(vals*k,y,"o-",color=col[t],lw=2); a.set_xlabel(xl); a.set_ylabel(T[t]); a.set_title(f"{T[t]} vs {xl.split(' (')[0].lower()}",fontsize=12,loc="left")
fig.tight_layout(); fig.savefig("figures/direction.png",dpi=200); plt.close(fig)

# surrogate-assisted search figure: model trained on 500 designs, scored against all unseen designs
te_n = d.sample(500, random_state=1); te = d.drop(te_n.index)
lo2 = {t: (d[t].min(), d[t].max()) for t in ["udi", "sdg", "penergy", "tcp"]}
n2b = lambda v, t: (v - lo2[t][0]) / (lo2[t][1] - lo2[t][0])
pred2 = {t: RandomForestRegressor(200, random_state=1, n_jobs=-1).fit(te_n[feats], te_n[t]).predict(te[feats]) for t in lo2}
ps = (n2b(pred2["udi"], "udi") + (1 - n2b(pred2["sdg"], "sdg")) + n2b(pred2["tcp"], "tcp") + (1 - n2b(pred2["penergy"], "penergy"))) / 4 * 100
te = te.assign(pred=ps); top = te.nlargest(50, "pred")
fig, ax = plt.subplots(figsize=(6.4, 4.3))
ax.scatter(te.score, te.pred, s=7, color="#b9b2a5", alpha=.5, label="Unseen designs")
ax.scatter(top.score, top.pred, s=26, color="#C0392B", label="50 designs picked by the model")
ax.plot([40, 85], [40, 85], "k--", lw=.8); ax.set_xlabel("Balanced score from simulation"); ax.set_ylabel("Balanced score predicted")
ax.legend(frameon=False, loc="upper left", fontsize=10.5); fig.tight_layout(); fig.savefig("figures/search_scatter.png", dpi=200); plt.close(fig)
