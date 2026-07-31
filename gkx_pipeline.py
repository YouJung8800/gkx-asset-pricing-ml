"""
Gu, Kelly & Xiu (2020, RFS) "Empirical Asset Pricing via Machine Learning" 재현 파이프라인
실행: python gkx_pipeline.py
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import SplineTransformer
from scipy.stats import norm

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1) 데이터 생성 (논문 Internet Appendix A 방식: 선형+비선형+상호작용+노이즈 합성 패널)
#    * 실제 CRSP 데이터가 있으면 이 블록만 교체하면 됩니다.
# ---------------------------------------------------------------------------
N_STOCKS, N_MONTHS = 300, 200
CHARS = ["mom1m", "mom12m", "mom36m", "chmom", "indmom", "mvel1", "dolvol", "turn",
         "retvol", "idiovol", "beta", "bm", "ep", "agr"]
MACRO = ["dp", "bm_agg", "ntis", "tbl"]

style = rng.normal(size=(N_STOCKS, len(CHARS)))
rows = []
for t in range(N_MONTHS):
    style = 0.85 * style + rng.normal(scale=0.3, size=(N_STOCKS, len(CHARS)))
    ranks = pd.DataFrame(style, columns=CHARS).rank(pct=True)
    df_t = (ranks - 0.5) * 2
    df_t["permno"], df_t["date"] = np.arange(N_STOCKS), t
    rows.append(df_t)
panel = pd.concat(rows, ignore_index=True)

macro = pd.DataFrame(rng.normal(scale=0.5, size=(N_MONTHS, len(MACRO))), columns=MACRO)
macro = macro.cumsum() * 0.02
macro = macro.clip(-1, 1)
macro["date"] = np.arange(N_MONTHS)

df = panel.merge(macro, on="date")
signal = (0.015 * df["mom12m"] - 0.010 * df["mvel1"] + 0.008 * df["bm"] - 0.010 * df["retvol"]
          + 0.008 * df["retvol"] ** 2 + 0.012 * df["mom1m"] * df["mvel1"]
          - 0.006 * df["mvel1"] * df["bm_agg"])
df["ret"] = signal + rng.normal(scale=0.08, size=len(df))
for j in range(3):
    df[f"placebo_{j}"] = rng.normal(size=len(df))   # 순수 노이즈 특성 (논문 robustness 취지)

feature_cols = CHARS + [f"placebo_{j}" for j in range(3)]
for c in CHARS:
    for m in MACRO:
        col = f"{c}_x_{m}"
        df[col] = df[c] * df[m]
        feature_cols.append(col)

months = np.sort(df["date"].unique())
n_train, n_val = int(len(months) * 0.5), int(len(months) * 0.25)
train_m, val_m, test_m = months[:n_train], months[n_train:n_train + n_val], months[n_train + n_val:]
train, val, test = [df[df["date"].isin(m)].copy() for m in (train_m, val_m, test_m)]
Xtr, ytr = train[feature_cols].values, train["ret"].values
Xval, yval = val[feature_cols].values, val["ret"].values
Xte, yte = test[feature_cols].values, test["ret"].values
print(f"train={len(train)}  val={len(val)}  test={len(test)}  features={len(feature_cols)}")

# ---------------------------------------------------------------------------
# 2) 평가 함수 — 논문 식(19) R2_oos, 식(20) Diebold-Mariano 검정
# ---------------------------------------------------------------------------
def r2_oos(y, yhat):
    return float(1 - np.sum((y - yhat) ** 2) / np.sum(y ** 2)) * 100

def diebold_mariano(e1, e2, dates):
    d = (e2 ** 2 - e1 ** 2)
    d_t = pd.DataFrame({"date": dates, "d": d}).groupby("date")["d"].mean().values
    T = len(d_t); lag = max(1, int(T ** 0.25))
    var = np.var(d_t, ddof=1)
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        var += 2 * w * np.cov(d_t[:-l], d_t[l:])[0, 1]
    se = np.sqrt(max(var, 1e-12) / T)
    stat = d_t.mean() / se if se > 0 else 0.0
    return stat, 2 * (1 - norm.cdf(abs(stat)))

# ---------------------------------------------------------------------------
# 3) 논문 식(3)~(18): OLS / OLS-3 / ElasticNet / PCR / PLS / GLM+GroupLasso / RF / GBRT / NN1-3
# ---------------------------------------------------------------------------
models = {}

models["OLS+H"] = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=500).fit(Xtr, ytr)

ols3_cols = [c for c in ["mvel1", "bm", "mom12m"] if c in feature_cols]
idx3 = [feature_cols.index(c) for c in ols3_cols]
ols3 = LinearRegression().fit(Xtr[:, idx3], ytr)
models["OLS-3"] = ("subset", ols3, idx3)

best = None
for l1 in (0.1, 0.5, 0.9):
    m = ElasticNetCV(l1_ratio=l1, cv=3, n_alphas=15, max_iter=5000, random_state=0).fit(Xtr, ytr)
    mse = np.mean((yval - m.predict(Xval)) ** 2)
    if best is None or mse < best[0]:
        best = (mse, m)
models["ENet+H"] = best[1]

best = None
for k in (5, 10, 20):
    pca = PCA(n_components=k, random_state=0).fit(Xtr)
    lr = LinearRegression().fit(pca.transform(Xtr), ytr)
    mse = np.mean((yval - lr.predict(pca.transform(Xval))) ** 2)
    if best is None or mse < best[0]:
        best = (mse, pca, lr)
models["PCR"] = ("pca", best[1], best[2])

best = None
for k in (2, 3, 5, 8):
    pls = PLSRegression(n_components=k).fit(Xtr, ytr)
    mse = np.mean((yval - pls.predict(Xval).ravel()) ** 2)
    if best is None or mse < best[0]:
        best = (mse, pls)
models["PLS"] = best[1]

models["RF"] = RandomForestRegressor(n_estimators=200, max_depth=4, max_features=1/3,
                                      random_state=0, n_jobs=-1).fit(Xtr, ytr)

best = None
for depth in (1, 2):
    for lr_ in (0.03, 0.1):
        m = GradientBoostingRegressor(n_estimators=150, max_depth=depth, learning_rate=lr_,
                                       loss="huber", random_state=0).fit(Xtr, ytr)
        mse = np.mean((yval - m.predict(Xval)) ** 2)
        if best is None or mse < best[0]:
            best = (mse, m)
models["GBRT+H"] = best[1]

# GLM + Group Lasso (식 14-15): 스플라인 확장 + proximal gradient group-lasso 직접 구현
def fit_glm_group_lasso(Xtr_df, ytr, Xval_df, yval, cols, lam_grid=(0.01, 0.05)):
    parts_tr, parts_val, groups = [], [], []
    splines = {}
    for c in cols:
        sp = SplineTransformer(n_knots=3, degree=2, include_bias=False).fit(Xtr_df[[c]].values)
        splines[c] = sp
        tr = sp.transform(Xtr_df[[c]].values); parts_tr.append(tr)
        parts_val.append(sp.transform(Xval_df[[c]].values))
        groups.append(tr.shape[1])
    Ztr, Zval = np.hstack(parts_tr), np.hstack(parts_val)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    Ztr, Zval = (Ztr - mu) / sd, (Zval - mu) / sd
    idx = np.cumsum([0] + groups)
    best = None
    for lam in lam_grid:
        theta = np.zeros(Ztr.shape[1])
        step = 1.0 / (np.linalg.norm(Ztr, 2) ** 2 / len(ytr) + 1e-6)
        for _ in range(200):
            grad = -Ztr.T @ (ytr - Ztr @ theta) / len(ytr)
            theta = theta - step * grad
            for j in range(len(groups)):
                g = theta[idx[j]:idx[j+1]]; n = np.linalg.norm(g)
                theta[idx[j]:idx[j+1]] = 0 if n <= step*lam else g * (1 - step*lam/n)
        mse = np.mean((yval - Zval @ theta) ** 2)
        if best is None or mse < best[0]:
            best = (mse, theta.copy())
    return best[1], splines, mu, sd

glm_cols = CHARS[:10]
glm_theta, glm_splines, glm_mu, glm_sd = fit_glm_group_lasso(train, ytr, val, yval, glm_cols)
models["GLM+H"] = ("glm", glm_theta, glm_splines, glm_mu, glm_sd, glm_cols)

# 신경망 NN1~NN3 (논문 피라미드 구조 32-16-8, ReLU, Adam, early stopping, 3개 시드 앙상블)
ARCHS = {"NN1": (32,), "NN2": (32, 16), "NN3": (32, 16, 8)}
for name, arch in ARCHS.items():
    ens = [MLPRegressor(hidden_layer_sizes=arch, activation="relu", solver="adam",
                         alpha=1e-3, early_stopping=True, n_iter_no_change=15,
                         max_iter=300, random_state=s).fit(Xtr, ytr) for s in range(3)]
    models[name] = ("ensemble", ens)

# ---------------------------------------------------------------------------
# 4) 예측 통일 인터페이스
# ---------------------------------------------------------------------------
def predict(name, X, Xdf=None):
    m = models[name]
    if name == "OLS-3":
        return m[1].predict(X[:, m[2]])
    if name == "PCR":
        return m[2].predict(m[1].transform(X))
    if name == "GLM+H":
        _, theta, splines, mu, sd, cols = m
        parts = [splines[c].transform(Xdf[[c]].values) for c in cols]
        Z = (np.hstack(parts) - mu) / sd
        return Z @ theta
    if name in ARCHS:
        return np.mean([e.predict(X) for e in m[1]], axis=0)
    return m.predict(X)

# ---------------------------------------------------------------------------
# 5) Table 1 (R2_oos) + Table 3 (Diebold-Mariano)
# ---------------------------------------------------------------------------
preds, rows_out = {}, []
for name in models:
    p = predict(name, Xte, test)
    preds[name] = p
    rows_out.append({"model": name, "R2_oos(%)": r2_oos(yte, p)})
table1 = pd.DataFrame(rows_out).sort_values("R2_oos(%)", ascending=False)
table1.to_csv(f"{RESULTS}/r2_oos_table.csv", index=False)
print("\n=== Table 1 : R2_oos (%) ===")
print(table1.to_string(index=False))

names = list(models.keys())
dm_mat = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        if a == b:
            dm_mat.loc[a, b] = np.nan; continue
        stat, _ = diebold_mariano(yte - preds[a], yte - preds[b], test["date"].values)
        dm_mat.loc[a, b] = stat
dm_mat.to_csv(f"{RESULTS}/diebold_mariano.csv")

# ---------------------------------------------------------------------------
# 6) 10-1 롱숏 포트폴리오 (Table 7) + GitHub용 시각자료 PNG 2종
# ---------------------------------------------------------------------------
best_model = table1.iloc[0]["model"]
test = test.copy()
test["pred"] = preds[best_model]
test["decile"] = test.groupby("date")["pred"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop"))
decile_ret = test.groupby(["date", "decile"])["ret"].mean().unstack()
hl = decile_ret[decile_ret.columns.max()] - decile_ret[decile_ret.columns.min()]
sharpe = hl.mean() / hl.std() * np.sqrt(12)
decile_ret.mean().to_frame("avg_ret(%)").to_csv(f"{RESULTS}/decile_portfolios.csv")
print(f"\n최고 성능 모델: {best_model} | 10-1 롱숏 연율화 Sharpe: {sharpe:.2f}")

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#c0392b" if "NN" in n else "#2980b9" if n in ("RF", "GBRT+H") else "#7f8c8d"
          for n in table1["model"]]
ax.bar(table1["model"], table1["R2_oos(%)"], color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Out-of-sample R² (%)")
ax.set_title("Gu, Kelly & Xiu (2020) 재현 — 모델별 예측 정확도 (Table 1)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{RESULTS}/fig1_r2_oos_by_model.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(9, 5))
cum_top = (1 + decile_ret[decile_ret.columns.max()]).cumprod()
cum_bot = (1 + decile_ret[decile_ret.columns.min()]).cumprod()
ax.plot(cum_top.values, label="Decile 10 (매수)", color="#27ae60")
ax.plot(cum_bot.values, label="Decile 1 (매도)", color="#c0392b")
ax.set_title(f"{best_model} 기반 10-1 롱숏 포트폴리오 누적수익률 (Figure 9)")
ax.set_xlabel("Out-of-sample month"); ax.set_ylabel("누적 성장 (1=시작시점)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS}/fig2_decile_portfolio.png", dpi=150)
plt.close()

print(f"\n결과 저장 완료: {os.path.abspath(RESULTS)}/ (csv 3개 + png 2개)")
