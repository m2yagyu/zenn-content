#!/usr/bin/env python3
"""記事の図を生成する。
記事本文のコードブロックと同じseed・同じ計算を使い、本文の数値と図を一致させる。"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 150
BLUE, ORANGE, GRAY = "#3b82f6", "#ef7d54", "#8a8a8a"
OUT = "/Users/mitsu/SideWork/images/"

def softmax(z):
    z = z - z.max()
    return np.exp(z) / np.exp(z).sum()

def make_cue(x, sigma, rng):
    n = rng.normal(size=x.shape)
    n /= np.linalg.norm(n)
    c = x + sigma * n
    return c / np.linalg.norm(c)

def eff(w):
    w = np.clip(w, 1e-30, 1)
    return float(np.exp(-(w * np.log(w)).sum()))

# ================================================= 記事ブロック[0] と同一 (seed 0)
rng = np.random.default_rng(0)
d, N = 64, 6
X = rng.normal(size=(d, N))
X /= np.linalg.norm(X, axis=0, keepdims=True)
target = 3
xi = make_cue(X[:, target], 0.5, rng)
beta = 100.0
w = softmax(beta * (X.T @ xi))
rec = X @ w
assert int(w.argmax()) == target

fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.2),
                       gridspec_kw={"width_ratios": [1.9, 2.2, 1.0, 2.2]})
ax[0].imshow(X.T, aspect="auto", cmap="RdBu_r", vmin=-.4, vmax=.4)
ax[0].set_title("① 6つの記憶を保存する", fontsize=11)
ax[0].set_ylabel("記憶の番号"); ax[0].set_xlabel("特徴の次元"); ax[0].grid(False)
ax[1].plot(X[:, target], color=BLUE, lw=1.6, label=f"本物の記憶{target}")
ax[1].plot(xi, color=GRAY, lw=1.1, label="ノイズを乗せた手がかり")
ax[1].set_title("② 崩れた手がかりを渡す", fontsize=11)
ax[1].legend(fontsize=8.5); ax[1].set_xlabel("特徴の次元")
ax[2].barh(np.arange(N), w, color=[ORANGE if i == target else BLUE for i in range(N)])
ax[2].set_title("③ 想起の重み", fontsize=11); ax[2].invert_yaxis()
ax[2].set_yticks(range(N)); ax[2].set_xlim(0, 1); ax[2].set_ylabel("記憶の番号")
ax[3].plot(X[:, target], label=f"本物の記憶{target}", color=BLUE, lw=2.4)
ax[3].plot(rec, label="想起した結果", color=ORANGE, lw=1.2, ls="--")
ax[3].set_title("④ 元の記憶が戻ってきた", fontsize=11)
ax[3].legend(fontsize=8.5); ax[3].set_xlabel("特徴の次元")
plt.tight_layout(); plt.savefig(OUT + "attention-hopfield-recall.png", bbox_inches="tight"); plt.close()
print(f"図1 記憶{target}を想起 重み={w.max():.3f}")

# ================================================= 記事ブロック[3] と同一 (X, xi を継承)
betas_show = [0.5, 2, 4, 8]
fig, ax = plt.subplots(1, 4, figsize=(13, 2.9), sharey=True)
for a, b in zip(ax, betas_show):
    ww = softmax(b * (X.T @ xi))
    a.bar(range(N), ww, color=[ORANGE if i == target else BLUE for i in range(N)])
    a.set_title(f"β={b:g}  (温度 T={1/b:.2f})\n有効記憶数 {eff(ww):.2f}", fontsize=10.5)
    a.set_ylim(0, 1); a.set_xlabel("記憶の番号")
ax[0].set_ylabel("想起の重み")
plt.tight_layout(); plt.savefig(OUT + "attention-hopfield-beta-bars.png", bbox_inches="tight"); plt.close()

bs = np.logspace(-1, 1.6, 80)
effs = [eff(softmax(b * (X.T @ xi))) for b in bs]
plt.figure(figsize=(6.2, 3.6))
plt.semilogx(bs, effs, color=BLUE, lw=2)
plt.axhline(N, color=GRAY, ls="--", lw=1); plt.text(0.11, N - .45, "全部を均等に混ぜた状態", fontsize=9, color=GRAY)
plt.axhline(1, color=GRAY, ls="--", lw=1); plt.text(0.11, 1.15, "1つの記憶に絞れた状態", fontsize=9, color=GRAY)
plt.xlabel("逆温度 β"); plt.ylabel("有効記憶数 exp(H)")
plt.title("温度を下げると記憶は1つに絞られる", fontsize=11)
plt.tight_layout(); plt.savefig(OUT + "attention-hopfield-beta-curve.png", bbox_inches="tight"); plt.close()
print(f"図3 β=0.5:{eff(softmax(0.5*(X.T@xi))):.2f} → β=8:{eff(softmax(8*(X.T@xi))):.2f}")

# ================================================= 記事ブロック[2] と同一 (seed 2)
rng = np.random.default_rng(2)
dims = [16, 64, 256, 1024, 4096]
std_raw, mx_with, mx_without = [], [], []
for dd in dims:
    q = rng.normal(size=(3000, dd)); k = rng.normal(size=(3000, dd))
    std_raw.append(float((q * k).sum(1).std()))
    a_, b_ = [], []
    for _ in range(200):
        K = rng.normal(size=(16, dd))
        sc = K @ rng.normal(size=dd)
        a_.append(softmax(sc / np.sqrt(dd)).max()); b_.append(softmax(sc).max())
    mx_with.append(np.mean(a_)); mx_without.append(np.mean(b_))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].loglog(dims, std_raw, "o-", color=ORANGE, label="Q·K をそのまま")
ax[0].loglog(dims, np.array(std_raw) / np.sqrt(dims), "o-", color=BLUE, label="√d で割った後")
ax[0].set_xlabel("ヘッドあたりの次元 d"); ax[0].set_ylabel("内積の標準偏差")
ax[0].set_title("内積の散らばりは d とともに育つ", fontsize=11); ax[0].legend(fontsize=9)
ax[1].semilogx(dims, mx_without, "o-", color=ORANGE, label="√d で割らない")
ax[1].semilogx(dims, mx_with, "o-", color=BLUE, label="√d で割る")
ax[1].axhline(1 / 16, color=GRAY, ls=":", lw=1)
ax[1].text(20, 1/16 + .03, "16本に均等に注目した場合", fontsize=8.5, color=GRAY)
ax[1].set_xlabel("ヘッドあたりの次元 d"); ax[1].set_ylabel("最大の注意重み(200回平均)")
ax[1].set_ylim(0, 1.05)
ax[1].set_title("割らないと注意が1点に寄っていく", fontsize=11); ax[1].legend(fontsize=9)
plt.tight_layout(); plt.savefig(OUT + "attention-hopfield-sqrtd.png", bbox_inches="tight"); plt.close()
print("図2 " + " / ".join(f"d={d_}:{mw:.3f}vs{mo:.3f}" for d_, mw, mo in zip(dims, mx_with, mx_without)))

# ================================================= 記事ブロック[4] と同一 (seed 4)
rng = np.random.default_rng(4)
d = 64
Ns = [8, 32, 128, 512, 2048, 8192]
acc = []
for NN in Ns:
    ok = 0
    for _ in range(400):
        Xr = rng.normal(size=(d, NN)); Xr /= np.linalg.norm(Xr, axis=0, keepdims=True)
        t = rng.integers(NN)
        q = make_cue(Xr[:, t], 2.0, rng)
        ok += int(softmax(100.0 * (Xr.T @ q)).argmax() == t)
    acc.append(ok / 400 * 100)
plt.figure(figsize=(6.8, 3.9))
plt.semilogx(Ns, acc, "o-", color=BLUE, lw=2, base=2, label="softmax版(=attention)")
plt.semilogx(Ns, [100 / n for n in Ns], "--", color=GRAY, lw=1.2, base=2, label="でたらめに選んだ場合")
plt.axvline(0.138 * d, color=ORANGE, ls="--", lw=1.5)
plt.text(0.138 * d * 1.15, 42, f"古典Hopfieldの容量\n0.138×d ≒ {0.138*d:.1f}個", fontsize=9, color=ORANGE)
plt.xlabel(f"詰め込んだ記憶の数 N  (特徴次元 d={d})"); plt.ylabel("正しく想起できた割合 [%]")
plt.title("記憶の2倍のノイズを乗せても、数千個から取り出せる", fontsize=11)
plt.ylim(0, 105); plt.legend(fontsize=9)
plt.tight_layout(); plt.savefig(OUT + "attention-hopfield-capacity.png", bbox_inches="tight"); plt.close()
print("図4 " + " / ".join(f"N={n}:{a:.1f}%" for n, a in zip(Ns, acc)))

# ================================================= 図5 エネルギー地形
gx = np.linspace(-2.2, 2.2, 300)
P = np.array([[-1.2, -0.9], [1.3, -0.6], [0.1, 1.3]]).T
GX, GY = np.meshgrid(gx, gx)
Z = np.stack([GX.ravel(), GY.ravel()])
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7))
for a, b in zip(ax, [1.0, 4.0, 20.0]):
    s = b * (P.T @ Z)
    lse = (np.log(np.exp(s - s.max(0)).sum(0)) + s.max(0)) / b
    E = (-lse + 0.5 * (Z ** 2).sum(0)).reshape(GX.shape)
    a.contourf(GX, GY, E, levels=28, cmap="Blues_r")
    a.contour(GX, GY, E, levels=14, colors="white", linewidths=.4, alpha=.6)
    a.scatter(P[0], P[1], c=ORANGE, s=70, edgecolor="white", zorder=5, label="記憶")
    a.set_title(f"β={b:g}  (温度 T={1/b:.2f})", fontsize=11); a.grid(False)
    a.set_xticks([]); a.set_yticks([])
ax[0].legend(fontsize=9, loc="upper left")
ax[0].set_ylabel("エネルギーの地形", fontsize=10)
fig.suptitle("温度を下げると、記憶のひとつひとつが別々の谷になる", fontsize=12, y=1.04)
plt.tight_layout(); plt.savefig(OUT + "attention-hopfield-energy.png", bbox_inches="tight"); plt.close()
print("図5 OK")
