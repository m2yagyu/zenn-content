#!/usr/bin/env python3
"""X投稿用のカード画像を作る（1600x900）。
数値は記事のコードブロックと同じ seed・同じ計算で出したものを使う。

  .venv/bin/python scripts/make_x_card_attention.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401

BG, FG, ACCENT, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
WARN = "#ef7d54"
OUT = "/Users/mitsu/SideWork/images/x/card_attention-hopfield.png"


def softmax(z):
    z = z - z.max()
    return np.exp(z) / np.exp(z).sum()


# ---- 記事の3つ目のコードブロックと同じ計算（seed 2）----
rng = np.random.default_rng(2)
dims = [16, 64, 256, 1024, 4096]
with_s, without_s = [], []
for dd in dims:
    rng.normal(size=(3000, dd)), rng.normal(size=(3000, dd))   # 記事と乱数列を揃える
    a_, b_ = [], []
    for _ in range(200):
        K = rng.normal(size=(16, dd))
        sc = K @ rng.normal(size=dd)
        a_.append(softmax(sc / np.sqrt(dd)).max())
        b_.append(softmax(sc).max())
    with_s.append(np.mean(a_))
    without_s.append(np.mean(b_))
print("÷√d 有:", [round(v, 3) for v in with_s])
print("÷√d 無:", [round(v, 3) for v in without_s])

fig = plt.figure(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.09, 0.20, 0.55, 0.50])
ax.set_facecolor(BG)

ax.semilogx(dims, without_s, "o-", color=WARN, lw=3.5, ms=11, label="√d で割らない")
ax.semilogx(dims, with_s, "o-", color=ACCENT, lw=3.5, ms=11, label="√d で割る")
ax.axhline(1 / 16, color=SUB, ls=":", lw=1.5)
ax.text(17, 1 / 16 + 0.035, "16本に均等に注目した場合", color=SUB, fontsize=13)

ax.set_ylim(0, 1.08)
ax.set_xlabel("ヘッドあたりの次元 d", color=FG, fontsize=16, labelpad=10)
ax.set_ylabel("最大の注意重み", color=FG, fontsize=16, labelpad=10)
ax.tick_params(colors=FG, labelsize=14)
for s in ax.spines.values():
    s.set_color("#3a3a3a")
ax.grid(alpha=0.15, color=FG)
leg = ax.legend(fontsize=15, facecolor=BG, edgecolor="#3a3a3a", loc="center right")
for t in leg.get_texts():
    t.set_color(FG)

# ---- 見出し ----
fig.text(0.06, 0.90, "√d で割らないと、次元を上げただけで", color=FG, fontsize=34, weight="bold")
fig.text(0.06, 0.825, "注意が1点に凍りつく", color=FG, fontsize=34, weight="bold")
fig.text(0.06, 0.765, "キー16本に対する最大の注意重み（200回の平均）", color=SUB, fontsize=17)

# ---- 右側の数値 ----
fig.text(0.70, 0.66, "d = 4096 では", color=SUB, fontsize=18)
fig.text(0.70, 0.545, f"{without_s[-1]:.3f}", color=WARN, fontsize=64, weight="bold")
fig.text(0.70, 0.495, "割らない → 実質1本しか見ない", color=SUB, fontsize=15)
fig.text(0.70, 0.345, f"{with_s[-1]:.3f}", color=ACCENT, fontsize=64, weight="bold")
fig.text(0.70, 0.295, "割る → d を変えても動かない", color=SUB, fontsize=15)

fig.text(0.06, 0.075, "この √d は、統計力学の温度そのものだった", color=FG, fontsize=20)
fig.text(0.06, 0.028, "zenn.dev/m2yagyu", color=SUB, fontsize=15)

fig.savefig(OUT, facecolor=BG, dpi=100)
print("保存:", OUT)
