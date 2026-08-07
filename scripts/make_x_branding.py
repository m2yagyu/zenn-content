"""X（旧Twitter）用のアイコンとヘッダー画像を生成する。

配色・フォントは instagram/ のスライドに合わせてある。
  背景 #1c1c1c / 文字 #f2f2f2 / 補助 #8a8a8a / アクセント #3b82f6

    .venv/bin/python scripts/make_x_branding.py

出力は images/x/ 以下。
  icon_a_blue.png    400x400  青地に白いガウス分布
  icon_b_dark.png    400x400  暗地に青いガウス分布（スライドと同じ配色）
  icon_c_moons.png   400x400  拡散の終端（構造が戻った状態）の散布図
  header.png        1500x500  ノイズ→構造の3コマ + テキスト
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

BG = "#1c1c1c"
FG = "#f2f2f2"
GRAY = "#8a8a8a"
BLUE = "#3b82f6"

plt.rcParams["font.family"] = ["Hiragino Sans", "Helvetica Neue", "sans-serif"]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "x")
os.makedirs(OUT, exist_ok=True)


def forward_diffuse(x0, alpha_bar, rng):
    """DDPMの前向き過程 q(x_t | x_0):  x_t = sqrt(a_bar) x_0 + sqrt(1 - a_bar) eps.

    alpha_bar が 1 に近いほど元の構造（x_0）が残り、0 に近いほど純ノイズになる。
    """
    eps = rng.normal(size=x0.shape)
    return np.sqrt(alpha_bar) * x0 + np.sqrt(1.0 - alpha_bar) * eps


def moons(n=700, seed=0):
    """二日月データを原点まわりに正規化して x_0 とする。"""
    x, y = make_moons(n_samples=n, noise=0.045, random_state=seed)
    x = x - x.mean(axis=0)
    x = x / x.std(axis=0).max()
    return x, y


# ---------------------------------------------------------------- アイコン
def gaussian_icon(path, bg, curve, edge=None):
    """ガウス分布（= 拡散の終端、ボルツマン分布の指数の形）を単純化した紋章。

    Xのアイコンは円形に切り抜かれ、タイムラインでは40px程度でしか表示されない。
    細部は捨てて、シルエットだけで読める形にしてある。
    """
    fig = plt.figure(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg)

    x = np.linspace(-3.4, 3.4, 600)
    y = np.exp(-(x ** 2) / 2.0)

    base = 0.30          # 円に切り抜かれても見切れない高さに山を置く
    height = 0.42
    ax.fill_between(x, base, base + height * y, color=curve, zorder=2)
    if edge is not None:
        ax.plot(x, base + height * y, color=edge, lw=3.0, zorder=3)
    ax.plot([-3.4, 3.4], [base, base], color=curve, lw=3.0, zorder=2)

    # 山の上の散らばり: ノイズから分布が立ち上がる含み
    rng = np.random.default_rng(3)
    for _ in range(26):
        px = rng.normal(0, 1.35)
        if abs(px) > 3.0:
            continue
        py = base + height * np.exp(-(px ** 2) / 2.0)
        ax.scatter(px, py + rng.uniform(0.04, 0.22), s=16,
                   color=curve, alpha=0.45, zorder=1, linewidths=0)

    ax.set_xlim(-3.9, 3.9)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.savefig(path, facecolor=bg)
    plt.close(fig)
    print("wrote", path)


def moons_icon(path):
    """拡散を逆にたどりきった状態（構造が戻った x_0 付近）そのものをアイコンにする。"""
    fig = plt.figure(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)

    rng = np.random.default_rng(0)
    x0, lab = moons(n=900, seed=1)
    xt = forward_diffuse(x0, 0.97, rng)

    ax.scatter(xt[lab == 0, 0], xt[lab == 0, 1], s=22, color=BLUE, alpha=0.95, linewidths=0)
    ax.scatter(xt[lab == 1, 0], xt[lab == 1, 1], s=22, color=FG, alpha=0.95, linewidths=0)

    # 円形に切り抜かれても両端の月が欠けないよう、内接円の内側に収まる範囲まで引く
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.25, 2.25)
    ax.axis("off")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- ヘッダー
def header(path):
    """1500x500。左下はプロフィール画像が重なるので、そこには何も置かない。"""
    W, H = 1500, 500
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)

    # --- テキスト（左）。アイコンの重なりを避けて x=330px から始める
    fig.text(330 / W, 312 / H, "物理で読み解く生成AI",
             color=FG, fontsize=34, va="center", ha="left")
    fig.text(330 / W, 242 / H, "拡散モデル・LLM・統計力学を、動くコードで。",
             color=GRAY, fontsize=16, va="center", ha="left")
    fig.text(330 / W, 190 / H, "zenn.dev/m2yagyu",
             color=BLUE, fontsize=16, va="center", ha="left")

    # --- 図（右）: 前向き過程を逆順に並べて「ノイズ→構造」に見せる
    rng = np.random.default_rng(0)
    x0, lab = moons(n=600, seed=1)
    alpha_bars = [0.02, 0.60, 0.97]      # 左からノイズ寄り→構造寄り
    labels = ["ノイズ", "", "構造"]

    panel = 150
    gap = 20
    left0 = W - 55 - (panel * 3 + gap * 2)
    bottom = (H - panel) / 2 + 8

    for i, ab in enumerate(alpha_bars):
        left = left0 + i * (panel + gap)
        ax = fig.add_axes([left / W, bottom / H, panel / W, panel / H])
        ax.set_facecolor("#232323")
        xt = forward_diffuse(x0, ab, rng)
        ax.scatter(xt[lab == 0, 0], xt[lab == 0, 1], s=7, color=BLUE, alpha=0.9, linewidths=0)
        ax.scatter(xt[lab == 1, 0], xt[lab == 1, 1], s=7, color=FG, alpha=0.9, linewidths=0)
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-2.6, 2.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#3a3a3a")
        if labels[i]:
            fig.text((left + panel / 2) / W, (bottom - 26) / H, labels[i],
                     color=GRAY, fontsize=14, ha="center", va="center")
        if i < 2:
            fig.text((left + panel + gap / 2) / W, (bottom + panel / 2) / H, "▶",
                     color="#4a4a4a", fontsize=11, ha="center", va="center")

    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    gaussian_icon(os.path.join(OUT, "icon_a_blue.png"), bg=BLUE, curve="#ffffff")
    gaussian_icon(os.path.join(OUT, "icon_b_dark.png"), bg=BG, curve=BLUE, edge=None)
    moons_icon(os.path.join(OUT, "icon_c_moons.png"))
    header(os.path.join(OUT, "header.png"))
