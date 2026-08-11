#!/usr/bin/env python3
"""Instagram用スライド(1080x1080)を作る。宛先はプログラミング未経験の完全初心者。

持ち帰ってもらう理解は3つだけ:
  ① AIは前に出てきた言葉を「思い出して」次の言葉を決めている
  ② 思い出し方には「ひとつをはっきり / たくさんをぼんやり」の幅がある
  ③ その幅を決めているのは、たとえ話ではなく本当に物理の「温度」

数式・コード・専門用語は出さない。棒グラフと地形図は記事と同じ計算の実データ。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
from matplotlib.patches import FancyBboxPatch

BG, FG, ACCENT, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
WARM = "#ef7d54"
OUT = "/Users/mitsu/SideWork/instagram/attention-hopfield-associative-memory/"
PX = 1080


def new_slide():
    fig = plt.figure(figsize=(PX / 100, PX / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


def box(fig, x, y, w, h, color=None, lw=2, alpha=1.0):
    """角丸の枠線ボックス"""
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=fig.transFigure, facecolor="none",
        edgecolor=color or "#3a3a3a", linewidth=lw, alpha=alpha, zorder=0))


def label(fig, x, y, text, size, color=FG, weight="normal", ha="left", va="baseline"):
    fig.text(x, y, text, fontsize=size, color=color, weight=weight, ha=ha, va=va)


def save(fig, name):
    fig.savefig(OUT + name, facecolor=BG, dpi=100)
    plt.close(fig)
    print("保存:", name)


def softmax(z):
    z = z - z.max()
    return np.exp(z) / np.exp(z).sum()


def make_cue(x, sigma, rng):
    n = rng.normal(size=x.shape)
    n /= np.linalg.norm(n)
    c = x + sigma * n
    return c / np.linalg.norm(c)


# =============================== 1枚目: タイトル
fig = new_slide()
box(fig, 0.07, 0.07, 0.86, 0.86)
label(fig, 0.5, 0.80, "AIは、前に出てきた言葉を", 40, ha="center")
label(fig, 0.5, 0.715, "「思い出して」いる", 46, ACCENT, "bold", ha="center")
fig.patches.append(plt.Rectangle((0.36, 0.665), 0.28, 0.006,
                                 transform=fig.transFigure, color=ACCENT, zorder=1))
label(fig, 0.5, 0.565, "しかも、その思い出し方を決めているのは", 25, ha="center")
label(fig, 0.5, 0.495, "物理の「温度」でした", 33, ha="center")
label(fig, 0.5, 0.395, "たとえ話ではありません。", 22, SUB, ha="center")
label(fig, 0.5, 0.345, "本当に同じ仕組みで動いています。", 22, SUB, ha="center")
label(fig, 0.5, 0.185, "プログラミングを知らなくても大丈夫", 21, SUB, ha="center")
label(fig, 0.5, 0.135, "この5枚で分かります", 21, SUB, ha="center")
save(fig, "slide1_title.png")

# =============================== 2枚目: ①思い出している
fig = new_slide()
label(fig, 0.09, 0.90, "①", 46, ACCENT, "bold")
label(fig, 0.20, 0.905, "AIは前の言葉を探しに行く", 34, weight="bold")

box(fig, 0.09, 0.545, 0.82, 0.235, color=ACCENT, lw=2)
label(fig, 0.5, 0.715, "日本の首都は東京です。", 27, ha="center")
label(fig, 0.5, 0.655, "日本で一番高い山は富士山です。", 27, ha="center")
label(fig, 0.5, 0.588, "日本の首都は  ___", 30, ACCENT, "bold", ha="center")

label(fig, 0.5, 0.470, "↓  次に来る言葉を決めるとき", 22, SUB, ha="center")

label(fig, 0.09, 0.375, "AIは前に出てきた言葉を見渡して、", 26)
label(fig, 0.09, 0.315, "いま関係のあるものを探します。", 26)
label(fig, 0.09, 0.225, "この文なら", 26)
label(fig, 0.265, 0.225, "「東京」", 30, ACCENT, "bold")
label(fig, 0.43, 0.225, "を思い出して答える。", 26)
label(fig, 0.09, 0.125, "人が記憶をたどるのと、よく似た働きです。", 23, SUB)
save(fig, "slide2_recall.png")

# =============================== 3枚目: ②思い出し方に幅がある（実データ）
rng = np.random.default_rng(0)
d, N = 64, 6
X = rng.normal(size=(d, N))
X /= np.linalg.norm(X, axis=0, keepdims=True)
target = 3
xi = make_cue(X[:, target], 0.5, rng)
w_hot = softmax(0.5 * (X.T @ xi))     # ぼんやり
w_cold = softmax(8.0 * (X.T @ xi))    # はっきり

fig = new_slide()
label(fig, 0.09, 0.90, "②", 46, ACCENT, "bold")
label(fig, 0.20, 0.905, "思い出し方には「幅」がある", 34, weight="bold")

for i, (ww, ttl, col) in enumerate([(w_hot, "ぼんやり全部を混ぜる", WARM),
                                    (w_cold, "ひとつをはっきり", ACCENT)]):
    ax = fig.add_axes([0.10 + i * 0.46, 0.335, 0.36, 0.36])
    ax.set_facecolor(BG)
    ax.bar(range(N), ww, color=col, width=0.72)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=BG)
    fig.text(0.28 + i * 0.46, 0.725, ttl, fontsize=23, color=col,
             ha="center", weight="bold")
    fig.text(0.28 + i * 0.46, 0.278, "候補の記憶", fontsize=17, color=SUB, ha="center")

label(fig, 0.09, 0.185, "棒の高さが「どれくらい思い出したか」。", 25)
label(fig, 0.09, 0.120, "AIの中には、この幅を切り替える", 25)
label(fig, 0.09, 0.062, "「つまみ」があります。", 25)
save(fig, "slide3_range.png")

# =============================== 4枚目: ③正体は温度（実データの地形）
P = np.array([[-1.2, -0.9], [1.3, -0.6], [0.1, 1.3]]).T
g = np.linspace(-2.2, 2.2, 260)
GX, GY = np.meshgrid(g, g)
Z = np.stack([GX.ravel(), GY.ravel()])

fig = new_slide()
label(fig, 0.09, 0.90, "③", 46, ACCENT, "bold")
label(fig, 0.20, 0.905, "そのつまみは、本当に「温度」", 34, weight="bold")

for i, (b, ttl, col) in enumerate([(1.0, "熱いとき", WARM), (20.0, "冷たいとき", ACCENT)]):
    s = b * (P.T @ Z)
    lse = (np.log(np.exp(s - s.max(0)).sum(0)) + s.max(0)) / b
    E = (-lse + 0.5 * (Z ** 2).sum(0)).reshape(GX.shape)
    ax = fig.add_axes([0.09 + i * 0.455, 0.435, 0.395, 0.29])
    ax.contourf(GX, GY, E, levels=26, cmap="Blues_r")
    ax.scatter(P[0], P[1], c=WARM, s=90, edgecolor="white", zorder=5)
    ax.set_xticks([]), ax.set_yticks([])
    fig.text(0.2875 + i * 0.455, 0.755, ttl, fontsize=25, color=col,
             ha="center", weight="bold")
fig.text(0.2875, 0.395, "3つの記憶がひと山に融ける", fontsize=17, color=SUB, ha="center")
fig.text(0.7425, 0.395, "ひとつずつの谷に分かれる", fontsize=17, color=SUB, ha="center")

label(fig, 0.09, 0.300, "熱いと混ざり、冷たいとはっきり決まる。", 25)
label(fig, 0.09, 0.235, "お湯の中で分子が動き回るのと", 25)
label(fig, 0.09, 0.175, "同じ言葉で説明できます。", 25)
box(fig, 0.09, 0.055, 0.82, 0.075, color=ACCENT, lw=2)
label(fig, 0.5, 0.088, "似ているのではなく、同じ仕組みです", 25, ACCENT, "bold", ha="center")
save(fig, "slide4_temperature.png")

# =============================== 5枚目: CTA
fig = new_slide()
box(fig, 0.07, 0.07, 0.86, 0.86)
label(fig, 0.5, 0.815, "この仕組みを考えたのは", 27, ha="center")
label(fig, 0.5, 0.735, "AIの研究者ではなく", 27, ha="center")
label(fig, 0.5, 0.640, "物理学者でした", 40, ACCENT, "bold", ha="center")
fig.patches.append(plt.Rectangle((0.36, 0.590), 0.28, 0.006,
                                 transform=fig.transFigure, color=ACCENT, zorder=1))
label(fig, 0.5, 0.505, "1982年に「記憶のしくみ」として作られた考えが、", 21, ha="center")
label(fig, 0.5, 0.450, "40年後のいま、AIの中で動いています。", 21, ha="center")
label(fig, 0.5, 0.350, "2024年のノーベル物理学賞", 26, WARM, "bold", ha="center")
label(fig, 0.5, 0.205, "もっと詳しく知りたい人は", 23, SUB, ha="center")
label(fig, 0.5, 0.150, "プロフィールのリンクから", 23, SUB, ha="center")
save(fig, "slide5_cta.png")

print("\n実データの確認:")
print(f"  ぼんやり(左)の最大 {w_hot.max():.3f} / はっきり(右)の最大 {w_cold.max():.3f}")
