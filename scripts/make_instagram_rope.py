"""rope-long-context-breakdown のInstagramスライドを作る。

宛先はZennと違い、プログラミング未経験の完全初心者。
数式・コード・専門用語を出さず、画像だけで理解が完結すること。
数値は記事のコードを実行して得た本物の実測値を使う（scratchpadのig.npz）。
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "Hiragino Sans"

BG, FG, ACC, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
OUT = "/Users/mitsu/SideWork/instagram/rope-long-context-breakdown"
d = np.load(sys.argv[1])
PROBE, local, theta, TRAIN = d["PROBE"], d["local"], d["theta"], int(d["TRAIN"])


def canvas():
    fig = plt.figure(figsize=(10.8, 10.8), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_facecolor(BG)
    return fig, ax


def box(ax, x, y, w, h, color=SUB, lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fill=False, edgecolor=color, lw=lw, transform=ax.transAxes))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", facecolor=BG)
    plt.close(fig)
    print("saved", name)


# 1枚目 ------------------------------------------------------------------
fig, ax = canvas()
ax.text(.5, .80, "AIに長い文章を貼ると", ha="center", fontsize=40, color=FG)
ax.text(.5, .72, "急に話が通じなくなる", ha="center", fontsize=40, color=FG)
ax.text(.5, .615, "あれ、なぜ？", ha="center", fontsize=46, color=ACC, weight="bold")
box(ax, .12, .34, .76, .17)
ax.text(.5, .455, "「量が多すぎて処理しきれない」", ha="center", fontsize=25, color=SUB)
ax.text(.5, .385, "ではありませんでした", ha="center", fontsize=25, color=FG)
ax.text(.5, .21, "AIの中を実際にのぞいて", ha="center", fontsize=25, color=SUB)
ax.text(.5, .15, "原因を突き止めた話", ha="center", fontsize=25, color=SUB)
save(fig, "slide1_title")

# 2枚目 ------------------------------------------------------------------
fig, ax = canvas()
ax.text(.5, .91, "AIは言葉の順番を", ha="center", fontsize=34, color=FG)
ax.text(.5, .845, "「針の向き」で覚えている", ha="center", fontsize=34, color=ACC, weight="bold")
sub = fig.add_axes([.10, .38, .80, .38]); sub.set_facecolor(BG); sub.axis("off")
sub.set_xlim(-3.6, 3.6); sub.set_ylim(-1.5, 1.5); sub.set_aspect("equal")
t = np.linspace(0, 2 * np.pi, 300)
for cx, ang, lab in [(-2.4, 0, "1番目の言葉"), (0, 1.1, "2番目の言葉"), (2.4, 2.2, "3番目の言葉")]:
    sub.plot(cx + np.cos(t), np.sin(t), color="#3a3a3a", lw=2)
    sub.annotate("", xy=(cx + np.cos(ang), np.sin(ang)), xytext=(cx, 0),
                 arrowprops=dict(color=ACC, width=4, headwidth=16, headlength=16))
    sub.text(cx, -1.42, lab, ha="center", fontsize=19, color=SUB)
ax.text(.5, .27, "何番目に出てきた言葉なのかを", ha="center", fontsize=25, color=FG)
ax.text(.5, .205, "針を少しずつ回して表している", ha="center", fontsize=25, color=FG)
ax.text(.5, .10, "針の向きの差が「どれだけ離れているか」になる",
        ha="center", fontsize=22, color=SUB)
save(fig, "slide2_clock")

# 3枚目 ------------------------------------------------------------------
deg = TRAIN * float(theta[31]) * 180 / np.pi
fig, ax = canvas()
ax.text(.5, .91, "針は1本ではなく32本ある", ha="center", fontsize=34, color=FG)
ax.text(.5, .845, "しかも速さがまるで違う", ha="center", fontsize=34, color=ACC, weight="bold")
sub = fig.add_axes([.12, .36, .76, .40]); sub.set_facecolor(BG); sub.axis("off")
sub.set_xlim(-2.6, 2.6); sub.set_ylim(-1.6, 1.4); sub.set_aspect("equal")
for cx, frac, col, lab in [(-1.3, 1.0, ACC, "速い針"), (1.3, deg / 360, "#d9534f", "いちばん遅い針")]:
    sub.plot(cx + np.cos(t), np.sin(t), color="#3a3a3a", lw=2)
    s = np.linspace(0, 2 * np.pi * frac, 300)
    sub.plot(cx + np.cos(s), np.sin(s), color=col, lw=11, solid_capstyle="butt")
    sub.text(cx, 1.28, lab, ha="center", fontsize=22, color=FG)
sub.text(-1.3, -1.5, "練習中に何百周も回った", ha="center", fontsize=19, color=SUB)
sub.text(1.3, -1.5, f"練習中に{deg:.0f}度しか回っていない", ha="center", fontsize=19, color="#d9534f")
box(ax, .10, .12, .80, .15)
ax.text(.5, .215, "遅い針にとっては、練習で見た向きの外は", ha="center", fontsize=23, color=FG)
ax.text(.5, .155, "生まれて初めて見る景色", ha="center", fontsize=23, color=ACC)
save(fig, "slide3_hands")

# 4枚目 ------------------------------------------------------------------
fig, ax = canvas()
ax.text(.5, .93, "練習した長さを超えた瞬間", ha="center", fontsize=34, color=FG)
ax.text(.5, .865, "AIは隣の言葉を見なくなる", ha="center", fontsize=34, color=ACC, weight="bold")
sub = fig.add_axes([.14, .40, .74, .38], facecolor=BG)
sub.plot(PROBE, local * 100, color=ACC, lw=3)
sub.axvline(TRAIN, color="#d9534f", ls="--", lw=2.5)
sub.text(TRAIN * 1.06, 46, "練習した長さ", color="#d9534f", fontsize=19)
sub.set_xlabel("読んだ文章の長さ", fontsize=19, color=FG)
sub.set_ylabel("すぐ隣の言葉を\n見ている割合", fontsize=19, color=FG)
sub.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
for sp in sub.spines.values():
    sp.set_color("#555")
sub.tick_params(colors=SUB, labelsize=15)
sub.set_facecolor(BG)
sub.grid(alpha=.15)
i4 = list(PROBE).index(4096); i6 = list(PROBE).index(6144)
ax.text(.5, .275, f"直前まで {local[i4] * 100:.0f}% だったのが", ha="center", fontsize=26, color=FG)
ax.text(.5, .205, "ほぼ 0% になる", ha="center", fontsize=30, color="#d9534f", weight="bold")
ax.text(.5, .10, "すぐ前に書いてあることを読まずに答えている",
        ha="center", fontsize=22, color=SUB)
save(fig, "slide4_collapse")

# 5枚目 ------------------------------------------------------------------
fig, ax = canvas()
ax.text(.5, .88, "つまり", ha="center", fontsize=30, color=SUB)
lines = ["AIは言葉の順番を針の向きで覚えている",
         "遅い針は練習中ほんの少ししか回っていない",
         "その先へ出ると、隣の言葉を見なくなる"]
for i, tx in enumerate(lines):
    y = .74 - i * .13
    box(ax, .08, y - .048, .84, .095, color="#3a3a3a", lw=1.4)
    ax.text(.125, y, f"{i + 1}", fontsize=27, color=ACC, weight="bold", va="center")
    ax.text(.195, y, tx, fontsize=23, color=FG, va="center")
box(ax, .10, .15, .80, .16, color=ACC)
ax.text(.5, .255, "もっと詳しく知りたい人は", ha="center", fontsize=24, color=FG)
ax.text(.5, .19, "プロフィールのリンクから", ha="center", fontsize=24, color=ACC)
save(fig, "slide5_cta")
