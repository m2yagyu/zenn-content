#!/usr/bin/env python3
"""note第1回「AIを疑っていた私が…」用の画像を作る（1200x675）。

宛先はプログラミング未経験の完全初心者。数式・コード・専門用語は画像にも出さない。
棒グラフの数値は articles/llm-temperature-boltzmann.md の検証済みコードと同じ計算。

  .venv/bin/python scripts/make_note_temperature.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
from matplotlib.patches import FancyBboxPatch

BG, FG, ACCENT, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
WARM = "#ef7d54"
OUT = "/Users/mitsu/SideWork/note/temperature/"
os.makedirs(OUT, exist_ok=True)


def new(w=12.0, h=6.75):
    fig = plt.figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


def save(fig, name):
    fig.savefig(OUT + name, facecolor=BG, dpi=100, bbox_inches=None)
    plt.close(fig)
    print("saved:", name)


def bubble(fig, x, y, w, h, color):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.025",
        transform=fig.transFigure, facecolor="none",
        edgecolor=color, linewidth=2.2, zorder=0))


# ---------- 記事と同じ計算 ----------
logits = np.array([2.1, 1.8, 0.3, -0.5])
labels = ["東京", "大阪", "京都", "札幌"]


def boltzmann(logits, T):
    z = np.exp(logits / T)
    return z / z.sum()


for T in (0.1, 1.0, 2.0):
    print(f"T={T}: " + " ".join(f"{l}{v*100:.0f}%" for l, v in zip(labels, boltzmann(logits, T))))

# ============ 1. 同じ質問、ちがう答え ============
fig = new()
fig.text(.5, .90, "同じ質問なのに、答えがちがう", color=FG, fontsize=36, ha="center", va="top")
fig.text(.5, .795, "私が最初に「信用できない」と思ったこと", color=SUB, fontsize=20, ha="center", va="top")

fig.text(.5, .655, "「この文章、直したほうがいいですか？」", color=FG, fontsize=25, ha="center")
for i, (txt, col) in enumerate((("1回目 　直したほうがいいです", ACCENT),
                                ("2回目 　このままで問題ありません", WARM),
                                ("3回目 　直したほうがいいです", ACCENT))):
    y = .47 - i * .135
    bubble(fig, .175, y - .035, .65, .095, col)
    fig.text(.205, y + .012, txt, color=col, fontsize=23, va="center")
fig.text(.5, .075, "道具なのに、毎回ちがう。これが引っかかっていました",
         color=SUB, fontsize=19, ha="center")
save(fig, "01_doubt.png")

# ============ 2. 温度を変えたときの選ばれやすさ ============
fig = new()
fig.text(.5, .935, "温度を変えると、選ばれやすさが変わる", color=FG, fontsize=34,
         ha="center", va="top")
fig.text(.5, .845, "「日本の首都は」の次に来る言葉の候補", color=SUB, fontsize=19,
         ha="center", va="top")

for j, (T, lab, col) in enumerate(((0.1, "温度を下げる", ACCENT),
                                   (1.0, "そのまま", "#9aa4b2"),
                                   (2.0, "温度を上げる", WARM))):
    ax = fig.add_axes([.075 + j * .308, .195, .245, .50])
    ax.set_facecolor(BG)
    p = boltzmann(logits, T) * 100
    ax.bar(range(4), p, color=col, width=.62)
    for i, v in enumerate(p):
        ax.text(i, v + 3.5, f"{v:.0f}%", ha="center", color=FG, fontsize=16)
    ax.set(ylim=(0, 118), xticks=range(4), yticks=[])
    ax.set_xticklabels(labels, fontsize=17)
    ax.tick_params(colors=FG, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(lab, color=col, fontsize=22, pad=16)

fig.text(.5, .075, "下げると東京でほぼ確定。上げると大阪や京都も出てくる",
         color=SUB, fontsize=19, ha="center")
save(fig, "02_bars.png")

# ============ 3. 冷たい水とお湯 ============
fig = new()
fig.text(.5, .935, "温度が高いと、散らばって決まらなくなる", color=FG, fontsize=34,
         ha="center", va="top")
fig.text(.5, .845, "水の中の、目に見えない小さな粒のようす", color=SUB, fontsize=19,
         ha="center", va="top")

rng = np.random.default_rng(3)
for j, (spread, lab, col) in enumerate(((0.30, "冷たい水", ACCENT),
                                        (1.15, "お　湯", WARM))):
    ax = fig.add_axes([.115 + j * .45, .195, .33, .50])
    ax.set_facecolor(BG)
    pts = rng.normal(0, spread, size=(160, 2))
    ax.scatter(pts[:, 0], pts[:, 1], s=34, color=col, alpha=.72)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color("#3a3a3a")
    ax.set_title(lab, color=col, fontsize=24, pad=16)
    ax.text(.5, -.13, "まとまっている" if j == 0 else "散らばっている",
            transform=ax.transAxes, ha="center", color=col, fontsize=20)

fig.text(.5, .045, "AIの「温度」も、これとまったく同じ仕組みで動いています",
         color=SUB, fontsize=19, ha="center")
save(fig, "03_water.png")
print("完了:", OUT)
