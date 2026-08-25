#!/usr/bin/env python3
"""note第8回「長い文章を読ませたら、ある長さから隣を読まなくなった」用の画像を作る（1200x675）。

宛先はプログラミング未経験の完全初心者。数式・コード・専門用語は画像にも出さない。
数値は articles/rope-long-context-breakdown.md のコードブロックをそのまま実行して得る
（別の測り方を書き写さないので、記事と図がずれない）。

  .venv/bin/python scripts/make_note_rope.py
"""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401

BG, FG, ACCENT, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
NG = "#ef4444"
ART = "/Users/mitsu/SideWork/articles/rope-long-context-breakdown.md"
OUT = "/Users/mitsu/SideWork/note/rope/"
os.makedirs(OUT, exist_ok=True)


def new(w=12.0, h=6.75):
    fig = plt.figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


def save(fig, name):
    fig.savefig(OUT + name, facecolor=BG, dpi=100)
    plt.close(fig)
    print("saved", name)


def panel(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.tick_params(colors=SUB, labelsize=12)
    return ax


# ---------- 記事のコードをそのまま実行して実測値を取る ----------
blocks = [m.group(2) for m in re.finditer(r"```(\S*)\n(.*?)```", open(ART).read(), re.S)
          if m.group(1) == "python"]
g = {}
_show = plt.show
plt.show = lambda *a, **k: plt.close(plt.gcf())     # 記事側の図は捨てる
for b in blocks[:5]:                                # 注意の重みを測るブロックまで
    exec(b, g)
plt.show = _show

PROBE, local, theta = g["PROBE"], g["local"], g["theta"]
nll, TRAIN, WINDOW = g["nll"], g["TRAIN"], g["WINDOW"]
i4096 = list(PROBE).index(4096)
i6144 = list(PROBE).index(6144)
deg = TRAIN * float(theta[31]) * 180 / np.pi

# ---------- 01 崖 ----------
fig = new()
ax = panel(fig, [.10, .17, .84, .60])
W = 64
pos = np.arange(0, len(nll) - W, W)
ax.plot(pos, [np.exp(nll[p:p + W].mean()) for p in pos], lw=2.2, color=ACCENT)
ax.axvline(TRAIN, color=NG, ls="--", lw=2.2)
ax.set_yscale("log")
ax.text(TRAIN * 1.04, 48, "AIが練習で読んだ最大の長さ", color=NG, fontsize=14)
ax.set_xlabel("読んだ場所", fontsize=14, color=FG)
ax.set_ylabel("次の言葉の当てにくさ", fontsize=14, color=FG)
ax.grid(alpha=.15)
fig.text(.5, .90, "ある長さを境に、急に当てられなくなる",
         ha="center", fontsize=22, color=FG)
fig.text(.5, .045, "だんだん悪くなるのではなく、崖のように落ちる",
         ha="center", fontsize=14, color=SUB)
save(fig, "01_cliff.png")

# ---------- 02 針の向きで順番を覚える ----------
fig = new()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG)
ax.set_xlim(-3.9, 3.9)
ax.set_ylim(-2.2, 2.2)
ax.set_aspect("equal")
ax.axis("off")
t = np.linspace(0, 2 * np.pi, 300)
for cx, ang, lab in [(-2.5, 0, "1番目の言葉"), (0, 1.0, "2番目の言葉"), (2.5, 2.0, "3番目の言葉")]:
    ax.plot(cx + np.cos(t), np.sin(t), color="#3a3a3a", lw=2)
    ax.annotate("", xy=(cx + np.cos(ang), np.sin(ang)), xytext=(cx, 0),
                arrowprops=dict(color=ACCENT, width=3.5, headwidth=14, headlength=14))
    ax.text(cx, -1.45, lab, ha="center", fontsize=15, color=SUB)
fig.text(.5, .90, "AIは「何番目か」を、針の向きで表している",
         ha="center", fontsize=22, color=FG)
fig.text(.5, .055, "2つの言葉がどれだけ離れているかは、向きの差になる",
         ha="center", fontsize=14, color=SUB)
save(fig, "02_clock.png")

# ---------- 03 速い針と遅い針 ----------
fig = new()
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG)
ax.set_xlim(-3.4, 3.4)
ax.set_ylim(-1.9, 1.9)
ax.set_aspect("equal")
ax.axis("off")
for cx, frac, col, lab, note in [
        (-1.7, 1.0, ACCENT, "一番速い針", "練習中に何百周も回った"),
        (1.7, deg / 360, NG, "一番遅い針", f"練習中に{deg:.0f}度しか回っていない")]:
    ax.plot(cx + np.cos(t), np.sin(t), color="#3a3a3a", lw=2)
    s = np.linspace(0, 2 * np.pi * frac, 300)
    ax.plot(cx + np.cos(s), np.sin(s), color=col, lw=9, solid_capstyle="butt")
    ax.text(cx, 1.32, lab, ha="center", fontsize=17, color=FG)
    ax.text(cx, -1.55, note, ha="center", fontsize=14, color=col)
fig.text(.5, .91, "針は32本あって、速さがまるで違う",
         ha="center", fontsize=22, color=FG)
fig.text(.5, .045, "遅い針にとっては、練習で見た向きの外がまるごと未知",
         ha="center", fontsize=14, color=SUB)
save(fig, "03_slow_hand.png")

# ---------- 04 隣を読まなくなる ----------
fig = new()
ax = panel(fig, [.10, .17, .84, .60])
ax.plot(PROBE, local * 100, lw=2.4, color=ACCENT)
ax.axvline(TRAIN, color=NG, ls="--", lw=2.2)
ax.text(TRAIN * 1.03, 46, "練習で読んだ最大の長さ", color=NG, fontsize=14)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.set_xlabel("読んだ場所", fontsize=14, color=FG)
ax.set_ylabel("すぐ隣の言葉を見ている割合", fontsize=14, color=FG)
ax.grid(alpha=.15)
fig.text(.5, .90, "そこを超えると、すぐ隣を読まなくなる",
         ha="center", fontsize=22, color=FG)
fig.text(.5, .045,
         f"手前では{local[i4096] * 100:.0f}%。超えたあとはほぼ0%",
         ha="center", fontsize=14, color=SUB)
save(fig, "04_neighbor.png")

print(f"\n実測値: 手前 {local[i4096] * 100:.1f}% / 超えたあと {local[i6144] * 100:.3f}% "
      f"/ 遅い針 {deg:.1f}度")
