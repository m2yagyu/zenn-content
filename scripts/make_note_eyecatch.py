#!/usr/bin/env python3
"""note連載の見出し画像（1280x670）。第2回・第3回ぶん。

第1回 `eyecatch_temperature.png` と同じ配色・同じ骨格にする。
フィードではカードとして小さく出るので、文字は2行まで・太く・左寄せ、
右に顔つきのアイコンを1つだけ置く。本文の暗い図版とは別の役割。

  .venv/bin/python scripts/make_note_eyecatch.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
from matplotlib.patches import Circle, FancyBboxPatch, Arc

OUT = "/Users/mitsu/SideWork/note/brand/"
os.makedirs(OUT, exist_ok=True)

CREAM = "#fff3e2"
ORANGE = "#f4813f"
BLUE = "#3b82f6"
NAVY = "#2a3346"
PINK = "#ff9db0"
GRAY = "#7b8494"


def face(ax, cx, cy, s=1.0, col=NAVY):
    ax.add_patch(Circle((cx - .16 * s, cy + .05 * s), .052 * s, color=col, zorder=6))
    ax.add_patch(Circle((cx + .16 * s, cy + .05 * s), .052 * s, color=col, zorder=6))
    ax.add_patch(Arc((cx, cy - .04 * s), .30 * s, .24 * s, theta1=200, theta2=340,
                     color=col, lw=5.5 * s, zorder=6, capstyle="round"))
    ax.add_patch(Circle((cx - .30 * s, cy - .04 * s), .062 * s, color=PINK, alpha=.85, zorder=5))
    ax.add_patch(Circle((cx + .30 * s, cy - .04 * s), .062 * s, color=PINK, alpha=.85, zorder=5))


def layout(l1, l2, sub):
    """第1回と同じ組み方（左に文字、右にアイコン用の枠を返す）"""
    fig = plt.figure(figsize=(12.8, 6.7), dpi=100)
    fig.patch.set_facecolor(CREAM)
    fig.text(.065, .80, l1, color=NAVY, fontsize=54, va="center")
    fig.text(.065, .635, l2, color=ORANGE, fontsize=54, va="center")
    fig.add_artist(plt.Line2D([.065, .30], [.525, .525], color=ORANGE, lw=4))
    fig.text(.065, .42, sub, color=NAVY, fontsize=27, va="center")
    fig.text(.065, .175, "理系大学院で学んだ私が、AIの中身を覗いてみた記録",
             color=GRAY, fontsize=21, va="center")
    ax = fig.add_axes([.60, .12, .36, .76])
    ax.set_facecolor(CREAM)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def save(fig, name):
    fig.savefig(OUT + name, facecolor=CREAM, dpi=100)
    plt.close(fig)
    print("saved:", name)


# ============ 第2回：Attention（思い出している）============
fig, ax = layout("AIが見ているのは", "文章のどこなのか", "覗いたら、記憶の仕組みでした")
# 頭（丸）に顔
ax.add_patch(Circle((-.10, -.18), .52, facecolor=ORANGE, edgecolor="none", zorder=3))
face(ax, -.10, -.16, s=1.05)
# 思い出し中の吹き出し（小→大）
for (x, y, r) in ((.34, .30, .075), (.50, .46, .115), (.70, .68, .17)):
    ax.add_patch(Circle((x, y), r, facecolor="#ffffff", edgecolor=BLUE, lw=5, zorder=4))
for dx in (-.06, 0, .06):
    ax.add_patch(Circle((.70 + dx, .68), .022, color=BLUE, zorder=5))
save(fig, "eyecatch_attention.png")

# ============ 第3回：拡散モデル（インクが広がる）============
fig, ax = layout("AIは絵を", "描いていなかった", "覗いたら、インクの話でした")
# 広がっていく波紋
for r, a in ((.92, .16), (.74, .24), (.56, .34)):
    ax.add_patch(Circle((-.02, -.10), r, facecolor=BLUE, alpha=a, edgecolor="none", zorder=1))
# 散った粒
rng = np.random.default_rng(3)
for _ in range(26):
    th, rr = rng.uniform(0, 2 * np.pi), rng.uniform(.62, 1.02)
    ax.add_patch(Circle((-.02 + rr * np.cos(th), -.10 + rr * np.sin(th)),
                        rng.uniform(.022, .045), color=BLUE, alpha=.55, zorder=2))
# インクの一滴（本体）
ax.add_patch(Circle((-.02, -.16), .40, facecolor=ORANGE, edgecolor="none", zorder=3))
ax.add_patch(plt.Polygon([[-.02, .44], [-.24, .04], [.20, .04]], facecolor=ORANGE,
                         edgecolor="none", zorder=3))
face(ax, -.02, -.18, s=.92)
save(fig, "eyecatch_diffusion.png")
