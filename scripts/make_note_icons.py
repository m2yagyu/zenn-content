#!/usr/bin/env python3
"""note用のプロフィールアイコン候補（320x320）と、第1回の見出し画像（1280x670）。

宛先が完全初心者なので、硬い図版ではなくポップで愛嬌のある方向に振る。
円形に切り抜かれ、フィードでは40px程度で表示されるので、
要素は少なく・太く・中央に寄せる。顔は点2つと弧1本まで。

  .venv/bin/python scripts/make_note_icons.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
from matplotlib.patches import Circle, FancyBboxPatch, Arc, Wedge

OUT = "/Users/mitsu/SideWork/note/brand/"
os.makedirs(OUT, exist_ok=True)

CREAM = "#fff3e2"
ORANGE = "#f4813f"
BLUE = "#3b82f6"
NAVY = "#2a3346"
PINK = "#ff9db0"


def canvas(bg, px=320):
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def face(ax, cx, cy, s=1.0, col=NAVY, happy=True):
    """点2つ + 弧1本だけの顔。小さくしても潰れない"""
    ax.add_patch(Circle((cx - .16 * s, cy + .05 * s), .052 * s, color=col, zorder=6))
    ax.add_patch(Circle((cx + .16 * s, cy + .05 * s), .052 * s, color=col, zorder=6))
    ax.add_patch(Arc((cx, cy - .04 * s), .30 * s, .24 * s,
                     theta1=200 if happy else 20, theta2=340 if happy else 160,
                     color=col, lw=5.5 * s, zorder=6, capstyle="round"))
    # ほっぺ
    ax.add_patch(Circle((cx - .30 * s, cy - .04 * s), .062 * s, color=PINK, alpha=.85, zorder=5))
    ax.add_patch(Circle((cx + .30 * s, cy - .04 * s), .062 * s, color=PINK, alpha=.85, zorder=5))


def save(fig, name, bg):
    fig.savefig(OUT + name, facecolor=bg, dpi=100)
    plt.close(fig)
    print("saved:", name)


# ============ A: 湯気の立つマグカップ（第1回の「温度」そのもの）============
fig, ax = canvas(CREAM)
# 取っ手
ax.add_patch(Wedge((.30, -.12), .34, -68, 68, width=.115, color=ORANGE, zorder=2))
# 本体
ax.add_patch(FancyBboxPatch((-.52, -.56), .92, .74,
                            boxstyle="round,pad=0.02,rounding_size=0.22",
                            facecolor=ORANGE, edgecolor="none", zorder=3))
face(ax, -.06, -.16, s=1.05)
# 湯気
for dx, amp in ((-.30, .055), (-.04, .07), (.22, .055)):
    t = np.linspace(0, 1, 90)
    ax.plot(dx + amp * np.sin(t * 7.0), .28 + t * .52,
            color=BLUE, lw=9, alpha=.55, solid_capstyle="round", zorder=1)
save(fig, "icon_a_mug.png", CREAM)

# ============ B: 分子くん（散らばる粒に顔）============
fig, ax = canvas(BLUE)
for (x, y, r) in ((-.66, .52, .105), (.62, .44, .085), (-.58, -.62, .085),
                  (.66, -.54, .105), (.02, .78, .07)):
    ax.add_patch(Circle((x, y), r, color=CREAM, alpha=.85, zorder=2))
ax.add_patch(Circle((0, -.04), .56, color=CREAM, zorder=3))
face(ax, 0, -.02, s=1.25)
save(fig, "icon_b_particle.png", BLUE)

# ============ C: 温度計くん ============
fig, ax = canvas(CREAM)
ax.add_patch(FancyBboxPatch((-.17, -.18), .34, .82,
                            boxstyle="round,pad=0.02,rounding_size=0.17",
                            facecolor="#ffffff", edgecolor=NAVY, lw=6, zorder=2))
ax.add_patch(FancyBboxPatch((-.085, -.10), .17, .52,
                            boxstyle="round,pad=0.01,rounding_size=0.08",
                            facecolor=ORANGE, edgecolor="none", zorder=3))
ax.add_patch(Circle((0, -.42), .34, facecolor=ORANGE, edgecolor=NAVY, lw=6, zorder=4))
face(ax, 0, -.42, s=.92)
for y in (.18, .36, .54):
    ax.plot([.20, .38], [y, y], color=NAVY, lw=5, solid_capstyle="round", zorder=2)
save(fig, "icon_c_thermometer.png", CREAM)

# ============ 見出し画像（1280x670）============
fig = plt.figure(figsize=(12.8, 6.7), dpi=100)
fig.patch.set_facecolor(CREAM)
fig.text(.065, .80, "AIの答えが", color=NAVY, fontsize=54, va="center")
fig.text(.065, .635, "毎回ちがう理由", color=ORANGE, fontsize=54, va="center")
fig.add_artist(plt.Line2D([.065, .30], [.525, .525], color=ORANGE, lw=4))
fig.text(.065, .42, "調べたら、物理の「温度」でした", color=NAVY, fontsize=27, va="center")
fig.text(.065, .175, "理系大学院で学んだ私が、AIの中身を覗いてみた記録",
         color="#7b8494", fontsize=21, va="center")

ax = fig.add_axes([.60, .12, .36, .76])
ax.set_facecolor(CREAM)
ax.set(xlim=(-1, 1), ylim=(-1, 1), xticks=[], yticks=[])
ax.set_aspect("equal")
for s in ax.spines.values():
    s.set_visible(False)
ax.add_patch(Wedge((.30, -.12), .34, -68, 68, width=.115, color=ORANGE, zorder=2))
ax.add_patch(FancyBboxPatch((-.52, -.56), .92, .74,
                            boxstyle="round,pad=0.02,rounding_size=0.22",
                            facecolor=ORANGE, edgecolor="none", zorder=3))
face(ax, -.06, -.16, s=1.05)
for dx, amp in ((-.30, .055), (-.04, .07), (.22, .055)):
    t = np.linspace(0, 1, 90)
    ax.plot(dx + amp * np.sin(t * 7.0), .28 + t * .52,
            color=BLUE, lw=9, alpha=.55, solid_capstyle="round", zorder=1)

fig.savefig(OUT + "eyecatch_temperature.png", facecolor=CREAM, dpi=100)
plt.close(fig)
print("saved: eyecatch_temperature.png")
print("完了:", OUT)
