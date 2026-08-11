#!/usr/bin/env python3
"""note第2回「AIが文章のどこを見ているのか…」用の画像を作る（1200x675）。

宛先はプログラミング未経験の完全初心者。数式・コード・専門用語は画像にも出さない。
数値はすべて articles/attention-hopfield-associative-memory.md の検証済みコードと
同じ計算を実行して得る（本物の llm-jp/llm-jp-3-150m を CPU で走らせる）。

  .venv/bin/python scripts/make_note_attention.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BG, FG, ACCENT, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
WARM, DIM = "#ef7d54", "#4a5464"
OUT = "/Users/mitsu/SideWork/note/attention/"
os.makedirs(OUT, exist_ok=True)


def new(w=12.0, h=6.75):
    fig = plt.figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


def save(fig, name):
    fig.savefig(OUT + name, facecolor=BG, dpi=100)
    plt.close(fig)
    print("saved:", name)


def style(ax, grid=False):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, length=0, labelsize=15)
    for s in ax.spines.values():
        s.set_visible(False)
    if grid:
        ax.grid(alpha=.18, axis="y", color=FG)
        ax.set_axisbelow(True)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def eff(w):
    """実効的に何個ぶんを見ているか（記事と同じ exp(エントロピー)）"""
    w = np.clip(w, 1e-30, 1)
    return np.exp(-(w * np.log(w)).sum())


# ============ 本物のLLMのattentionを測る（記事の検証済みコードと同一） ============
MODEL = "llm-jp/llm-jp-3-150m"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
model = model.float()
model.eval()

text = "日本の首都は東京です。日本で一番高い山は富士山です。日本の首都は"
ids = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model(**ids, output_attentions=True)

T = ids["input_ids"].shape[1]
toks = [t.replace("▁", "") or "␣" for t in tok.convert_ids_to_tokens(ids["input_ids"][0])]
labels = ["文頭の印" if t == "<s>" else t for t in toks]
L, H = len(out.attentions), out.attentions[0].shape[1]

E = np.zeros((L, H))        # ヘッドごとの有効注目数
W = np.zeros((L, H, T))     # ヘッドごとの注意の重み
for l, a in enumerate(out.attentions):
    w = a[0, :, -1, :].float().numpy()
    for h in range(H):
        E[l, h], W[l, h] = eff(w[h]), w[h]

print(f"有効注目数: 最小 {E.min():.2f} / 中央値 {np.median(E):.2f} / 最大 {E.max():.2f} (全{T}語)")
for l, h in ((6, 5), (6, 2)):
    o = np.argsort(W[l, h])[::-1][:3]
    print(f"  第{l}層ヘッド{h}: " + " ".join(f"{labels[i]}{W[l,h,i]*100:.0f}%" for i in o))

# ============ 1. AIは前に出てきたどの言葉を見ているか ============
fig = new()
fig.text(.5, .955, "AIは、前に出てきた言葉のどれを見るか配っている", color=FG,
         fontsize=33, ha="center", va="top")
fig.text(.5, .868, "「日本の首都は東京です。日本で一番高い山は富士山です。日本の首都は」の続きを考えるとき",
         color=SUB, fontsize=16, ha="center", va="top")

for j, ((l, h), title, key, col) in enumerate((
        ((6, 5), "ある係は、「首都」をじっと見ていた", "首都", ACCENT),
        ((6, 2), "別の係は、答えである「東京」を見ていた", "東京", WARM))):
    ax = fig.add_axes([.07, .50 - j * .315, .88, .225])
    style(ax)
    w = W[l, h] * 100
    cols = [col if labels[i] == key else DIM for i in range(T)]
    ax.bar(range(T), w, color=cols, width=.68)
    top = int(np.argmax([w[i] if labels[i] == key else -1 for i in range(T)]))
    ax.text(top, w[top] + 8, f"{w[top]:.0f}%", ha="center", color=col, fontsize=17)
    if j == 1:
        ax.text(0, w[0] + 8, f"{w[0]:.0f}%", ha="center", color=SUB, fontsize=15)
    ax.set(ylim=(0, 108), yticks=[], xticks=range(T))
    ax.set_xticklabels(labels if j == 1 else [""] * T, fontsize=12, rotation=45, ha="right")
    ax.set_title(title, color=col, fontsize=21, loc="left", pad=6)

fig.text(.5, .035, "棒の高さが「どれくらい見ているか」。本物のAIの中身を測った数字です",
         color=SUB, fontsize=16, ha="center")
save(fig, "01_where.png")

# ============ 2. 思い出すとは、谷に落ちること ============
# 記事と同じエネルギー関数 E(ξ) = -(1/β)logΣexp(β x_i·ξ) + ½|ξ|²
P = np.array([[-1.2, -0.9], [1.3, -0.6], [0.1, 1.3]]).T
g = np.linspace(-2.2, 2.2, 400)
GX, GY = np.meshgrid(g, g)
Z = np.stack([GX.ravel(), GY.ravel()])
beta = 8.0
s = beta * (P.T @ Z)
energy = (-(1 / beta) * (np.log(np.exp(s - s.max(0)).sum(0)) + s.max(0))
          + .5 * (Z ** 2).sum(0)).reshape(GX.shape)

fig = new()
fig.text(.5, .945, "「思い出す」とは、一番近い谷に転がり落ちること", color=FG,
         fontsize=33, ha="center", va="top")
fig.text(.5, .858, "物理学者が四十年あまり前に作った、記憶の模型", color=SUB,
         fontsize=18, ha="center", va="top")

ax = fig.add_axes([.30, .14, .40, .66])
ax.set_facecolor(BG)
ax.contourf(GX, GY, energy, levels=40, cmap="magma")
ax.contour(GX, GY, energy, levels=14, colors="#00000055", linewidths=.6)
ax.scatter(P[0], P[1], s=150, color=FG, zorder=5)
for x, y, t in zip(P[0], P[1], ("記憶A", "記憶B", "記憶C")):
    ax.text(x, y + .28, t, color=FG, fontsize=17, ha="center")
cue = np.array([0.72, 0.95])
ax.scatter(*cue, s=180, color=WARM, marker="X", zorder=6)
ax.annotate("", xy=(P[0, 2] + .06, P[1, 2] + .04), xytext=cue,
            arrowprops=dict(arrowstyle="-|>", color=WARM, lw=2.6, shrinkA=9, shrinkB=9))
ax.text(cue[0] + .22, cue[1] - .55, "ぼんやりした\n手がかり", color=WARM, fontsize=16)
ax.set(xticks=[], yticks=[], xlim=(-2.2, 2.2), ylim=(-2.2, 2.2))
ax.set_aspect("equal")
for sp in ax.spines.values():
    sp.set_color("#3a3a3a")

fig.text(.075, .60, "明るいところが\n高い場所、\n暗いところが谷", color=SUB, fontsize=18, va="center")
fig.text(.925, .60, "記憶は谷として\nしまわれている。\n手がかりを置くと\n坂を下って\n一番近い記憶に\n着く", color=SUB,
         fontsize=18, va="center", ha="right")
fig.text(.5, .05, "AIが言葉を選ぶ計算は、この「谷に落ちる」計算と同じ形をしていました",
         color=SUB, fontsize=17, ha="center")
save(fig, "02_valley.png")

# ============ 3. 前回のつまみが、ここにもあった ============
rng = np.random.default_rng(0)
d, N = 64, 6
X = rng.normal(size=(d, N))
X /= np.linalg.norm(X, axis=0, keepdims=True)
n = rng.normal(size=d)
n /= np.linalg.norm(n)
xi = X[:, 3] + 0.5 * n
xi /= np.linalg.norm(xi)

fig = new()
fig.text(.5, .945, "前回の「温度」のつまみが、ここにもあった", color=FG,
         fontsize=34, ha="center", va="top")
fig.text(.5, .858, "6つの記憶のうち、どれをどれくらい思い出すか", color=SUB,
         fontsize=18, ha="center", va="top")

for j, (b, lab, col) in enumerate(((0.5, "つまみを熱い側へ", WARM),
                                   (8.0, "つまみを冷たい側へ", ACCENT))):
    ax = fig.add_axes([.10 + j * .47, .275, .36, .43])
    style(ax, grid=True)
    w = softmax(b * (X.T @ xi)) * 100
    ax.bar(range(N), w, color=col, width=.62)
    ax.set(ylim=(0, 118), yticks=[], xticks=range(N))
    ax.set_xticklabels([f"記憶{i+1}" for i in range(N)], fontsize=14)
    ax.set_title(lab, color=col, fontsize=23, pad=14)
    note = "ぜんぶが薄く混ざる\n（およそ6つぶんを同時に思い出している）" if j == 0 \
        else "1つに決まる\n（ほぼ記憶4だけを思い出している）"
    ax.text(.5, -.28, note, transform=ax.transAxes, ha="center", color=col, fontsize=17)

fig.text(.5, .035, "同じ仕組みで、同じつまみが効いています", color=SUB, fontsize=16, ha="center")
save(fig, "03_knob.png")

# ============ 4. 見る係は96人いて、性格がばらばら ============
fig = new()
fig.text(.5, .955, "「見る係」は九十六人いて、性格がばらばらだった", color=FG,
         fontsize=32, ha="center", va="top")
fig.text(.5, .868, "十七語の文を読ませて、係ごとに「何語ぶんを見ているか」を測った",
         color=SUB, fontsize=17, ha="center", va="top")

ax = fig.add_axes([.075, .275, .40, .46])
style(ax, grid=True)
ax.hist(E.ravel(), bins=22, color=ACCENT, edgecolor=BG)
ax.set(yticks=[], ylim=(0, 17))
ax.set_xlabel("何語ぶんを見ているか", color=SUB, fontsize=16)
ax.annotate("1語だけを\nじっと見る係", xy=(1.3, 10.5), xytext=(2.8, 13.2), color=WARM, fontsize=15,
            arrowprops=dict(arrowstyle="-|>", color=WARM, lw=1.8))
ax.annotate("十二語ぶんを\nならして見る係", xy=(12.1, 1.6), xytext=(7.0, 9.5), color=SUB, fontsize=15,
            arrowprops=dict(arrowstyle="-|>", color=SUB, lw=1.8))
ax.set_title("係ごとの性格の散らばり", color=FG, fontsize=20, pad=10)

l, h = np.unravel_index(E.argmin(), E.shape)
ax = fig.add_axes([.575, .275, .40, .46])
style(ax, grid=True)
w = W[l, h] * 100
ax.bar(range(T), w, color=[WARM if i == 0 else DIM for i in range(T)], width=.68)
ax.text(0.7, 93, f"{w[0]:.1f}%", color=WARM, fontsize=17)
ax.set(ylim=(0, 118), yticks=[], xticks=range(T))
ax.set_xticklabels(labels, fontsize=11, rotation=45, ha="right")
ax.set_title("一番鋭い係が見ていたのは「文頭の印」", color=FG, fontsize=20, pad=10)

fig.text(.5, .035, "思い出すものが無いとき、置き場所に困った重みがここに捨てられます",
         color=SUB, fontsize=16, ha="center")
save(fig, "04_heads.png")
