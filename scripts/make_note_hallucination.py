#!/usr/bin/env python3
"""note第6回「AIは間違えているときのほうが自信がある」用の画像を作る（1200x675）。

宛先はプログラミング未経験の完全初心者。数式・コード・専門用語は画像にも出さない。
測り方は articles/llm-hallucination-confidence.md の検証済みコードと同一で、
そこから出てきた本物の数値だけを図にする。

  .venv/bin/python scripts/make_note_hallucination.py
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401

BG, FG, ACCENT, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
OK, NG = "#22c55e", "#ef4444"
OUT = "/Users/mitsu/SideWork/note/hallucination/"
os.makedirs(OUT, exist_ok=True)


def new(w=12.0, h=6.75):
    fig = plt.figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


def save(fig, name):
    fig.savefig(OUT + name, facecolor=BG, dpi=100)
    plt.close(fig)
    print("saved", name)


def axes(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return ax


# ---------- 記事と同じ測り方 ----------
MODEL = "llm-jp/llm-jp-3-150m"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

SHOT = "日本の首都は東京です。フランスの首都はパリです。"
REAL = [("イタリア", "ローマ"), ("ドイツ", "ベルリン"), ("イギリス", "ロンドン"),
        ("オーストラリア", "キャンベラ"), ("アメリカ", "ワシントン"), ("ブラジル", "ブラジリア"),
        ("カナダ", "オタワ"), ("トルコ", "アンカラ"), ("スイス", "ベルン"),
        ("中国", "北京"), ("韓国", "ソウル"), ("インド", "ニューデリー"),
        ("スペイン", "マドリード"), ("エジプト", "カイロ"), ("ニュージーランド", "ウェリントン")]


def next_logits(country):
    ids = tok(SHOT + f"{country}の首都は", return_tensors="pt").input_ids
    with torch.no_grad():
        return model(ids).logits[0, -1]


def gold_id(country, answer):
    a = tok(SHOT + f"{country}の首都は").input_ids
    b = tok(SHOT + f"{country}の首都は{answer}").input_ids
    return b[len(a)]


def greedy(country, n=8):
    ids = tok(SHOT + f"{country}の首都は", return_tensors="pt").input_ids
    k = 0
    for _ in range(n):
        with torch.no_grad():
            nxt = int(model(ids).logits[0, -1].argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        k += 1
        if tok.decode([nxt]).endswith("。"):
            break
    return tok.decode(ids[0, -k:]).replace("です。", "")


rows = []
for country, gold in REAL:
    lg = next_logits(country)
    p = F.softmax(lg, dim=-1)
    gid = gold_id(country, gold)
    rows.append(dict(country=country, gold=gold, out=greedy(country), logits=lg,
                     conf=float(p.max()), ok=(int(p.argmax()) == gid)))
n_ok = sum(r["ok"] for r in rows)
print(f"正解 {n_ok}/{len(rows)}")
for r in rows:
    print(f"  {r['country']:<9}{r['out']:<12}{r['conf']:.3f} {'○' if r['ok'] else '×'}")

# ---------- 図1 まちがえ方に癖がある ----------
FAMOUS = ["オーストラリア", "アメリカ", "ブラジル", "カナダ", "トルコ"]
fig = new(); ax = axes(fig)
ax.text(50, 91, f"15問聞いて、{len(rows) - n_ok}問まちがえた", ha="center", color=FG, fontsize=31)
ax.text(50, 82, "しかも、まちがえ方に癖があった", ha="center", color=ACCENT, fontsize=25)
ax.text(37, 71, "AIの答え", ha="center", color=SUB, fontsize=17)
ax.text(76, 71, "本当の首都", ha="center", color=SUB, fontsize=17)
for i, c in enumerate(FAMOUS):
    r = next(x for x in rows if x["country"] == c)
    y = 62 - i * 10.5
    ax.text(20, y, r["country"], ha="right", va="center", color=SUB, fontsize=15)
    ax.text(37, y, r["out"], ha="center", va="center", color=NG, fontsize=21)
    ax.annotate("", xy=(66, y), xytext=(55, y),
                arrowprops=dict(arrowstyle="->", color=SUB, lw=1.4))
    ax.text(76, y, r["gold"], ha="center", va="center", color=FG, fontsize=21)
ax.text(50, 6, "どれも、その国で一番有名な都市だった", ha="center", color=FG, fontsize=21)
save(fig, "01_quiz.png")

# ---------- 図2 合っているときより自信がある ----------
de = next(r for r in rows if r["country"] == "ドイツ")
br = next(r for r in rows if r["country"] == "ブラジル")
fig = new(); ax = axes(fig)
ax.text(50, 91, "合っているときより", ha="center", color=FG, fontsize=29)
ax.text(50, 82, "まちがえているときのほうが自信があった", ha="center", color=ACCENT, fontsize=27)
for i, (r, tag, col) in enumerate([(de, "正解", OK), (br, "まちがい", NG)]):
    y = 58 - i * 26
    ax.text(8, y + 11, f"「{r['country']}の首都は？」", ha="left", color=SUB, fontsize=17)
    ax.text(8, y + 3, r["out"], ha="left", va="center", color=FG, fontsize=27)
    ax.text(92, y + 11, tag, ha="right", color=col, fontsize=19)
    ax.barh([y - 5], [r["conf"] * 62], left=8, height=4.2, color=col)
    ax.text(8 + r["conf"] * 62 + 2, y - 5, f"自信 {r['conf'] * 100:.0f}%",
            ha="left", va="center", color=FG, fontsize=18)
ax.text(50, 5, "自信の大きさを見ても、正しいかは分からない", ha="center", color=FG, fontsize=20)
save(fig, "02_confidence.png")

# ---------- 図3 正解も知っているが、うすい ----------
p_br = F.softmax(br["logits"], dim=-1)
p_rio = float(p_br.max())
p_bra = float(p_br[gold_id("ブラジル", "ブラジリア")])
fig = new(); ax = axes(fig)
ax.text(50, 91, "本当の首都も、知ってはいた", ha="center", color=FG, fontsize=30)
ax.text(50, 82, f"ただ、{p_rio / p_bra:.0f}倍うすかった", ha="center", color=ACCENT, fontsize=26)
ax.text(50, 71, "「ブラジルの首都は」の次に来る言葉の、選ばれやすさ",
        ha="center", color=SUB, fontsize=17)
for i, (w, v, col) in enumerate([("リオデジャネイロ", p_rio, NG), ("ブラジリア", p_bra, OK)]):
    y = 55 - i * 22
    ax.text(8, y + 8, w, ha="left", color=FG, fontsize=23)
    ax.barh([y], [max(v, 0.008) * 70], left=8, height=5.5, color=col)
    ax.text(8 + max(v, 0.008) * 70 + 2, y, f"{v * 100:.1f}%",
            ha="left", va="center", color=FG, fontsize=20)
ax.text(50, 6, "知らないのではなく、うすいところに沈んでいる", ha="center", color=FG, fontsize=20)
save(fig, "03_thin.png")

# ---------- 図4 つまみを回しても直らない ----------
L = torch.stack([r["logits"] for r in rows])
gold_ids = [gold_id(r["country"], r["gold"]) for r in rows]
labels = [("慎重ぎみ", 0.1), ("ふつう", 1.0), ("大胆ぎみ", 5.0)]
vals = []
for _, T in labels:
    pT = F.softmax(L / T, dim=-1)
    vals.append(float(pT[range(len(rows)), gold_ids].mean()))
print("温度ごとの正解率:", [f"{v:.3f}" for v in vals])

fig = new(); ax = axes(fig)
ax.text(50, 91, "「慎重に答えて」と設定しても", ha="center", color=FG, fontsize=29)
ax.text(50, 82, "この間違いは直らない", ha="center", color=ACCENT, fontsize=30)
for i, ((lab, _), v) in enumerate(zip(labels, vals)):
    x = 20 + i * 30
    h = v * 100 * 0.9
    if h > 0.5:                                   # 0%のときは棒を描かない（線が残るため）
        ax.bar([x], [h], bottom=22, width=13, color=NG, linewidth=0)
    ax.text(x, 16, lab, ha="center", color=FG, fontsize=21)
    ax.text(x, 26 + h, f"正解率 {v * 100:.0f}%", ha="center", color=FG, fontsize=19)
ax.text(50, 8, "つまみが変えるのは答えのばらつきだけで、正しさではない",
        ha="center", color=FG, fontsize=20)
save(fig, "04_knob.png")
