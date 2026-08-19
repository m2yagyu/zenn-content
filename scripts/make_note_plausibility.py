#!/usr/bin/env python3
"""note第7回「AIに違う質問をしたのに同じ答えが返ってきた」用の画像を作る（1200x675）。

宛先はプログラミング未経験の完全初心者。数式・コード・専門用語は画像にも出さない。
測り方は articles/llm-hallucination-confidence.md の検証済みコードと同一で、
そこに「一番大きい都市は」を聞く2本目の問いを足しただけ。
図に出る数値はすべてこのスクリプトの実行結果そのもの。

  .venv/bin/python scripts/make_note_plausibility.py
"""
import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401

BG, FG, ACCENT, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
OK, NG = "#22c55e", "#ef4444"
OUT = "/Users/mitsu/SideWork/note/plausibility/"
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


# ---------- 第6回と同じ測り方に、2本目の問いを足す ----------
MODEL = "llm-jp/llm-jp-3-150m"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

SHOT_CAP = "日本の首都は東京です。フランスの首都はパリです。"
SHOT_BIG = "日本で一番大きい都市は東京です。フランスで一番大きい都市はパリです。"
REAL = [("イタリア", "ローマ"), ("ドイツ", "ベルリン"), ("イギリス", "ロンドン"),
        ("オーストラリア", "キャンベラ"), ("アメリカ", "ワシントン"), ("ブラジル", "ブラジリア"),
        ("カナダ", "オタワ"), ("トルコ", "アンカラ"), ("スイス", "ベルン"),
        ("中国", "北京"), ("韓国", "ソウル"), ("インド", "ニューデリー"),
        ("スペイン", "マドリード"), ("エジプト", "カイロ"), ("ニュージーランド", "ウェリントン")]


def greedy(prompt, n=8):
    """確率が最大の語をつないで答えを取る（第6回と同じ手続き）。"""
    ids = tok(prompt, return_tensors="pt").input_ids
    k = 0
    for _ in range(n):
        with torch.no_grad():
            nxt = int(model(ids).logits[0, -1].argmax())
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        k += 1
        if tok.decode([nxt]).endswith("。"):
            break
    return tok.decode(ids[0, -k:]).replace("です。", "").replace("。", "")


rows = []
for country, gold in REAL:
    cap = greedy(SHOT_CAP + f"{country}の首都は")
    big = greedy(SHOT_BIG + f"{country}で一番大きい都市は")
    rows.append(dict(country=country, gold=gold, cap=cap, big=big,
                     same=(cap == big), ok=(cap == gold)))

n_same = sum(r["same"] for r in rows)
n_ok = sum(r["ok"] for r in rows)
print(f"{'国':<12}{'首都は':<14}{'一番大きい都市は':<14}{'正解':<12}")
for r in rows:
    print(f"{r['country']:<12}{r['cap']:<14}{r['big']:<14}{r['gold']:<12}"
          f"{'同じ' if r['same'] else '':<5}{'○' if r['ok'] else '×'}")
print(f"\n2つの問いに同じ答え: {n_same}/{len(rows)}   首都問の正解: {n_ok}/{len(rows)}")

# ---------- 図1 違う質問に、同じ答え ----------
SHOW = ["ブラジル", "アメリカ", "オーストラリア", "カナダ"]
fig = new(); ax = axes(fig)
ax.text(50, 92, "2つの違う質問をしたのに", ha="center", color=FG, fontsize=29)
ax.text(50, 83, "同じ答えが返ってきた", ha="center", color=ACCENT, fontsize=30)
ax.text(30, 72, "「首都は？」", ha="center", color=SUB, fontsize=18)
ax.text(72, 72, "「一番大きい都市は？」", ha="center", color=SUB, fontsize=18)
for i, c in enumerate(SHOW):
    r = next(x for x in rows if x["country"] == c)
    y = 60 - i * 12
    ax.text(9, y, r["country"], ha="left", va="center", color=SUB, fontsize=15)
    ax.text(30, y, r["cap"], ha="center", va="center", color=FG, fontsize=20)
    ax.text(72, y, r["big"], ha="center", va="center", color=FG, fontsize=20)
    ax.text(51, y, "＝", ha="center", va="center", color=ACCENT, fontsize=20)
ax.text(50, 8, f"{len(rows)}カ国のうち{n_same}カ国で、答えが同じだった",
        ha="center", color=FG, fontsize=21)
save(fig, "01_same_answer.png")

# ---------- 図2 質問を聞き分けていない ----------
fig = new(); ax = axes(fig)
ax.text(50, 90, "AIは質問を聞き分けていなかった", ha="center", color=FG, fontsize=30)
for i, q in enumerate(["「首都は？」", "「一番大きい都市は？」"]):
    y = 72 - i * 26
    ax.text(23, y, q, ha="center", va="center", color=FG, fontsize=22)
    ax.annotate("", xy=(55, 59), xytext=(41, y),
                arrowprops=dict(arrowstyle="->", color=SUB, lw=1.6))
ax.text(78, 59, "その国といえば、で\n真っ先に出てくる都市",
        ha="center", va="center", color=ACCENT, fontsize=23, linespacing=1.5)
ax.text(50, 27, "どちらを聞いても、返ってくるのは同じもの",
        ha="center", color=FG, fontsize=22)
ax.text(50, 14, "「よく一緒に書かれている名前」を出しているだけ",
        ha="center", color=SUB, fontsize=20)
save(fig, "02_not_listening.png")

# ---------- 図3 それでも当たる場所がある ----------
hit = [r for r in rows if r["ok"]]
FAMOUS = ["オーストラリア", "アメリカ", "ブラジル", "カナダ", "トルコ"]   # 有名な非首都を答えた国
miss = [r for r in rows if r["country"] in FAMOUS]
fig = new(); ax = axes(fig)
ax.text(50, 93, "それでも当たるのは、重なっている国", ha="center", color=FG, fontsize=29)
ax.text(27, 82, "有名な都市 ＝ 首都", ha="center", color=OK, fontsize=22)
ax.text(74, 82, "有名な都市 ≠ 首都", ha="center", color=NG, fontsize=22)
for i, r in enumerate(hit):
    ax.text(27, 71 - i * 9.5, f"{r['country']}　{r['cap']}",
            ha="center", va="center", color=FG, fontsize=18)
for i, r in enumerate(miss):
    ax.text(74, 71 - i * 9.5, f"{r['country']}　{r['cap']}",
            ha="center", va="center", color=FG, fontsize=18)
ax.plot([50, 50], [10, 78], color=SUB, lw=1.0)
ax.text(27, 8, f"{len(hit)}問すべて正解", ha="center", color=OK, fontsize=22)
ax.text(74, 8, "すべてまちがい", ha="center", color=NG, fontsize=22)
save(fig, "03_overlap.png")

# ---------- 図4 世の中の大半は重なっている ----------
fig = new(); ax = axes(fig)
ax.text(50, 93, "ふだん困らないのは、世の中の大半で", ha="center", color=FG, fontsize=27)
ax.text(50, 83, "この2つが重なっているから", ha="center", color=ACCENT, fontsize=28)
# 図の縦横比が12:6.75なので、円に見せるには横半径を縦半径の0.5625倍にする
RY, RX = 26.0, 26.0 * 6.75 / 12
for cx, col in [(45, ACCENT), (55, OK)]:
    ax.add_patch(matplotlib.patches.Ellipse((cx, 45), 2 * RX, 2 * RY,
                                            color=col, alpha=0.45))
ax.text(35, 45, "よく\n書かれて\nいること", ha="center", va="center",
        color=FG, fontsize=17, linespacing=1.4)
ax.text(65, 45, "本当の\nこと", ha="center", va="center",
        color=FG, fontsize=17, linespacing=1.4)
ax.text(50, 45, "たいてい\nここ", ha="center", va="center", color=FG, fontsize=19,
        linespacing=1.4)
ax.annotate("ずれるのはここ", xy=(34, 62), xytext=(16, 73),
            ha="center", color=NG, fontsize=19,
            arrowprops=dict(arrowstyle="->", color=NG, lw=1.6))
ax.text(50, 7, "AIが強いのは重なっている場所、あぶないのはずれている場所",
        ha="center", color=FG, fontsize=20)
save(fig, "04_world.png")
