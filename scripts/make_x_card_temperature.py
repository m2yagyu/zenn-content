"""llm-temperature-boltzmann の記事をXで告知するときの添付画像を作る。

数値は記事本文に載っている実測結果（rinna/japanese-gpt2-medium に
「日本の首都は」を入力したときの上位6トークンの確率と exp(S)）をそのまま使う。
articles/llm-temperature-boltzmann.md の出力表と一致していること。

記事中の図は 1430x374 と横長すぎてタイムラインで潰れるので、
16:9 に組み直し、配色をXのプロフィール（images/x/）に合わせてある。

    .venv/bin/python scripts/make_x_card_temperature.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#1c1c1c"
FG = "#f2f2f2"
GRAY = "#8a8a8a"
BLUE = "#3b82f6"

plt.rcParams["font.family"] = ["Hiragino Sans", "Helvetica Neue", "sans-serif"]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images", "x")
os.makedirs(OUT, exist_ok=True)

TOKENS = ["東京", "首都", "「", "新宿", "渋谷", "日本"]

# (T, 上位6トークンの確率, exp(S) = 実効的な選択肢の数)
PANELS = [
    (0.3, [0.965, 0.023, 0.008, 0.000, 0.000, 0.000], 1.2),
    (1.0, [0.146, 0.047, 0.035, 0.015, 0.014, 0.013], 534.5),
    (2.0, [0.005, 0.003, 0.003, 0.002, 0.002, 0.002], 19438.3),
]
CAPTIONS = ["低温", "標準", "高温"]


def card(path):
    W, H = 1600, 900
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    fig.patch.set_facecolor(BG)

    fig.text(0.5, 0.925, "「日本の首都は」の次に来るトークンの確率",
             color=FG, fontsize=30, ha="center", va="center")
    fig.text(0.5, 0.862, "temperature を変えて、実際のLLMで測ったもの",
             color=GRAY, fontsize=17, ha="center", va="center")

    panel_w, panel_h = 400, 400
    gap = 60
    left0 = (W - (panel_w * 3 + gap * 2)) / 2
    bottom = 250

    for i, (T, probs, ppl) in enumerate(PANELS):
        left = left0 + i * (panel_w + gap)
        ax = fig.add_axes([left / W, bottom / H, panel_w / W, panel_h / H])
        ax.set_facecolor(BG)

        p = np.array(probs)
        ax.bar(range(len(p)), p / p.max(), color=BLUE, width=0.62)

        ax.set_ylim(0, 1.18)
        ax.set_xlim(-0.7, len(p) - 0.3)
        ax.set_yticks([])
        ax.set_xticks(range(len(p)))
        ax.set_xticklabels(TOKENS, color=GRAY, fontsize=14, rotation=45, ha="right")
        ax.tick_params(axis="x", length=0, pad=6)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(0, color="#3a3a3a", lw=1.2)

        fig.text((left + panel_w / 2) / W, (bottom + panel_h + 66) / H,
                 f"{CAPTIONS[i]}  T = {T}", color=FG, fontsize=25, ha="center", va="center")
        fig.text((left + panel_w / 2) / W, (bottom + panel_h + 22) / H,
                 f"最大 {probs[0]:.3f}", color=GRAY, fontsize=17, ha="center", va="center")
        # exp(S): エントロピーの指数 = 実効的に選ばれうるトークン数
        fig.text((left + panel_w / 2) / W, (bottom - 100) / H,
                 f"{ppl:,.0f}" if ppl >= 10 else f"{ppl:.1f}",
                 color=BLUE, fontsize=34, ha="center", va="center")

    fig.text(0.5, (bottom - 152) / H, "実効的な選択肢の数  exp(S)",
             color=GRAY, fontsize=17, ha="center", va="center")
    # 縦軸をパネルごとに正規化していることは明記する（そうしないと確率の絶対値を誤読させる）
    fig.text(0.5, 0.045, "縦軸はパネルごとに最大値で規格化（形の比較用）／ zenn.dev/m2yagyu",
             color="#5a5a5a", fontsize=14, ha="center", va="center")

    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    card(os.path.join(OUT, "card_llm-temperature.png"))
