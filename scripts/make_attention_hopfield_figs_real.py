#!/usr/bin/env python3
"""実物のTransformer(llm-jp-3-150m)のattentionを測って図にする。"""
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["figure.dpi"] = 150
BLUE, ORANGE, GRAY = "#3b82f6", "#ef7d54", "#8a8a8a"
OUT = "/Users/mitsu/SideWork/images/"

MODEL = "llm-jp/llm-jp-3-150m"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager",
                                             dtype=torch.float32)
model.eval()
cfg = model.config
L, H = cfg.num_hidden_layers, cfg.num_attention_heads
d_head = cfg.hidden_size // H

text = "日本の首都は東京です。日本で一番高い山は富士山です。日本の首都は"
ids = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model(**ids, output_attentions=True)
T = ids["input_ids"].shape[1]
toks = [t.replace("▁", "") or "␣" for t in tok.convert_ids_to_tokens(ids["input_ids"][0])]

def eff(w):
    w = np.clip(w, 1e-30, 1)
    return float(np.exp(-(w * np.log(w)).sum()))

E = np.zeros((L, H))
W = np.zeros((L, H, T))
for l, a in enumerate(out.attentions):
    w = a[0, :, -1, :].float().numpy()
    for h in range(H):
        E[l, h] = eff(w[h]); W[l, h] = w[h]

# --- 図6: ヘッドごとの有効注目数 ---
fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9),
                       gridspec_kw={"width_ratios": [1.35, 1]})
im = ax[0].imshow(E, aspect="auto", cmap="viridis", vmin=1, vmax=T)
ax[0].set_xlabel("ヘッド番号"); ax[0].set_ylabel("層")
ax[0].set_title(f"最終トークンから見た「有効注目数」({L}層×{H}ヘッド)", fontsize=11)
ax[0].set_xticks(range(H)); ax[0].set_yticks(range(0, L, 2)); ax[0].grid(False)
cb = plt.colorbar(im, ax=ax[0]); cb.set_label(f"何語ぶんを見ているか (最大{T})", fontsize=9)
ax[1].hist(E.ravel(), bins=22, color=BLUE, edgecolor="white")
ax[1].axvline(1, color=ORANGE, ls="--", lw=1.5)
ax[1].text(1.25, ax[1].get_ylim()[1] * .82, "1語だけを\n見ている", fontsize=9, color=ORANGE)
ax[1].set_xlabel("有効注目数"); ax[1].set_ylabel("ヘッド数")
ax[1].set_title("鋭いヘッドと混ぜるヘッドが共存する", fontsize=11); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT + "attention-hopfield-real-heads.png", bbox_inches="tight"); plt.close()

flat = [(E[l, h], l, h) for l in range(L) for h in range(H)]
flat.sort()
print(f"有効注目数: 最小{flat[0][0]:.2f}(層{flat[0][1]}頭{flat[0][2]}) "
      f"中央値{np.median(E):.2f} 最大{flat[-1][0]:.2f}(層{flat[-1][1]}頭{flat[-1][2]}) / 全{T}トークン")

# --- 図7: 最も鋭いヘッド vs 最も混ぜるヘッド ---
(e_lo, l_lo, h_lo), (e_hi, l_hi, h_hi) = flat[0], flat[-1]
fig, ax = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
for a, (e, l, h), c, lab in [(ax[0], (e_lo, l_lo, h_lo), ORANGE, "最も鋭いヘッド"),
                             (ax[1], (e_hi, l_hi, h_hi), BLUE, "最も混ぜるヘッド")]:
    a.bar(range(T), W[l, h], color=c)
    a.set_ylabel("注意の重み")
    a.set_title(f"{lab}(第{l}層 ヘッド{h})  有効注目数 {e:.2f}", fontsize=11, loc="left")
    a.set_ylim(0, 1); a.grid(alpha=.3, axis="y")
ax[1].set_xticks(range(T))
ax[1].set_xticklabels(toks, rotation=55, ha="right", fontsize=9)
plt.tight_layout(); plt.savefig(OUT + "attention-hopfield-real-sink.png", bbox_inches="tight"); plt.close()
print(f"最も鋭いヘッドが見ていたトークン: {toks[int(W[l_lo,h_lo].argmax())]!r} "
      f"(重み{W[l_lo,h_lo].max():.3f})")

# 鋭いヘッド上位10個が何を見ているか
print("\n最も鋭いヘッド上位10個が見ていたトークン:")
for e, l, h in flat[:10]:
    print(f"  第{l:>2}層 ヘッド{h}  有効注目数{e:.2f}  → {toks[int(W[l,h].argmax())]!r}")
sink = sum(1 for e, l, h in flat[:10] if int(W[l, h].argmax()) == 0)
print(f"  → 上位10個のうち {sink} 個が先頭トークンを見ている")

# --- スコアの分散(√dの効果を実機で) ---
with torch.no_grad():
    hs = model(**ids, output_hidden_states=True).hidden_states
print()
for layer in [0, 6, 11]:
    blk = model.model.layers[layer].self_attn
    # q_proj/k_proj には input_layernorm を通した後の値が入る。
    # 生の hidden_states をそのまま渡すとスケールが違い、誤った分散になる
    with torch.no_grad():
        x = model.model.layers[layer].input_layernorm(hs[layer])
    with torch.no_grad():
        q = blk.q_proj(x)[0].view(T, -1, d_head).transpose(0, 1)
        k = blk.k_proj(x)[0].view(T, -1, d_head).transpose(0, 1)
        s = torch.matmul(q, k.transpose(-1, -2))
    print(f"第{layer:>2}層: 生スコアの標準偏差 {s.std().item():6.2f} → "
          f"√d({np.sqrt(d_head):.0f})で割った後 {(s/np.sqrt(d_head)).std().item():5.2f}")
