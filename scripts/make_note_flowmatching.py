#!/usr/bin/env python3
"""note第4回「私が覚えたばかりのAIの絵の描き方は…」用の画像を作る（1200x675）。

宛先はプログラミング未経験の完全初心者。数式・コード・専門用語は画像にも出さない。
学習・生成・評価のコードは articles/flow-matching-vs-diffusion.md の検証済みコードと同一。
（two moons、同じネットワーク、当てる対象だけ差し替え、6000ステップ）

  .venv/bin/python scripts/make_note_flowmatching.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
from sklearn.datasets import make_moons

BG, FG, ACCENT, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
WARM = "#ef7d54"
OUT = "/Users/mitsu/SideWork/note/flow-matching/"
os.makedirs(OUT, exist_ok=True)


def new(w=12.0, h=6.75):
    fig = plt.figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


def save(fig, name):
    fig.savefig(OUT + name, facecolor=BG, dpi=100)
    plt.close(fig)
    print("saved:", name)


def panel(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color("#3a3a3a")
    return ax


# ============ 記事と同じ設定 ============
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alpha_bars = torch.cumprod(1.0 - betas, dim=0)
signal, noise = alpha_bars.sqrt(), (1 - alpha_bars).sqrt()

np.random.seed(0)
X0, _ = make_moons(n_samples=2000, noise=0.05)
X0 = X0.astype(np.float32)
X0 = (X0 - X0.mean(0)) / X0.std(0)
X0 = torch.from_numpy(X0)


def train(mode, steps=6000, bs=512, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                        nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x0 = X0[torch.randint(0, len(X0), (bs,))]
        eps = torch.randn_like(x0)
        if mode == "ddpm":
            t = torch.randint(0, T, (bs,))
            xt, target, tin = (signal[t, None] * x0 + noise[t, None] * eps, eps,
                               t[:, None].float() / T)
        else:
            t = torch.rand(bs, 1)
            xt, target, tin = ((1 - t) * x0 + t * eps, eps - x0, t)
        loss = ((net(torch.cat([xt, tin], 1)) - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


net_ddpm, net_fm = train("ddpm"), train("fm")
print("trained.")


@torch.no_grad()
def sample_fm(net, n=2000, steps=50, seed=2, path=False):
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    ts = torch.linspace(1.0, 0.0, steps + 1)
    tr = [x.clone()]
    for i in range(steps):
        x = x + (ts[i + 1] - ts[i]) * net(torch.cat([x, ts[i].repeat(n, 1)], 1))
        tr.append(x.clone())
    return (x, torch.stack(tr)) if path else x


@torch.no_grad()
def sample_ddpm(net, n=2000, steps=1000, seed=2, noise_seed=None, path=False):
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    gn = torch.Generator().manual_seed(seed if noise_seed is None else noise_seed)
    idx = torch.linspace(T - 1, 0, steps).long()
    tr = [x.clone()]
    for i, t in enumerate(idx):
        ab = alpha_bars[t]
        ab_p = alpha_bars[idx[i + 1]] if i + 1 < len(idx) else torch.tensor(1.0)
        eps = net(torch.cat([x, (t.float() / T).repeat(n, 1)], 1))
        x0h = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-4, 4)
        eps = (x - ab.sqrt() * x0h) / (1 - ab).sqrt()
        s2 = ((1 - ab_p) / (1 - ab) * (1 - ab / ab_p)).clamp(min=0)
        x = ab_p.sqrt() * x0h + (1 - ab_p - s2).clamp(min=0).sqrt() * eps
        if i + 1 < len(idx):
            x = x + s2.sqrt() * torch.randn(x.shape, generator=gn)
        tr.append(x.clone())
    return (x, torch.stack(tr)) if path else x


def sliced_w2(a, b, n_proj=256, seed=0):
    r = np.random.default_rng(seed)
    d = r.normal(size=(n_proj, 2))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return float(np.sqrt(((np.sort(np.asarray(a) @ d.T, 0) - np.sort(np.asarray(b) @ d.T, 0)) ** 2).mean()))


# ============ 1. 道のりの形 ============
_, tr_fm = sample_fm(net_fm, n=8, steps=50, seed=5, path=True)
_, tr_dd = sample_ddpm(net_ddpm, n=8, steps=200, seed=5, path=True)

fig = new()
fig.text(.5, .945, "同じ場所から始めて、通る道がまるで違う", color=FG, fontsize=35,
         ha="center", va="top")
fig.text(.5, .855, "砂嵐から絵へ向かう道を、八本ぶんたどったもの", color=SUB,
         fontsize=18, ha="center", va="top")
for j, (tr, lab, col, sub) in enumerate((
        (tr_dd, "これまでのやり方", ACCENT, "ふらつきながら進む"),
        (tr_fm, "いまのやり方", WARM, "ほぼ最短距離で進む"))):
    ax = panel(fig, [.115 + j * .45, .175, .33, .52])
    ax.scatter(X0[:, 0], X0[:, 1], s=2, color="#3f4652")
    p = tr.numpy()
    for k in range(p.shape[1]):
        ax.plot(p[:, k, 0], p[:, k, 1], color=col, lw=1.3, alpha=.8)
        ax.scatter(p[0, k, 0], p[0, k, 1], s=26, color=FG, zorder=5)
        ax.scatter(p[-1, k, 0], p[-1, k, 1], s=30, color=col, zorder=5)
    ax.set_title(lab, color=col, fontsize=23, pad=14)
    ax.text(.5, -.11, sub, transform=ax.transAxes, ha="center", color=col, fontsize=18)
fig.text(.5, .045, "白い丸が出発点。うっすら見える灰色の雲が、目指している形",
         color=SUB, fontsize=17, ha="center")
save(fig, "01_paths.png")

# ============ 2. 歩数を減らすとどうなるか ============
show = [2, 5, 50]
res = {}
fig = new()
fig.text(.5, .945, "歩数を減らすと、差がはっきり出る", color=FG, fontsize=35,
         ha="center", va="top")
fig.text(.5, .855, "同じ学習量・同じ出発点で、作るときの歩数だけを変えた", color=SUB,
         fontsize=18, ha="center", va="top")
for i, (fn, net, lab, col) in enumerate(((sample_ddpm, net_ddpm, "これまでのやり方", ACCENT),
                                         (sample_fm, net_fm, "いまのやり方", WARM))):
    for j, s in enumerate(show):
        g = fn(net, steps=s)
        res[(i, s)] = sliced_w2(g, X0)
        ax = panel(fig, [.30 + j * .155, .43 - i * .275, .145, .258])
        ax.scatter(g[:, 0], g[:, 1], s=1.6, color=col, alpha=.55)
        if i == 0:
            ax.set_title(f"{s}歩", color=FG, fontsize=20, pad=10)
    fig.text(.28, .555 - i * .275, lab, color=col, fontsize=20, ha="right", va="center")
fig.text(.5, .045, "上は五十歩でようやく形になる。下は五歩でもう形が見えている",
         color=SUB, fontsize=17, ha="center")
save(fig, "02_steps.png")
for k, v in sorted(res.items()):
    print(f"  {'DDPM' if k[0]==0 else 'FM  '} {k[1]:3}歩 : 距離 {v:.4f}")

# ============ 3. 同じ出発点から二回 ============
d1 = sample_ddpm(net_ddpm, n=500, steps=200, seed=7, noise_seed=11)
d2 = sample_ddpm(net_ddpm, n=500, steps=200, seed=7, noise_seed=12)
f1 = sample_fm(net_fm, n=500, steps=50, seed=7)
f2 = sample_fm(net_fm, n=500, steps=50, seed=7)
md, mf = float((d1 - d2).abs().max()), float((f1 - f2).abs().max())
print(f"  同じ出発点から2回の最大差: これまで {md:.2f} / いま {mf:.1e}")

fig = new()
fig.text(.5, .945, "同じところから始めて、二回作ってみる", color=FG, fontsize=35,
         ha="center", va="top")
fig.text(.5, .855, "一回目を大きな点、二回目を黒い点で重ねた", color=SUB,
         fontsize=18, ha="center", va="top")
for j, (a, b, lab, col, note) in enumerate((
        (d1, d2, "これまでのやり方", ACCENT, f"毎回ちがう場所に着く（ずれ {md:.1f}）"),
        (f1, f2, "いまのやり方", WARM, "二回とも寸分たがわず同じ（ずれ 零）"))):
    ax = panel(fig, [.115 + j * .45, .175, .33, .52])
    ax.scatter(a[:, 0], a[:, 1], s=22, color=col, alpha=.75)
    ax.scatter(b[:, 0], b[:, 1], s=6, color="#000000", alpha=.85)
    ax.set_title(lab, color=col, fontsize=23, pad=14)
    ax.text(.5, -.11, note, transform=ax.transAxes, ha="center", color=col, fontsize=18)
fig.text(.5, .045, "右は二回目が一回目の真上に乗っている。左は別の場所に散っている",
         color=SUB, fontsize=17, ha="center")
save(fig, "03_same.png")

# ============ 4. 歩数と出来ばえ ============
steps_list = [1, 2, 5, 10, 50, 200]
r_fm = {s: sliced_w2(sample_fm(net_fm, steps=s), X0) for s in steps_list}
r_dd = {s: sliced_w2(sample_ddpm(net_ddpm, steps=s), X0) for s in steps_list}
for s in steps_list:
    print(f"  {s:4d}歩   いま {r_fm[s]:.4f}   これまで {r_dd[s]:.4f}")

fig = new()
fig.text(.5, .945, "ただし、歩数をかけられるなら逆転する", color=FG, fontsize=35,
         ha="center", va="top")
fig.text(.5, .855, "縦は元のデータからの遠さ。下にあるほど上手にできている", color=SUB,
         fontsize=18, ha="center", va="top")
ax = fig.add_axes([.16, .195, .68, .49])
ax.set_facecolor(BG)
ax.plot(steps_list, [r_dd[s] for s in steps_list], "o-", color=ACCENT, lw=2.6, ms=9,
        label="これまでのやり方")
ax.plot(steps_list, [r_fm[s] for s in steps_list], "o-", color=WARM, lw=2.6, ms=9,
        label="いまのやり方")
ax.set(xscale="log", yscale="log", ylim=(.06, 4.2))
ax.set_xticks(steps_list)
ax.set_xticklabels([f"{s_}歩" for s_ in steps_list])
ax.set_yticks([])
ax.minorticks_off()
ax.set_xlabel("作るときの歩数", color=SUB, fontsize=17)
ax.tick_params(colors=SUB, labelsize=14)
ax.grid(alpha=.18, which="both", color=FG)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_color("#3a3a3a")
leg = ax.legend(fontsize=17, facecolor=BG, edgecolor="#3a3a3a", loc="upper right")
for t_, c_ in zip(leg.get_texts(), (ACCENT, WARM)):
    t_.set_color(c_)
ax.annotate("少ない歩数では大差", xy=(2, r_dd[2]), xytext=(2.4, r_dd[2] * 1.9),
            color=SUB, fontsize=16,
            arrowprops=dict(arrowstyle="-|>", color=SUB, lw=1.6))
ax.annotate("二百歩まで使うと追い越される", xy=(200, r_dd[200]), xytext=(4.2, r_dd[200] * .78),
            color=SUB, fontsize=16,
            arrowprops=dict(arrowstyle="-|>", color=SUB, lw=1.6))
fig.text(.5, .045, "速いから偉い、という単純な話ではありませんでした",
         color=SUB, fontsize=17, ha="center")
save(fig, "04_curve.png")
