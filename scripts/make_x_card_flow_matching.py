#!/usr/bin/env python3
"""X投稿用のカード画像を作る（1600x900）。
数値・軌跡は記事のコードブロックと同じ seed・同じ計算で出したものを使う。

  .venv/bin/python scripts/make_x_card_flow_matching.py
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
WARN = "#ef7d54"
OUT = "/Users/mitsu/SideWork/images/x/card_flow-matching.png"

# ---- 記事と同じデータ・同じスケジュール ----
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
    """記事の train() と同一"""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                        nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x0 = X0[torch.randint(0, len(X0), (bs,))]
        eps = torch.randn_like(x0)
        if mode == "ddpm":
            t = torch.randint(0, T, (bs,))
            xt, target, tin = (signal[t, None]*x0 + noise[t, None]*eps, eps, t[:, None].float()/T)
        else:
            t = torch.rand(bs, 1)
            xt, target, tin = ((1-t)*x0 + t*eps, eps - x0, t)
        loss = ((net(torch.cat([xt, tin], 1)) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def blank_net():
    return nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                         nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))


# 学習は6000ステップ×2で数分かかるので、重みをキャッシュして作図をやり直せるようにする
CACHE = "/tmp/x_card_fm_nets.pt"
if os.path.exists(CACHE):
    sd = torch.load(CACHE)
    net_ddpm, net_fm = blank_net(), blank_net()
    net_ddpm.load_state_dict(sd["ddpm"])
    net_fm.load_state_dict(sd["fm"])
else:
    net_ddpm, net_fm = train("ddpm"), train("fm")
    torch.save({"ddpm": net_ddpm.state_dict(), "fm": net_fm.state_dict()}, CACHE)


def traj_fm(n=12, steps=50, seed=5):
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    ts, out = torch.linspace(1.0, 0.0, steps+1), [x.clone()]
    with torch.no_grad():
        for i in range(steps):
            x = x + (ts[i+1]-ts[i]) * net_fm(torch.cat([x, ts[i].repeat(n, 1)], 1))
            out.append(x.clone())
    return torch.stack(out)


def traj_ddpm(n=12, steps=200, seed=5):
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    gn = torch.Generator().manual_seed(seed)
    idx, out = torch.linspace(T-1, 0, steps).long(), [x.clone()]
    with torch.no_grad():
        for i, t in enumerate(idx):
            ab = alpha_bars[t]
            ab_p = alpha_bars[idx[i+1]] if i+1 < len(idx) else torch.tensor(1.0)
            eps = net_ddpm(torch.cat([x, (t.float()/T).repeat(n, 1)], 1))
            x0h = ((x-(1-ab).sqrt()*eps)/ab.sqrt()).clamp(-4, 4)
            eps = (x-ab.sqrt()*x0h)/(1-ab).sqrt()
            s2 = ((1-ab_p)/(1-ab)*(1-ab/ab_p)).clamp(min=0)
            x = ab_p.sqrt()*x0h + (1-ab_p-s2).clamp(min=0).sqrt()*eps
            if i+1 < len(idx):
                x = x + s2.sqrt()*torch.randn(x.shape, generator=gn)
            out.append(x.clone())
    return torch.stack(out)


def wiggle(tr):
    L = (tr[1:]-tr[:-1]).norm(dim=2).sum(0)
    return float((L / (tr[-1]-tr[0]).norm(dim=1)).median())


tf, td = traj_fm(), traj_ddpm()
w_fm, w_dd = wiggle(tf), wiggle(td)
print(f"経路長/直線距離  FM {w_fm:.2f}  DDPM {w_dd:.2f}")

# ---- カード ----
fig = plt.figure(figsize=(16, 9), dpi=100)
fig.patch.set_facecolor(BG)

fig.text(.055, .945, "ノイズから絵へ、どんな道を通っているか",
         color=FG, fontsize=42, va="top")
fig.text(.055, .845, "拡散モデルは酔歩。FLUXが使うフローマッチングはほぼ直線。",
         color=SUB, fontsize=22, va="top")

for rect, tr, c, ttl, val in (
        ([.075, .175, .37, .52], td, ACCENT, "拡散モデル（DDPM）", w_dd),
        ([.545, .175, .37, .52], tf, WARN, "フローマッチング", w_fm)):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.scatter(X0[:, 0], X0[:, 1], s=2.5, color=SUB, alpha=.28)
    for k in range(tr.shape[1]):
        ax.plot(tr[:, k, 0], tr[:, k, 1], color=c, lw=1.5, alpha=.95)
    ax.scatter(tr[0, :, 0], tr[0, :, 1], s=34, color=FG, zorder=3)
    ax.scatter(tr[-1, :, 0], tr[-1, :, 1], s=34, color=c, zorder=3)
    ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color("#3a3a3a")
    ax.set_title(ttl, color=FG, fontsize=23, pad=13)
    ax.text(.5, -.135, f"直線距離の {val:.1f} 倍の道のり", transform=ax.transAxes,
            ha="center", color=c, fontsize=21)

fig.text(.055, .055, "同じデータ・同じネットワークで、学習則だけ差し替えて測定",
         color=SUB, fontsize=17.5)
fig.text(.945, .055, "zenn.dev/m2yagyu", color=SUB, fontsize=17.5, ha="right")

fig.savefig(OUT, facecolor=BG, dpi=100)
print("saved:", OUT)
