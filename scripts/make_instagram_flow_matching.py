#!/usr/bin/env python3
"""Instagram用スライド(1080x1080)を作る。宛先はプログラミング未経験の完全初心者。

持ち帰ってもらう理解は3つだけ:
  ① 絵を作るAIは、砂嵐のような画面から少しずつ形を掘り出している
  ② その「道のり」には、ぐるぐる回る道と、ほぼまっすぐな道がある
  ③ 最新の画像生成AIはまっすぐな道に変えた。だから手数が減って速い

数式・コード・専門用語は出さない。図はすべて記事と同じ計算の実データ。

  .venv/bin/python scripts/make_instagram_flow_matching.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401
from matplotlib.patches import FancyBboxPatch
from sklearn.datasets import make_moons

BG, FG, ACCENT, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
WARM = "#ef7d54"
OUT = "/Users/mitsu/SideWork/instagram/flow-matching-vs-diffusion/"
PX = 1080
os.makedirs(OUT, exist_ok=True)


def new_slide():
    fig = plt.figure(figsize=(PX / 100, PX / 100), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


def box(fig, x, y, w, h, color=None, lw=2):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=fig.transFigure, facecolor="none",
        edgecolor=color or "#3a3a3a", linewidth=lw, zorder=0))


def save(fig, name):
    fig.savefig(OUT + name, facecolor=BG, dpi=100)
    plt.close(fig)
    print("saved:", name)


# ============ 記事と同じ計算 ============
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alpha_bars = torch.cumprod(1.0 - betas, dim=0)
signal, noise = alpha_bars.sqrt(), (1 - alpha_bars).sqrt()

np.random.seed(0)
X0, _ = make_moons(n_samples=2000, noise=0.05)
X0 = X0.astype(np.float32)
X0 = (X0 - X0.mean(0)) / X0.std(0)
X0 = torch.from_numpy(X0)


def blank_net():
    return nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                         nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))


def train(mode, steps=6000, bs=512, seed=0):
    torch.manual_seed(seed)
    net, opt = blank_net(), None
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x0 = X0[torch.randint(0, len(X0), (bs,))]
        eps = torch.randn_like(x0)
        if mode == "ddpm":
            t = torch.randint(0, T, (bs,))
            xt, tg, tin = (signal[t, None]*x0 + noise[t, None]*eps, eps, t[:, None].float()/T)
        else:
            t = torch.rand(bs, 1)
            xt, tg, tin = ((1-t)*x0 + t*eps, eps - x0, t)
        loss = ((net(torch.cat([xt, tin], 1)) - tg) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


CACHE = "/tmp/x_card_fm_nets.pt"
if os.path.exists(CACHE):
    sd = torch.load(CACHE)
    net_ddpm, net_fm = blank_net(), blank_net()
    net_ddpm.load_state_dict(sd["ddpm"]); net_fm.load_state_dict(sd["fm"])
else:
    net_ddpm, net_fm = train("ddpm"), train("fm")
    torch.save({"ddpm": net_ddpm.state_dict(), "fm": net_fm.state_dict()}, CACHE)


@torch.no_grad()
def sample_fm(steps, n=2000, seed=2):
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        x = x + (ts[i+1]-ts[i]) * net_fm(torch.cat([x, ts[i].repeat(n, 1)], 1))
    return x


@torch.no_grad()
def sample_ddpm(steps, n=2000, seed=2):
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    gn = torch.Generator().manual_seed(seed)
    idx = torch.linspace(T-1, 0, steps).long()
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
    return x


def traj(kind, n=12, seed=5):
    if kind == "fm":
        x = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
        ts, out = torch.linspace(1.0, 0.0, 51), [x.clone()]
        with torch.no_grad():
            for i in range(50):
                x = x + (ts[i+1]-ts[i]) * net_fm(torch.cat([x, ts[i].repeat(n, 1)], 1))
                out.append(x.clone())
        return torch.stack(out)
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    gn = torch.Generator().manual_seed(seed)
    idx, out = torch.linspace(T-1, 0, 200).long(), [x.clone()]
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


def wig(tr):
    L = (tr[1:]-tr[:-1]).norm(dim=2).sum(0)
    return float((L / (tr[-1]-tr[0]).norm(dim=1)).median())


tf, td = traj("fm"), traj("ddpm")
w_fm, w_dd = wig(tf), wig(td)
print(f"道のり  まっすぐ {w_fm:.1f}倍 / ぐるぐる {w_dd:.1f}倍")


def scatter_ax(fig, rect, pts, color, s=3.0, alpha=.6):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.scatter(pts[:, 0], pts[:, 1], s=s, color=color, alpha=alpha)
    ax.set(xlim=(-3, 3), ylim=(-3, 3), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_color("#3a3a3a")
    return ax


# ============ 1枚目 ============
fig = new_slide()
box(fig, .07, .07, .86, .86)
fig.text(.5, .80, "AIが絵にたどり着くまでの", color=FG, fontsize=44, ha="center")
fig.text(.5, .715, "「道のり」", color=WARM, fontsize=62, ha="center")
fig.text(.5, .60, "ずいぶん遠回りしていました", color=FG, fontsize=34, ha="center")
ax = scatter_ax(fig, [.255, .215, .49, .34], X0, SUB, s=2.0, alpha=.40)
for k in range(6):
    ax.plot(tf[:, k, 0], tf[:, k, 1], color=WARM, lw=1.6, alpha=.95)
for k in range(2):
    ax.plot(td[:, k, 0], td[:, k, 1], color=ACCENT, lw=.8, alpha=.55)
fig.text(.5, .145, "その道を、実際に描いてみました", color=SUB, fontsize=27, ha="center")
save(fig, "slide1_title.png")

# ============ 2枚目 ============
fig = new_slide()
fig.text(.5, .915, "AIは「砂嵐」から形を掘り出している", color=FG, fontsize=38, ha="center")
fig.text(.5, .845, "いきなり描くのではなく、少しずつ整えていきます",
         color=SUB, fontsize=25, ha="center")
stages = [(1, "はじめ"), (3, "とちゅう"), (50, "できあがり")]
for j, (st, lab) in enumerate(stages):
    pts = torch.randn(2000, 2, generator=torch.Generator().manual_seed(2)) if j == 0 \
        else sample_fm(st)
    ax = scatter_ax(fig, [.07 + j*.312, .40, .275, .275], pts, WARM if j == 2 else SUB,
                    s=2.2, alpha=.5)
    ax.set_title(lab, color=FG, fontsize=27, pad=14)
    if j < 2:
        fig.text(.352 + j*.312, .535, "▶", color=FG, fontsize=26, ha="center")
fig.text(.5, .285, "ここでは分かりやすいように、\n「三日月がふたつ」という簡単な形を作らせています",
         color=SUB, fontsize=24, ha="center", linespacing=1.7)
fig.text(.5, .13, "この「少しずつ整える」やり方が、\nいまの画像生成AIの基本です",
         color=FG, fontsize=27, ha="center", linespacing=1.7)
save(fig, "slide2_process.png")

# ============ 3枚目 ============
fig = new_slide()
fig.text(.5, .935, "道のりを描いたら、こうだった", color=FG, fontsize=40, ha="center")
for rect, tr, c, ttl, val in (
        ([.075, .40, .38, .38], td, ACCENT, "これまでのやり方", w_dd),
        ([.545, .40, .38, .38], tf, WARM, "最新のやり方", w_fm)):
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    ax.scatter(X0[:, 0], X0[:, 1], s=2, color=SUB, alpha=.25)
    for k in range(tr.shape[1]):
        ax.plot(tr[:, k, 0], tr[:, k, 1], color=c, lw=1.4, alpha=.95)
    ax.scatter(tr[0, :, 0], tr[0, :, 1], s=26, color=FG, zorder=3)
    ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for sp in ax.spines.values():
        sp.set_color("#3a3a3a")
    ax.set_title(ttl, color=FG, fontsize=29, pad=14)
    ax.text(.5, -.14, f"まっすぐ行く場合の\n{val:.1f}倍の長さ", transform=ax.transAxes,
            ha="center", va="top", color=c, fontsize=25, linespacing=1.6)
fig.text(.5, .10, "白い点が出発地点。ゴールは同じなのに、\n左はぐるぐる回ってから着いています",
         color=SUB, fontsize=25, ha="center", linespacing=1.7)
save(fig, "slide3_path.png")

# ============ 4枚目 ============
fig = new_slide()
fig.text(.5, .935, "まっすぐにしたら、手数が減った", color=FG, fontsize=40, ha="center")
fig.text(.5, .868, "どちらも「5回だけ」整えた結果です", color=SUB, fontsize=26, ha="center")
for rect, pts, c, ttl in (
        ([.075, .40, .38, .38], sample_ddpm(5), ACCENT, "これまでのやり方"),
        ([.545, .40, .38, .38], sample_fm(5), WARM, "最新のやり方")):
    ax = scatter_ax(fig, rect, pts, c, s=2.6, alpha=.55)
    ax.set_title(ttl, color=FG, fontsize=29, pad=14)
ax = fig.axes[-2]
fig.text(.265, .345, "まだ ぼんやりした塊", color=ACCENT, fontsize=26, ha="center")
fig.text(.735, .345, "もう形が見えている", color=WARM, fontsize=26, ha="center")
fig.text(.5, .175, "道がまっすぐなら、大またで歩いても迷わない。\nだから最新のAIは、少ない手数で速く絵が出せます",
         color=FG, fontsize=27, ha="center", linespacing=1.75)
save(fig, "slide4_steps.png")

# ============ 5枚目 ============
fig = new_slide()
box(fig, .07, .07, .86, .86)
fig.text(.5, .865, "まとめ", color=WARM, fontsize=46, ha="center", va="top")
for i, t1 in enumerate([
        "① 絵を作るAIは、砂嵐から\n　　少しずつ形を掘り出している",
        "② その道のりには、ぐるぐる回る道と\n　　ほぼまっすぐな道がある",
        "③ 最新の画像生成AIはまっすぐな道に変え、\n　　少ない手数で絵を出せるようになった"]):
    fig.text(.115, .730 - i*.150, t1, color=FG, fontsize=26, ha="left", va="top",
             linespacing=1.8)
fig.text(.5, .300, "図はすべて、実際に計算して\n出てきたものをそのまま載せています",
         color=SUB, fontsize=23, ha="center", va="top", linespacing=1.7)
fig.text(.5, .180, "もっと詳しく知りたい人は\nプロフィールのリンクから",
         color=FG, fontsize=26, ha="center", va="top", linespacing=1.7)
save(fig, "slide5_cta.png")

print("完了:", OUT)
