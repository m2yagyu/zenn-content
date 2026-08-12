#!/usr/bin/env python3
"""note第3回「AIが絵を描く仕組みを覗いてみたら…」用の画像を作る（1200x675）。

宛先はプログラミング未経験の完全初心者。数式・コード・専門用語は画像にも出さない。
モデルとforward/reverseの式は articles/diffusion-model-mnist-unet.md の検証済みコードと同一。
MNISTでU-Netをその場で学習させ、そこから出てきた本物の生成結果だけを図にする。

  .venv/bin/python scripts/make_note_diffusion.py
"""
import gzip
import math
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib_fontja  # noqa: F401

BG, FG, ACCENT, SUB = "#1c1c1c", "#f2f2f2", "#3b82f6", "#8a8a8a"
WARM = "#ef7d54"
ROOT = "/Users/mitsu/SideWork/"
OUT = ROOT + "note/diffusion/"
DATA = ROOT + "mnist_data/"
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)


def new(w=12.0, h=6.75):
    fig = plt.figure(figsize=(w, h), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


def save(fig, name):
    fig.savefig(OUT + name, facecolor=BG, dpi=100)
    plt.close(fig)
    print("saved:", name)


# ============ データ（記事と同じ正規化） ============
def load_mnist():
    os.makedirs(DATA, exist_ok=True)
    f = DATA + "train-images-idx3-ubyte.gz"
    if not os.path.exists(f):
        urllib.request.urlretrieve(
            "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz", f)
    with gzip.open(f) as fh:
        raw = np.frombuffer(fh.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
    return torch.from_numpy(raw.copy()).float().unsqueeze(1)


X_raw = load_mnist()
mean, std = X_raw.mean(), X_raw.std()
X0 = ((X_raw - mean) / std).to(device)
print("normalization mean/std:", mean.item(), std.item())

# ============ forward process（記事と同じ式） ============
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
betas_d, alphas_d, alpha_bars_d = betas.to(device), alphas.to(device), alpha_bars.to(device)


def forward_sample(x0, t):
    a_bar = alpha_bars_d[t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps, eps


# ============ 軽量U-Net（記事のセル3と同一） ============
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None, :]
        return self.mlp(torch.cat([torch.sin(ang), torch.cos(ang)], dim=1))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(tdim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.temb(self.act(temb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class NoisePredictorUNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        tdim = hidden * 4
        self.time = TimeEmbedding(hidden)
        self.stem = nn.Conv2d(1, hidden, 3, padding=1)
        self.down1 = ResBlock(hidden, hidden, tdim)
        self.pool1 = nn.Conv2d(hidden, hidden, 3, stride=2, padding=1)
        self.down2 = ResBlock(hidden, hidden * 2, tdim)
        self.pool2 = nn.Conv2d(hidden * 2, hidden * 2, 3, stride=2, padding=1)
        self.mid = ResBlock(hidden * 2, hidden * 2, tdim)
        self.up2 = nn.ConvTranspose2d(hidden * 2, hidden * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(hidden * 4, hidden, tdim)
        self.up1 = nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1)
        self.dec1 = ResBlock(hidden * 2, hidden, tdim)
        self.out = nn.Sequential(nn.GroupNorm(8, hidden), nn.SiLU(),
                                 nn.Conv2d(hidden, 1, 3, padding=1))

    def forward(self, x, t):
        temb = self.time(t)
        s = self.stem(x)
        h1 = self.down1(s, temb)
        h2 = self.down2(self.pool1(h1), temb)
        m = self.mid(self.pool2(h2), temb)
        u2 = self.dec2(torch.cat([self.up2(m), h2], dim=1), temb)
        u1 = self.dec1(torch.cat([self.up1(u2), h1], dim=1), temb)
        return self.out(u1)


model = NoisePredictorUNet(hidden=64).to(device)
print("n_params:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

# ============ 学習（記事のセル4と同一：6000ステップ + EMA） ============
CKPT = DATA + "note_diffusion_ema.pt"
n_steps, batch_size, ema_decay = 6000, 128, 0.995
if os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT, map_location=device))
    print("loaded checkpoint:", CKPT)
else:
    msd = model.state_dict()
    ema = {k: v.detach().clone() for k, v in msd.items()}
    losses, t0 = [], time.time()
    for step in range(n_steps):
        idx = torch.randint(0, X0.shape[0], (batch_size,))
        x0 = X0[idx]
        t = torch.randint(0, T, (batch_size,), device=device)
        x_t, eps = forward_sample(x0, t)
        loss = ((model(x_t, t) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for k, v in msd.items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)
                else:
                    ema[k].copy_(v)
        losses.append(loss.item())
        if (step + 1) % 500 == 0:
            print(f"step {step+1}: loss = {np.mean(losses[-500:]):.4f} "
                  f"({time.time()-t0:.0f}s)")
    print("最終的な誤差:", np.mean(losses[-500:]))
    model.load_state_dict(ema)
    torch.save(model.state_dict(), CKPT)


# ============ reverse process（記事と同じ式） ============
@torch.no_grad()
def sample(n_samples=32, record_steps=()):
    x = torch.randn(n_samples, 1, 28, 28, device=device)
    frames = {}
    for t in reversed(range(T)):
        tb = torch.full((n_samples,), t, dtype=torch.long, device=device)
        eps_theta = model(x, tb)
        a_t, ab_t, b_t = alphas_d[t], alpha_bars_d[t], betas_d[t]
        m = (1 / torch.sqrt(a_t)) * (x - (b_t / torch.sqrt(1 - ab_t)) * eps_theta)
        x = m + torch.sqrt(b_t) * torch.randn_like(x) if t > 0 else m
        if t in record_steps:
            frames[t] = x.clone().cpu()
    return frames


model.eval()
show_t = [999, 700, 400, 100, 0]
FRAMES = DATA + "note_diffusion_frames.pt"
if os.path.exists(FRAMES):
    frames = torch.load(FRAMES)
    print("loaded frames:", FRAMES)
else:
    frames = sample(n_samples=32, record_steps=set(show_t))
    torch.save(frames, FRAMES)
    print("sampled.")


def cell(fig, j, i, x0=.155, y0=.52, dx=.145, dy=.20, w=.10):
    """図の中に正方形のマス目を置く（figureが12x6.75なので縦横比を補正する）"""
    return fig.add_axes([x0 + j * dx, y0 - i * dy, w, w * 12 / 6.75])


def to_img(x):
    """正規化を戻して0〜1の画素に直す"""
    v = x * std.item() + mean.item()
    return np.clip(v / 255.0, 0, 1)


# ============ 1. インクは、ひとりでに広がる ============
rng = np.random.default_rng(0)
n_p = 900
pos = rng.normal(0, 0.12, size=(n_p, 2))
snaps = {}
for step in range(1, 401):
    pos = pos + rng.normal(0, 0.055, size=(n_p, 2))
    if step in (1, 40, 400):
        snaps[step] = pos.copy()

fig = new()
fig.text(.5, .945, "インクは、ひとりでに広がる", color=FG, fontsize=36, ha="center", va="top")
fig.text(.5, .855, "水に落とした小さな粒を、時間が経つ順に並べたもの", color=SUB,
         fontsize=19, ha="center", va="top")
for j, (step, lab) in enumerate(((1, "落とした直後"), (40, "しばらく後"), (400, "ずっと後"))):
    ax = fig.add_axes([.075 + j * .308, .215, .245, .49])
    ax.set_facecolor(BG)
    p = snaps[step]
    ax.scatter(p[:, 0], p[:, 1], s=12, color=ACCENT if j == 0 else (ACCENT if j == 1 else WARM),
               alpha=.55)
    ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xticks=[], yticks=[])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color("#3a3a3a")
    ax.set_title(lab, color=FG, fontsize=22, pad=14)
fig.text(.5, .075, "広がる向きにしか進まない。散らばったインクは、勝手には戻らない",
         color=SUB, fontsize=19, ha="center")
save(fig, "01_ink.png")

# ============ 2. 絵も同じように、壊れていく ============
fig = new()
fig.text(.5, .945, "絵も、同じように壊れていく", color=FG, fontsize=36, ha="center", va="top")
fig.text(.5, .855, "手書きの数字に、少しずつノイズを混ぜたもの", color=SUB,
         fontsize=19, ha="center", va="top")
digits = X0[:3].cpu()
steps = [0, 100, 300, 600, 999]
for j, t in enumerate(steps):
    for i in range(3):
        ax = cell(fig, j, i)
        if t == 0:
            img = digits[i, 0].numpy()
        else:
            ab = alpha_bars[t]
            g = torch.Generator().manual_seed(100 + i * 10 + j)
            eps = torch.randn(digits[i, 0].shape, generator=g)
            img = (torch.sqrt(ab) * digits[i, 0] + torch.sqrt(1 - ab) * eps).numpy()
        ax.imshow(to_img(img), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    lab = {0: "もとの絵", 100: "少し混ぜる", 300: "もっと混ぜる",
           600: "かなり混ぜる", 999: "砂嵐"}[t]
    fig.text(.205 + j * .145, .745, lab, color=FG if t == 0 else (WARM if t == 999 else SUB),
             fontsize=19, ha="center")
fig.text(.5, .045, "最後は、もとが何の数字だったか誰にも分からなくなる",
         color=SUB, fontsize=19, ha="center")
save(fig, "02_break.png")

# ============ 3. 砂嵐から、数字が現れる ============
fig = new()
fig.text(.5, .945, "その逆再生を覚えさせると、砂嵐から数字が出てくる", color=FG,
         fontsize=33, ha="center", va="top")
fig.text(.5, .855, "何も無いところから始めて、少しずつノイズを取り除いた（数字は戻した歩数）", color=SUB,
         fontsize=19, ha="center", va="top")
for j, t in enumerate(show_t):
    for i in range(3):
        ax = cell(fig, j, i)
        ax.imshow(to_img(frames[t][i, 0].numpy()), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    lab = {999: "はじめの砂嵐", 700: "三百歩ぶん", 400: "六百歩ぶん",
           100: "九百歩ぶん", 0: "できあがり"}[t]
    fig.text(.205 + j * .145, .745, lab, color=WARM if t == 999 else (ACCENT if t == 0 else SUB),
             fontsize=19, ha="center")
fig.text(.5, .045, "左から右へ、一歩ずつ戻している。誰も数字の形を教えていない",
         color=SUB, fontsize=19, ha="center")
save(fig, "03_reverse.png")

# ============ 4. 出てきた数字 ============
fig = new()
fig.text(.5, .945, "出てきたのは、どこにも無かった数字だった", color=FG,
         fontsize=35, ha="center", va="top")
fig.text(.5, .855, "同じやり方を32回くり返して作ったもの", color=SUB,
         fontsize=19, ha="center", va="top")
g0 = frames[0]
for k in range(32):
    r, c = divmod(k, 8)
    ax = cell(fig, c, r, x0=.155, y0=.60, dx=.0885, dy=.155, w=.082)
    ax.imshow(to_img(g0[k, 0].numpy()), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
fig.text(.5, .045, "手書きの見本を覚えたのではなく、書き方のほうを覚えている",
         color=SUB, fontsize=19, ha="center")
save(fig, "04_grid.png")
