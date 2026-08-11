---
title: "FLUXが使うフローマッチングって結局何なの？"
emoji: "🏹"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["deeplearning", "生成ai", "llm", "diffusion", "物理"]
published: true
---

この記事を読むと、いまの画像生成の主流であるフローマッチングが、拡散モデルから「熱浴」を取り外しただけのものであることを、自分の手で確かめられます。同じデータ・同じネットワークで学習則だけを差し替えて、2つを並べて測ります。

GPUは不要です。2次元の点2000個と、隠れ層3枚のMLPをCPUで動かすだけで、最後まで通ります。載せているコードは作図まで含んでいるので、上から順にコピペすると記事と同じ図が手元に出ます。

[前回の記事](https://zenn.dev/m2yagyu/articles/attention-hopfield-associative-memory)の最後に、「$\beta$ がもう一箇所、思わぬところにも隠れている」と書きました。その回収です。結論から言うと、拡散モデルの `betas` は温度に関係する量で、[FLUX.1](https://github.com/black-forest-labs/flux) や [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) はそれを捨てています。

:::message
**記号の約束**：行列は大文字、ベクトルは小文字で書きます。進行度はどちらの方式でも「0がデータ、1がノイズ」に揃えます。拡散モデル側は離散ステップ $t \in \{0, \dots, 999\}$、フローマッチング側は連続時刻 $t \in [0,1]$ で、同じ軸の目盛りが違うだけです。物理の温度は、総ステップ数 $T$ と紛らわしいので $\mathcal{T}$ と書きます。
:::

## 拡散モデルの β は、何の速さだったのか

[拡散モデルの記事](https://zenn.dev/m2yagyu/articles/diffusion-model-toy-physics)で、こんなコードを書きました。

```python:抜粋
betas = torch.linspace(1e-4, 0.02, T)
```

これは「各ステップでどれだけノイズを混ぜるか」のスケジュールです。ここで注意したいのは、**$\beta_t$ は温度そのものではない**ということです。よくある誤解なので、先に潰しておきます。

拡散モデルのforward processは、物理でいえばデータを熱浴に浸す操作です。熱浴の温度は最初から最後まで一定で、[前回の拡散モデル記事](https://zenn.dev/m2yagyu/articles/diffusion-model-toy-physics)で見たとおり $k_B\mathcal{T} = 1$、つまり平衡状態が標準正規分布 $\mathcal{N}(0, I)$ になる大きさに固定されています。$\beta_t$ が決めているのは温度ではなく、**その平衡へ向かう速さ**です。熱いお湯に手を入れる話ではなく、どれくらいの勢いで浸すかの話です。

実際に測ってみます。

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# グラフの日本語が豆腐にならないようにする。無ければこの場で入れる（初回のみ）
try:
    import matplotlib_fontja  # noqa: F401
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib-fontja"])
    import matplotlib_fontja  # noqa: F401

BLUE, ORANGE, GRAY = "#3b82f6", "#ef7d54", "#8a8a8a"

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alpha_bars = torch.cumprod(1.0 - betas, dim=0)   # alpha_bar_t = (1-beta_1)...(1-beta_t)
signal, noise = alpha_bars.sqrt(), (1 - alpha_bars).sqrt()

print(f"beta_t: {betas[0]:.4f} -> {betas[-1]:.4f}")
for t in (0, 99, 299, 599, 999):
    print(f"  t={t:4d}  残る信号 {signal[t]:.3f}   混ざったノイズの分散 {1-alpha_bars[t]:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
ax[0].plot(betas, color=ORANGE, lw=2)
ax[0].set(title=r"$\beta_t$ = 1ステップで熱浴に浸す強さ", xlabel="ステップ $t$", ylabel=r"$\beta_t$")
ax[1].plot(1 - alpha_bars, color=BLUE, lw=2, label=r"混ざったノイズの分散 $1-\bar\alpha_t$")
ax[1].axhline(1.0, color=GRAY, ls=":", lw=1.4)
ax[1].text(30, 1.03, "熱浴の温度（分散1）", color=GRAY, fontsize=9)
ax[1].set(title="行き先は最初から決まっている", xlabel="ステップ $t$", ylim=(-.05, 1.15))
ax[1].legend(fontsize=9, loc="lower right")
for a in ax:
    a.grid(alpha=.3)
plt.tight_layout()
plt.savefig("flow-matching-beta-schedule.png", bbox_inches="tight", dpi=150)
plt.show()
```

```
beta_t: 0.0001 -> 0.0200
  t=   0  残る信号 1.000   混ざったノイズの分散 0.0001
  t=  99  残る信号 0.947   混ざったノイズの分散 0.1030
  t= 299  残る信号 0.630   混ざったノイズの分散 0.6036
  t= 599  残る信号 0.161   混ざったノイズの分散 0.9741
  t= 999  残る信号 0.006   混ざったノイズの分散 1.0000
```

![betaスケジュールと到達点](/images/flow-matching-beta-schedule.png)

右の図が言いたいことのすべてです。分散は1に張り付いて、それ以上は上がりません。$t=599$ の時点でもう0.974まで来ていて、残りの400ステップはほとんど何も起きていません。

```python:抜粋
alpha_bars = torch.cumprod(1.0 - betas, dim=0)
signal, noise = alpha_bars.sqrt(), (1 - alpha_bars).sqrt()
```

$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon$ の2つの係数がこれです。$\bar\alpha_t$ は総乗なので、$\beta_t$ を少し変えても最終的な行き先（分散1）は変わりません。変わるのは**そこへ着くまでの道のり**だけです。

つまり $\beta$ は目的地ではなく速度プロファイルでした。ここで自然な疑問が出ます。行き先が最初から $\mathcal{N}(0,I)$ と決まっているなら、わざわざ熱浴に浸して**確率的に**たどり着く必要があるのでしょうか。出発点と到着点を直接つないではいけないのでしょうか。

## その熱浴を外すと、何が残るのか

外してみます。ノイズを混ぜるのをやめて、データ点 $x_0$ とノイズ点 $x_1$ を**直線で結ぶ**だけにします。

$$
x_t = (1-t)\, x_0 + t\, x_1, \qquad t \in [0,1]
$$

これだけです。確率過程でも、マルコフ連鎖でも、微分方程式でもありません。ただの線形補間です。この置き換えが[フローマッチング](https://arxiv.org/abs/2210.02747)、より具体的には[rectified flow](https://arxiv.org/abs/2209.03003)と呼ばれるものの出発点になります。

同じデータで並べて見てみます。データは[拡散モデルの記事](https://zenn.dev/m2yagyu/articles/diffusion-model-toy-physics)と同じ「二つの月（two moons）」、三日月のような弧が2つ噛み合った形の点の集まりです。

```python
from sklearn.datasets import make_moons

np.random.seed(0)
X0, _ = make_moons(n_samples=2000, noise=0.05)
X0 = X0.astype(np.float32)
X0 = (X0 - X0.mean(0)) / X0.std(0)   # 平衡が分散1なので、データ側も揃えておく
X0 = torch.from_numpy(X0)

torch.manual_seed(1)
X1 = torch.randn_like(X0)            # 終点：標準正規ノイズ

fr = [0.0, 0.25, 0.5, 0.75, 1.0]
fig, ax = plt.subplots(2, 5, figsize=(12.6, 5.4))
for j, f in enumerate(fr):
    t = min(int(f * (T - 1)), T - 1)
    xt = signal[t] * X0 + noise[t] * X1            # 拡散モデルの周辺分布
    ax[0, j].scatter(xt[:, 0], xt[:, 1], s=2, color=BLUE, alpha=.5)
    ax[0, j].set_title(f"DDPM   t={t}", fontsize=10)
    xs = (1 - f) * X0 + f * X1                     # ただの直線補間
    ax[1, j].scatter(xs[:, 0], xs[:, 1], s=2, color=ORANGE, alpha=.5)
    ax[1, j].set_title(f"直線補間   t={f:.2f}", fontsize=10)
for a in ax.ravel():
    a.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xticks=[], yticks=[])
    a.set_aspect("equal")
ax[0, 0].set_ylabel("熱浴に浸していく", fontsize=10.5)
ax[1, 0].set_ylabel("まっすぐ運ぶ", fontsize=10.5)
plt.tight_layout()
plt.savefig("flow-matching-forward-compare.png", bbox_inches="tight", dpi=150)
plt.show()

ts = np.linspace(0, 1, 51)
v_ddpm = [float((signal[min(int(f*(T-1)), T-1)]*X0 + noise[min(int(f*(T-1)), T-1)]*X1).var()) for f in ts]
v_lin = [float(((1-f)*X0 + f*X1).var()) for f in ts]
print(f"中間 t=0.5 の分散   DDPM {v_ddpm[25]:.3f}   直線補間 {v_lin[25]:.3f}")

plt.figure(figsize=(5.4, 3.5))
plt.plot(ts, v_ddpm, color=BLUE, lw=2, label="DDPM")
plt.plot(ts, v_lin, color=ORANGE, lw=2, label="直線補間")
plt.axhline(1.0, color=GRAY, ls=":", lw=1.3)
plt.xlabel("進行度（0=データ, 1=ノイズ）")
plt.ylabel("分布の分散")
plt.grid(alpha=.3)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("flow-matching-variance.png", bbox_inches="tight", dpi=150)
plt.show()
```

```
中間 t=0.5 の分散   DDPM 1.006   直線補間 0.497
```

![forward processの比較](/images/flow-matching-forward-compare.png)

上の段（DDPM）は、25%進んだ時点でもう形が消えています。下の段（直線補間）は、同じ25%でまだ2つの塊が見えています。同じ距離を進むのに、熱浴に浸す方は早々に構造を壊してしまう。

そしてもう一つ、見逃せない違いが出ています。

![分散の推移](/images/flow-matching-variance.png)

```python:抜粋
v_ddpm = ...  # DDPM: 途中もずっと分散1のまま
v_lin  = ...  # 直線補間: 中間で 0.497 まで縮む
```

DDPMは途中も一貫して分散1を保ちますが、直線補間は中間で分散が半分に縮みます。$(1-t)x_0 + t x_1$ の分散が $(1-t)^2 + t^2$ になるからで、$t=0.5$ で $0.5$ です。これはrectified flowの既知の弱点で、SD3が[timestep shift](https://arxiv.org/abs/2403.03206)という補正を入れているのはこの縮みへの対処です。**直線にすれば全部よくなる、という話ではありません。**

## 直線で結ぶだけで、本当に学習できるのか

直線を引いただけでは生成モデルになりません。生成時には $x_0$ が分かっていない（それを作りたい）ので、「いまいる場所と時刻から、どっちへ動けばいいか」をネットワークに覚えさせる必要があります。

直線補間の場合、動く向きは微分すれば出ます。

$$
\frac{dx_t}{dt} = x_1 - x_0
$$

時刻によらず一定です。これを**速度**と呼びます。学習は、$x_t$ と $t$ を入力して $x_1 - x_0$ を当てさせるだけです。

拡散モデル側は、同じ $x_t$ と $t$ から**混ぜたノイズ $\varepsilon$** を当てさせます（[DDPM](https://arxiv.org/abs/2006.11239)の標準的な作り方です）。当てさせるものが違うだけで、ネットワークも入力も同じにできます。そこで、**同じネットワーク・同じデータで、当てる対象だけを差し替えて**比べます。

```python
def train(mode, steps=6000, bs=512, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                        nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x0 = X0[torch.randint(0, len(X0), (bs,))]
        eps = torch.randn_like(x0)
        if mode == "ddpm":
            t = torch.randint(0, T, (bs,))                       # 離散ステップ
            xt, target, tin = (signal[t, None]*x0 + noise[t, None]*eps, eps, t[:, None].float()/T)
        else:
            t = torch.rand(bs, 1)                                # 連続時刻 [0,1]
            xt, target, tin = ((1-t)*x0 + t*eps, eps - x0, t)    # 直線補間と、その速度
        loss = ((net(torch.cat([xt, tin], 1)) - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


net_ddpm, net_fm = train("ddpm"), train("fm")


@torch.no_grad()
def sample_fm(net, n=2000, steps=50, seed=2):
    """ODEをEuler法で t=1 から t=0 へ解くだけ。乱数は最初の1回しか使わない"""
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        x = x + (ts[i+1] - ts[i]) * net(torch.cat([x, ts[i].repeat(n, 1)], 1))
    return x


@torch.no_grad()
def sample_ddpm(net, n=2000, steps=1000, seed=2, noise_seed=None):
    """DDIM系の更新式。eta=1 なので毎ステップ乱数を足す"""
    x = torch.randn(n, 2, generator=torch.Generator().manual_seed(seed))
    gn = torch.Generator().manual_seed(seed if noise_seed is None else noise_seed)
    idx = torch.linspace(T - 1, 0, steps).long()
    for i, t in enumerate(idx):
        ab = alpha_bars[t]
        ab_p = alpha_bars[idx[i+1]] if i+1 < len(idx) else torch.tensor(1.0)
        eps = net(torch.cat([x, (t.float()/T).repeat(n, 1)], 1))
        x0h = ((x - (1-ab).sqrt()*eps) / ab.sqrt()).clamp(-4, 4)
        eps = (x - ab.sqrt()*x0h) / (1-ab).sqrt()     # クランプ後のx0と辻褄を合わせ直す
        s2 = ((1-ab_p)/(1-ab) * (1-ab/ab_p)).clamp(min=0)
        x = ab_p.sqrt()*x0h + (1-ab_p-s2).clamp(min=0).sqrt()*eps
        if i+1 < len(idx):
            x = x + s2.sqrt() * torch.randn(x.shape, generator=gn)
    return x


def sliced_w2(a, b, n_proj=256, seed=0):
    """2つの点群の距離。ランダムな向きに射影して1次元ずつ比べる"""
    r = np.random.default_rng(seed)
    d = r.normal(size=(n_proj, 2)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    return float(np.sqrt(((np.sort(np.asarray(a)@d.T, 0) - np.sort(np.asarray(b)@d.T, 0))**2).mean()))


gen_fm, gen_dd = sample_fm(net_fm, steps=50), sample_ddpm(net_ddpm, steps=1000)
d_fm, d_dd = sliced_w2(gen_fm, X0), sliced_w2(gen_dd, X0)
print(f"データとの距離   FM 50ステップ {d_fm:.4f}   DDPM 1000ステップ {d_dd:.4f}")

fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
for a, g, ttl, c in ((ax[0], X0, "学習データ", GRAY),
                     (ax[1], gen_dd, f"DDPM 1000ステップ（距離 {d_dd:.3f}）", BLUE),
                     (ax[2], gen_fm, f"フローマッチング 50ステップ（距離 {d_fm:.3f}）", ORANGE)):
    a.scatter(g[:, 0], g[:, 1], s=3, color=c, alpha=.6)
    a.set(title=ttl, xlim=(-3, 3), ylim=(-3, 3), xticks=[], yticks=[])
    a.set_aspect("equal")
ax[1].title.set_fontsize(10.5); ax[2].title.set_fontsize(10.5)
plt.tight_layout()
plt.savefig("flow-matching-generated.png", bbox_inches="tight", dpi=150)
plt.show()
```

```
データとの距離   FM 50ステップ 0.1061   DDPM 1000ステップ 0.1198
```

![生成結果の比較](/images/flow-matching-generated.png)

両方とも二つの月が出ています。フローマッチングは50ステップ、DDPMは1000ステップです。

### 変わったのは、実質2行だけ

差分はここに凝縮されています。

```python:抜粋
# DDPM
t = torch.randint(0, T, (bs,))
xt, target = signal[t, None]*x0 + noise[t, None]*eps, eps        # ノイズを当てる

# フローマッチング
t = torch.rand(bs, 1)
xt, target = (1-t)*x0 + t*eps, eps - x0                          # 速度を当てる
```

ネットワークの形も、最適化も、学習ステップ数も同じです。`alpha_bars` が消えて $(1-t)$ と $t$ になり、当てる対象が $\varepsilon$ から $\varepsilon - x_0$ になっただけ。**`betas` が使われなくなったことに注目してください。** スケジュールという概念自体がフローマッチング側には存在しません。

サンプリング側の差はもっと極端です。

```python:抜粋
# フローマッチング：乱数は最初の1回だけ
x = x + (ts[i+1] - ts[i]) * net(...)

# DDPM：毎ステップ乱数を足す
x = x + s2.sqrt() * torch.randn(x.shape, generator=gn)
```

前者は常微分方程式（ODE）をEuler法で解いているだけ、後者は確率微分方程式（SDE）を解いています。この違いが後の章で効いてきます。

:::message alert
`sample_ddpm` の `eps = (x - ab.sqrt()*x0h) / (1-ab).sqrt()` の行は、消すと壊れます。$\hat{x}_0$ を `clamp` した後に生の $\varepsilon$ をそのまま使うと、両者が矛盾したまま更新が積み上がり、1000ステップ回したときに一部のサンプルが発散して `nan` になります（実際にこれで一度壊しました）。クランプした $\hat{x}_0$ から $\varepsilon$ を逆算し直すことで辻褄を合わせています。
:::

## なぜ「速度」を覚えると、絵が出てくるのか

ここが直感的に引っかかるところです。学習時に当てさせているのは $x_1 - x_0$、つまり**特定のペアを結ぶ向き**です。生成時には $x_0$ が分かりません。それなのに、なぜ動く向きが決まるのでしょうか。

### 朝の駅を、真上から眺める

朝の駅の通路を、真上から見下ろしていると思ってください。

一人ひとりは、自分の目的地へ向かってまっすぐ歩いています。改札へ行く人、階段へ向かう人、売店に寄る人。行き先はばらばらで、本人たちは他人がどこへ行くかを知りません。

それでも上から眺めると、「この辺りにいる人は、だいたいこっちへ流れている」という模様が見えます。通路の入口付近は奥へ向かう流れ、階段の手前は上りへ吸い込まれる流れ。**個々の人の軌跡は互いに交差しますが、「その地点にいる人たちの平均の向き」は各地点でひとつに決まります。** だから流れの模様そのものは交差しません。

フローマッチングが学んでいるのは、この上から見た流れです。学習時に見せるのは個々の人（$x_0$ と $x_1$ のペア）が引いた直線ですが、二乗誤差で回帰させると、モデルはその場所を通る全員の向きの平均に収束します。

そして生成時にやっているのは、流れに乗ることです。自分がどの月へ行きたいのかを知らないまま、いまいる場所の流れに従って一歩進む。それを繰り返すと、勝手に月の上へ運ばれます。

### 式で言えば、条件付き分布の一次モーメント

いま言ったことは、そのまま式になります。

$$
v_\theta(x, t) \;\longrightarrow\; \mathbb{E}\!\left[\, x_1 - x_0 \;\middle|\; x_t = x \,\right]
$$

これは $p(x_1 - x_0 \mid x_t = x)$ の**一次モーメント**、つまり条件付き期待値そのものです。二乗誤差で回帰させると条件付き期待値に収束する、という一般的な事実がそのまま効いています。これが[フローマッチングの中心的な主張](https://arxiv.org/abs/2210.02747)で、条件付きの単純な経路を平均すると、周辺分布を正しく運ぶ流れが得られる、というものです。

同じ見方をするとDDPMの $\varepsilon$ 予測も $\mathbb{E}[\varepsilon \mid x_t]$ で、やはり一次モーメントです。**両者の違いは「何の一次モーメントを取るか」だけ**で、片方は混ぜたノイズ、もう片方は速度を対象にしています。学習則を2行差し替えるだけで移れたのは、そもそも同じ種類の量を推定していたからでした。

ここで見落とせないのは、モデルが持っているのが一次モーメント**だけ**だということです。その地点を通る人たちがどれくらいばらけているか（二次以降のモーメント）は、どこにも残っていません。ばらつきは、時間をかけて流れをたどる過程で復元されます。

学習後の流れ場と、その帰結を見てみます。

```python
gx, gy = np.meshgrid(np.linspace(-3, 3, 22), np.linspace(-3, 3, 22))
grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32)

fig, ax = plt.subplots(1, 3, figsize=(11.4, 4))
for a, tv in zip(ax, [0.9, 0.5, 0.1]):
    with torch.no_grad():
        v = net_fm(torch.cat([grid, torch.full((len(grid), 1), tv)], 1))
    # 生成は t を減らす向きに進むので、速度の符号を反転して描く
    a.quiver(gx, gy, -v[:, 0].reshape(gx.shape), -v[:, 1].reshape(gx.shape),
             color=GRAY, alpha=.8, width=.004)
    a.scatter(X0[:, 0], X0[:, 1], s=1.5, color=BLUE, alpha=.25)
    a.set(title=f"t = {tv}", xlim=(-3, 3), ylim=(-3, 3), xticks=[], yticks=[])
    a.set_aspect("equal")
plt.tight_layout()
plt.savefig("flow-matching-velocity-field.png", bbox_inches="tight", dpi=150)
plt.show()

# 一次モーメントしか持っていないことの帰結を測る
print(f"学習データ        標準偏差 {X0.std():.3f}")
for s in (1, 2, 5, 50):
    print(f"FM {s:2d}ステップ生成   標準偏差 {sample_fm(net_fm, steps=s).std():.3f}")
with torch.no_grad():
    x1 = torch.randn(2000, 2, generator=torch.Generator().manual_seed(2))
    v1 = net_fm(torch.cat([x1, torch.ones(2000, 1)], 1))
print(f"v(x,1) と x1 のずれ（理論上は0）: {float((v1-x1).abs().mean()):.4f}")
```

```
学習データ        標準偏差 1.000
FM  1ステップ生成   標準偏差 0.082
FM  2ステップ生成   標準偏差 0.567
FM  5ステップ生成   標準偏差 0.850
FM 50ステップ生成   標準偏差 0.971
v(x,1) と x1 のずれ（理論上は0）: 0.0635
```

![学習された速度場](/images/flow-matching-velocity-field.png)

$t=0.9$（ノイズ寄り）ではどこも中心へ向かう大まかな流れですが、$t=0.1$（データ寄り）になると二つの月の形に沿って向きが細かく分かれています。時刻によって流れ場が切り替わり、点を目的の形へ運んでいます。

### 一次モーメントしか無いと、1ステップでは潰れる

出力の標準偏差を見てください。1ステップだと0.082、データの1.000に対してほぼ一点です。

理由は式から出ます。$t=1$ では $x_t$ は $x_1$ そのものなので、

$$
v(x_1, 1) = \mathbb{E}[\,x_1 - x_0 \mid x_1\,] = x_1 - \mathbb{E}[x_0] = x_1
$$

となります（$x_0$ と $x_1$ は独立で、データは平均0に正規化してあります）。1ステップの更新は $x_1 - v(x_1, 1) = 0$、つまり**全員が原点＝データの平均に着地します**。実測でも $v(x,1)$ と $x_1$ のずれは平均0.0635しかなく、式のとおりでした。

ステップを刻むほど標準偏差は 0.567 → 0.850 → 0.971 と戻っていきます。平均の向きに一歩進むと、次の地点では流れが枝分かれしていて、行き先が分かれる。**一次モーメントしか持たないモデルから、ばらつきが再構成されていく過程**がここに見えています。1ステップ生成が難しいのは実装の問題ではなく、この構造から来ています。

## なぜ最初から直線にしなかったのか

ここまで来ると、はじめに置いた疑問がもう一度立ち上がります。直線を引いて速度を回帰させるだけでこれだけ動くなら、**なぜ2015年から7年も、わざわざ確率過程を経由していたのでしょうか。**

意外に思われるかもしれませんが、決定的なODEで生成するという発想の方が、拡散モデルより**古い**です。[Neural ODE](https://arxiv.org/abs/1806.07366)（2018）や[FFJORD](https://arxiv.org/abs/1810.01367)（2019）は、まさに常微分方程式でノイズをデータへ運ぶモデルでした。アイデアが無かったわけではありません。

### 詰まっていたのは学習の方だった

問題は、当時のODEモデルの学習方法にありました。パラメータを1回更新するたびに、ODEを最初から最後まで数値積分して結果を見る必要があったのです。生成を1回丸ごと走らせないと勾配が計算できない。この方式は**シミュレーションを伴う学習**と呼ばれ、画像のような高次元データではコストが現実的でありませんでした。

拡散モデル（[Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)）が突破したのは、実はここです。forward processを確率過程にすると、副産物として大きな性質が手に入ります。**任意の時刻 $t$ の $x_t$ が、$x_0$ から一発で書けるようになる**のです。

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \varepsilon
$$

この記事の最初のコードで使った式です。$t$ をランダムに1つ引いて、その場で $x_t$ を作り、ノイズを当てさせる。途中経過を積分する必要がありません。**シミュレーション不要の学習目標**が手に入りました。

つまり歴史の順序としては、確率的にしたかったから確率過程を選んだのではありません。**確率過程にすると周辺分布が閉じた形で書けて、学習が回るようになったから**です。揺らぎは目的ではなく、計算を成立させるために払った代償でした。

### 揺らぎが要らないと分かるまで

2021年、[スコアベース生成モデルの理論](https://arxiv.org/abs/2011.13456)によって、拡散モデルのSDEには同じ周辺分布をたどる確率流ODEが必ず存在することが示されます。同じ年の[DDIM](https://arxiv.org/abs/2010.02502)は、実際に拡散モデルから決定的にサンプリングしてみせました。この時点で「揺らぎは生成に必須ではない」ことは分かっていました。ただし学習の方は、依然としてスコアマッチング経由のままです。

最後のピースが2022年に揃います。[Lipman ら](https://arxiv.org/abs/2210.02747)、[Liu ら](https://arxiv.org/abs/2209.03003)、[Albergo と Vanden-Eijnden](https://arxiv.org/abs/2209.15571) が、ほぼ同時期に独立して同じ結論に到達しました。前節で見た「条件付きの経路を平均する」という目標を使えば、**ODEを直接、シミュレーション不要で学習できる**。

拡散モデルが確率過程と引き換えに手に入れていた計算上の利点を、決定的な枠組みのまま得る方法が、ここで初めて見つかりました。直線でつないでよかったのではなく、**直線でつないだものを安く学習する方法が無かった**、というのが答えです。

## 一直線に進んでいるのか、実際に測るとどうか

「フローマッチングはまっすぐ進む」という説明をよく見ますが、**厳密には正しくありません**。まっすぐなのは学習時に引いた条件付きの直線であって、生成時にたどるのは平均された流れ場の上の軌跡です。平均をとった時点で曲がります。

どれくらい曲がっているのか測ります。経路の長さを、出発点と到着点の直線距離で割ります。1.00なら完全な直線です。

```python
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
    """経路長 / 直線距離。1.00 なら完全な直線"""
    L = (tr[1:]-tr[:-1]).norm(dim=2).sum(0)
    return float((L / (tr[-1]-tr[0]).norm(dim=1)).median())


tf, td = traj_fm(), traj_ddpm()
w_fm, w_dd = wiggle(tf), wiggle(td)
print(f"経路長 / 直線距離 の中央値   FM {w_fm:.2f}   DDPM {w_dd:.2f}   (1.00 なら完全な直線)")

fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
for a, tr, c, ttl in ((ax[0], td, BLUE, f"DDPM（直線の {w_dd:.1f} 倍の道のり）"),
                      (ax[1], tf, ORANGE, f"フローマッチング（{w_fm:.1f} 倍）")):
    a.scatter(X0[:, 0], X0[:, 1], s=2, color=GRAY, alpha=.22)
    for k in range(tr.shape[1]):
        a.plot(tr[:, k, 0], tr[:, k, 1], color=c, lw=1.1, alpha=.9)
    a.scatter(tr[0, :, 0], tr[0, :, 1], s=26, color="k", zorder=3, label="出発（ノイズ）")
    a.scatter(tr[-1, :, 0], tr[-1, :, 1], s=26, color=c, zorder=3, label="到着（データ）")
    a.set(title=ttl, xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), xticks=[], yticks=[])
    a.set_aspect("equal"); a.legend(fontsize=8.5, loc="upper left")
plt.tight_layout()
plt.savefig("flow-matching-trajectory.png", bbox_inches="tight", dpi=150)
plt.show()
```

```
経路長 / 直線距離 の中央値   FM 1.48   DDPM 20.48   (1.00 なら完全な直線)
```

![軌跡の比較](/images/flow-matching-trajectory.png)

フローマッチングは直線の1.48倍。完全な直線ではありませんが、DDPMの20.48倍と比べると桁が違います。左の図でDDPMがやっているのは、実質的に**酔歩**です。毎ステップ乱数を足しているので、行ったり来たりしながら少しずつデータへ寄っていきます。

この差が、次のステップ数の話に直結します。**道が曲がりくねっているほど、細かく刻まないと近似できない。** 直線なら大股で歩いても目的地に着きます。

## ステップ数を1/50にしても壊れないのはなぜか

前節で測った1.48倍と20.48倍が、そのまま効いてくるのがここです。この節で確かめたいのは、次の一点だけです。

**道がほとんど直線なら、Euler法の刻み幅を大きくしても目的地を外さないはずだ。**

Euler法は、いまいる地点の進行方向にまっすぐ一歩進む、という近似です。道が曲がっていれば、一歩が大きいほど本来の道から外れます。逆に道が直線なら、どんなに大股で歩いても外れません。誤差は刻み幅と**経路の曲がり具合**の積で決まるからです。

フローマッチングの経路は直線の1.48倍、DDPMは20.48倍でした。この比がそのまま「大股で歩けるかどうか」の差になっているなら、ステップ数を削ったときの壊れ方に大きな違いが出るはずです。振ってみます。

```python
steps_list = [1, 2, 5, 10, 50, 200]
r_fm = {s: sliced_w2(sample_fm(net_fm, steps=s), X0) for s in steps_list}
r_dd = {s: sliced_w2(sample_ddpm(net_ddpm, steps=s), X0) for s in steps_list}
for s in steps_list:
    print(f"  {s:4d}ステップ   FM {r_fm[s]:.4f}   DDPM {r_dd[s]:.4f}")
print(f"  1000ステップ            DDPM {d_dd:.4f}")

show = [1, 2, 5, 10, 50]
fig, ax = plt.subplots(2, len(show), figsize=(12.6, 5.6))
for j, s in enumerate(show):
    ax[0, j].scatter(*sample_fm(net_fm, steps=s).T, s=2, color=ORANGE, alpha=.55)
    ax[0, j].set_title(f"FM {s}ステップ\n距離 {r_fm[s]:.3f}", fontsize=9.5)
    ax[1, j].scatter(*sample_ddpm(net_ddpm, steps=s).T, s=2, color=BLUE, alpha=.55)
    ax[1, j].set_title(f"DDPM {s}ステップ\n距離 {r_dd[s]:.3f}", fontsize=9.5)
for a in ax.ravel():
    a.set(xlim=(-3, 3), ylim=(-3, 3), xticks=[], yticks=[])
    a.set_aspect("equal")
plt.tight_layout()
plt.savefig("flow-matching-steps.png", bbox_inches="tight", dpi=150)
plt.show()

plt.figure(figsize=(5.6, 3.7))
plt.plot(steps_list, [r_fm[s] for s in steps_list], "o-", color=ORANGE, lw=2, label="フローマッチング")
plt.plot(steps_list, [r_dd[s] for s in steps_list], "o-", color=BLUE, lw=2, label="DDPM")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("サンプリングのステップ数"); plt.ylabel("データとの距離（小さいほど良い）")
plt.grid(alpha=.3, which="both"); plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("flow-matching-steps-curve.png", bbox_inches="tight", dpi=150)
plt.show()
```

```
     1ステップ   FM 0.9395   DDPM 2.8013
     2ステップ   FM 0.4610   DDPM 2.7537
     5ステップ   FM 0.2012   DDPM 0.3933
    10ステップ   FM 0.1375   DDPM 0.2094
    50ステップ   FM 0.1061   DDPM 0.1300
   200ステップ   FM 0.1031   DDPM 0.0804
  1000ステップ            DDPM 0.1198
```

![ステップ数を振った生成結果](/images/flow-matching-steps.png)

![ステップ数と距離](/images/flow-matching-steps-curve.png)

2ステップの列を見てください。フローマッチングはもう二つの月らしい形が出ているのに、DDPMはまだ点が散らばっているだけです。5ステップでフローマッチングは距離0.201、DDPMは0.393。

### ただし「常にFMが勝つ」ではない

数字を最後まで見ると、200ステップではDDPMが0.0804でフローマッチングの0.1031を**下回っています**。つまりDDPMの方が良い。ステップ数を十分に使えるなら、DDPMは追いついて追い越します。

**差が出るのは少ないステップ数の領域だけ**です。これは実務的にはきわめて重要な差で、画像生成の推論コストはステップ数にほぼ比例するからです。4ステップで絵が出るか50ステップ必要かは、そのままGPU代の一桁の差になります。ただし「フローマッチングの方が生成品質が高い」という言い方は、少なくともこの実験では支持されません。

なお1000ステップのDDPM（0.1198）が200ステップ（0.0804）より悪化しているのは、刻みすぎて毎ステップの乱数が積もった結果です。SDEを解いている以上、刻めば刻むほど良くなるとは限りません。

## 揺らぎを捨てると、何が変わるのか

ここが物理的にいちばん面白いところです。フローマッチングの生成は、乱数を最初の1回しか使いません。出発点さえ決まれば、あとは決定的です。

$$
\frac{dx}{dt} = v_\theta(x, t) \qquad \text{（ODE：揺らぎなし）}
$$

一方DDPMは、[前回の拡散モデル記事](https://zenn.dev/m2yagyu/articles/diffusion-model-toy-physics)で見たとおりランジュバン方程式、つまり毎ステップ揺らぎが入るSDEです。

$$
dx = -\nabla U(x)\, dt + \sqrt{2D}\, dW \qquad \text{（SDE：揺らぎあり）}
$$

同じ出発点から2回まわして確かめます。

```python
f1 = sample_fm(net_fm, n=500, steps=50, seed=7)
f2 = sample_fm(net_fm, n=500, steps=50, seed=7)
# 出発点は同じ（seed=7）、途中の乱数だけ変える
d1 = sample_ddpm(net_ddpm, n=500, steps=200, seed=7, noise_seed=11)
d2 = sample_ddpm(net_ddpm, n=500, steps=200, seed=7, noise_seed=12)
mf, md = float((f1-f2).abs().max()), float((d1-d2).abs().max())
print(f"同じ出発点から2回まわしたときの最大差   FM {mf:.1e}   DDPM {md:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.3))
ax[0].scatter(d1[:, 0], d1[:, 1], s=18, color=BLUE, alpha=.75, label="1回目")
ax[0].scatter(d2[:, 0], d2[:, 1], s=7, color="k", alpha=.75, label="2回目")
ax[0].set_title(f"DDPM：毎回ちがう場所に着く（最大差 {md:.1f}）", fontsize=10.5)
ax[1].scatter(f1[:, 0], f1[:, 1], s=18, color=ORANGE, alpha=.75, label="1回目")
ax[1].scatter(f2[:, 0], f2[:, 1], s=7, color="k", alpha=.75, label="2回目")
ax[1].set_title(f"フローマッチング：完全に一致（最大差 {mf:.0e}）", fontsize=10.5)
for a in ax:
    a.set(xlim=(-3, 3), ylim=(-3, 3), xticks=[], yticks=[])
    a.set_aspect("equal"); a.legend(fontsize=8.5, loc="upper left")
plt.tight_layout()
plt.savefig("flow-matching-determinism.png", bbox_inches="tight", dpi=150)
plt.show()
```

```
同じ出発点から2回まわしたときの最大差   FM 0.0e+00   DDPM 3.48
```

![決定性の比較](/images/flow-matching-determinism.png)

フローマッチングの最大差は**厳密に0**です。浮動小数の丸め誤差すら出ません。同じ入力に同じ演算を通しているので当然ですが、生成モデルとしては大きな性質です。右の図では2回分の点が完全に重なっているので、黒い点しか見えません。

DDPMは最大3.48ずれています。左の図で青と黒がバラバラの位置にあるとおりです。

この違いは、実務では「シードを固定すれば同じ絵が出る」という再現性として現れます。そしてもう一つ、ODEは**逆向きにも解ける**ので、実画像をノイズへ厳密に戻して編集する（inversion）ことができます。SDEでは原理的にできません。

## 本物のFLUXでも、そうなっているのか

ここまではすべて、2次元の点で作ったトイモデルの話でした。この記事の主張のうち一つだけは、実物で裏を取れます。

確かめたいのはこれです。**フローマッチングにはノイズスケジュールが存在しない、という主張は、本物の実装でも成り立っているのか。**

見るのはスケジューラが返す `sigma` の列です。これは「進行度ごとに、ノイズをどの割合で混ぜるか」を並べた配列で、DDPMの $\sqrt{1-\bar\alpha_t}$ に対応します。判定は単純です。

- $\beta$ に相当するスケジュールが**まだ生きている**なら、`sigma` は $\bar\alpha_t$ の総乗を反映した**曲線**になるはず
- スケジュールが**消えている**なら、0から1を等分しただけの**直線**になるはず

`diffusers` のスケジューラはモデルの重みを落とさなくても単体で動くので、数秒で確認できます。実際にFLUX.1とStable Diffusion 3が使っているクラスをそのまま呼びます。

```python
try:
    from diffusers import FlowMatchEulerDiscreteScheduler
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "diffusers"])
    from diffusers import FlowMatchEulerDiscreteScheduler

sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
sch.set_timesteps(10)
sig = np.array([float(s) for s in sch.sigmas])
print("FLUX / SD3 のスケジューラが返す sigma:")
print("  " + "  ".join(f"{s:.3f}" for s in sig))
step = np.abs(np.diff(sig[:10]))
print(f"隣り合う差: 平均 {step.mean():.4f}  ばらつき {step.std():.2e}  → 等間隔")

plt.figure(figsize=(5.8, 3.8))
plt.plot(np.linspace(0, 1, 10), sig[:10], "o-", color=ORANGE, lw=2, label="FLUX / SD3 の sigma")
plt.plot(np.linspace(0, 1, T), noise.numpy()[::-1], color=BLUE, lw=2,
         label=r"DDPM の $\sqrt{1-\bar\alpha_t}$")
plt.xlabel("進行度（0=ノイズ側の端, 1=データ側の端）")
plt.ylabel("ノイズの割合")
plt.grid(alpha=.3); plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("flow-matching-flux-sigmas.png", bbox_inches="tight", dpi=150)
plt.show()
```

```
FLUX / SD3 のスケジューラが返す sigma:
  1.000  0.889  0.778  0.667  0.556  0.445  0.334  0.223  0.112  0.001  0.000
隣り合う差: 平均 0.1110  ばらつき 4.80e-09  → 等間隔
```

![FLUXのsigmaとDDPMのスケジュール](/images/flow-matching-flux-sigmas.png)

**後者でした。** 隣り合う差のばらつきが $4.8 \times 10^{-9}$、つまり完全な等間隔です。図のオレンジは定規で引いた直線そのもので、青い曲線（DDPMの $\sqrt{1-\bar\alpha_t}$）とは似ても似つきません。

`betas` に相当する配列は[このスケジューラ](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete)のどこにも存在しません。あるのは0から1までを等分する目盛りだけです。**温度スケジュールという概念が、実物のコードから消えています。**

## これは物理のどこから来たのか

### SDEからODEへ

拡散モデルとフローマッチングの関係は、[スコアベース生成モデルの理論](https://arxiv.org/abs/2011.13456)で言うところの、SDEと確率流ODEの関係です。ある確率分布の時間発展を与えるSDEに対して、**まったく同じ周辺分布をたどる決定的なODE**が必ず存在します。

$$
dx = \left[ f(x,t) - \tfrac{1}{2} g(t)^2 \nabla_x \log p_t(x) \right] dt
$$

揺らぎの項 $g(t)\,dW$ が消えて、代わりにスコア関数の項が半分になっています。個々の粒子の軌跡は変わりますが、粒子の**分布**は時刻ごとに一致します。分布さえ合っていれば生成モデルとしては十分なので、揺らぎは捨ててよい、というのがここでの理屈です。

物理でいえば、拡散する粒子1個のブラウン運動を追いかける代わりに、流体としての流れだけを追うようなものです。個々の分子はランダムに動きますが、インクの濃度分布の時間発展は決定的な方程式で書けます。

### 連続の式

その決定的な方程式が[連続の式](https://ja.wikipedia.org/wiki/%E9%80%A3%E7%B6%9A%E6%80%A7%E5%BC%8F)です。

$$
\frac{\partial p_t(x)}{\partial t} + \nabla \cdot \left( p_t(x)\, v(x,t) \right) = 0
$$

「密度の時間変化は、流れ込む量と流れ出る量の差に等しい」という、質量保存そのものです。流体力学でも電磁気学でも同じ形で出てきます。フローマッチングが学習している $v_\theta(x,t)$ は、まさにこの $v$ です。**生成モデルの学習が、密度を運ぶ速度場の推定になっている。**

拡散モデルが解いていたのは[フォッカー・プランク方程式](https://ja.wikipedia.org/wiki/%E3%83%95%E3%82%A9%E3%83%83%E3%82%AB%E3%83%BC%3D%E3%83%97%E3%83%A9%E3%83%B3%E3%82%AF%E6%96%B9%E7%A8%8B%E5%BC%8F)で、拡散項 $\nabla^2$ が入った形でした。フローマッチングはその拡散項を落として、移流項だけにしています。熱浴を外すというのは、方程式のレベルではこの項を消すことでした。

### 最適輸送との関係

なぜ直線なのか、という問いには輸送理論からの答えがあります。2つの分布を最小のコストで移し替える問題（[最適輸送](https://arxiv.org/abs/2209.03003)）では、コストが二乗距離のとき、最適な経路は直線になります。rectified flowが直線補間を選んだのは、この最適解を最初から使っているからです。

ただし前述のとおり、学習後の流れ場は平均をとった結果として曲がります（測定値1.48倍）。rectified flowの論文が提案している「reflow」という反復手続きは、生成したペアで学習し直すことでこの曲がりをさらに減らし、1に近づけていく操作です。

## まとめ

同じデータ・同じネットワークで、学習則だけを差し替えて比較しました。

| | 拡散モデル（DDPM） | フローマッチング |
|---|---|---|
| forward | 熱浴に浸す確率過程 | 直線補間 $(1-t)x_0 + t x_1$ |
| スケジュール | $\beta_t$ が必要 | **存在しない** |
| 学習の対象 | 混ぜたノイズ $\varepsilon$ | 速度 $\varepsilon - x_0$ |
| 生成 | SDE（毎ステップ乱数） | ODE（乱数は最初だけ） |
| 経路の直線度 | 20.48倍 | 1.48倍 |
| 少ステップ（5） | 距離 0.393 | 距離 0.201 |
| 十分なステップ（200） | 距離 0.080 | 距離 0.103 |
| 再現性 | 最大差 3.48 | 最大差 0 |

$\beta$ の正体は温度そのものではなく、平衡へ向かう速さでした。行き先が最初から決まっているなら道中を確率的にする必要はない、というのがフローマッチングの発想で、実際にFLUXやSD3のスケジューラからは $\beta$ に相当するものが消えています。

得たものは、少ないステップ数での性能と、完全な再現性。失ったものは、中間分布の分散が保たれる性質（SD3がtimestep shiftで補正しています）と、十分なステップ数を使えるときの品質でした。**まっすぐにすれば良くなる、という単純な話ではありません。**

softmaxのボルツマン分布、Attentionの連想記憶に続いて、生成AIの中身がまた物理の言葉で書けました。今回は熱力学ではなく、流体の連続の式です。

## 参考文献

- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) (ICLR 2023) — フローマッチングの原論文
- Liu et al., [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) (ICLR 2023) — 直線補間とreflow
- Albergo & Vanden-Eijnden, [Building Normalizing Flows with Stochastic Interpolants](https://arxiv.org/abs/2209.15571) (ICLR 2023) — 同時期の独立した定式化
- Esser et al., [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) (ICML 2024) — Stable Diffusion 3。timestep shiftはここ
- Sohl-Dickstein et al., [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) (ICML 2015) — 拡散モデルの原点。非平衡熱力学から出発している
- Chen et al., [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366) (NeurIPS 2018) — 決定的なODEで生成する枠組み。拡散モデルより古い
- Grathwohl et al., [FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models](https://arxiv.org/abs/1810.01367) (ICLR 2019) — シミュレーションを伴う学習の限界が見える
- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (NeurIPS 2020) — DDPM
- Song et al., [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) (ICLR 2021) — 確率流ODEの導出
- Song et al., [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (ICLR 2021) — この記事のDDPMサンプラーの更新式
- [FLUX.1 (black-forest-labs)](https://github.com/black-forest-labs/flux) — 実装
- [FlowMatchEulerDiscreteScheduler / diffusers](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete) — この記事で覗いたスケジューラ
- 前回の記事: [Attentionは結局、何を思い出しているのか](https://zenn.dev/m2yagyu/articles/attention-hopfield-associative-memory) — $\beta$ が温度の逆数だと確かめた回
- [拡散モデルの中身を覗いてみる](https://zenn.dev/m2yagyu/articles/diffusion-model-toy-physics) — 同じtwo moonsでDDPMを実装した回

---

**このシリーズの続き**

この記事は「生成AIの中身を物理から読む」シリーズの 7 本目です。

次に読む → [Attentionは結局、何を思い出しているのか](https://zenn.dev/m2yagyu/articles/attention-hopfield-associative-memory) — softmax(QKᵀ/√d)V がHopfieldの連想記憶の想起則と同じ式だと確かめる

:::details シリーズ全7本

1. [Colabのセル3つで作るLLMチャットボット](https://zenn.dev/m2yagyu/articles/first-ai-chatbot-colab) — まずLLMを自分の手で動かす
2. [LLMのtemperatureは本当に温度だった](https://zenn.dev/m2yagyu/articles/llm-temperature-boltzmann) — softmaxが統計力学のボルツマン分布そのものだと測って確かめる
3. [Attentionは結局、何を思い出しているのか](https://zenn.dev/m2yagyu/articles/attention-hopfield-associative-memory) — softmax(QKᵀ/√d)V がHopfieldの連想記憶の想起則と同じ式だと確かめる
4. [Hugging Face推論APIで動かすtext-to-image](https://zenn.dev/m2yagyu/articles/text-to-image-huggingface-colab) — 文章から画像を作るところまでを最小構成で
5. [拡散モデルの中身を覗いてみる](https://zenn.dev/m2yagyu/articles/diffusion-model-toy-physics) — 2次元のトイデータで拡散モデルをゼロから実装し、ランジュバン方程式と繋ぐ
6. [拡散モデルをMNISTで動かす](https://zenn.dev/m2yagyu/articles/diffusion-model-mnist-unet) — 784次元の画像へ拡張しても forward / reverse の式は変わらないことを確かめる
7. **FLUXが使うフローマッチングって結局何なの？**（この記事）— 拡散モデルから熱浴を外すと何が残るのか、同じデータ・同じネットで学習則だけ差し替えて測る

:::
