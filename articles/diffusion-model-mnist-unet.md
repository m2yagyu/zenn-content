---
title: "拡散モデルをMNISTで動かす：トイモデルから本物のU-Netへ"
emoji: "🔢"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["生成ai", "machinelearning", "初心者", "diffusion", "物理"]
published: true
---

この記事を読むと、[前回](https://zenn.dev/m2yagyu/articles/diffusion-model-toy-physics)2次元の点で確認した拡散モデルの仕組みを、実際の手書き数字画像（MNIST）に拡張し、小さなU-Netを自分の手でゼロから学習させて「ノイズから数字を生み出す」ところまで動かせるようになります。

[前回の記事](https://zenn.dev/m2yagyu/articles/diffusion-model-toy-physics)では、2次元の「二つの月」という点の集まりを使って、拡散モデルのforward process（データにノイズを混ぜる）とreverse process（ノイズからデータを復元する）を実装し、物理の拡散現象・ランジュバン方程式・統計力学との対応まで見てきました。今回はその続きとして、「この仕組みが実際の画像データでどう使われているか」に踏み込みます。

## 今回やること

前回のモデルは、点の座標$(x, y)$と時刻$t$を受け取って全結合層（MLP）に通すだけの、とても小さなネットワークでした。画像はこのままでは扱えません。28×28ピクセルの画像を「784個の独立な数字」として全結合層に投げると、どのピクセルが隣り合っているかという情報がすべて失われてしまうからです。そこで今回は、

- 画像の「近さ」を保ったまま処理できる**CNN（畳み込みニューラルネットワーク）**をベースにした軽量な**U-Net**を実装する
- forward process・reverse processの数式そのものは前回と完全に同じものを使う（変わるのはネットワークの構造だけ）
- 学習データには手書き数字画像**MNIST**を使う

という3点だけを変更します。数式が前回とほとんど変わらないことは、後半で確認します。

:::message alert
今回は画像を扱うぶん、前回の2次元データよりも計算量が大きくなります。CPUでも動きますが、実測で1ステップあたり数十倍遅く、6000ステップの学習に数時間かかってしまいます。ColabのランタイムをGPU（T4など）に変更してから実行してください。変更方法は[こちらの記事](https://zenn.dev/usagi1975/articles/2025-12-06-000_google_colab)を参照してください。
:::

## さっさと動かしてみよう

Colabで新しいノートブックを開き、ランタイムをGPUに変更した上で、次のセルを順番に実行してください。

### セル1：データを見てみる

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

mnist = torchvision.datasets.MNIST(root="./mnist_data", train=True, download=True)
X_raw = mnist.data.float()  # (60000, 28, 28), 画素値は0〜255
print("raw shape:", X_raw.shape)

fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(X_raw[i].numpy(), cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

```
device: cuda
raw shape: torch.Size([60000, 28, 28])
```

![MNISTの学習データの一部（0〜9の手書き数字）](/images/mnist-diffusion-raw-samples.png)

6万枚の手書き数字画像が入っています。前回の「二つの月」は2000個の点でしたが、今回はこれを28×28＝784個の数字が並んだ画像として扱います。

```python
X0 = X_raw.unsqueeze(1)  # (60000, 1, 28, 28)。1はチャンネル数（グレースケールなので1）
mean, std = X0.mean(), X0.std()
X0 = (X0 - mean) / std
print("normalization mean/std:", mean.item(), std.item())
print("X0 mean/std after norm:", X0.mean().item(), X0.std().item())
```

```
normalization mean/std: 33.31842041015625 78.56748962402344
X0 mean/std after norm: 2.0262010735905278e-08 1.0
```

前回と同じく「分散1に揃える」正規化ですが、1つだけ注意点があります。ここで使っている`mean`と`std`は、6万枚・784ピクセル全部をまとめた**1つの数字**です。ピクセルごとに平均・標準偏差を計算してしまうと、常に真っ黒（画素値0）な背景のピクセルは標準偏差が0になり、0除算でエラーになります。画像全体で1つの統計量にまとめておくのが安全です。

### セル2：forward process（前回と全く同じ式）

```python
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

alpha_bars_dev = alpha_bars.to(device)
alphas_dev = alphas.to(device)
betas_dev = betas.to(device)

def forward_sample(x0, t):
    """x0にステップtまでノイズを混ぜたx_tを返す（前回のforward_sampleと同じ式）"""
    a_bar = alpha_bars_dev[t].view(-1, 1, 1, 1)  # 画像用に形を(N,1,1,1)に
    eps = torch.randn_like(x0)
    x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps
    return x_t, eps

print("alpha_bar[T-1]      =", alpha_bars[-1].item())
print("sqrt(alpha_bar[T-1]) =", alpha_bars[-1].sqrt().item())
```

```
alpha_bar[T-1]      = 4.035830352222547e-05
sqrt(alpha_bar[T-1]) = 0.006352818571031094
```

$T$や`betas`の値、`sqrt(alpha_bar[T-1])`が0.006まで落ちていることまで、前回とまったく同じです。変わったのは`.view(-1, 1, 1, 1)`の部分だけで、これは$\bar\alpha_t$を画像の形（バッチ×チャンネル×高さ×幅）にブロードキャストできるようにする調整です。**forward processの数式自体は、点でも画像でも変わりません。**

### セル3：軽量U-Net

```python
class TimeEmbedding(nn.Module):
    """時刻tを三角関数で複数の周波数に展開してベクトルにする。
    ゆっくり振動する成分と速く振動する成分を同時に持たせることで、
    「前半か後半か」も「隣のステップとの細かな違い」も表せるようにする。"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim * 4)
        )

    def forward(self, t):
        half = self.dim // 2
        # 周波数を対数的に並べる（粗いスケールと細かいスケールを両方カバーする）
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        ang = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> Conv を2回。その間に時刻埋め込みを足し込む"""
    def __init__(self, in_ch, out_ch, tdim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(tdim, out_ch)   # 時刻埋め込みをこの層のチャンネル数に合わせる
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        # 入出力のチャンネル数が違うときだけ1x1畳み込みで合わせる
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.temb(self.act(temb))[:, :, None, None]  # 全ピクセルに同じ値を足す
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)   # 残差接続：入力をそのまま足す


class NoisePredictorUNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        tdim = hidden * 4
        self.time = TimeEmbedding(hidden)
        self.stem = nn.Conv2d(1, hidden, 3, padding=1)                                 # 28x28
        self.down1 = ResBlock(hidden, hidden, tdim)                                    # 28x28
        self.pool1 = nn.Conv2d(hidden, hidden, 3, stride=2, padding=1)                 # 14x14に縮小
        self.down2 = ResBlock(hidden, hidden * 2, tdim)                                # 14x14
        self.pool2 = nn.Conv2d(hidden * 2, hidden * 2, 3, stride=2, padding=1)         # 7x7に縮小
        self.mid = ResBlock(hidden * 2, hidden * 2, tdim)                              # 一番狭い層(bottleneck)
        self.up2 = nn.ConvTranspose2d(hidden * 2, hidden * 2, 4, stride=2, padding=1)  # 14x14に拡大
        self.dec2 = ResBlock(hidden * 4, hidden, tdim)                                 # 14x14
        self.up1 = nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1)          # 28x28に拡大
        self.dec1 = ResBlock(hidden * 2, hidden, tdim)                                 # 28x28
        self.out = nn.Sequential(
            nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv2d(hidden, 1, 3, padding=1)
        )   # 出力は予測ノイズ1枚

    def forward(self, x, t):
        temb = self.time(t)                 # 時刻を1本のベクトルにして各ブロックに配る
        s = self.stem(x)
        h1 = self.down1(s, temb)
        h2 = self.down2(self.pool1(h1), temb)
        m = self.mid(self.pool2(h2), temb)
        u2 = self.dec2(torch.cat([self.up2(m), h2], dim=1), temb)   # skip connection
        u1 = self.dec1(torch.cat([self.up1(u2), h1], dim=1), temb)  # skip connection
        return self.out(u1)

model = NoisePredictorUNet(hidden=64).to(device)
print("n_params:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=2e-4)
```

```
n_params: 1632129
```

入口が1チャンネル（ノイズ画像）、出口が1チャンネル（予測ノイズ）で、時刻`t`は別ルートで各層に配られます。前回の「入口3・出口2」の入出力の役割と、担っている仕事は同じです。

違うのは中身で、$28\times28 \to 14\times14 \to 7\times7$と画像を縮小しながら特徴を抜き出し（エンコーダ）、$7\times7 \to 14\times14 \to 28\times28$と元の解像度に戻しながら細部を復元する（デコーダ）という形をしています。この「縮小してから拡大する」形がU字に見えるので**U-Net**と呼ばれます。`TimeEmbedding`と`ResBlock`が何をしているかは、動かしたあとの[後半](#cnnとu-netが必要な理由)でまとめて説明します。ここでは「時刻を三角関数で展開して各層に配っている」「各ブロックで正規化と残差接続を使っている」とだけ押さえて先に進んでください。

### セル4：学習する

```python
X0 = X0.to(device)
n_steps = 6000
batch_size = 128

# EMA：学習中の重みの指数移動平均。生成にはこの平滑化した重みを使う
ema_decay = 0.995
msd = model.state_dict()          # テンソルへの参照なので、一度取れば常に最新を指す
ema = {k: v.detach().clone() for k, v in msd.items()}

losses = []
for step in range(n_steps):
    idx = torch.randint(0, X0.shape[0], (batch_size,))
    x0 = X0[idx]
    t = torch.randint(0, T, (batch_size,), device=device)
    x_t, eps = forward_sample(x0, t)
    eps_pred = model(x_t, t)
    loss = ((eps_pred - eps) ** 2).mean()

    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():         # 平滑化した副本を更新する
        for k, v in msd.items():
            if v.dtype.is_floating_point:
                ema[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)
            else:
                ema[k].copy_(v)

    losses.append(loss.item())

    if (step + 1) % 500 == 0:
        print(f"step {step+1}: loss = {np.mean(losses[-500:]):.4f}")

print("最終的な誤差:", np.mean(losses[-500:]))

model.load_state_dict(ema)        # 生成前に、平滑化した重みへ差し替える
```

```
step 500: loss = 0.0734
step 1000: loss = 0.0402
step 1500: loss = 0.0362
step 2000: loss = 0.0345
step 2500: loss = 0.0335
step 3000: loss = 0.0328
step 3500: loss = 0.0320
step 4000: loss = 0.0316
step 4500: loss = 0.0313
step 5000: loss = 0.0311
step 5500: loss = 0.0309
step 6000: loss = 0.0307
最終的な誤差: 0.030724045377224684
<All keys matched successfully>
```

最後の`<All keys matched successfully>`は、`model.load_state_dict(ema)`が返した結果がノートブックに表示されたものです。エラーではなく、平滑化した重みが正しく読み込まれたという意味なので、そのまま先に進んでください。

学習ループの本体は前回のセル4と1行も変わっていません（`model`と`forward_sample`の中身だけが変わっています）。追加したのはEMAの更新だけです。

**Colab（T4 GPU）ではこの6000ステップの学習に10分弱かかりました。** 前回の2次元トイモデルは20000ステップがCPUで約10秒だったので、画像を扱うようになったことで計算コストが大きく増えたことが分かります。

計算コストについて、正直に書いておきたいことがあります。ステップ数は前回の記事から40000まで増やす必要はなく6000で足りたのですが、**1ステップあたりのコストが約10倍になったため、待ち時間は結局増えています。** 時刻埋め込みを各層に注入し、GroupNormを通し、チャンネル数を32から64に増やした分がそのまま効いています。「ステップ数を減らせたから速くなった」とはならない、という点に注意してください。もし待ち時間を詰めたい場合は`n_steps`を4000に落とすと約7分になります。そのぶん生成される数字の質は落ちます（後述の指標で0.801から0.766へ）。

:::message
**EMA（指数移動平均）とは何をしているのか。** 学習中の重みは、ステップごとに引いたミニバッチに振られて細かく揺れ続けています。その揺れを含んだ瞬間の値ではなく、直近の軌跡を平均した値を使うほうが、生成される画像が安定します。`ema_decay = 0.995`は「直近200ステップ分をおおよそ平均する」設定です。物理で言えば、熱揺らぎを含む瞬間値ではなく時間平均量を見るのに近い操作です。

筆者が測った範囲では、EMAを入れると「はっきり数字と読める割合」が6000ステップで0.793から0.801へ、4000ステップでは0.676から0.766へ改善しました。減衰率は大きすぎると逆効果で、`0.999`（直近1000ステップ相当）にすると4000ステップ以下では素の重みより悪くなりました。学習が短いほど、平均する範囲も短くする必要があります。
:::

**なぜ6000ステップなのか。** 損失は6000ステップでもまだわずかに下がり続けています（12000ステップまで回すと0.0294まで下がります）。それでもここで止めているのは、生成される数字の読みやすさが6000から8000でほぼ変わらなかったからです。一方で12000まで回すと読みやすさは目に見えて上がるので、学習時間を気にしないのであれば`n_steps`を増やす価値はあります。**損失の値だけを見て「まだ下がるから回すべきだ」とも「頭打ちだから止めてよい」とも判断できない**、というのがここで測って分かったことです。

### セル5：reverse process（前回と全く同じ式）

```python
@torch.no_grad()
def sample(n_samples=16, record_steps=()):
    x = torch.randn(n_samples, 1, 28, 28, device=device)  # 完全なノイズ画像からスタート
    frames = {}
    for t in reversed(range(T)):
        t_batch = torch.full((n_samples,), t, dtype=torch.long, device=device)
        eps_theta = model(x, t_batch)
        alpha_t, alpha_bar_t, beta_t = alphas_dev[t], alpha_bars_dev[t], betas_dev[t]

        mean = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
        )
        if t > 0:
            sigma_t = torch.sqrt(beta_t)
            x = mean + sigma_t * torch.randn_like(x)
        else:
            x = mean

        if t in record_steps:
            frames[t] = x.clone().cpu()
    return frames

show_t = [0, 100, 300, 600, 999]
frames = sample(n_samples=32, record_steps=set(show_t))
```

前回のセル5と比べても、`x`が2次元の点から`(32, 1, 28, 28)`の画像バッチに変わっただけで、更新式は一字一句同じです。1000ステップかけて32枚のノイズ画像を数字に戻すのに、筆者の手元では数秒しかかかりませんでした。**学習に10分近くかかったのに対して、生成は一瞬で終わります。** 一度学習を済ませてしまえば、あとは何枚でも安価に作れる、というのが拡散モデルの実用上の性質です。

:::message
**この先の生成結果は、実行するたびに変わります。** セル1で`torch.manual_seed(0)`を設定していますが、それでも筆者が載せた図と同じ絵にはなりません。理由は3つあります。第一に、GPU上の畳み込み演算は既定では非決定的で、まったく同じ入力でも実行ごとに微小な差が出ます。第二に、GPUの機種やPyTorchのバージョンが違えば呼ばれる計算カーネル自体が変わります。第三に、セルを再実行するとその時点の乱数の状態が変わるため、`torch.randn`が引く初期ノイズが別のものになります。以降の図は「筆者の環境で1回実行したときの例」として見てください。**あなたの手元で違う形が出てくるのは失敗ではありません。**
:::

### セル6：forwardとreverseを並べて可視化する

```python
def denorm(x):
    return (x * std.cpu() + mean.cpu()).clamp(0, 255)  # 正規化を戻して画素値0-255に

x0_single = X0[0:1]
fwd_frames = {}
for t in show_t:
    t_batch = torch.full((1,), t, dtype=torch.long, device=device)
    x_t, _ = forward_sample(x0_single, t_batch)
    fwd_frames[t] = x_t.cpu()

fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
for i, t in enumerate(show_t):
    axes[0, i].imshow(denorm(fwd_frames[t])[0, 0].numpy(), cmap="gray", vmin=0, vmax=255)
    axes[0, i].set_title(f"t = {t}")
    axes[0, i].axis("off")
for i, t in enumerate(reversed(show_t)):
    axes[1, i].imshow(denorm(frames[t][0:1])[0, 0].numpy(), cmap="gray", vmin=0, vmax=255)
    axes[1, i].set_title(f"t = {t}")
    axes[1, i].axis("off")
axes[0, 0].text(-6, 14, "forward\n(data->noise)", fontsize=11, ha="right", va="center")
axes[1, 0].text(-6, 14, "reverse\n(noise->data)", fontsize=11, ha="right", va="center")
plt.tight_layout()
plt.show()
```

![上段はforward processで「5」の画像が徐々にノイズになる様子、下段はreverse processで別のノイズから数字が生成される様子](/images/mnist-diffusion-forward-reverse.png)

前回の月型の図と同じ構成です。上段（forward）は学習データの中の実際の「5」という画像が、ステップを重ねるごとに完全なノイズへと埋もれていく様子です。下段（reverse）は、学習済みのU-Netが完全なノイズから1000回のノイズ除去を繰り返した結果です。

ここは誤解しやすいので明示しておきます。**上段の`t=999`と下段の`t=999`は、別のノイズです。** 上段は本物の「5」に`forward_sample`でノイズを混ぜたもの、下段はセル5の`torch.randn`が引いた乱数で、両者はつながっていません。ですから下段が「5」に戻る理由はなく、実際に現れているのは別の数字です。上段の終点を下段の出発点につなぎ直しても同じで、$\sqrt{\bar\alpha_T} = 0.006$まで信号が落ちている以上、元画像の情報はほとんど残っていません。この図は「1枚の画像を壊して元に戻す」実験ではなく、「壊す向き」と「作る向き」を並べて見せた図だと捉えてください。

そのうえで下段に注目すると、`t=600`ではまだ一面のノイズだったものが、`t=300`付近で薄い輪郭として立ち上がり、`t=100`ではもう数字らしい線としてはっきり見えています。誰も「数字はこういう形だ」と教えていないのに、ノイズを削っていく過程で形が湧き上がってくる、これがreverse processの働きです。

### セル7：生成された32枚をまとめて見る

```python
gen0 = denorm(frames[0])

n = gen0.shape[0]                       # 実際に生成された枚数を見る
ncol = 8
nrow = (n + ncol - 1) // ncol           # 端数が出ても足りる行数を計算する
fig, axes = plt.subplots(nrow, ncol, figsize=(1.5 * ncol, 1.5 * nrow))
for i, ax in enumerate(axes.ravel()):
    if i < n:                           # 余ったマスは空欄にする
        ax.imshow(gen0[i, 0].numpy(), cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

:::message
行数を`gen0.shape[0]`から計算しているので、セル5の`n_samples`を変えてもこのセルは書き換え不要です。逆に、`n_samples`を変えたときは**セル5を実行し直してから**このセルを実行してください。セル5を再実行せずにここへ来ると、`frames`の中身が前回の枚数のままなので`IndexError`になります。再学習は不要で、セル5だけなら数秒で終わります。
:::

![学習済みU-Netがゼロから生成した手書き数字32枚。約8割が数字として読み取れる](/images/mnist-diffusion-gen-grid.png)

セル5の`sample`関数は32枚を同時に生成していたので、そのうちの`t=0`の32枚をまとめて並べました。新しく計算しているものは何もなく、セル6で1枚だけ見ていたものの残り31枚も含めて見ているだけです。

「0」「1」「2」「3」「4」「5」「6」「7」「9」がはっきり読み取れます。**出来を正直に書くと、数字として自信をもって読めるのは32枚のうち25枚前後、およそ8割です。** 残りの2割は線が崩れていたり、二つの数字が混ざったような形をしていて、どの数字とも言い切れません。

この「約8割」は目で数えた印象ではなく、測った値です。筆者は別途MNISTの分類器（訓練精度99.4%）を学習させ、生成した256枚を分類させて確信度を測りました。確信度0.9を超えた割合は0.801でした。参考までに、本物のMNIST画像を同じ分類器にかけると0.973です。つまり本物にはまだ届かないものの、ノイズから出発してここまで来ている、という水準です。

:::message
**「損失が下がった」だけでは生成の質を測れません。** 筆者は最初、この記事のモデルを40000ステップ学習させ、損失が0.0334まで下がったので十分だと判断しました。ところが生成された画像を見ると、数字として読めるものは16枚に1枚程度しかありませんでした。

さらに紛らわしい例もあります。学習を500ステップで止めてEMAをかけた重みで生成すると、分類器の確信度は0.998、本物のMNIST（0.992）すら上回りました。しかし実際に出ていたのは「2」「3」「5」に見えるぼやけた原型が数種類だけで、予測クラスの分布は`[0,5,80,94,5,59,0,1,11,1]`と激しく偏っていました。滑らかで平均的な形は分類器にとって分類しやすいため、確信度が高く出てしまうのです。

生成モデルを評価するには、1枚ごとの「もっともらしさ」と、集団としての「多様性」の両方を見る必要があります。この記事では確信度に加えて、予測クラス分布のエントロピー（10クラスが均等なら$\ln 10 = 2.303$）も測っています。上の32枚を生成したモデルは2.277で、本物のMNIST（2.283）とほぼ同じでした。
:::

そしてこの図で見てほしいのは、1枚ごとの出来だけではなく、**32枚がそれぞれ違う数字になっている**ことです。このモデルはラベルを一切受け取っていません（セル1で読み込んでいるのは`mnist.data`だけで、正解ラベルの`mnist.targets`は使っていません）。ですから「3を描いて」と指定する手段はそもそもなく、モデルが学習したのは「MNIST全体の字形の分布」そのものです。32枚を分けているのは、出発点として引いた32通りの乱数だけです。784次元空間のどこから降り始めるかによって、たどり着く谷が「1」の谷になるか「8」の谷になるかが決まる、というのがここで起きていることです。

## CNNとU-Netが必要な理由

前回のMLP（全結合層だけのネットワーク）から、今回CNN・U-Netに変えた理由を短くまとめておきます。

**なぜ全結合層ではだめなのか。** MLPに画像を渡すには、28×28の画像を784個の数字が並んだ1本の長いリストに変形する必要があります。これは、写真を「左上から順に読み上げた明るさの値のリスト」に変えてしまうようなもので、「このピクセルの右隣は何か」という、画像にとって最も重要な位置関係の情報が失われます。

**畳み込み（Convolution）は何をしているのか。** `nn.Conv2d`は、3×3の小さな窓を画像の上でスライドさせながら、「その場所の周辺にどんな模様があるか」を調べる操作です。窓の位置を保ったまま計算するので、隣り合うピクセルの関係が最後まで保たれます。これが画像を扱うときにCNNを使う理由です。

**U-Net（エンコーダ→デコーダ）は何をしているのか。** セル3の`pool1`・`pool2`（`stride=2`の畳み込み）は画像を$28\to14\to7$と縮小しながら、「全体としてどんな形をしているか」という大まかな情報を抜き出します。逆に`up2`・`up1`（`ConvTranspose2d`）は$7\to14\to28$と元の解像度に戻しながら、線の位置を細かく描き直します。ラフな下描きをしてから細部を仕上げる、という2段階の作業に近いイメージです。`torch.cat([self.up2(m), h2], dim=1)`という**skip connection**（縮小前の情報をもう一度混ぜる操作）を入れているのは、縮小する過程で失われがちな「線がどこにあったか」という位置の情報を、デコーダ側に直接渡してあげるためです。これを外すと、生成される数字の線がぼやけたり、位置がずれたりします。

**時刻`t`はどう渡しているのか。** ここが前回から一番大きく変わった部分で、そして生成される数字の質を最も左右した部分でもあります。

前回は座標`(x, y)`に`t`を1つの数として連結するだけで足りていました。画像でも同じように「全ピクセルが$t/T$で埋まった28×28のシートを1枚重ねる」という素直な方法が考えられます。実際にそれを試したのですが、うまくいきませんでした。理由は2つあります。第一に、$t/T$という1つの数は入口で1チャンネル分の情報しか持たず、層を深く進むうちに他の特徴に埋もれてしまいます。第二に、$t=500$と$t=501$の違いと、$t=0$と$t=999$の違いを、同じ1つのスケールで表さなければなりません。

そこで実用的な拡散モデルでは、**時刻埋め込み**という方法を使います。`TimeEmbedding`がしているのは、$t$を複数の周波数の三角関数に展開することです。

$$
\mathrm{emb}(t) = \bigl[\sin(\omega_1 t), \ldots, \sin(\omega_k t), \cos(\omega_1 t), \ldots, \cos(\omega_k t)\bigr]
$$

周波数$\omega_i$を対数的に並べておくと、ゆっくり振動する成分が「前半か後半か」という大まかな時期を、速く振動する成分が「隣のステップとの細かな違い」を担います。物理をやっていれば見慣れた発想で、1つの量をフーリエ成分に分解して、粗いスケールと細かいスケールを同時に扱えるようにしているだけです。

そしてこのベクトルを、入口で1回混ぜるのではなく、`ResBlock`の中で**各層ごとに**足し込みます（`h + self.temb(self.act(temb))[:, :, None, None]`）。`self.temb`はその層のチャンネル数に合わせる線形変換で、`[:, :, None, None]`は1つの値を28×28（あるいは14×14、7×7）の全ピクセルに同じだけ足すための形の調整です。こうすると、どの深さの層も「いまノイズがどれくらい残っているか」を直接参照しながら処理できます。

**GroupNormと残差接続は何のためか。** `ResBlock`には`nn.GroupNorm`が入っています。層を通るたびに値の大きさが暴れるのを抑え、学習を安定させるためです。また各ブロックの出口で`h + self.skip(x)`と入力をそのまま足しています（**残差接続**）。ネットワークが「入力をどう修正するか」だけを学べばよくなるので、層を重ねても学習が進みやすくなります。どちらも拡散モデル特有の工夫ではなく、画像を扱うネットワークの標準的な部品です。

## 物理との対応（前回のふりかえり）

forward process・reverse processの数式、そしてスコア関数$s(x,t) = \nabla_x \log p_t(x)$やランジュバン方程式との対応は、前回導出した内容から一切変わっていません。変わったのは、$x$が2次元のベクトルから$28\times28=784$次元のベクトル（画像）になったという次元の数だけです。物理的に言えば、今回のポテンシャル$U(x) = -\log p_t(x)$は、2次元平面ではなく784次元空間の中で定義された、もっと複雑な関数になっています。U-Netがしているのは、この784次元空間の中で「今の位置からどの方向に力がかかっているか」を学習することで、依然として同じランジュバン方程式

$$
dx = -\nabla U(x)\, dt + \sqrt{2D}\, dW
$$

に従って、ノイズという平衡状態からデータという非平衡状態へ時間を遡っている、という点は前回とまったく同じです。次元が増えても物理の骨格は変わらない、というのがこの記事の実験結果が示していることです。

## まとめ

前回の2次元トイモデルを、実際の手書き数字画像MNISTに拡張し、軽量なU-Netをスクラッチで学習させることで、ノイズから数字を生成できることを確認しました。変更点は「ネットワークをMLPからCNN・U-Netにしたこと」と「学習データを画像にしたこと」の2点だけで、forward process・reverse processの数式自体、そして物理（ランジュバン方程式・統計力学）との対応は前回からまったく変わっていません。次元が2次元から784次元に増えても、拡散モデルの骨格が同じままであることが、実際に動くコードで確認できたのではないかと思います。

---

**このシリーズの続き**

この記事は「生成AIの中身を物理から読む」シリーズの 6 本目です。

次に読む → [FLUXが使うフローマッチングって結局何なの？](https://zenn.dev/m2yagyu/articles/flow-matching-vs-diffusion) — 拡散モデルから熱浴を外すと何が残るのか、同じデータ・同じネットで学習則だけ差し替えて測る

:::details シリーズ全7本

1. [Colabのセル3つで作るLLMチャットボット](https://zenn.dev/m2yagyu/articles/first-ai-chatbot-colab) — まずLLMを自分の手で動かす
2. [LLMのtemperatureは本当に温度だった](https://zenn.dev/m2yagyu/articles/llm-temperature-boltzmann) — softmaxが統計力学のボルツマン分布そのものだと測って確かめる
3. [Attentionは結局、何を思い出しているのか](https://zenn.dev/m2yagyu/articles/attention-hopfield-associative-memory) — softmax(QKᵀ/√d)V がHopfieldの連想記憶の想起則と同じ式だと確かめる
4. [Hugging Face推論APIで動かすtext-to-image](https://zenn.dev/m2yagyu/articles/text-to-image-huggingface-colab) — 文章から画像を作るところまでを最小構成で
5. [拡散モデルの中身を覗いてみる](https://zenn.dev/m2yagyu/articles/diffusion-model-toy-physics) — 2次元のトイデータで拡散モデルをゼロから実装し、ランジュバン方程式と繋ぐ
6. **拡散モデルをMNISTで動かす**（この記事）— 784次元の画像へ拡張しても forward / reverse の式は変わらないことを確かめる
7. [FLUXが使うフローマッチングって結局何なの？](https://zenn.dev/m2yagyu/articles/flow-matching-vs-diffusion) — 拡散モデルから熱浴を外すと何が残るのか、同じデータ・同じネットで学習則だけ差し替えて測る

:::
