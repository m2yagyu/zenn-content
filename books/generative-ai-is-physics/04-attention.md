---
title: "Attentionは結局、何を思い出しているのか"
---

前章で、softmaxがボルツマン分布であり、その中の $\beta = 1/\mathcal{T}$ が温度の逆数だと確かめました。

ここで気づいてほしいことがあります。Transformerの中心にある $\mathrm{softmax}(qK^\top/\sqrt{d})V$ にも、同じsoftmaxが入っています。ということは、**Attentionにも温度があるはずです。** どこにあるでしょうか。

答えは $\sqrt{d}$ です。そしてこの $\beta$ を追いかけていくと、1982年の磁性体の模型にたどり着きます。

この記事を読むと、Transformerの `softmax(QKᵀ/√d)V` が「連想記憶から記憶を思い出す計算」と一字一句同じ式であることを、自分の手で確かめられます。GPUは不要で、前半はnumpyだけ、後半も150Mパラメータの小さなモデルをCPUで動かすだけです。

載せているコードはすべて図の出力までセットになっているので、上から順にコピペすると記事と同じ図がそのまま手元に出ます。グラフの日本語表示に必要なパッケージは最初のセルが自動で入れるので、事前の準備は要りません（Google Colabでもそのまま動きます）。

結論を先に言うと、Attentionの式にある `/ math.sqrt(d_k)` の `√d` は**温度**です。しかも比喩ではなく、[前回の記事](https://zenn.dev/m2yagyu/articles/llm-temperature-boltzmann)で扱った統計力学の温度と同じものです。そしてその温度が支配しているのは、Attentionが「記憶をひとつだけ思い出すか、複数を混ぜて思い出すか」という切り替えでした。

## Attentionを知らない人のための3分の復習

Attentionの中身をまだ触ったことがない人向けに、必要な分だけ先に説明します。すでに実装を読んだことがある人はこの節を飛ばして構いません。

LLMは文章を左から読み、次の1語を予測します。たとえばこういう文を考えます。

> 日本の首都は東京です。日本で一番高い山は富士山です。日本の首都は

最後の「は」の次に来る語を当てたい。人間なら、前に出てきた「東京」を思い出して答えます。このとき頭の中でやっているのは、**いま必要としている情報を手がかりに、前に出てきた語の中から関係あるものを探す**という作業です。

Attentionはこれを3つのベクトルで表現します。

| 名前 | 役割 | さっきの例でいうと |
|---|---|---|
| **クエリ** $q$ | いま探しているもの | 「日本の首都、といえば？」 |
| **キー** $k$ | 各語が掲げている見出し | 「東京」は"首都に関する語"という見出しを持つ |
| **バリュー** $v$ | 各語が実際に渡す中身 | 「東京」という語の情報そのもの |

クエリと各キーの内積をとると「その語がどれくらい関係あるか」の点数が出ます。点数をsoftmaxで重みに変え、その重みでバリューを平均する。これがAttentionの出力です。

重要なのは、$q, k, v$ が固定のものではなく、**各語の埋め込みベクトルに、学習で決まった行列 $W_Q, W_K, W_V$ を掛けて作られる**という点です。「何を探すか」「何を見出しに掲げるか」をモデルが自分で学習します。

Transformer全体の構造（複数ヘッド、複数層、位置エンコーディングなど）はこの記事では扱いません。必要なのは上の3つだけです。もっと丁寧な入門は [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) や原論文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) が読みやすいです。

## そもそもAttentionは何を計算しているのか？

式で書くとこうなります。

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

記号の使い分けだけ先に決めておきます。この記事では**大文字と小文字で「行列か、ベクトル1本か」を区別**します。

| 表記 | 意味 |
|---|---|
| $Q, K, V$ | クエリ・キー・バリューを縦に並べた**行列** |
| $q, k, v$ | そのうちの**1本のベクトル** |

以降はクエリを1本だけに絞って話を進めるので、主役は小文字の $q$ です。キーとバリューは「文脈中の全部の語ぶん」を相手にするので、大文字の $K, V$ のままです。

その $q$ 1本について、やっていることは3ステップです。

```mermaid
graph TD
    A["① クエリ q と、N本のキー k_i の内積をとる<br/>= 各キーとの「相性の点数」を出す"] --> B["② 点数を √d で割って softmax にかける<br/>= 点数を、合計1になる重みに変える"]
    B --> C["③ その重みで、N本のバリュー v_i を平均する<br/>= 重み付き平均が出力になる"]
```

つまりAttentionは「*問い合わせに近いものほど重く数える、重み付き平均*」です。ここまでは実装を読めばわかる話です。

問題は、この計算に名前がついていることに気づきにくい点です。「手がかりを渡すと、それに近いものを引っ張り出してくる」という装置は、機械学習より40年前から物理の世界にありました。**連想記憶**です。

## 「思い出す」という計算を、先に書いてみる

Attentionのことはいったん忘れて、連想記憶をゼロから作ってみます。やりたいことはこうです。

- いくつかのパターン（記憶）を保存しておく
- 一部が欠けたり、ノイズで崩れたパターン（手がかり）を渡す
- 元の完全なパターンを返してもらう

人間でいえば「ぼんやりした思い出しかけの記憶から、元の出来事を思い出す」に当たります。

素直に書くとこうなります。記憶を並べた行列 $X$（1列が1つの記憶）と手がかり $\xi$ を用意して、「*手がかりと似ている記憶ほど重く数えて、記憶たちを平均する*」だけです。

```python
import numpy as np
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


def softmax(z):
    z = z - z.max()
    return np.exp(z) / np.exp(z).sum()


def make_cue(x, sigma, rng):
    """記憶 x に、x の大きさに対して相対的に sigma 倍のノイズを乗せた手がかりを作る"""
    n = rng.normal(size=x.shape)
    n /= np.linalg.norm(n)
    c = x + sigma * n
    return c / np.linalg.norm(c)


rng = np.random.default_rng(0)
d, N = 64, 6                                    # 特徴の次元 / 記憶の数
X = rng.normal(size=(d, N))
X /= np.linalg.norm(X, axis=0, keepdims=True)   # 記憶を単位ノルムに揃える

target = 3
xi = make_cue(X[:, target], 0.5, rng)           # 記憶3を半分の大きさのノイズで崩す

beta = 100.0                                     # 逆温度。あとで意味を説明します
w = softmax(beta * (X.T @ xi))                   # ① 各記憶との相性 → ② 重みに変換
recalled = X @ w                                 # ③ 重み付き平均

print("選ばれた記憶:", w.argmax(), " 重み:", w.max().round(3))
print("元の記憶との差:", np.abs(recalled - X[:, target]).max().round(4))

# ---- 作図 ----
fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.2),
                       gridspec_kw={"width_ratios": [1.9, 2.2, 1.0, 2.2]})
ax[0].imshow(X.T, aspect="auto", cmap="RdBu_r", vmin=-.4, vmax=.4)
ax[0].set(title="① 6つの記憶を保存する", ylabel="記憶の番号", xlabel="特徴の次元")
ax[1].plot(X[:, target], color=BLUE, lw=1.6, label=f"本物の記憶{target}")
ax[1].plot(xi, color=GRAY, lw=1.1, label="ノイズを乗せた手がかり")
ax[1].set(title="② 崩れた手がかりを渡す", xlabel="特徴の次元")
ax[1].legend(fontsize=8.5)
ax[2].barh(range(N), w, color=[ORANGE if i == target else BLUE for i in range(N)])
ax[2].set(title="③ 想起の重み", xlim=(0, 1), yticks=range(N), ylabel="記憶の番号")
ax[2].invert_yaxis()
ax[3].plot(X[:, target], color=BLUE, lw=2.4, label=f"本物の記憶{target}")
ax[3].plot(recalled, color=ORANGE, lw=1.2, ls="--", label="想起した結果")
ax[3].set(title="④ 元の記憶が戻ってきた", xlabel="特徴の次元")
ax[3].legend(fontsize=8.5)
for a in ax[1:]:
    a.grid(alpha=.3)
plt.tight_layout()
plt.savefig("attention-hopfield-recall.png", bbox_inches="tight", dpi=150)
plt.show()
```

長く見えますが、想起をやっているのは次の2行だけです。

```python:抜粋
w = softmax(beta * (X.T @ xi))   # ①② 全記憶との相性 → 重み
recalled = X @ w                 # ③ その重みで記憶を平均
```

`X.T @ xi` が、手がかりと6つの記憶すべての内積を一度に計算しています。残りの行は記憶を作る準備と作図です。

なお `make_cue` でノイズを**相対的な大きさ**として定義しているのは、ここを間違えると実験が壊れるからです。単位ノルムに揃えた記憶に `rng.normal` をそのまま足すと、1成分あたりのノイズが信号の $\sqrt{d}$ 倍になり、手がかりが実質ただのノイズになってしまいます。

実行するとこうなります。

```
選ばれた記憶: 3  重み: 1.0
元の記憶との差: 0.0
```

崩れた手がかりから、元の記憶3が完全に戻ってきました。

![連想記憶の想起](/images/attention-hopfield-recall.png)

②の手がかり（灰色）は、元の記憶（青）からかなり崩れています。それでも③で記憶3だけに重みが集まり、④で元の波形がぴったり戻っています。

### なぜこれが「記憶」だと言えるのか？

やっているのは内積とsoftmaxと行列積だけで、どこにも「検索」や「照合」のような手続きは書いていません。それでも想起が成立するのは、**高次元ではランダムな2本のベクトルがほぼ直交する**からです。

$d=64$ 次元でランダムに引いた記憶どうしの内積はほぼ0になり、自分自身との内積だけが1になります。手がかりが記憶3の近くにあれば、記憶3との内積だけが突出し、softmaxがそれを1に、他を0に押し上げます。次元が高いほど記憶どうしが干渉しにくくなる、というのがこの仕組みの土台です。

## なぜその式がAttentionと一致するのか？

さて、いま書いた想起の式はこれでした。

$$
\xi_{\text{new}} = X \,\mathrm{softmax}(\beta X^\top \xi)
$$

そしてAttentionはこれでした。

$$
\mathrm{Attention}(q,K,V) = \mathrm{softmax}\!\left(\frac{qK^\top}{\sqrt{d}}\right)V
$$

対応をとってみます。

| 連想記憶 | Attention |
|---|---|
| 手がかり $\xi$（ベクトル1本） | クエリ $q$ |
| 保存された記憶 $X$（行列） | キー $K$ |
| 取り出す中身 $X$（行列） | バリュー $V$ |
| 逆温度 $\beta$ | $1/\sqrt{d}$ |

$K$ と $V$ に同じ記憶行列を入れ、$\beta = 1/\sqrt{d}$ と置けば、2つの式は同じものになります。実際に確かめます。

```python
rng = np.random.default_rng(1)
M = rng.normal(size=(64, 8))          # 記憶を並べた行列（1列が1つの記憶）
cue = M[:, 3] + 0.6 * rng.normal(size=64)

# (A) 連想記憶の想起則
beta = 1.0 / np.sqrt(64)
recalled_A = M @ softmax(beta * (M.T @ cue))


# (B) Transformerのattention（K=V=記憶パターン, Q=手がかり）
def softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=-1, keepdims=True)


# クエリは1本だが、行列として扱うため [None, :] で1行の行列にする（だから大文字Q）
Q, K, V = cue[None, :], M.T, M.T
out_B = (softmax_rows(Q @ K.T / np.sqrt(64)) @ V)[0]   # [0] で1行だけ取り出す

print("最大絶対誤差:", np.abs(recalled_A - out_B).max())
print("一致:", np.allclose(recalled_A, out_B))

# ---- 作図 ----
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.5),
                       gridspec_kw={"width_ratios": [2.2, 1]})
ax[0].plot(recalled_A, color=BLUE, lw=3.2, label="(A) 連想記憶の想起則")
ax[0].plot(out_B, color=ORANGE, lw=1.2, ls="--", label="(B) Attentionの式")
ax[0].set(title="2つの式の出力は完全に重なる", xlabel="特徴の次元", ylabel="出力の値")
ax[0].legend(fontsize=9)
attn = softmax_rows(Q @ K.T / np.sqrt(64))[0]      # ②で作られた重み
ax[1].bar(range(len(attn)), attn,
          color=[ORANGE if i == attn.argmax() else BLUE for i in range(len(attn))])
ax[1].set(title=f"このとき使われた重み（差の最大 {np.abs(recalled_A - out_B).max():.0f}）",
          xlabel="記憶の番号", ylabel="重み", ylim=(0, 1))
for a in ax:
    a.grid(alpha=.3)
plt.tight_layout()
plt.savefig("attention-hopfield-equivalence.png", bbox_inches="tight", dpi=150)
plt.show()
```

比べているのはこの2行です。

```python:抜粋
recalled_A = M @ softmax(beta * (M.T @ cue))              # 想起則
out_B = (softmax_rows(Q @ K.T / np.sqrt(64)) @ V)[0]      # attention
```

`Q, K, V = cue[None, :], M.T, M.T` と置いた行が要で、**キーとバリューに同じ記憶行列 `M` を入れています**。ここを別々にすると一致しません。そして `/ np.sqrt(64)` が (A) の `beta` と同じ値になっています。この2つを揃えると、残りの演算は完全に同じ順序になります。

手がかりは1本のベクトル $q$ ですが、`cue[None, :]` で1行だけの行列にしてから渡しています。Attentionの式が行列で書かれているためで、最後の `[0]` でその1行を取り出して元のベクトルに戻しています。

```
最大絶対誤差: 0.0
一致: True
```

![2つの式の一致](/images/attention-hopfield-equivalence.png)

誤差は $10^{-16}$ 程度ではなく、**ちょうど0**です。近似的に似ているのではなく、同じ順序で同じ演算をしているので当然そうなります。

これは後付けの解釈ではありません。Ramsauerらの論文 [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217)（ICLR 2021）が、現代版Hopfieldネットワークの想起則を導出したうえで「これはTransformerのattentionそのものである」と示しています。タイトルが元論文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) のもじりになっているのはそのためです。

つまり**Transformerの各ヘッドは、毎回「その文脈の中から記憶をひとつ思い出す」という計算を実行している**ことになります。

## なぜ √d で割る必要があるのか？

ここからが本題です。$\beta = 1/\sqrt{d}$ という対応が出てきましたが、なぜこの値なのでしょうか。

直感的な説明から入ります。softmaxに渡す点数が大きすぎると、1位だけが1になって他が全部0になります。逆に点数が小さすぎると、全部が同じくらいの重みになって「平均しただけ」になります。**点数の大きさは、softmaxがどれくらい鋭く効くかを決めるつまみ**です。

ここで問題になるのが、内積の大きさが次元 $d$ に依存することです。成分がそれぞれ分散1の $q, k$ について、内積 $q \cdot k = \sum_{i=1}^{d} q_i k_i$ は $d$ 個の項の和なので、その分散は $d$ に比例します。つまり標準偏差は $\sqrt{d}$ に比例して育ちます。

$d$ を大きくすると点数が勝手に大きくなり、softmaxが勝手に鋭くなってしまう。これを打ち消すのが $\sqrt{d}$ で割る操作です。実際に測ります。

```python
rng = np.random.default_rng(2)

dims = [16, 64, 256, 1024, 4096]
stds, with_s, without_s = [], [], []

for dd in dims:
    # 内積そのものの散らばり
    q = rng.normal(size=(3000, dd))
    k = rng.normal(size=(3000, dd))
    stds.append((q * k).sum(1).std())

    # 16本のキーに対する注意重みの最大値。1回だと振れるので200回の平均をとる
    a_, b_ = [], []
    for _ in range(200):
        Kk = rng.normal(size=(16, dd))
        sc = Kk @ rng.normal(size=dd)
        a_.append(softmax(sc / np.sqrt(dd)).max())
        b_.append(softmax(sc).max())
    with_s.append(np.mean(a_))
    without_s.append(np.mean(b_))

    print(f"d={dd:5d}  内積の標準偏差={stds[-1]:7.1f}  "
          f"最大重み(÷√d有)={with_s[-1]:.3f}  (÷√d無)={without_s[-1]:.3f}")

# ---- 作図 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].loglog(dims, stds, "o-", color=ORANGE, label="Q·K をそのまま")
ax[0].loglog(dims, np.array(stds) / np.sqrt(dims), "o-", color=BLUE, label="√d で割った後")
ax[0].set(title="内積の散らばりは d とともに育つ",
          xlabel="ヘッドあたりの次元 d", ylabel="内積の標準偏差")
ax[0].legend(fontsize=9)
ax[1].semilogx(dims, without_s, "o-", color=ORANGE, label="√d で割らない")
ax[1].semilogx(dims, with_s, "o-", color=BLUE, label="√d で割る")
ax[1].axhline(1 / 16, color=GRAY, ls=":", lw=1)
ax[1].text(20, 1 / 16 + .03, "16本に均等に注目した場合", fontsize=8.5, color=GRAY)
ax[1].set(title="割らないと注意が1点に寄っていく", xlabel="ヘッドあたりの次元 d",
          ylabel="最大の注意重み(200回平均)", ylim=(0, 1.05))
ax[1].legend(fontsize=9)
for a in ax:
    a.grid(alpha=.3)
plt.tight_layout()
plt.savefig("attention-hopfield-sqrtd.png", bbox_inches="tight", dpi=150)
plt.show()
```

比較しているのは、同じスコア `sc` に対する2つのsoftmaxです。

```python:抜粋
a_.append(softmax(sc / np.sqrt(dd)).max())   # √d で割る
b_.append(softmax(sc).max())                 # 割らない
```

違いは `/ np.sqrt(dd)` の有無だけで、入力するスコアも記憶の数（16本）も同じにしてあります。200回の平均をとっているのは、1回だけだと引いた乱数によって最大重みが大きく振れて、$d$ 依存の傾向が読み取れなくなるためです。

```
d=   16  内積の標準偏差=    4.0  最大重み(÷√d有)=0.240  (÷√d無)=0.671
d=   64  内積の標準偏差=    8.0  最大重み(÷√d有)=0.245  (÷√d無)=0.824
d=  256  内積の標準偏差=   15.8  最大重み(÷√d有)=0.247  (÷√d無)=0.916
d= 1024  内積の標準偏差=   32.0  最大重み(÷√d有)=0.235  (÷√d無)=0.940
d= 4096  内積の標準偏差=   63.6  最大重み(÷√d有)=0.248  (÷√d無)=0.978
```

![√dで割ることの効果](/images/attention-hopfield-sqrtd.png)

内積の標準偏差は 4.0 → 63.6 と、ちょうど $\sqrt{d}$ に比例して育っています（$d$ が256倍で標準偏差は16倍）。

そして $\sqrt{d}$ で割らない場合、最大の注意重みは 0.671 → 0.978 と上がり続けます。16本のキーがあるのに、$d=4096$ では実質1本しか見ていません。**しかもこれは学習の結果ではなく、次元を上げただけで自動的にそうなります。**

一方 $\sqrt{d}$ で割ると、$d$ を256倍にしても最大重みは 0.240 → 0.248 とほぼ動きません（均等に見た場合が $1/16 = 0.0625$ なので、適度に絞れている状態です）。**次元をいくら変えても同じ鋭さが保たれる**わけです。

### √d は温度を次元によらず一定に保つ係数だった

いま起きたことを物理の言葉に翻訳します。softmaxの中身を

$$
\mathrm{softmax}(\beta s_i) = \frac{e^{\beta s_i}}{\sum_j e^{\beta s_j}}
$$

と書くと、これは[前回の記事](https://zenn.dev/m2yagyu/articles/llm-temperature-boltzmann)で扱ったボルツマン分布そのものです。$\beta = 1/T$ が逆温度、$-s_i$ がエネルギーに当たります。

$\sqrt{d}$ で割らない場合、実効的な温度は $T \propto 1/\sqrt{d}$ となり、次元を上げるほど温度が下がります。**次元を上げただけで系が絶対零度に向かって凍りつく**わけです。凍りついた系は1つの状態しか取らないので、softmaxの出力は完全なone-hotになり、勾配はほぼ消えます。

$\sqrt{d}$ で割るというのは、**次元をいくら変えても温度が一定に保たれるように正規化する操作**でした。「スケーリングのため」という説明は正しいのですが、何をスケーリングしているかというと、温度です。

## 温度を変えると、記憶の想起はどう変わるか？

温度だとわかったので、振ってみます。$\beta$ を変えながら想起の重みを見ます。

```python
def eff(w):
    """有効記憶数 exp(H)。1なら1つの記憶に絞れている"""
    w = np.clip(w, 1e-30, 1)
    return np.exp(-(w * np.log(w)).sum())


betas = [0.5, 2, 4, 8]
for b in betas:
    ww = softmax(b * (X.T @ xi))
    print(f"β={b:4.1f} (T={1/b:.2f})  最大重み={ww.max():.3f}  有効記憶数={eff(ww):.2f}")

# ---- 作図1: βごとの重みの棒グラフ ----
fig, ax = plt.subplots(1, 4, figsize=(13, 2.9), sharey=True)
for a, b in zip(ax, betas):
    ww = softmax(b * (X.T @ xi))
    a.bar(range(N), ww, color=[ORANGE if i == target else BLUE for i in range(N)])
    a.set(title=f"β={b:g}  (温度 T={1/b:.2f})\n有効記憶数 {eff(ww):.2f}",
          ylim=(0, 1), xlabel="記憶の番号")
    a.grid(alpha=.3, axis="y")
ax[0].set_ylabel("想起の重み")
plt.tight_layout()
plt.savefig("attention-hopfield-beta-bars.png", bbox_inches="tight", dpi=150)
plt.show()

# ---- 作図2: 有効記憶数の温度依存 ----
bs = np.logspace(-1, 1.6, 80)
effs = [eff(softmax(b * (X.T @ xi))) for b in bs]
plt.figure(figsize=(6.2, 3.6))
plt.semilogx(bs, effs, color=BLUE, lw=2)
plt.axhline(N, color=GRAY, ls="--", lw=1)
plt.text(0.11, N - .45, "全部を均等に混ぜた状態", fontsize=9, color=GRAY)
plt.axhline(1, color=GRAY, ls="--", lw=1)
plt.text(0.11, 1.15, "1つの記憶に絞れた状態", fontsize=9, color=GRAY)
plt.xlabel("逆温度 β")
plt.ylabel("有効記憶数 exp(H)")
plt.title("温度を下げると記憶は1つに絞られる")
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("attention-hopfield-beta-curve.png", bbox_inches="tight", dpi=150)
plt.show()
```

要は `softmax(b * (X.T @ xi))` の `b` を差し替えているだけで、記憶 `X` も手がかり `xi` も最初のコードのまま使い回しています。変えたのは温度だけです。

もう一つの要点は `eff` です。

```python:抜粋
return np.exp(-(w * np.log(w)).sum())    # exp(エントロピー)
```

`-(w * np.log(w)).sum()` がエントロピー $H$ で、その指数をとっています。前回の記事で出てきたperplexityと同じ量で、確率が1つに集中していれば1、均等にばらけていれば選択肢の数に一致します。

```
β= 0.5 (T=2.00)  最大重み=0.242  有効記憶数=5.88
β= 2.0 (T=0.50)  最大重み=0.556  有効記憶数=4.00
β= 4.0 (T=0.25)  最大重み=0.880  有効記憶数=1.72
β= 8.0 (T=0.12)  最大重み=0.995  有効記憶数=1.04
```

![βを振ったときの想起の重み](/images/attention-hopfield-beta-bars.png)

「有効記憶数」は $\exp(H)$、つまりエントロピーの指数です。前回perplexityとして出てきたものと同じ量で、**実効的に何個の記憶を混ぜているか**を表します。記憶が6個なので、6なら全部を均等に混ぜた状態、1なら1個に絞れた状態です。

$\beta$ を上げていくと 5.88 → 4.00 → 1.72 → 1.04 と落ちていきます。連続的に見るとこうなります。

![有効記憶数の温度依存](/images/attention-hopfield-beta-curve.png)

高温では全部の記憶がぼんやり混ざり、低温では1つの記憶がくっきり立ち上がる。前回「温度を下げると1位のトークンに確率が集中する」という現象を見ましたが、まったく同じことが記憶の想起でも起きています。**同じ式なので当たり前ですが、その当たり前が面白いところです。**

これは実務上の意味も持ちます。Attentionのヘッドが「1つのトークンを鋭く参照する」のか「文脈全体をならして見る」のかは、学習で決まる別々の機能のように見えて、**実体は同じ式のスコアの大きさ、つまり実効温度の違い**でしかありません。

## 何個まで記憶を詰め込めるのか？

連想記憶には容量の限界があります。詰め込みすぎると記憶どうしが干渉して、手がかりを渡しても正しいものが出てこなくなります。

古典的なHopfieldネットワーク（1982年）の容量は、スピングラスの理論から $0.138 \times d$ 個と計算されています。$d=64$ なら**8.8個**しか覚えられません。えらく少ないと感じますが、これがsoftmax版だとどうなるか測ってみます。手がかりには記憶の2倍の大きさのノイズを乗せる、かなり厳しい条件にします。

```python
rng = np.random.default_rng(4)

d = 64
Ns = [8, 32, 128, 512, 2048, 8192]
acc = []
for n in Ns:
    ok = 0
    for _ in range(400):
        Xc = rng.normal(size=(d, n))
        Xc /= np.linalg.norm(Xc, axis=0, keepdims=True)
        t = rng.integers(n)
        cue = make_cue(Xc[:, t], 2.0, rng)
        ok += int(softmax(100.0 * (Xc.T @ cue)).argmax() == t)
    acc.append(ok / 400 * 100)
    print(f"N={n:5d}  正解率 {acc[-1]:5.1f}%  (でたらめなら {100/n:.3f}%)")

# ---- 作図 ----
plt.figure(figsize=(6.8, 3.9))
plt.semilogx(Ns, acc, "o-", color=BLUE, lw=2, base=2, label="softmax版(=attention)")
plt.semilogx(Ns, [100 / n for n in Ns], "--", color=GRAY, lw=1.2, base=2,
             label="でたらめに選んだ場合")
plt.axvline(0.138 * d, color=ORANGE, ls="--", lw=1.5)
plt.text(0.138 * d * 1.15, 42, f"古典Hopfieldの容量\n0.138×d ≒ {0.138*d:.1f}個",
         fontsize=9, color=ORANGE)
plt.xlabel(f"詰め込んだ記憶の数 N  (特徴次元 d={d})")
plt.ylabel("正しく想起できた割合 [%]")
plt.title("記憶の2倍のノイズを乗せても、数千個から取り出せる")
plt.ylim(0, 105)
plt.legend(fontsize=9)
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("attention-hopfield-capacity.png", bbox_inches="tight", dpi=150)
plt.show()
```

判定しているのはこの1行です。

```python:抜粋
ok += int(softmax(100.0 * (Xc.T @ cue)).argmax() == t)
```

想起した結果が元の記憶と近いかではなく、**最も重みが大きかった記憶の番号 `argmax` が、崩す前の番号 `t` と一致したか**で数えています。曖昧さの残らない基準にしたかったためです。`make_cue(..., 2.0, rng)` の `2.0` が、記憶の2倍の大きさのノイズを乗せるという条件を作っています。

```
N=    8  正解率  99.5%  (でたらめなら 12.500%)
N=   32  正解率  94.8%  (でたらめなら 3.125%)
N=  128  正解率  87.0%  (でたらめなら 0.781%)
N=  512  正解率  78.0%  (でたらめなら 0.195%)
N= 2048  正解率  61.8%  (でたらめなら 0.049%)
N= 8192  正解率  50.0%  (でたらめなら 0.012%)
```

![記憶容量](/images/attention-hopfield-capacity.png)

$d=64$ のまま、**8192個**の記憶から正しいものを50.0%で取り出せています。でたらめに選んだ場合の0.012%と比べると4000倍以上です。古典の限界8.8個とは桁が違います。

この差が生まれるのは、softmaxの指数関数が効いているからです。古典Hopfieldは内積を線形に足し合わせるので干渉がそのまま蓄積しますが、softmaxは1位とのわずかな差を指数的に増幅するため、干渉を押しつぶせます。Ramsauerらはこの容量が $d$ に対して指数的に増えることを示しています。

Transformerが長い文脈から必要な1語を引っ張ってこられるのは、この容量に支えられています。

## 本物のTransformerでもそうなっているのか？

ここまでは自作の連想記憶の話でした。実物のLLMのattentionが本当にこう振る舞っているかを測ります。モデルは前回と同じ `llm-jp/llm-jp-3-150m`（150Mパラメータ、12層×8ヘッド、ヘッドあたり $d=64$）で、CPUで動きます。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "llm-jp/llm-jp-3-150m"
tok = AutoTokenizer.from_pretrained(MODEL)
# attention の重みを取り出したいので eager 実装を指定する
model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager")
model = model.float()   # bfloat16のままだとnumpyに変換できないのでfloat32にする
model.eval()

text = "日本の首都は東京です。日本で一番高い山は富士山です。日本の首都は"
ids = tok(text, return_tensors="pt")
with torch.no_grad():
    out = model(**ids, output_attentions=True)

T = ids["input_ids"].shape[1]
toks = [t.replace("▁", "") or "␣" for t in tok.convert_ids_to_tokens(ids["input_ids"][0])]
L, H = len(out.attentions), out.attentions[0].shape[1]

Emap = np.zeros((L, H))          # ヘッドごとの有効注目数
Wmap = np.zeros((L, H, T))       # ヘッドごとの注意の重み
for l, a in enumerate(out.attentions):
    w = a[0, :, -1, :].float().numpy()      # (ヘッド, トークン) 最終トークンの行
    for h in range(H):
        Emap[l, h], Wmap[l, h] = eff(w[h]), w[h]

print(f"有効注目数: 最小 {Emap.min():.2f} / 中央値 {np.median(Emap):.2f} "
      f"/ 最大 {Emap.max():.2f}  (全{T}トークン)")

# ---- 作図1: ヘッドごとの有効注目数 ----
fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.35, 1]})
im = ax[0].imshow(Emap, aspect="auto", cmap="viridis", vmin=1, vmax=T)
ax[0].set(title=f"最終トークンから見た「有効注目数」({L}層×{H}ヘッド)",
          xlabel="ヘッド番号", ylabel="層", xticks=range(H), yticks=range(0, L, 2))
plt.colorbar(im, ax=ax[0]).set_label(f"何語ぶんを見ているか (最大{T})", fontsize=9)
ax[1].hist(Emap.ravel(), bins=22, color=BLUE, edgecolor="white")
ax[1].axvline(1, color=ORANGE, ls="--", lw=1.5)
ax[1].text(1.25, ax[1].get_ylim()[1] * .82, "1語だけを\n見ている", fontsize=9, color=ORANGE)
ax[1].set(title="鋭いヘッドと混ぜるヘッドが共存する",
          xlabel="有効注目数", ylabel="ヘッド数")
ax[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig("attention-hopfield-real-heads.png", bbox_inches="tight", dpi=150)
plt.show()

# ---- 作図2: 最も鋭いヘッド vs 最も混ぜるヘッド ----
lo = np.unravel_index(Emap.argmin(), Emap.shape)
hi = np.unravel_index(Emap.argmax(), Emap.shape)
fig, ax = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True)
for a, (l, h), c, lab in [(ax[0], lo, ORANGE, "最も鋭いヘッド"),
                          (ax[1], hi, BLUE, "最も混ぜるヘッド")]:
    a.bar(range(T), Wmap[l, h], color=c)
    a.set(ylabel="注意の重み", ylim=(0, 1))
    a.set_title(f"{lab}(第{l}層 ヘッド{h})  有効注目数 {Emap[l, h]:.2f}",
                fontsize=11, loc="left")
    a.grid(alpha=.3, axis="y")
ax[1].set_xticks(range(T))
ax[1].set_xticklabels(toks, rotation=55, ha="right", fontsize=9)
plt.tight_layout()
plt.savefig("attention-hopfield-real-sink.png", bbox_inches="tight", dpi=150)
plt.show()

# 鋭いヘッド上位10個が何を見ているか
order = np.dstack(np.unravel_index(np.argsort(Emap, axis=None), Emap.shape))[0]
print("\n最も鋭いヘッド上位10個が見ていたトークン:")
for l, h in order[:10]:
    print(f"  第{l:>2}層 ヘッド{h}  有効注目数{Emap[l,h]:.2f}  → {toks[int(Wmap[l,h].argmax())]!r}")
print(f"  → 上位10個のうち {sum(1 for l,h in order[:10] if Wmap[l,h].argmax()==0)} 個が先頭トークン")
```

このコードの要は次の1行です。

```python:抜粋
w = a[0, :, -1, :].float().numpy()      # (ヘッド, トークン) 最終トークンの行
```

`out.attentions` は層ごとに `(バッチ, ヘッド, 問い合わせ側, 参照側)` の4次元で入っています。`-1` で**最終トークンの行だけ**を取り出しているので、「次の1語を予測しようとしている今、どの語を見ているか」が得られます。この行が、自作の連想記憶でいう `softmax(beta * (X.T @ xi))` の中身にあたります。

`attn_implementation="eager"` を指定しているのは、高速化された実装だと重みが返らないためです。`model.float()` は、このモデルの既定がbfloat16でnumpyに変換できないので挟んでいます。

```
有効注目数: 最小 1.04 / 中央値 5.61 / 最大 12.43  (全17トークン)
```

![実物のヘッドごとの有効注目数](/images/attention-hopfield-real-heads.png)

96個のヘッドが、1.04（1語だけを鋭く見る）から12.43（17語中12語をならして見る）まで幅広く分布しています。**すべて同じ式・同じ $\sqrt{d}$ を使っているのに、実効的な温度がヘッドごとに違う**わけです。温度を決めているのは $\sqrt{d}$ ではなく、学習で決まった $W_Q, W_K$ が作るスコアの大きさです。

$\sqrt{d}$ は温度の基準点を揃えるだけで、そこからどれだけ上下させるかはモデルが学習で決めている、という分業になっています。実際、$\sqrt{d}$ で割る前のスコアの大きさを層ごとに測ると違います。

```python
with torch.no_grad():
    hs = model(**ids, output_hidden_states=True).hidden_states

d_head = model.config.hidden_size // H
raw, scaled = [], []
for layer in range(L):
    blk = model.model.layers[layer].self_attn
    with torch.no_grad():
        # q_proj/k_proj には input_layernorm を通した後の値が入る。
        # 生の hidden_states を渡すとスケールが違い、誤った分散になる
        x = model.model.layers[layer].input_layernorm(hs[layer])
        q = blk.q_proj(x)[0].view(T, -1, d_head).transpose(0, 1)
        k = blk.k_proj(x)[0].view(T, -1, d_head).transpose(0, 1)
        s = torch.matmul(q, k.transpose(-1, -2))
    raw.append(s.std().item())
    scaled.append((s / np.sqrt(d_head)).std().item())
    if layer in (0, 6, 11):
        print(f"第{layer:>2}層: 割る前 {raw[-1]:6.2f} → √d({np.sqrt(d_head):.0f})で割った後 {scaled[-1]:5.2f}")

# ---- 作図 ----
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(range(L), raw, "o-", color=ORANGE, label="√d で割る前")
ax.plot(range(L), scaled, "o-", color=BLUE, label="√d で割った後")
ax.axhline(1, color=GRAY, ls=":", lw=1)
ax.text(0.1, 1.6, "softmaxがほどよく効く目安", fontsize=8.5, color=GRAY)
ax.set(title="実効温度は層ごとに違う", xlabel="層", ylabel="スコアの標準偏差",
       yscale="log", xticks=range(L))
ax.legend(fontsize=9)
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("attention-hopfield-layer-temp.png", bbox_inches="tight", dpi=150)
plt.show()
```

ここでは `out.attentions` の完成品ではなく、**softmaxに入る前のスコア**を見たいので、$Q$ と $K$ を自分で作り直しています。

```python:抜粋
x = model.model.layers[layer].input_layernorm(hs[layer])
q = blk.q_proj(x)[0].view(T, -1, d_head).transpose(0, 1)
```

`input_layernorm` を通しているのが要点です。`q_proj` に入るのは正規化を済ませた後の値なので、`hidden_states` をそのまま渡すとスケールが違い、分散が実際とは別の値になります（最初これを忘れて、第0層の標準偏差が0.00という明らかにおかしな値が出ました）。

```
第 0層: 割る前  27.73 → √d(8)で割った後  3.47
第 6層: 割る前  29.67 → √d(8)で割った後  3.71
第11層: 割る前  10.14 → √d(8)で割った後  1.27
```

![層ごとの実効温度](/images/attention-hopfield-layer-temp.png)

同じ $\sqrt{d}$ で割っていても、割った後の散らばりが第6層で3.71、第11層で1.27と3倍近く違います。**層ごとに実効温度が違う**ということです。

### 最も鋭いヘッドが見ていたのは単語ではなかった

ここで予想外の結果が出ました。最も鋭い（＝最も低温の）ヘッド上位10個が何を見ているかを調べると、**10個中10個が文頭トークン `<s>` を見ていました**。

```
最も鋭いヘッド上位10個が見ていたトークン:
  第 9層 ヘッド1  有効注目数1.04  → '<s>'
  第 3層 ヘッド0  有効注目数1.15  → '<s>'
  第10層 ヘッド4  有効注目数1.33  → '<s>'
  ...
  → 上位10個のうち 10 個が先頭トークン
```

![最も鋭いヘッドと最も混ぜるヘッド](/images/attention-hopfield-real-sink.png)

第9層ヘッド1は、重みの99.6%を `<s>` に置いています。文頭トークンは内容を持たないので、意味のある想起をしているとは考えにくい。

これは **attention sink** として知られる現象です（[Xiao et al., 2023](https://arxiv.org/abs/2309.17453)）。softmaxは重みの合計を必ず1にするので、「今回は思い出すべきものが何もない」という状況でも、どこかに重みを置かざるを得ません。その捨て場所として、どの文でも必ず存在する先頭トークンが使われます。

連想記憶の言葉でいえば、**手がかりにマッチする記憶が無いときに落ち着く既定の状態**です。物理でいう基底状態に近い役割を、学習の結果としてモデルが自分で用意したことになります。

なお、この記事の実験は17トークンの1文だけで測ったものなので、「このモデルのヘッドの性格」を一般化して語れるものではありません。長い文や別の文で測れば分布は変わります。ここで確かめられるのは「同じ式のまま、ヘッドによって実効温度が大きく違う」という点までです。

## これは物理のどこから来たのか？

最後に出自を確認します。連想記憶をこの形で定式化したのは、物理学者のJohn Hopfieldが1982年に発表した論文です。磁性体のモデル（イジング模型・スピングラス）をそのまま記憶の模型として読み替える、という発想でした。

Hopfieldネットワークにはエネルギー関数があり、記憶はそのエネルギーの**極小点**として保存されます。想起とは、手がかりの位置から坂を下って一番近い谷に落ちることです。

現代版のエネルギー関数はこう書けます。

$$
E(\xi) = -\frac{1}{\beta}\log\sum_{i=1}^{N} e^{\beta\, x_i^\top \xi} + \frac{1}{2}\|\xi\|^2
$$

第1項は log-sum-exp で、統計力学では**自由エネルギー** $-\frac{1}{\beta}\log Z$ そのものです（$Z$ が分配関数）。前回の記事でsoftmaxの分母が分配関数だと確認しましたが、それがここで自由エネルギーとして再登場しています。このエネルギーを $\xi$ で微分して更新式を作ると、先ほどの $X\,\mathrm{softmax}(\beta X^\top\xi)$ が出てきます。

温度によってこの地形がどう変わるかを、2次元で描いてみます。

```python
# 2次元の記憶を3つ置いて、エネルギーの地形を温度ごとに描く
P = np.array([[-1.2, -0.9], [1.3, -0.6], [0.1, 1.3]]).T
g = np.linspace(-2.2, 2.2, 300)
GX, GY = np.meshgrid(g, g)
Z = np.stack([GX.ravel(), GY.ravel()])

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7))
for a, b in zip(ax, [1.0, 4.0, 20.0]):
    s = b * (P.T @ Z)
    lse = (np.log(np.exp(s - s.max(0)).sum(0)) + s.max(0)) / b   # 自由エネルギー
    E = (-lse + 0.5 * (Z ** 2).sum(0)).reshape(GX.shape)
    a.contourf(GX, GY, E, levels=28, cmap="Blues_r")
    a.contour(GX, GY, E, levels=14, colors="white", linewidths=.4, alpha=.6)
    a.scatter(P[0], P[1], c=ORANGE, s=70, edgecolor="white", zorder=5, label="記憶")
    a.set(title=f"β={b:g}  (温度 T={1/b:.2f})", xticks=[], yticks=[])
ax[0].legend(fontsize=9, loc="upper left")
ax[0].set_ylabel("エネルギーの地形")
fig.suptitle("温度を下げると、記憶のひとつひとつが別々の谷になる", fontsize=12, y=1.04)
plt.tight_layout()
plt.savefig("attention-hopfield-energy.png", bbox_inches="tight", dpi=150)
plt.show()
```

エネルギーを計算しているのはこの2行で、上の式をそのまま書き下したものです。

```python:抜粋
lse = (np.log(np.exp(s - s.max(0)).sum(0)) + s.max(0)) / b   # 自由エネルギー
E = (-lse + 0.5 * (Z ** 2).sum(0)).reshape(GX.shape)
```

`s.max(0)` を引いてから足し直しているのは、$e^{\beta x}$ が桁あふれするのを避けるためで、値は変わりません。第2項の `0.5 * (Z ** 2).sum(0)` が $\frac{1}{2}\|\xi\|^2$ で、これが無いと谷が無限に深くなってしまいます。

![エネルギーの地形](/images/attention-hopfield-energy.png)

高温（左）では3つの記憶が1つの大きな谷に融けていて、どこから落ちても同じ場所に着きます。区別ができないので、想起は「全部の平均」になります。温度を下げる（右）と谷が3つに分裂し、記憶それぞれが独立した極小点になります。これが「1つの記憶に絞れた」状態の正体です。

さきほど $\beta$ を振って見た有効記憶数 5.88 → 1.04 の変化は、この地形が融合状態から分離状態へ移る様子を数値で見ていたことになります。

ちなみにHopfieldは、この一連の仕事で2024年のノーベル物理学賞をGeoffrey Hintonと共同受賞しています。受賞理由は「人工ニューラルネットワークによる機械学習を可能にした基礎的発見と発明」でした。物理の道具で作られた記憶の模型が、40年後にTransformerの中で毎秒何十億回も実行されている、というのがいまの状況です。

## まとめ

- Attentionの `softmax(QKᵀ/√d)V` は、連想記憶の想起則 $X\,\mathrm{softmax}(\beta X^\top\xi)$ と**同一の式**でした。実際に計算すると誤差はちょうど0です
- $\sqrt{d}$ で割るのは**温度を次元によらず一定に保つ**ためでした。割らないと最大重みが $d$ とともに 0.671 → 0.978 と上がり、16本のキーのうち実質1本しか見なくなります
- $\beta$（逆温度）は、記憶を1つに絞るか複数を混ぜるかを決めるつまみです。前回の記事の温度と同じものが、同じ働きをしています
- 容量は古典Hopfieldの $0.138d \simeq 8.8$ 個をはるかに超え、$d=64$ で8192個の記憶から50.0%で取り出せました
- 実物のLLMでは、96ヘッドの有効注目数が1.04から12.43まで分布していました。$\sqrt{d}$ が基準点を揃え、そこからの上下は学習が決めています
- 最も鋭いヘッドは単語ではなく文頭トークンを見ていました。思い出すものが無いときの重みの捨て場所（attention sink）です

「Attentionは重み付き平均だ」という理解は間違っていませんが、**何の重み付き平均なのか**まで降りると、40年前の磁性体の模型に行き着きます。次回は、この $\beta$ がもう一箇所、思わぬところにも隠れている話を扱う予定です。

## 参考文献

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017) — $\sqrt{d}$ による正規化の初出
- Ramsauer et al., [Hopfield Networks is All You Need](https://arxiv.org/abs/2008.02217) (ICLR 2021) — 現代版Hopfieldとattentionの同一性、指数的な容量
- J. J. Hopfield, [Neural networks and physical systems with emergent collective computational abilities](https://www.pnas.org/doi/10.1073/pnas.79.8.2554) (PNAS 1982) — 連想記憶の原論文
- Amit, Gutfreund & Sompolinsky, [Storing infinite numbers of patterns in a spin-glass model of neural networks](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.55.1530) (PRL 1985) — 容量 $0.138N$ の導出
- Xiao et al., [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (ICLR 2024) — attention sink
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Attention/Transformerの図解入門
- [The Nobel Prize in Physics 2024](https://www.nobelprize.org/prizes/physics/2024/summary/) — Hopfield と Hinton の受賞
- 前回の記事: [LLMのtemperatureは本当に温度だった](https://zenn.dev/m2yagyu/articles/llm-temperature-boltzmann) — softmaxとボルツマン分布の対応

---

この章の元記事: [attention-hopfield-associative-memory](https://zenn.dev/m2yagyu/articles/attention-hopfield-associative-memory)
