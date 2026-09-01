---
title: "長いプロンプトで大事な文を置く場所は、結局どこが正解なのか"
emoji: "📍"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["deeplearning", "生成ai", "llm", "chatgpt", "huggingface"]
published: true
---

この記事を読むと、長いプロンプトの中で「答えの根拠になる1文をどこに置くべきか」を、他人の記事ではなく自分の手元の測定で決められるようになります。あわせて、その測定が**そもそも成立しているか**を確かめる方法が身につきます。今回はそこで一度つまずいたので、その顛末も含めて書きます。

結論を先に書きます。

- 学習長（このモデルでは4,096）の**内側**では、根拠文を先頭に置いても末尾に置いても正解率は変わりませんでした。細かく見ると差はありますが、それは根拠文の**絶対位置**ではなく、根拠文と質問の**距離**で説明できます
- 学習長の**外側**では、根拠文を同じ長さの無関係な文に差し替えても**答えが1問も変わりません**でした。文脈が読まれていないので、そこでは置き場所を論じる意味がありません
- つまり「どこに置くか」を気にする前に、**その長さで文脈が読まれているか**を確かめる必要があります

GPUは不要です。152Mパラメータのモデルと numpy をCPUで動かすだけで最後まで通ります。ただしCPU推論を何度もまわすので、全部で30分ほどかかります（手元の8スレッドのCPUでの目安です）。コードは作図まで含んでいるので、上から順にコピペすると記事と同じ図が手元に出ます。

## 大事な文は、先頭と末尾のどちらに置くべきなのか

長い資料をプロンプトに貼るとき、誰もが一度は迷う実務的な問題があります。**答えの根拠になる1文を、資料のどこに置くのが有利なのか。**

よく引かれるのが [Lost in the Middle](https://arxiv.org/abs/2307.03172) です。関連する情報が文脈の先頭か末尾にあるときに性能が最も高く、中央にあるときに大きく落ちる、という報告で、GPT-3.5-Turbo の20文書設定では先頭75.8%・中央53.8%・末尾63.2%という数字が出ています。中央に置いた場合は、文書を一切与えない56.1%すら下回ります。

ここから「大事なことは先頭か末尾に置け」という実務上の言い伝えが広まりました。ただ、これは特定のモデルと特定のタスクでの測定です。手元のモデルで同じことが起きるとは限りません。

そこで、自分で測れる形に落とします。設計はこうです。

- 長い詰め物の文章の中に、**答えが書いてある1文だけ**を埋める
- その1文を埋める位置を、先頭から質問の直前まで動かす
- 最後に質問し、4つの候補から選ばせる
- 正解率を「埋めた位置」と「全体の長さ」の関数として見る

いわゆる [Needle in a Haystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) の形式ですが、生成させて目で見るのではなく、選択肢の対数尤度で機械的に採点します。

この記事は [前回](https://zenn.dev/m2yagyu/articles/rope-long-context-breakdown) の続きです。前回はRoPEの幾何を追って、学習長を超えると注意の向き先が壊れることを perplexity で見ました。今回はその壊れ方が、**具体的なタスクの成績としてどう出るか**を測ります。同じモデルと同じ書籍を使いますが、前処理と計算精度は変えているので、前回の数値がそのまま再現するわけではありません。

## なぜ実在の国名を使ってはいけないのか

ここが設計でいちばん大事なところです。

もし「日本の首都は東京である」を埋めて「日本の首都は」と聞いたら、モデルは**文脈を読まずに**東京と答えられます。[前々回](https://zenn.dev/m2yagyu/articles/llm-hallucination-confidence)で測ったとおり、このモデルは資料なしでも15カ国中6カ国の首都を当てます。つまり正解しても、それが「読んだから」なのか「覚えていたから」なのか区別できません。

なので、国名も都市名も**架空のもの**を使います。事前学習に存在しない組み合わせなら、正解する唯一の道は文脈中のその1文を読むことだけです。

ただし、これだけでは足りませんでした。**架空の名前にしても、モデルは文脈を読まずに何かを選びます。** 4択なので当てずっぽうでも25%は当たる、と言いたくなりますが、後で見るようにそれも違います。そこで最初から、**根拠文を同じ長さの無関係な文に差し替えた対照**を用意します。これがこの記事の背骨です。

## 測る道具をひととおり用意する

モデルの読み込み、詰め物のテキスト、架空の問題セット、採点関数、そして対照条件。全部このブロックに入っています。長いので、後ろで要の数行だけ取り出して説明します。

```python
try:
    import matplotlib_fontja  # noqa: F401
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib-fontja"])
    import matplotlib_fontja  # noqa: F401

import io, re, math, copy, zipfile, urllib.request
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.set_grad_enabled(False)
MODEL = "llm-jp/llm-jp-3-150m"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).float()   # CPUではfp32のほうが速い（後述）
model.eval()
TRAIN = model.config.max_position_embeddings
BASE  = getattr(model.config, "rope_theta", None) or model.config.rope_scaling["rope_theta"]
print(f"学習で見た最大の位置 TRAIN = {TRAIN} / RoPEのbase = {BASE:,.0f}")

url = "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip"
with urllib.request.urlopen(url) as r:
    z = zipfile.ZipFile(io.BytesIO(r.read()))
    raw = z.read(z.namelist()[0]).decode("shift_jis", errors="ignore")
raw = re.sub(r"《[^》]*》|［[^］]*］|｜", "", re.split(r"-{10,}", raw)[-1])
FILLER = tok.encode(re.sub(r"[\r\n　]+", "", raw), add_special_tokens=False)
print(f"詰め物 {len(FILLER):,} トークン")

CITIES = ['ミリタ','サルド','ガルミ','キノス','メロナ','パナラ','セイフ','ノルダ','ベルカ','テミス',
          'カルナ','ソルビ','ネビス','ドゥカ','ヴィナ','オリド','ゼノア','タルモ','ユニカ','ロゼル']
COUNTRIES = ['ザラン','ブリード','オスカニア','タシュカ','リンデル','ヨーカス','ヴェルタ','メロディ',
             'カンタル','ペルシオ','ドナウス','キルヴァ']
NITEM = 10
rng = np.random.default_rng(0)
golds = list(rng.permutation(CITIES))[:NITEM]
ITEMS = []
for i in range(NITEM):
    others = [c for c in CITIES if c != golds[i]]
    opts = rng.permutation([golds[i]] + list(rng.permutation(others)[:3]))
    ITEMS.append((COUNTRIES[i], str(golds[i]), [str(c) for c in opts]))
print(f"例: {ITEMS[0][0]}国 → 正解 {ITEMS[0][1]} / 選択肢 {ITEMS[0][2]}")
print(f"都市名のトークン数: {sorted({len(tok.encode(c, add_special_tokens=False)) for c in CITIES})}")

PREFIX   = "次の文章を読んで、あとの問題に答えなさい。\n"
NEEDLE   = lambda c, city: f"{c}国の首都は{city}である。"
QUESTION = lambda c: f"\n問題：{c}国の首都はどこか。\n答え：{c}国の首都は"

def build(country, gold, ctx_len, depth, drop=False):
    """全長が ctx_len になるよう詰め物を敷き、その depth の位置に根拠文を1つ埋める。
       drop=True なら根拠文を同じ長さの別の詰め物に差し替える（＝対照条件）"""
    pre = tok.encode(PREFIX, add_special_tokens=False)
    ndl = tok.encode(NEEDLE(country, gold), add_special_tokens=False)
    qst = tok.encode(QUESTION(country), add_special_tokens=False)
    body = FILLER[:max(0, ctx_len - len(pre) - len(ndl) - len(qst))]
    cut = int(round(depth * len(body)))
    if drop:
        ndl = FILLER[100_000:100_000 + len(ndl)]   # 長さは保ったまま中身だけ無関係な文に
    return pre + body[:cut] + ndl + body[cut:] + qst, len(pre) + cut

def choose(m, ids, opts):
    """4候補の対数尤度。長い前置きは1回だけ通し、KVキャッシュを使い回す"""
    out = m.model(torch.tensor([ids]), use_cache=True)
    past, L = out.past_key_values, len(ids)
    lp = torch.log_softmax(m.lm_head(out.last_hidden_state[:, -1])[0].float(), -1)
    res = {}
    for o in opts:
        oid = tok.encode(o, add_special_tokens=False)
        tot = lp[oid[0]].item()
        for j in range(1, len(oid)):
            o2 = m.model(torch.tensor([[oid[j-1]]]), past_key_values=past, use_cache=True)
            tot += torch.log_softmax(m.lm_head(o2.last_hidden_state[:, -1])[0].float(), -1)[oid[j]].item()
        past.crop(L)                       # 次の候補のためキャッシュを前置きの長さへ戻す
        res[o] = tot
    return res

def run(m, ctx_len, depth, drop=False):
    """正解率 / マージン（正解−最良の誤答）/ 実際に選んだ候補の列"""
    ok, mg, picks = [], [], []
    for country, gold, opts in ITEMS:
        ids, _ = build(country, gold, ctx_len, depth, drop)
        s = choose(m, ids, opts)
        p = max(s, key=s.get); picks.append(p)
        ok.append(p == gold)
        mg.append(s[gold] - max(v for k, v in s.items() if k != gold))
    return float(np.mean(ok)), float(np.mean(mg)), picks

for L in (256, 1024):
    a, g, _ = run(model, L, 0.5)
    a0, g0, _ = run(model, L, 0.5, drop=True)
    print(f"全長 {L:>5}: 根拠文あり 正解率{a:.2f}／マージン{g:+.2f}   "
          f"根拠文なし 正解率{a0:.2f}／マージン{g0:+.2f}")

# 候補は全て2トークン。1トークン目だけで採点すると何が起きるかを確かめる
firsts = [tok.encode(c, add_special_tokens=False)[0] for c in CITIES]
print(f"\n候補20語のうち、1トークン目が他と重なる語: {len(firsts) - len(set(firsts))} 語")
disagree = 0
for country, gold, opts in ITEMS:
    ids, _ = build(country, gold, 1024, 0.5)
    out = model.model(torch.tensor([ids]), use_cache=True)
    lp = torch.log_softmax(model.lm_head(out.last_hidden_state[:, -1])[0].float(), -1)
    first = {o: lp[tok.encode(o, add_special_tokens=False)[0]].item() for o in opts}
    full = choose(model, ids, opts)
    disagree += max(first, key=first.get) != max(full, key=full.get)
print(f"1トークン目だけの採点と、全トークンの採点で選択が違った問: {disagree}/{NITEM}")
```

```
学習で見た最大の位置 TRAIN = 4096 / RoPEのbase = 10,000
詰め物 186,458 トークン
例: ザラン国 → 正解 メロナ / 選択肢 ['ソルビ', 'ネビス', 'パナラ', 'メロナ']
都市名のトークン数: [2]
全長   256: 根拠文あり 正解率1.00／マージン+9.94   根拠文なし 正解率0.30／マージン-1.08
全長  1024: 根拠文あり 正解率1.00／マージン+9.98   根拠文なし 正解率0.20／マージン-1.10

候補20語のうち、1トークン目が他と重なる語: 0 語
1トークン目だけの採点と、全トークンの採点で選択が違った問: 1/10
```

短い文脈では、根拠文があれば満点、なければ0.20〜0.30です。マージン（正解の対数尤度から最も強い誤答の対数尤度を引いた値）も、あれば約+10、なければ約−1.1。**この差があって初めて「読んで答えている」と言えます。**

ここで早くも1つ、思い込みが崩れています。**4択なのに、根拠文なしの正解率は0.25ではありません。** 全長256で0.30、全長1,024で0.20です。モデルには候補ごとの好みがあるので、当てずっぽうにはなりません。あとで見るように、この床は長さや設定によって0.20から0.60まで動きます。**床は計算するものではなく、測るものでした。**

効いている行は4つです。

```python:抜粋
model = AutoModelForCausalLM.from_pretrained(MODEL).float()
```

`.float()` を外すと大幅に遅くなります。このモデルは既定で bfloat16 で読み込まれますが、CPUのbf16はハードウェア支援が効かず、内部で変換しながら計算します。8,192トークンの順伝播1回で、手元の8スレッドでは約40秒対約3.2秒（12.5倍）、4スレッドでは約44.5秒対約4.7秒（9.5倍）でした。**環境に依存しますが、10倍前後は変わります。** GPUに載せるなら逆になるので、そのときは外してください。

```python:抜粋
    if drop:
        ndl = FILLER[100_000:100_000 + len(ndl)]   # 長さは保ったまま中身だけ無関係な文に
```

これがこの記事の背骨です。**根拠文を取り除くのではなく、同じトークン数の無関係な文に差し替えます。** 取り除くと全長が変わってしまい、比べたい「長さ」が動きます。長さを固定したまま中身だけ無意味にすることで、「その1文を読んで答えたのか」を分離できます。

```python:抜粋
body = FILLER[:max(0, ctx_len - len(pre) - len(ndl) - len(qst))]
cut = int(round(depth * len(body)))
return pre + body[:cut] + ndl + body[cut:] + qst, len(pre) + cut
```

詰め物を `cut` で切って、あいだに根拠文を挟みます。`depth` が0なら先頭、1なら質問の直前です。**全長がどの深さでも同じになるように**、詰め物の長さから根拠文と質問のぶんを先に引いています。ここを引き忘れると、深さを変えたときに長さも一緒に動き、この記事で見たい切り分けができなくなります。

```python:抜粋
        past.crop(L)                       # 次の候補のためキャッシュを前置きの長さへ戻す
```

候補は4つあり、すべて2トークンです。前置きが8,000トークンあるので、候補ごとに全部を通し直すと4倍の時間がかかります。前置きを1回だけ通してKVキャッシュを作り、候補のトークンだけを足して、次の候補に移る前に `crop` でキャッシュを前置きの長さへ戻します。**この1行を忘れると、前の候補のトークンがキャッシュに残ったまま次の候補を採点することになり、静かに間違った値が出ます。**

候補の全トークンぶんを足しているのは、1トークン目だけでは足りないからです。20語の1トークン目に重複はありませんが、それでも1トークン目だけの採点は10問中1問で全トークン採点と違う候補を選びました。2トークン目にも情報があるので、片方だけ見ると順位が変わります。

## 根拠文はどこに埋まっているのか

言葉だけだと分かりにくいので、実際に作ったトークン列を図にします。

```python
fig, ax = plt.subplots(figsize=(10, 3.6))
SHOW_D = [0.0, 0.5, 1.0]
for i, d in enumerate(SHOW_D):
    y = len(SHOW_D) - 1 - i
    ids, npos = build(*ITEMS[0][:2], 4096, d)
    nlen = len(tok.encode(NEEDLE(*ITEMS[0][:2]), add_special_tokens=False))
    ax.barh(y, len(ids), color="#e5e7eb", height=.55)
    w = min(nlen * 40, len(ids) - npos)      # 見えるよう誇張するが、文脈の外へはみ出させない
    ax.barh(y, w, left=npos, color="#ef4444", height=.55)
    ax.barh(y, 220, left=len(ids) - 220, color="#3b82f6", height=.55)   # 質問も同様に誇張
    ax.text(-120, y, f"深さ {d:.1f}", ha="right", va="center", fontsize=10)
ax.set_xlim(-900, 4400); ax.set_yticks([])
ax.set_xlabel("トークン位置")
ax.set_title("赤 = 答えが書いてある1文／青 = 質問（どちらも見やすさのため幅を誇張）")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
plt.tight_layout(); plt.show()
```

![根拠文を埋める位置。深さ0が先頭、深さ1が質問の直前](/images/needle-design.png)

赤が答えの書いてある1文、青が質問です。赤は実際には9〜11トークンしかないので、図では見えるように幅を誇張しています。**変えるのは赤の位置だけで、全長と質問は同じです。**

## 置く場所を変えると、正解率はどう動くのか

では本題です。深さを0から1まで動かし、同時に各長さの床（根拠文なし）も測ります。

```python
# 深さ（置く場所）と全長を振る。同時に「根拠文なし」の床も測る
LENGTHS = [1024, 4096, 8192]
DEPTHS  = [0.0, 0.25, 0.5, 0.75, 1.0]
acc = np.zeros((len(LENGTHS), len(DEPTHS))); mar = np.zeros_like(acc)
floor_a, floor_m, agree = {}, {}, {}
for i, L in enumerate(LENGTHS):
    floor_a[L], floor_m[L], pk0 = run(model, L, 0.5, drop=True)
    ag = []
    for j, d in enumerate(DEPTHS):
        acc[i, j], mar[i, j], pk = run(model, L, d)
        ag.append(sum(x == y for x, y in zip(pk, pk0)))   # 根拠文なしと同じ答えを選んだ問の数
    agree[L] = ag
    print(f"全長 {L:>5} 正解率 " + " ".join(f"{acc[i,j]:.2f}" for j in range(len(DEPTHS)))
          + f" | 床 {floor_a[L]:.2f}")
    print(f"{'':11}マージン " + " ".join(f"{mar[i,j]:+5.1f}" for j in range(len(DEPTHS)))
          + f" | 床 {floor_m[L]:+.1f}")
    print(f"{'':11}根拠文なしと同じ答えだった問 " + " ".join(f"{a:>2}" for a in ag) + f" / {NITEM}")

plt.figure(figsize=(9, 4.8))
for i, L in enumerate(LENGTHS):
    c = ["#3b82f6", "#10b981", "#ef4444"][i]
    plt.plot(DEPTHS, acc[i], "o-", lw=2, color=c, label=f"全長 {L:,}（根拠文あり）")
    plt.axhline(floor_a[L], ls=":", lw=1.8, color=c)
plt.ylim(0, 1.08); plt.xlabel("根拠文を置いた深さ（0=先頭, 1=質問の直前）")
plt.ylabel("正解率"); plt.title("実線=根拠文あり／点線=根拠文なし（その長さの床）")
plt.legend(fontsize=9); plt.grid(alpha=.3); plt.tight_layout(); plt.show()

plt.figure(figsize=(9, 4.8))
for i, L in enumerate(LENGTHS):
    c = ["#3b82f6", "#10b981", "#ef4444"][i]
    plt.plot(DEPTHS, mar[i], "o-", lw=2, color=c, label=f"全長 {L:,}（根拠文あり）")
    plt.axhline(floor_m[L], ls=":", lw=1.8, color=c)
plt.xlabel("根拠文を置いた深さ（0=先頭, 1=質問の直前）")
plt.ylabel("マージン（正解 − 最良の誤答, 対数尤度）")
plt.title("正解率が天井の4,096でも、マージンは深さで動いている")
plt.legend(fontsize=9); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
```

```
全長  1024 正解率 0.90 1.00 1.00 1.00 1.00 | 床 0.20
           マージン +10.0 +10.1 +10.0  +9.6  +8.2 | 床 -1.1
           根拠文なしと同じ答えだった問  3  2  2  2  2 / 10
全長  4096 正解率 1.00 1.00 1.00 1.00 1.00 | 床 0.20
           マージン  +5.8  +8.3  +9.8  +9.5  +8.2 | 床 -1.3
           根拠文なしと同じ答えだった問  2  2  2  2  2 / 10
全長  8192 正解率 0.40 0.40 0.40 0.40 0.40 | 床 0.40
           マージン  -0.4  -0.5  -0.5  -0.5  -0.4 | 床 -0.5
           根拠文なしと同じ答えだった問 10 10 10 10 10 / 10
```

![実線が根拠文あり、点線が根拠文なしの床](/images/needle-depth-accuracy.png)

まず1,024と4,096を見てください。深さをどこにしても正解率はほぼ満点で、床の0.20をはるかに上回っています。根拠文なしと同じ答えを選んだ問も10問中2〜3問だけです。**置く場所は効いていません。**

問題は8,192です。ここも横一線ですが、**床も0.40で、しかもどの深さでも根拠文なしと10問すべて同じ答えです。** つまりこの長さでは、根拠文があってもなくても、どこに置いても、返ってくる答えが変わりません。

ここが今回いちばん危なかったところです。最初にこの実験を組んだときは対照を取っておらず、読み方はこうでした。「8,192では深さを振っても0.40で横一線。だから位置は効かない」。

**これは間違いです。** 位置が効かないのではなく、**動かしている対象が読まれていない**のです。読まれていないものを動かせば、結果が一定になるのは当たり前です。8,192の横一線は、位置について何も語っていません。

対照を取らず正解率だけを見ていると、この誤りに気づけません。

## 正解率が天井なら、何を見ればいいのか

1,024と4,096の正解率は天井に貼り付いています。この状態では、位置による小さい差があっても見えません。飽和していない指標が要ります。マージンです。

![マージンは、正解率が天井でも動いている](/images/needle-depth-margin.png)

4,096の行を見てください。マージンは +5.8 / +8.3 / +9.8 / +9.5 / +8.2 です。**正解率が全部1.00でも、余裕の量は場所によって違います。** 最も小さいのは先頭（+5.8）で、最も大きいのは中央（+9.8）、末尾はやや下がって +8.2。1,024でも末尾だけ +8.2 に下がります。

形としては**中央が最も高い逆U字**で、Lost in the Middle が言う「中央がへこむ」とは逆向きです。

つまり「置く場所は効かない」は、正解率という粗い物差しでの話でした。細かく見れば深さは効いています。

ただし、ここから「位置が効く」と結論するのは早すぎます。深さを変えると、根拠文の**絶対位置**と、根拠文から質問までの**距離**が同時に動くからです。どちらが効いているのかを分けないと、何も言えません。

## それは位置なのか、距離なのか

分ける方法があります。**距離をそろえたまま、絶対位置だけを動かします。** たとえば距離約1,020トークンは、全長1,024の先頭でも、全長2,048の中央でも、全長4,096の深さ0.75でも作れます。このとき根拠文の絶対位置は13、1,016、3,053と大きく違います。

```python
# 深さを変えると「絶対位置」と「距離」が同時に動く。距離をそろえて絶対位置だけを振る
MATCHED = [("約4,080", [(4096, 0.000)]),
           ("約2,040", [(2048, 0.000), (4096, 0.500)]),
           ("約1,020", [(1024, 0.000), (2048, 0.500), (4096, 0.750)]),
           ("約520",   [(1024, 0.500), (2048, 0.750), (4096, 0.875)])]
pts = []
for name, cfgs in MATCHED:
    print(f"距離 {name} トークン")
    for L, d in cfgs:
        a, g, _ = run(model, L, d)
        ids, npos = build(*ITEMS[0][:2], L, d)
        pts.append((npos, len(ids) - npos, g, name))
        print(f"  全長{L:>5} 深さ{d:<6.3f} 根拠文の位置{npos:>5} 距離{len(ids)-npos:>5}"
              f" → マージン{g:6.2f} 正解率{a:.2f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
cols = {n: c for (n, _), c in zip(MATCHED, ["#ef4444", "#f59e0b", "#10b981", "#3b82f6"])}
for name in cols:
    s = [p for p in pts if p[3] == name]
    a1.plot([p[0] for p in s], [p[2] for p in s], "o-", color=cols[name], lw=2, ms=9,
            label=f"距離 {name}")
    a2.plot([p[1] for p in s], [p[2] for p in s], "o", color=cols[name], ms=9)
a1.set_xlabel("根拠文の絶対位置（トークン）"); a1.set_ylabel("マージン")
a1.set_title("横に動かしても平ら＝絶対位置は効いていない")
a1.legend(fontsize=9); a1.grid(alpha=.3); a1.set_ylim(4, 12)
a2.set_xlabel("根拠文から質問までの距離（トークン）"); a2.set_ylabel("マージン")
a2.set_title("距離を変えると落ちる＝効いているのはこちら")
a2.grid(alpha=.3); a2.set_ylim(4, 12)
plt.tight_layout(); plt.show()
```

```
距離 約4,080 トークン
  全長 4096 深さ0.000  根拠文の位置   13 距離 4083 → マージン  5.78 正解率1.00
距離 約2,040 トークン
  全長 2048 深さ0.000  根拠文の位置   13 距離 2035 → マージン 10.28 正解率1.00
  全長 4096 深さ0.500  根拠文の位置 2040 距離 2056 → マージン  9.80 正解率1.00
距離 約1,020 トークン
  全長 1024 深さ0.000  根拠文の位置   13 距離 1011 → マージン 10.01 正解率0.90
  全長 2048 深さ0.500  根拠文の位置 1016 距離 1032 → マージン  9.84 正解率1.00
  全長 4096 深さ0.750  根拠文の位置 3053 距離 1043 → マージン  9.52 正解率1.00
距離 約520 トークン
  全長 1024 深さ0.500  根拠文の位置  504 距離  520 → マージン  9.98 正解率1.00
  全長 2048 深さ0.750  根拠文の位置 1517 距離  531 → マージン 10.07 正解率1.00
  全長 4096 深さ0.875  根拠文の位置 3560 距離  536 → マージン  9.61 正解率1.00
```

![左は絶対位置を動かした場合、右は距離を動かした場合](/images/needle-distance-vs-position.png)

左のパネルがはっきりしています。**距離をそろえたまま絶対位置を13から3,560まで動かしても、マージンは9.52〜10.28に収まります。** 絶対位置が3,500トークン動いても、マージンは1もずれません。

**根拠文の絶対位置は効いていません。**

これは前回の記事で見た性質そのものです。RoPEは $R(m)^\top R(n) = R(n-m)$ を満たすので、注意スコアに残るのは $n-m$、つまり相対位置だけです。絶対位置は打ち消し合って消えます。前回はこれを乱数ベクトルの内積で確かめました（距離が同じなら位置5でも7,000でも小数10桁まで一致する）。**今回は同じことが、学習済みモデルのタスクの成績として出ています。** 前回は幾何の性質、今回は振る舞いです。

一方、右のパネル（距離を変えた場合）は、同じ強さでは読めません。マージンが明確に落ちるのは距離約4,080の1点だけで、520・1,020・2,040のあいだにはほとんど差がありません。しかもこの設計では、**距離を伸ばすと根拠文と質問のあいだに挟まる詰め物の量も増えます。** 「距離が遠いから」なのか「あいだに文章が多いから」なのかを、この対比だけでは分けられません。

言えるのは、**絶対位置は効いていない**ということと、**距離（あるいは介在量）が学習長に近づくと弱る**ということまでです。

## では、何が崖を作っているのか

根拠文を必ず真ん中に固定して、全長だけを動かします。床も同じ長さで測ります。

```python
# 根拠文は必ず真ん中。全長だけを変える。同じ長さで「根拠文なし」も測る
SCAN = [2048, 3072, 3968, 4096, 4224, 4608, 5120, 6144, 8192]
s_acc, s_floor, s_same = [], [], []
for L in SCAN:
    a, g, pk = run(model, L, 0.5)
    a0, g0, pk0 = run(model, L, 0.5, drop=True)
    s_acc.append(a); s_floor.append(a0); s_same.append(sum(x == y for x, y in zip(pk, pk0)))
    print(f"全長 {L:>5}: 根拠文あり{a:.2f} 根拠文なし{a0:.2f} "
          f"マージン{g:+6.2f}/{g0:+6.2f} 選択の一致{s_same[-1]:>2}/{NITEM}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
a1.plot(SCAN, s_acc, "o-", color="#3b82f6", lw=2, label="根拠文あり")
a1.plot(SCAN, s_floor, "s--", color="#8a8a8a", lw=1.8, label="根拠文なし（床）")
a1.axvline(TRAIN, color="#ef4444", ls="--", lw=1.5)
a1.text(TRAIN * 1.03, 0.75, f"学習長 {TRAIN:,}", color="#ef4444")
a1.set_ylim(0, 1.08); a1.set_xlabel("全長（トークン）"); a1.set_ylabel("正解率")
a1.set_title("根拠文を入れた効果が、どこで消えるか"); a1.legend(); a1.grid(alpha=.3)
a2.plot(SCAN, [s / NITEM for s in s_same], "o-", color="#8b5cf6", lw=2)
a2.axvline(TRAIN, color="#ef4444", ls="--", lw=1.5)
a2.axhline(1.0, ls=":", c="#8a8a8a", lw=1.4)
a2.text(2100, 1.02, "全問で出力が同じ＝文脈を読んでいない", color="#8a8a8a", fontsize=9)
a2.set_ylim(0, 1.12); a2.set_xlabel("全長（トークン）")
a2.set_ylabel("根拠文の有無で選択が一致した割合")
a2.set_title("学習長を超えると、根拠文があってもなくても同じ答えになる")
a2.grid(alpha=.3)
plt.tight_layout(); plt.show()

# 崖のふもとだけ拡大する
plt.figure(figsize=(9, 4.4))
zoom = [(L, a, f) for L, a, f in zip(SCAN, s_acc, s_floor) if 3500 <= L <= 5300]
plt.plot([z[0] for z in zoom], [z[1] for z in zoom], "o-", color="#ef4444", lw=2.2, ms=8,
         label="根拠文あり")
plt.plot([z[0] for z in zoom], [z[2] for z in zoom], "s--", color="#8a8a8a", lw=1.8,
         label="根拠文なし（床）")
for L, a, f in zoom:
    plt.annotate(f"{L:,}\n{a:.2f}", (L, a), textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=9)
plt.axvline(TRAIN, color="#8a8a8a", ls="--", lw=1.5)
plt.text(TRAIN * 1.005, 0.05, f"学習長 {TRAIN:,}", color="#8a8a8a", fontsize=9)
plt.ylim(0, 1.18); plt.xlabel("全長（トークン）"); plt.ylabel("正解率")
plt.title("崖は学習長のすぐ外側に立っている")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
```

```
全長  2048: 根拠文あり1.00 根拠文なし0.20 マージン +9.84/ -0.91 選択の一致 2/10
全長  3072: 根拠文あり1.00 根拠文なし0.20 マージン +9.19/ -1.39 選択の一致 2/10
全長  3968: 根拠文あり1.00 根拠文なし0.20 マージン +9.76/ -1.19 選択の一致 2/10
全長  4096: 根拠文あり1.00 根拠文なし0.20 マージン +9.80/ -1.29 選択の一致 2/10
全長  4224: 根拠文あり0.70 根拠文なし0.60 マージン +2.79/ -0.52 選択の一致 9/10
全長  4608: 根拠文あり0.20 根拠文なし0.20 マージン -1.80/ -1.82 選択の一致10/10
全長  5120: 根拠文あり0.30 根拠文なし0.30 マージン -1.70/ -1.70 選択の一致10/10
全長  6144: 根拠文あり0.50 根拠文なし0.50 マージン -0.39/ -0.39 選択の一致10/10
全長  8192: 根拠文あり0.40 根拠文なし0.40 マージン -0.46/ -0.46 選択の一致10/10
```

![根拠文を入れた効果が、どこで消えるか](/images/needle-length-cliff.png)

左の図で、実線（根拠文あり）と点線（床）の**あいだの隙間**を見てください。これが「根拠文を入れた効果」です。4,096までは1.00対0.20で大きく開いていますが、4,608以降は完全に閉じます。

右の図は、根拠文の有無で選択が一致した割合です。4,096までは2/10、つまりほとんどの問題で答えが変わります。4,608以降は10/10、**すべての問題で答えが変わりません。**

崖の正体はこれでした。正解率が下がるのではなく、**文脈を読むこと自体が止まります。**

落ち際だけ拡大します。

![崖は学習長のすぐ外側に立っている](/images/needle-cliff-zoom.png)

4,096で1.00、128トークン外側の4,224で0.70、512トークン外側の4,608で0.20（床と同じ）。**階段ではなく、数百トークンかけて崩れます。**

前回の記事の注意の測定と並べると、どちらも「境界で階段状に落ちるのではなく、数百トークンかけて崩れる」形をしています。前回、あるひとつの層で測った「直近64トークンへの重み」は、位置4,096で0.4107、4,224で0.3302、4,352で0.1259、5,120で0.0051でした。**4,224ではまだ8割方残っていて、実質ゼロになるのは5,120です。** 今回の正解率も4,224で0.70とまだ持ちこたえ、4,608で床に着きます。

ただし2つは別のモデル設定・別のタスクの測定なので、崩れきるまでの幅までは一致しません（注意は約1,000トークンかけて落ち、今回の正解率は約500トークンで床に着きます）。前回は詰め物を bfloat16 のまま読み、改行の扱いも違うので、そもそも同じトークン列を見ていません。**「同じ形をしている」以上のことは言えません。**

一方、前回の測定で境界の直後にすでに跳ねている量がありました。**距離4,096を超えたキーへ配る重みで、位置4,224で49.5%、4,352で73.6%、5,120で97.2%です。** 学習で見たことのない距離が現れた最初の測定点で、注意の半分がもうそちらへ移っています。

:::message
この記事のコードは1条件あたり10問です（`NITEM = 10`）。正解率の刻みが0.1になるので、崩れたあとの0.20と0.50のような差は読めません（0.40の95%信頼区間はおおよそ0.12〜0.74です）。手元で1条件40問に増やして測り直したところ、4,096で1.00、4,224で0.84、4,608で0.24でした。**崖の位置と、崖の上下（1.00と床）は変わりません。** 細かい上下は問題数が少ないことによる揺れなので、読まないでください。
:::

## 質問のすぐ手前に置いても駄目なのか

ここで、まだ2つの説明が残っています。

1. 根拠文と質問の**距離**が離れるから読めない
2. 文脈のどこかに**学習で見たことのない距離**が現れるから壊れる

分けるには、根拠文を質問の直前に固定したまま、全長だけを変えます。こうすると根拠文までの距離は一定のまま、文脈に含まれる最大の距離だけが動きます。

```python
# 根拠文を質問の直前に固定したまま、全長だけを変える
NEAR = [2048, 4096, 4608, 6144, 8192]
n_acc, n_floor = [], []
for L in NEAR:
    a, _, pk = run(model, L, 1.0)
    a0, _, pk0 = run(model, L, 1.0, drop=True)
    n_acc.append(a); n_floor.append(a0)
    ids, npos = build(*ITEMS[0][:2], L, 1.0)
    print(f"全長 {L:>5}: 根拠文の位置{npos:>5} 末尾まで{len(ids)-npos:>3}トークン "
          f"→ 正解率{a:.2f}（根拠文なし{a0:.2f}）選択の一致{sum(x==y for x,y in zip(pk,pk0)):>2}/{NITEM}")

plt.figure(figsize=(9, 4.6))
plt.plot(NEAR, n_acc, "o-", color="#8b5cf6", lw=2, label="根拠文あり")
plt.plot(NEAR, n_floor, "s--", color="#8a8a8a", lw=1.8, label="根拠文なし（床）")
plt.axvline(TRAIN, color="#ef4444", ls="--", lw=1.5)
plt.text(TRAIN * 1.03, 0.75, f"学習長 {TRAIN:,}", color="#ef4444")
plt.ylim(0, 1.08); plt.xlabel("全長（トークン）"); plt.ylabel("正解率")
plt.title("根拠文はどれも質問の直前にある。それでも全長で壊れる")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
```

```
全長  2048: 根拠文の位置 2019 末尾まで 29トークン → 正解率1.00（根拠文なし0.20）選択の一致 2/10
全長  4096: 根拠文の位置 4067 末尾まで 29トークン → 正解率1.00（根拠文なし0.20）選択の一致 2/10
全長  4608: 根拠文の位置 4579 末尾まで 29トークン → 正解率0.20（根拠文なし0.20）選択の一致 8/10
全長  6144: 根拠文の位置 6115 末尾まで 29トークン → 正解率0.50（根拠文なし0.50）選択の一致10/10
全長  8192: 根拠文の位置 8163 末尾まで 29トークン → 正解率0.40（根拠文なし0.40）選択の一致10/10
```

![根拠文はどれも質問の直前。それでも全長で壊れる](/images/needle-adjacent.png)

根拠文はどれも末尾から29トークンの位置にあります（根拠文9〜11トークンと質問18〜20トークンぶんで、両者は隣接しています）。それでも2,048と4,096では満点、4,608では床の0.20です。

**根拠文までの距離ではありません。** 答えがすぐ隣に書いてあっても壊れます。

ただし、この実験だけでは2番目の説明を確定できません。質問は常に末尾にあるので、「全長が4,096を超える」ことと「距離4,096超のキーが文脈に現れる」ことは、この設計では同じ意味になります。さらに「単に詰め物が多いから薄まった」という説明もまだ残っています。

分けるには、**文脈をまったく変えずに、位置の与え方だけを変える**必要があります。次の節がそれです。

## 位置の与え方だけを変えたら、読めるようになるのか

決着をつける実験があります。**トークン列を1つも変えず、位置インデックスの与え方だけを変えます。** 文脈の中身も長さも同じなので、「詰め物が多すぎて薄まった」という説明は使えません。

前回みた2つの手当てを使います。位置の目盛りを詰める [PI](https://arxiv.org/abs/2306.15595)（`rope_type: "linear"`）と、回転の基準となる base を広げる NTK-aware（`rope_type: "dynamic"`）です。前回は perplexity で効果を見ました。今回は**正しい1文を選べるようになるのか**を見ます。

```python
# トークン列は一切変えず、位置インデックスの与え方だけを変える
CONDS = {"そのまま": None,
         "PI（linear）":   {"rope_theta": BASE, "rope_type": "linear",  "factor": 2.0},
         "NTK（dynamic）": {"rope_theta": BASE, "rope_type": "dynamic", "factor": 2.0}}
FIX_D, res = [0.0, 0.25, 0.5, 0.75, 1.0], {}
models = {}
for name, sc in CONDS.items():
    if sc is None:
        m = model
    else:
        cfg = copy.deepcopy(model.config)      # 生きているmodelのconfigを書き換えない
        cfg.rope_scaling = dict(sc)
        m = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg).float(); m.eval()
    models[name] = m
    res[name] = [run(m, 8192, d)[:2] for d in FIX_D]
    a0 = run(m, 8192, 0.5, drop=True)[0]
    hit = int(round(sum(a for a, _ in res[name]) * NITEM))
    print(f"{name:14s} " + " ".join(f"深さ{d:.2f}→{a:.2f}" for d, (a, _) in zip(FIX_D, res[name]))
          + f" | 合計{hit:>2}/{len(FIX_D)*NITEM} | 床 {int(round(a0*NITEM))}/{NITEM}")

plt.figure(figsize=(9, 4.8))
for i, (name, v) in enumerate(res.items()):
    plt.plot(FIX_D, [a for a, _ in v], "o-", lw=2,
             color=["#8a8a8a", "#3b82f6", "#10b981"][i], label=name)
plt.ylim(0, 1.08); plt.xlabel("根拠文を置いた深さ（0=先頭, 1=質問の直前）")
plt.ylabel("正解率"); plt.title(f"全長 {8192:,} トークン。トークン列は同じで、位置の与え方だけが違う")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
```

```
そのまま           深さ0.00→0.40 深さ0.25→0.40 深さ0.50→0.40 深さ0.75→0.40 深さ1.00→0.40 | 合計20/50 | 床 4/10
PI（linear）     深さ0.00→0.60 深さ0.25→0.50 深さ0.50→0.50 深さ0.75→0.60 深さ1.00→0.90 | 合計31/50 | 床 5/10
NTK（dynamic）   深さ0.00→0.20 深さ0.25→0.90 深さ0.50→0.90 深さ0.75→1.00 深さ1.00→1.00 | 合計40/50 | 床 2/10
```

![トークン列は同じで、位置の与え方だけが違う](/images/needle-rope-scaling.png)

そのままは20/50、床は4/10。**まったく同じ水準で、読めていません。**

NTKは40/50に対し床2/10。Fisherの正確検定で $p = 0.0005$ です。**中身を1トークンも変えず、位置の与え方だけで読めるようになりました。** 重みも1バイト変えていません。文脈の中身も長さも同じなので、「詰め物が多すぎて薄まった」という説明はここでは使えません。**壊していたのは位置の与え方のほうでした。**

前回の記事は、この可能性を予想として書いていました。

> 内積に効くのは距離 $m-n$ だけなら、壊れる原因も距離の側にあるはずです。位置8,000そのものが特別なのではなく、そこで初めて現れる距離が特別なのではないか。

トークン列を変えずに位置の与え方だけで復活したことは、この予想と整合します。

一方、**PIは31/50に対し床5/10で $p = 0.50$** です。数字の上では床より高く見えますが、**この問題数では床と区別できません。** 前回 perplexity で見た「NTKのほうが良い」という順序とは矛盾しませんが、今回の標本で「PIも効いた」とは言えません。

もう1つ、**NTKは深さ0.00でだけ0.20**、つまり自分の床ちょうどになります。他の深さでは0.90〜1.00なのに、根拠文を先頭に置いたときだけ何も読めていません。

なお床は条件ごとに違います（そのまま0.40、PI 0.50、NTK 0.20）。**条件をまたいで正解率を直接比べることはできません。** それぞれ自分の床と比べる必要があります。

## モデルの中では、どこに注意が行っているのか

注意の重みを直接見ます。ここで基準の取り方に落とし穴がありました。

最初に使った基準は「均等にばらまいた場合の何倍か」でした。8,192トークンの文脈に9トークンの根拠文があるなら、均等配分なら 9/8,192 ≒ 0.0011。実際の重みがこれの何倍か、という見方です。

**これは基準になりません。** softmaxの重みは総和が1なので、必ず一部のキーが平均を下回ります。しかも学習済みモデルの注意は先頭と直近に偏るので、**文脈の中盤にある任意のスパンは、正常な状態でも均等を下回るのが普通です。** 下の出力で確かめられるとおり、**全問正解する4,096でも12層中9層が均等を下回ります。** この基準では「全問正解しながら目を逸らしている」ことになってしまいます。

正しい帰無は「同じ長さ・似た場所にある、ただの詰め物」です。根拠文の少し手前に同じ長さの対照スパンを取り、そちらへの重みと比べます。

```python
# 注意の重みを見る。基準は「均等」ではなく、同じ場所にある普通の詰め物
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
H  = model.config.num_attention_heads
Dh = model.config.hidden_size // H
NL = model.config.num_hidden_layers
assert model.config.num_key_value_heads == H, "GQAのモデルではこのview方法は使えない"

def attention_spans(m, ids, npos, nlen):
    """最後のクエリ位置が配る重みを層ごとに。(根拠文, 同じ長さの対照スパン, 先頭16, 距離4096超)"""
    cap, hooks = {}, []
    for i in range(NL):
        hooks.append(m.model.layers[i].self_attn.register_forward_pre_hook(
            (lambda i: lambda mod, a, kw: cap.__setitem__(
                i, kw.get("hidden_states", a[0] if a else None)))(i), with_kwargs=True))
    m(torch.tensor([ids]))
    for h in hooks: h.remove()
    out = []
    ctrl = max(0, npos - 3 * nlen)            # 根拠文と同じ長さの、ただの詰め物
    for i in range(NL):
        hs, at = cap[i], m.model.layers[i].self_attn
        q = at.q_proj(hs).view(1, -1, H, Dh).transpose(1, 2)
        k = at.k_proj(hs).view(1, -1, H, Dh).transpose(1, 2)
        cos, sin = m.model.rotary_emb(hs, torch.arange(hs.shape[1]).unsqueeze(0))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        p = len(ids) - 1
        a = torch.softmax((q[0].float()[:, p:p+1] * k[0].float()[:, :p+1]).sum(-1) / math.sqrt(Dh), -1)
        far = max(0, p - TRAIN)               # 距離4,096を超えたキーの範囲
        out.append((float(a[:, npos:npos+nlen].sum(-1).mean()),
                    float(a[:, ctrl:ctrl+nlen].sum(-1).mean()),
                    float(a[:, :16].sum(-1).mean()),
                    float(a[:, :far].sum(-1).mean()) if far else 0.0))
    return np.array(out)                      # (層, 4)

ATT_L, att = [1024, 4096, 8192], {}
for L in ATT_L:
    vs = []
    for country, gold, opts in ITEMS[:5]:
        ids, npos = build(country, gold, L, 0.5)
        nlen = len(tok.encode(NEEDLE(country, gold), add_special_tokens=False))
        vs.append(attention_spans(model, ids, npos, nlen))
    v = np.mean(vs, 0); att[L] = v
    ratio = v[:, 0] / np.maximum(v[:, 1], 1e-9)     # 層ごとの「根拠文 ÷ 対照スパン」
    j = int(ratio.argmax())                          # 比が最大の層を代表にする
    uni = np.mean([len(tok.encode(NEEDLE(c, g), add_special_tokens=False)) for c, g, _ in ITEMS[:5]]) / L
    below = int((v[:, 0] < uni).sum())          # 根拠文への重みが「均等配分」を下回った層の数
    print(f"全長 {L:>5}: 比が最大の層={j:>2} 根拠文{v[j,0]:.2e} 対照スパン{v[j,1]:.2e} "
          f"→ 比 {ratio[j]:7.1f}倍 | 距離{TRAIN:,}超へ {v[:,3].max():.3f} "
          f"| 均等({uni:.2e})を下回る層 {below}/{NL}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
for i, L in enumerate(ATT_L):
    c = ["#3b82f6", "#10b981", "#ef4444"][i]
    a1.plot(range(NL), att[L][:, 0] / np.maximum(att[L][:, 1], 1e-9), "o-", lw=2, color=c,
            label=f"全長 {L:,}")
    a2.plot(range(NL), att[L][:, 3], "o-", lw=2, color=c, label=f"全長 {L:,}")
a1.axhline(1, ls="--", c="#8a8a8a", lw=1.2)
a1.text(0.1, 1.3, "対照スパンと同じ＝根拠文を狙っていない", color="#8a8a8a", fontsize=9)
a1.set_yscale("log"); a1.set_xlabel("層"); a1.set_ylabel("根拠文への重み ÷ 対照スパンへの重み")
a1.set_title("根拠文を、ただの詰め物より狙えているか"); a1.legend(fontsize=9); a1.grid(alpha=.3)
a2.set_xlabel("層"); a2.set_ylabel(f"距離{TRAIN:,}超のキーへ配った重み")
a2.set_title("学習で見たことのない距離へ、どれだけ吸われているか")
a2.legend(fontsize=9); a2.grid(alpha=.3)
plt.tight_layout(); plt.show()

# NTKが深さ0.00でだけ落ちる件
print()
for name in ("そのまま", "NTK（dynamic）"):
    for d in (0.0, 0.5):
        vs = []
        for country, gold, opts in ITEMS[:5]:
            ids, npos = build(country, gold, 8192, d)
            nlen = len(tok.encode(NEEDLE(country, gold), add_special_tokens=False))
            vs.append(attention_spans(models[name], ids, npos, nlen))
        v = np.mean(vs, 0)                                  # 上と同じ順序で平均してから比を取る
        r = (v[:, 0] / np.maximum(v[:, 1], 1e-9)).max()
        print(f"{name:14s} 深さ{d:.1f}: 根拠文÷対照 {r:7.1f}倍 / "
              f"先頭16トークンへの重み {v[:, 2].max():.3f}")
```

```
全長  1024: 比が最大の層= 5 根拠文1.06e-01 対照スパン2.93e-04 → 比   362.4倍 | 距離4,096超へ 0.000 | 均等(8.98e-03)を下回る層 4/12
全長  4096: 比が最大の層= 6 根拠文8.53e-02 対照スパン1.53e-04 → 比   558.0倍 | 距離4,096超へ 0.000 | 均等(2.25e-03)を下回る層 9/12
全長  8192: 比が最大の層= 8 根拠文1.74e-05 対照スパン2.19e-06 → 比     7.9倍 | 距離4,096超へ 1.000 | 均等(1.12e-03)を下回る層 12/12

そのまま           深さ0.0: 根拠文÷対照   208.7倍 / 先頭16トークンへの重み 0.009
そのまま           深さ0.5: 根拠文÷対照     7.9倍 / 先頭16トークンへの重み 0.020
NTK（dynamic）   深さ0.0: 根拠文÷対照     1.0倍 / 先頭16トークンへの重み 0.394
NTK（dynamic）   深さ0.5: 根拠文÷対照   527.7倍 / 先頭16トークンへの重み 0.394
```

![根拠文を、ただの詰め物より狙えているか](/images/needle-attention.png)

左の図が対照スパンとの比です。1,024で362倍、4,096で558倍。**モデルはその1文をはっきり狙っています。** 8,192では7.9倍まで落ちます。

右の図が、距離4,096を超えたキーへ配った重みです。8,192では**1.000**、つまり比が最大の層では注意のほぼ全部が、学習で一度も見ていない距離へ行っています。1,024と4,096で0.000なのは当然で、そもそもその距離のキーが存在しません。

絶対値も見てください。8,192で根拠文が受け取る重みは 1.74e-05 で、1,024の 1.06e-01 の**約6,000分の1**です。対照スパンとの比が7.9倍残っていても、量そのものが無視できる水準まで落ちています。

ここで、この指標の限界をはっきり書いておきます。**注意の集中は、振る舞いを予測しませんでした。**

下段の出力を見てください。8,192で根拠文を先頭（深さ0.0）に置くと、対照スパンとの比は208.7倍あります。中央に置いたとき（7.9倍）よりずっと高く、よく狙えているように見えます。**それでも、そのどちらも根拠文を消したときと答えが10問すべて同じでした。** 上の深さ走査の出力にあるとおり、8,192では全ての深さで一致数が10/10です。集中していることと、その情報が使われることは別でした。

NTKについても同じことが言えます。NTKは深さ0.00で1.0倍（読めない）、深さ0.50で527.7倍（読める）と、比のほうは結果と揃っています。ところが**先頭16トークンへの重みは、失敗する深さ0.00でも成功する深さ0.50でも同じ0.394**です。**同じ値のものは、片方だけ失敗する理由になりません。** 先頭への集中でNTKの失敗を説明することはできませんでした。

これらは、最後のクエリ位置1点・ヘッド平均という粗い測り方の限界でもあります。根拠文の情報がいったん隣のトークンへ写り、後段で読み出される経路は、この測り方では見えません。**機構の代理指標より、根拠文を消す対照のほうが信頼できました。**

## Lost in the Middle はどこへ行ったのか

冒頭で引いた「中央に置くと落ちる」というU字は、この測定では出ませんでした。ただし「出なかった」を正しく言うには、どこを見られてどこを見られなかったかを分ける必要があります。

**学習長の内側**では、正解率は天井に貼り付いていて位置差を検出できません。ただしマージンは飽和していないので、そちらでは位置差が見えます。そして見えた差は、上の対比のとおり**絶対位置ではなく距離**で説明できました。中央がへこむ形にはなっていません。

**学習長の外側**では、そもそも文脈が読まれていないので、位置を論じられません。

残るのは、読めていて、かつ飽和していない唯一の領域 — 位置エンコーディングを伸ばした条件です。ここは正解率が0.2から1.0のあいだに散らばるので、位置の形が見えます。

上の表をもう一度見てください。NTKは深さ0.00で0.20、そこから0.90 / 0.90 / 1.00 / 1.00 と、**質問に近いほど単調に上がります**。PIも0.60 / 0.50 / 0.50 / 0.60 / 0.90 で、最も高いのは末尾です。**どちらも「中央がへこむ」形ではありません。** 端が高く中央が低いU字であれば Lost in the Middle ですが、出ているのは末尾ほど有利という傾き、つまり距離の効果です。

手元で問題数を40に増やしても同じ形でした。端（深さ0と0.125）と中央（深さ0.375〜0.625）の差は、PIの8,192で $+0.06 \pm 0.06$、6,144では符号が逆で、**U字と呼べる差はありません。**

つまりこの設定では、どの領域を見ても中央のへこみは出ませんでした。理由の候補は3つあります。

1. **モデルの規模とタスクが違う。** 元論文はGPT-3.5クラスに10〜30個の文書を並べて順番を入れ替える設計です。こちらは152Mのモデルに1文だけ埋めています
2. **根拠文と質問の字面が重なっている。** 根拠文は「◯◯国の首都は△△である」、質問は「◯◯国の首都はどこか」で、「◯◯国の首都は」がそのまま一致します。[NoLiMa](https://arxiv.org/abs/2502.05167) は、この一致を意図的に外すと難度が跳ね上がることを報告していて、128Kを謳う13モデルのうち11モデルが32Kで短文脈時の半分未満に落ち、GPT-4o も99.3%から69.7%へ下がっています。つまりこの記事の測定は**易しい側**の設定です
3. **中央のへこみは、指示追従の失敗として出る現象かもしれない。** 元論文のタスクは「文書群を読んで質問に答えよ」という指示への追従を含みますが、こちらは対数尤度の比較なので、その経路が最初から無い

**「Lost in the Middle は起きない」と読まないでください。** ここで言えるのは「この設定では出なかった」だけです。位置の効果を見たいなら、正解率が天井にも床にも張り付かない難度に調整したうえで測る必要があります。

## 実務ではどうすればいいのか

測定から言えることだけを書きます。

**1. まず、文脈が読まれているかを確かめる。** 根拠文を同じ長さの無関係な文に差し替えて、答えが変わるかを見ます。変わらなければ、そのモデルはその長さで文脈を使っていません。**この対照を取らずに正解率だけを見ると、「たまたま当たった」と「読んで当てた」の区別がつきません。** この記事も、対照を取るまで「位置は効かない」という誤った結論を書いていました。

**2. 正解率の床は、選択肢の数から計算しない。** 4択だから0.25、ではありませんでした。実測の床は条件によって0.20〜0.60と動きます。床は計算するものではなく、根拠文なしで測るものです。

**3. マージンの符号を診断に使わない。** マージンは「正解 − 3つの誤答の最大値」なので、4つの候補が交換可能なら、情報がゼロでも期待値は必ず負になります。実際、全長4,608では根拠文ありが−1.80、根拠文なしが−1.82で、**ほとんど同じ**でした。符号は壊れの指標になりません。

**4. 読めている範囲では、根拠文は質問に近いほうが有利。** 位置エンコーディングを伸ばして読めるようにした条件では、深さを先頭から末尾へ動かすと正解率が単調に上がりました（NTKで0.20 → 1.00）。ただしこれは中央がへこむ形ではなく、質問に近いほど良いという傾きです。**タイトルの問いに対する答えを1つ選ぶなら、「質問の近く」です。** ただしその効きは、全長が学習長を超えているかどうかに比べれば小さいものでした。

**5. 公称の窓の長さを信用しない。** configに書いてある値は「入る長さ」であって「使える長さ」ではありません。[RULER](https://arxiv.org/abs/2404.06654) は、32K以上を主張するモデルのうち32Kで満足に動くのは半数だけだと報告しています。同じ表で GPT-4 は公称128Kに対し実効64K、Llama3.1 (70B) も公称128Kに対し実効64Kです。自分の使うモデルの実効長は、この記事の対照つきの測り方で決められます。

**6. 窓を広げたいなら、まず base 拡大（NTK-aware）を試す。** この測定で自分の床を明確に超えたのはNTKだけでした（40/50 対 床2/10、Fisherの正確検定で $p = 0.0005$）。PIは31/50に対し床5/10で $p = 0.50$、**この問題数では床と区別できません。** 手当てなしは20/50対4/10で $p = 1.00$ です。

ただしNTKにも穴があります。**根拠文を先頭に置いた条件だけは自分の床ちょうど**で、そこでは何も読めていません。「大事なことは先頭に置く」という、まさに冒頭で読者が想定した使い方です。

なお条件ごとに床が違う（手当てなし0.40、PI 0.50、NTK 0.20）ので、**条件をまたいで正解率を直接比べることはできません。** それぞれ自分の床と比べる必要があります。

## この測定から言えないこと

- **1つのモデルの1つのタスクの結果です。** 152Mの日本語モデルで、架空の固有名詞を選ばせる4択です
- **位置の効果が無いことは示せていません。** 学習長の内側では正解率が天井、外側では文脈が読まれていないので、位置差を検出できる領域が限られています。マージンで見たかぎり絶対位置には依存しませんでしたが、これは1つの指標での結果です
- **崖の位置が特定の文に依存していないかは未検証です。** 詰め物は『吾輩は猫である』の同じ箇所だけを使っています
- **対照条件の差し替え文は、10問すべて同じ文字列です。** 床の条件では国名と選択肢以外の文脈が全問で共通になるので、床が長さごとに振れる一因になっている可能性があります
- **1条件10問しかありません。** 正解率0.40の95%信頼区間はおおよそ0.12〜0.74です。**0.20と0.50の差は、この問題数では読めません。** 崖の有無のような大きい差は読めますが、細かい上下は読まないでください。本文で床との比較に検定を付けたのは位置エンコーディングの節だけで、他の比較には付けていません
- **NTKが先頭で落ちる理由は特定できていません。** NTKが先頭16トークンに注意の39.4%を集めていることは測れましたが、**その値は失敗する深さ0.0でも成功する深さ0.5でも同じ**なので、なぜ先頭だけ落ちるかの説明にはなりません
- **注意の測定は最後のクエリ位置1点・ヘッド平均・12層のみです。** 根拠文の情報がいったん隣のトークンへ写り、後段で読み出される経路は、この測り方では見えません
- 追加学習は一切していません。[PI](https://arxiv.org/abs/2306.15595) も [YaRN](https://arxiv.org/abs/2309.00071) も、本来は短い追加学習とセットで使う手法です

## まとめ

- 長い文脈に根拠文を1つ埋めて選ばせると、**学習長の内側では、置く場所を先頭から末尾まで動かしても正解率は変わらなかった**（どこでもほぼ満点）
- 正解率は天井なので、飽和していないマージンで見た。マージンには差があったが、**距離をそろえて絶対位置だけを動かすと平らになった**。効いているのは距離のほうだった
- これは前回みた RoPE の性質 $R(m)^\top R(n) = R(n-m)$ の、学習済みモデルの振る舞いによる裏づけになる
- **全長4,608以降では、根拠文を同じ長さの無関係な文に差し替えても答えが1問も変わらなくなった。** その領域では文脈が読まれていないので、「どこに置くか」を論じる意味がない（境界のすぐ外側の4,224は10問中9問が一致で、途中段階にある）
- 崖は学習長のすぐ外側にある。ただし階段ではなく、128トークン外側ではまだ持ちこたえ、512トークン外側で崩れきる
- 根拠文を**質問の直前**に置いても、全長が学習長を超えていれば壊れた。効いているのは根拠文の場所ではなく、**文脈のどこかに距離4,096超のキーが現れること**
- `rope_type` を `dynamic`（NTK-aware）にすると、追加学習なしで読めるようになった。**自分の床を明確に超えたのはNTKだけ**で、PIはこの問題数では床と区別できなかった
- ただしNTKも、根拠文を**先頭**に置いた条件だけは床ちょうどで、そこでは読めていない
- 読めている条件では、**根拠文が質問に近いほど有利**という単調な傾向が出た。中央がへこむ形（Lost in the Middle）ではない
- 条件ごとに床が違うので（手当てなし0.40、PI 0.50、NTK 0.20）、条件をまたいで正解率を直接比べることはできない

## 参考

1. [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — 位置によって長文脈の使われ方が変わることを示した論文。[TACL版](https://aclanthology.org/2024.tacl-1.9/) が正式引用
2. [nelson-liu/lost-in-the-middle](https://github.com/nelson-liu/lost-in-the-middle) — 上の公式実装とデータ
3. [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — Needle in a Haystack のオリジナル実装
4. [Long context prompting for Claude 2.1](https://claude.com/blog/claude-2-1-prompting) — needleテストの低スコアがモデルの検索能力ではなかった件。応答の冒頭に1文を置くだけで27%→98%
5. [Multi Needle in a Haystack](https://www.langchain.com/blog/multi-needle-in-a-haystack) — 複数needleへの拡張。末尾側を優先して先頭側を落とす挙動の報告
6. [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) — 公称の窓と実効長の乖離
7. [NVIDIA/RULER](https://github.com/NVIDIA/RULER) — 実効長テーブル。この記事で引いたのはこちらの現行版
8. [NoLiMa: Long-Context Evaluation Beyond Literal Matching](https://arxiv.org/abs/2502.05167) — 質問と根拠文の字面一致を外した評価
9. [Same Task, More Tokens](https://arxiv.org/abs/2402.14848) — 同じタスクのまま入力長だけ伸ばすと推論性能が落ちる
10. [LongBench](https://arxiv.org/abs/2308.14508) / [LongBench v2](https://arxiv.org/abs/2412.15204) — 長文脈のベンチマーク
11. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE の原典。相対位置が残ることの導出
12. [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — この記事の `rope_type: "linear"`
13. [kaiokendev, Extending Context to 8K](https://kaiokendev.github.io/til) — 線形補間のコミュニティ側の発祥。HFの実装がクレジットしている
14. [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — NTK-aware の系譜をまとめた論文
15. [jquesnelle/yarn](https://github.com/jquesnelle/yarn) — YaRN の公式実装
16. [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — 非一様な補間で2Mトークンまで
17. [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) — attention sink と StreamingLLM
18. [Why do LLMs attend to the first token?](https://arxiv.org/abs/2504.02732) — 先頭への集中がなぜ起きるのか
19. [Utilities for Rotary Embedding](https://huggingface.co/docs/transformers/internal/rope_utils) — `rope_type` に指定できる値の一覧。transformers v5 のドキュメント
20. [modeling_rope_utils.py](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — `linear` / `dynamic` がどう計算されているか
21. [llm-jp/llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m) — この記事で使ったモデル
22. [吾輩は猫である（青空文庫）](https://www.aozora.gr.jp/cards/000148/card789.html) — 詰め物に使ったテキスト
