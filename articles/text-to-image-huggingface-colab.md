---
title: "文章から絵を描いてもらう。ブラウザだけで作る、はじめての画像生成AI"
emoji: "🎨"
type: "tech" # tech: 技術記事 / idea: アイデア
topics: ["colab", "python", "ai", "初心者", "huggingface"]
published: false
---

この記事を読むと、日本語や英語の文章を書くだけで、AIに自分だけの画像を描いてもらえるようになります。プログラミング経験・環境構築・課金は一切不要です。

## さっそく動かしてみよう

使うのは前回の記事と同じく [Google Colab](https://colab.research.google.com/) だけです。パソコンに何もインストールする必要はありません。

### 事前準備（1回だけ）

1. [Hugging Face](https://huggingface.co/join) の無料アカウントを作る（すでに持っている人はサインインしておく）
2. [こちらのリンク](https://huggingface.co/settings/tokens/new?ownUserPermissions=inference.serverless.write&tokenType=fineGrained)を開く（トークンの種類が自動的に「Fine-grained」になり、「Make calls to Inference Providers」という権限にあらかじめチェックが入った状態でトークン作成画面が開きます）。そのまま「Create token」を押してトークンを発行し、表示された値をコピーしておく（この画面を閉じると二度と表示されないので注意）
3. Colabで新しいノートブックを開き、左側の🔑（シークレット）アイコンをクリック
4. 「新しいシークレットを追加」で、名前を `HF_TOKEN`、値に先ほどコピーしたトークンを貼り付けて登録し、「ノートブックからのアクセス」をオンにする（貼り付けるときに余分な空白や改行が入らないよう注意してください）

:::message
今回の画像生成では、旧来の「Inference」チェックボックスではなく「Make calls to Inference Providers」という権限が必要です。[前回のチャットボット記事](https://zenn.dev/m2yagyu/articles/first-ai-chatbot-colab)で使ったトークンを使い回す場合は、上のリンクから権限を確認・再発行してください。Colabのシークレット登録画面のより詳しいスクリーンショット付き解説は前回記事も参考にしてください。
:::

これで準備完了です。あとは次の3つのセルを、上から順番にコピペして実行するだけです。

### セル1：ライブラリを準備してトークンを取り出す

```python
!pip install -q -U huggingface_hub

from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
```

`huggingface_hub` はColabに標準で入っていることもありますが、画像生成に必要な機能が古いバージョンだと足りない場合があるため、念のため最新版に更新しています。

### セル2：画像を生成する関数を作る

```python
from huggingface_hub import InferenceClient

client = InferenceClient(api_key=HF_TOKEN)
MODEL_NAME = "black-forest-labs/FLUX.1-schnell"

def generate_image(prompt, **params):
    return client.text_to_image(prompt, model=MODEL_NAME, **params)
```

`FLUX.1-schnell` は無料枠でも試せる、軽量で高速な画像生成モデルです。

### セル3：プロンプトを渡して画像を表示する

```python
image = generate_image("a small robot reading a book, watercolor style")
image
```

実行すると、数秒〜十数秒でこんな画像が出てきます。

![水彩画風の、本を読む小さなロボット](/images/text-to-image-robot-book.png)

プロンプト（`"a small robot reading a book, watercolor style"`）は日本語でも通じますが、英語の方が意図通りの絵になりやすい傾向があります。まずは英語の単語をいくつか並べる程度でも十分です。

## コードを1行ずつ読んでみよう

### セル1：ライブラリとトークン

`!pip install`のセルはColabのセルに直接シェルコマンドを書く書き方で、行頭の`!`が「これはPythonではなくシェルのコマンドだよ」という合図です。トークンの取り出し方は前回の記事と同じで、Colabのシークレット機能を使うことでトークンをコードに直接書かずに済みます。

### セル2：`InferenceClient`と関数

```python
client = InferenceClient(api_key=HF_TOKEN)
```

これは「Hugging Faceの色々なAIモデルにお願いを送るための窓口」を1つ作るイメージです。窓口を1つ作っておけば、あとは`client.text_to_image(...)`と呼ぶだけで、裏側のどのサーバーにどう問い合わせるかはライブラリが自動で面倒を見てくれます。

```python
def generate_image(prompt, **params):
    return client.text_to_image(prompt, model=MODEL_NAME, **params)
```

`**params`は「他にもオプションがあれば全部まとめて渡す」という書き方です。次の章で使う`negative_prompt`や`width`などは、ここを通じて`text_to_image`にそのまま渡されます。

### セル3：呼び出しと結果

`generate_image(...)`の戻り値は、Pythonの画像を扱うライブラリ（Pillow）の`Image`オブジェクトです。Colabのセルは、最後の行に書いた値をそのまま画面に表示してくれるので、`image`とだけ書けばノートブック上に絵が表示されます。

## なぜ文章から画像が作れるの？（直感編）

種明かしをすると、この裏側で動いている「拡散モデル」というAIは、最初にランダムなノイズ（砂嵐のような画像）を用意して、そこから「これは指定された文章にどれくらい近いか」を何度も確認しながら、少しずつノイズを取り除いていく、という作業を繰り返しています。

砂粒を少しずつ払っていくと、そこに埋もれていた絵が浮かび上がってくる——そんなイメージです。この「ノイズを取り除く」仕組みの数学的な中身（実は物理の拡散現象ととても似た構造をしています）は、また別の記事でじっくり扱う予定です。今回はまず「そういう仕組みで絵ができている」ということだけ知っておけば十分です。

## プロンプトを変えて遊んでみよう

`generate_image`関数には、プロンプト以外にもいくつかオプションを渡せます。

```python
image = generate_image(
    "a cat sitting on a chair, photo realistic",
    negative_prompt="blurry, low quality",
    width=512,
    height=512,
)
image
```

![512x512で生成した、椅子に座る猫の写真風画像](/images/text-to-image-cat-params.png)

- `negative_prompt`：「描いてほしくないもの」を指定できます。上の例では「ぼやけた、低品質な」画像を避けるようにお願いしています
- `width` / `height`：出力画像の縦横のピクセル数を指定できます（何も指定しないと1024×1024になります）

いろいろな組み合わせを試して、思い通りの1枚を探してみてください。

## つまずきやすいポイント Q&A

**Q. `seed`というパラメータもあるみたいですが？**
A. 「乱数の種」を固定する値で、本来は同じ`seed`・同じプロンプトなら同じ画像が再現されるはずのパラメータです。ただし今回使っている無料の推論APIでは、裏側でどのサーバーが処理するかによって、同じ`seed`を指定しても毎回微妙に違う画像になることを確認しています。「だいたい似た雰囲気を試したいときの目印」くらいに考えておくとよいでしょう。

**Q. `401 Unauthorized`（`Invalid or expired token`）というエラーが出ます**
A. ほぼ確実に、トークンに「Make calls to Inference Providers」権限が付いていないことが原因です。事前準備の手順2のリンクから、この権限にチェックが入った状態でトークンを発行し直し、Colabのシークレットを新しい値で上書きしてください。貼り付け時に余分な空白・改行が混ざっていないかも合わせて確認してください。

**Q. 無料でどれくらい使えますか？**
A. Hugging Faceの無料アカウントには、毎月一定量の推論クレジットが付与されています。この記事で試す程度（数枚〜数十枚の生成）であれば、無料の範囲で十分に収まります。

## まとめと次のステップ

文章だけで画像を作る体験ができました。プロンプトを変えるだけでも十分に遊べますが、実は裏側では「ノイズを少しずつ取り除いていく」という、物理の拡散現象ととてもよく似た仕組みが働いています。次回は、この拡散モデルの中身を物理の視点から覗いてみようと思います。
