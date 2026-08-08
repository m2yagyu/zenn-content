#!/usr/bin/env python3
"""Zenn公開APIから記事を集めて「伸びた記事 vs 伸びなかった記事」を判別的に比較する。

docs/zenn-analysis-2026-08.md はこのスクリプトの出力にもとづく。再測するときはこれを使う。

  python3 scripts/zenn_analyze.py collect   # 記事メタデータを収集(数分)
  python3 scripts/zenn_analyze.py titles    # タイトルの型を比較
  python3 scripts/zenn_analyze.py topics    # トピックの天井/回転速度/転換率を実測
  python3 scripts/zenn_analyze.py mine      # 自分の記事を同じ物差しにかける

母集団の作り方(ここを間違えると誤った結論が出る):
  WIN  = 100いいね以上
  BASE = 公開30日以上経過 かつ 10いいね未満
  公開直後で0いいねの新着は「失敗」ではないのでBASEから除外する。
"""
import json, re, sys, time, os, datetime, statistics, urllib.parse, urllib.request
from collections import Counter

CACHE = os.path.join(os.path.dirname(__file__), "..", ".zenn-cache.json")
UA = {"User-Agent": "Mozilla/5.0 (personal content analysis)"}
TODAY = datetime.date.today()

TOPICS = ["生成ai", "machinelearning", "chatgpt", "claude", "初心者", "python", "llm", "ai",
          "diffusion", "huggingface", "物理", "数学", "deeplearning", "colab"]


def api(params):
    url = "https://zenn.dev/api/articles?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def feed(params, max_pages=3):
    out, page = [], 1
    while page and page <= max_pages:
        p = dict(params, page=page)
        try:
            d = api(p)
        except Exception as e:
            print(f"  ! {p}: {e}", file=sys.stderr)
            break
        out += d.get("articles", [])
        page = d.get("next_page")
        time.sleep(0.2)
    return out


def age(a):
    return (TODAY - datetime.date.fromisoformat(a["published_at"][:10])).days


def load():
    if not os.path.exists(CACHE):
        sys.exit("先に `python3 scripts/zenn_analyze.py collect` を実行してください")
    return json.load(open(CACHE))


def groups(arts, since="2024"):
    recent = [a for a in arts if a["published_at"][:4] >= since]
    win = [a for a in recent if a["liked_count"] >= 100]
    base = [a for a in recent if age(a) >= 30 and a["liked_count"] < 10]
    return recent, win, base


# ---------------------------------------------------------------- collect
def cmd_collect():
    seen = {}
    for order in ["daily", "weekly", "alltime", "latest"]:
        arts = feed({"order": order}, 3)
        print(f"global {order}: {len(arts)}")
        for a in arts:
            seen.setdefault(a["id"], a)
    for t in TOPICS:
        for order in ["alltime", "monthly", "latest"]:
            arts = feed({"topicname": t, "order": order}, 2)
            print(f"{t} {order}: {len(arts)}")
            for a in arts:
                cur = seen.setdefault(a["id"], a)
                cur["_src"] = sorted(set(cur.get("_src", []) + [t]))
    json.dump(list(seen.values()), open(CACHE, "w"), ensure_ascii=False)
    print(f"\n{len(seen)}件を {CACHE} に保存")


# ---------------------------------------------------------------- titles
FEATS = {
    "「：」で副題を足す構造": lambda t: bool(re.match(r"^[^：:]{4,}[：:].{4,}", t)),
    "口語・感情の語彙": lambda t: bool(re.search(
        r"(だよね|なの[？?]|結局|そもそも|正直|雰囲気|面白い|しんどい|最強|ぶっちゃけ|って何|話$|けど)", t)),
    "論文調の名詞止め": lambda t: bool(re.search(
        r"(の解説$|解説$|の導出|入門$|まとめ$|について$|の考察$|論文解説|Explained)", t)),
    "〜してみた/やってみた": lambda t: bool(re.search(r"(してみた|やってみた|試してみた)", t)),
    "疑問形(?/か)": lambda t: bool(re.search(r"[?？]|のか$", t)),
    "具体的なモデル/製品名": lambda t: bool(re.search(
        r"(Claude|ChatGPT|GPT|Cursor|Gemini|MCP|LLM|RAG|Llama|Transformer|VAE|Stable ?Diffusion)", t, re.I)),
    "一人称が出る": lambda t: bool(re.search(r"(私|僕|俺|自分)", t)),
}


def cmd_titles():
    arts = load()
    recent, win, base = groups(arts)
    print(f"母集団: {len(recent)}件(2024年以降) / WIN {len(win)} / BASE {len(base)}")
    n = len(win) / (len(win) + len(base)) * 100
    print(f"WIN率の母集団平均 = {n:.1f}%  ※これより高ければ有利\n")
    print(f"{'タイトルの型':<24} {'WIN%':>7} {'BASE%':>7} {'差':>8} {'WIN率':>7}")
    print("-" * 58)
    rows = []
    for k, f in FEATS.items():
        w = sum(f(a["title"]) for a in win)
        b = sum(f(a["title"]) for a in base)
        wr = w / (w + b) * 100 if w + b else float("nan")
        rows.append((k, w / len(win) * 100, b / len(base) * 100, wr))
    for k, w, b, wr in sorted(rows, key=lambda r: -(r[1] - r[2])):
        print(f"{k:<24} {w:6.1f}% {b:6.1f}% {w-b:+7.1f}pt {wr:6.1f}%")

    for label, sel in [("タイトル長", lambda a: len(a["title"])),
                       ("本文の文字数", lambda a: a["body_letters_count"])]:
        print(f"\n{label}: WIN中央値 {statistics.median([sel(a) for a in win]):.0f}"
              f" / BASE中央値 {statistics.median([sel(a) for a in base]):.0f}")

    print("\n" + "=" * 70)
    print("同ジャンル(機械学習系, エージェント運用記事を除く)の上位タイトル")
    print("=" * 70)
    gen = ("machinelearning", "deeplearning", "数学", "物理", "diffusion", "huggingface")
    agent = re.compile(r"(Claude ?Code|Cursor|Copilot|Codex|Cline|MCP|AIエージェント)", re.I)
    core = [a for a in recent
            if set(a.get("_src", [])) & set(gen) and not agent.search(a["title"])]
    for a in sorted(core, key=lambda x: -x["liked_count"])[:25]:
        print(f'{a["liked_count"]:>5}  {a["body_letters_count"]:>6}字  {a["title"]}')


# ---------------------------------------------------------------- topics
def cmd_topics():
    print("各トピックの実力を直接測る。")
    print("  天井    = alltimeフィード上位の到達いいね数")
    print("  回転    = 新着48件が何日ぶんか(小さいほど早く埋もれる)")
    print("  転換率  = 公開30日以上経過した記事が10いいねに届いた割合\n")
    print(f"{'topic':<18} {'歴代1位':>7} {'歴代10位':>8} {'回転(日)':>8} {'転換率':>7} {'0いいね率':>8}")
    print("-" * 62)
    rows = []
    for t in TOPICS:
        try:
            top = api({"topicname": t, "order": "alltime"})["articles"]
        except Exception as e:
            print(f"{t:<18} error {e}")
            continue
        lk = sorted((x["liked_count"] for x in top), reverse=True)
        latest = feed({"topicname": t, "order": "latest"}, 12)
        if len(latest) >= 48:
            d0 = datetime.date.fromisoformat(latest[0]["published_at"][:10])
            d47 = datetime.date.fromisoformat(latest[47]["published_at"][:10])
            span = (d0 - d47).days
        else:
            span = None
        old = [x["liked_count"] for x in latest if age(x) >= 30]
        conv = sum(1 for v in old if v >= 10) / len(old) * 100 if len(old) >= 20 else None
        zero = sum(1 for v in old if v == 0) / len(old) * 100 if len(old) >= 20 else None
        rows.append((t, lk[0] if lk else 0, lk[9] if len(lk) > 9 else 0, span, conv, zero))
        time.sleep(0.2)
    for t, a1, a10, span, conv, zero in sorted(rows, key=lambda r: -r[1]):
        sp = f"{span}" if span is not None else "-"
        cv = f"{conv:.1f}%" if conv is not None else "-"
        zr = f"{zero:.1f}%" if zero is not None else "-"
        print(f"{t:<18} {a1:>7} {a10:>8} {sp:>8} {cv:>7} {zr:>8}")


# ---------------------------------------------------------------- mine
def cmd_mine(username="m2yagyu"):
    arts = api({"username": username, "order": "latest"})["articles"]
    print(f"{'いいね':>5} {'字数':>6} {'長さ':>4}  該当した型 / タイトル")
    print("-" * 76)
    for a in sorted(arts, key=lambda x: -x["liked_count"]):
        t = a["title"]
        hit = [k for k, f in FEATS.items() if f(t)]
        print(f'{a["liked_count"]:>5} {a["body_letters_count"]:>6} {len(t):>4}字  '
              f'{"/".join(hit) if hit else "(該当なし)"}\n        {t}')


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"collect": cmd_collect, "titles": cmd_titles,
          "topics": cmd_topics, "mine": cmd_mine}.get(cmd)
    if not fn:
        sys.exit(__doc__)
    fn()
