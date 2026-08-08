#!/usr/bin/env python3
"""記事のPythonコードブロックを順に実行し、記事に載せた出力と一致するか照合する。

CLAUDE.md の「記事に載せるコードは必ず実行して動作確認する」を機械化したもの。
コードブロックが図を保存する場合、カレントディレクトリに出力されるので
images/ で実行すれば記事用の図がそのまま更新される。

  cd images && python3 ../scripts/verify_article_code.py ../articles/<slug>.md

照合のしかた:
  ```python ... ``` を実行対象とし、その後ろに現れる最初の「言語指定なしの ``` ブロック」
  を、そのコードの出力として載せたものとみなす（次の ```python が来る前まで探す）。
  あいだに解説文や ```python:抜粋 を挟んでも対応づけは崩れない。
  出力に "..." を含む場合（一部を省略して載せている場合）は前方一致で判定する。

  補足: 記事に載せる抜粋コードは ```python:抜粋 と書く。断片なので実行対象から外れる。
"""
import re, sys, io, os, contextlib

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    src = open(path).read()

    import matplotlib
    matplotlib.use("Agg")          # 画面を出さずに savefig だけ効かせる

    # 文書中のフェンス付きブロックを、言語指定つきで出現順に並べる
    fences = [(m.group(1), m.group(2))
              for m in re.finditer(r"```(\S*)\n(.*?)```", src, re.S)]

    blocks, expected = [], []
    for i, (lang, body) in enumerate(fences):
        if lang != "python":            # python:抜粋 / mermaid / bash などは実行しない
            continue
        blocks.append(body)
        out = None
        for lang2, body2 in fences[i + 1:]:
            if lang2 == "python":       # 次のコードに達したら、その手前までが対象
                break
            if lang2 == "":             # 言語指定なし = 実行結果として載せたもの
                out = body2
                break
        expected.append(out)

    print(f"{os.path.basename(path)}: コードブロック {len(blocks)} 個 / "
          f"記事に載せた出力 {sum(1 for e in expected if e is not None)} 個\n")

    g, ng = {}, 0
    for i, b in enumerate(blocks):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(b, g)
        except Exception as e:
            print(f"  ブロック[{i}] 実行エラー: {type(e).__name__}: {e}")
            ng += 1
            continue
        out = buf.getvalue().rstrip()
        if not out:
            print(f"  ブロック[{i}] 出力なし（作図のみ）")
            continue
        if expected[i] is None:
            print(f"  ブロック[{i}] 出力があるのに記事に載せていない")
            ng += 1
            continue
        exp = expected[i].rstrip()
        head = exp.split("...")[0].strip()
        ok = out.strip() == exp.strip() or (head and out.strip().startswith(head))
        print(f"  ブロック[{i}] 記載と一致: {'YES' if ok else 'NO'}")
        if not ok:
            ng += 1
            print("    --- 実際の出力 ---"); print("    " + out.replace("\n", "\n    "))
            print("    --- 記事の記載 ---"); print("    " + exp.replace("\n", "\n    "))

    print("\n" + ("すべて一致" if ng == 0 else f"不一致 {ng} 件。記事を実際の出力に合わせること"))
    sys.exit(1 if ng else 0)


if __name__ == "__main__":
    main()
