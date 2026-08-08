#!/usr/bin/env python3
"""記事のPythonコードブロックを順に実行し、記事に載せた出力と一致するか照合する。

CLAUDE.md の「記事に載せるコードは必ず実行して動作確認する」を機械化したもの。
コードブロックが図を保存する場合、カレントディレクトリに出力されるので
images/ で実行すれば記事用の図がそのまま更新される。

  cd images && python3 ../scripts/verify_article_code.py ../articles/<slug>.md

照合のしかた:
  ```python ... ``` の直後にある ``` ... ``` を「記事に載せた出力」とみなす。
  出力に "..." を含む場合（一部を省略して載せている場合）は前方一致で判定する。
"""
import re, sys, io, os, contextlib

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    src = open(path).read()

    import matplotlib
    matplotlib.use("Agg")          # 画面を出さずに savefig だけ効かせる

    blocks = re.findall(r"```python\n(.*?)```", src, re.S)
    expected = re.findall(
        r"```python\n.*?```\n\n(?:[^\n`]*\n\n)?```\n(.*?)```", src, re.S)

    print(f"{os.path.basename(path)}: コードブロック {len(blocks)} 個 / "
          f"記事に載せた出力 {len(expected)} 個\n")

    g, ei, ng = {}, 0, 0
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
        if ei >= len(expected):
            print(f"  ブロック[{i}] 記事に対応する出力が見つからない")
            ng += 1
            continue
        exp = expected[ei].rstrip(); ei += 1
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
