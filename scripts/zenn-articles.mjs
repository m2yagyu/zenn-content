#!/usr/bin/env node
// Zennの非公式公開API (https://zenn.dev/api/articles) を叩いて記事情報を取得する。
// 認証不要。ダッシュボードのPV数は非公開APIかつセッションCookieが必要なため対象外。
//
// 使い方:
//   node scripts/zenn-articles.mjs mine [--username m2yagyu]
//   node scripts/zenn-articles.mjs trend --topic machinelearning [--order liked|latest] [--limit 20]

const ZENN_API = "https://zenn.dev/api/articles";

function parseArgs(argv) {
  const [mode, ...rest] = argv;
  const opts = {};
  for (let i = 0; i < rest.length; i++) {
    if (rest[i].startsWith("--")) {
      const key = rest[i].slice(2);
      opts[key] = rest[i + 1];
      i++;
    }
  }
  return { mode, opts };
}

async function fetchArticles(params) {
  const url = new URL(ZENN_API);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined) url.searchParams.set(k, v);
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Zenn API request failed: ${res.status} ${res.statusText}`);
  const { articles } = await res.json();
  return articles;
}

function printTable(articles) {
  const rows = articles.map((a) => ({
    published_at: a.published_at?.slice(0, 10),
    likes: a.liked_count,
    comments: a.comments_count,
    author: a.user.username,
    title: a.title,
    url: `https://zenn.dev${a.path}`,
  }));
  console.table(rows);
}

async function main() {
  const { mode, opts } = parseArgs(process.argv.slice(2));

  if (mode === "mine") {
    const username = opts.username ?? "m2yagyu";
    const articles = await fetchArticles({ username, order: "latest" });
    articles.sort((a, b) => b.liked_count - a.liked_count);
    printTable(articles);
    return;
  }

  if (mode === "trend") {
    if (!opts.topic) throw new Error("--topic <topicname> is required for trend mode");
    const articles = await fetchArticles({
      topicname: opts.topic,
      order: opts.order ?? "liked",
    });
    const limit = opts.limit ? Number(opts.limit) : 20;
    printTable(articles.slice(0, limit));
    return;
  }

  console.error("Usage:\n  node scripts/zenn-articles.mjs mine [--username <name>]\n  node scripts/zenn-articles.mjs trend --topic <topicname> [--order liked|latest] [--limit N]");
  process.exit(1);
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
