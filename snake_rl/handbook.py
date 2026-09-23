"""Generate docs/custom-train-config.html from form tips + schema."""

from __future__ import annotations

import html
from pathlib import Path

from .form_field_tips import FIELD_TIPS, form_meta
from .web_console.paths import PROJECT_ROOT

HANDBOOK_PATH = PROJECT_ROOT / "docs" / "custom-train-config.html"


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def render_handbook_html() -> str:
    sections = form_meta()["sections"]
    toc_items = [
        ("#files", "关联文件"),
        ("#impact", "高影响参数"),
    ]
    for sec in sections:
        toc_items.append((f"#{sec['tab']}", sec["tabTitle"]))
    toc_items.extend(
        [
            ("#advanced-notes", "课程 / 随机地图 / 并行"),
            ("#validate", "校验与排错"),
        ]
    )
    toc_html = "\n".join(
        f'        <li><a href="{href}">{_esc(title)}</a></li>' for href, title in toc_items
    )

    field_blocks: list[str] = []
    for sec in sections:
        groups_html: list[str] = []
        for group in sec.get("groups") or []:
            rows = []
            for field in group.get("fields") or []:
                key = field["key"]
                label, tip = FIELD_TIPS.get(key, (field.get("label", key), field.get("tip", "")))
                rows.append(
                    "<tr>"
                    f"<td><code>{_esc(key)}</code><div class='muted'>{_esc(label)}</div></td>"
                    f"<td>{_esc(tip)}</td>"
                    "</tr>"
                )
            groups_html.append(
                f"      <h3>{_esc(group.get('title', ''))}</h3>\n"
                "      <table><thead><tr><th>字段</th><th>说明</th></tr></thead><tbody>\n"
                + "\n".join(f"        {row}" for row in rows)
                + "\n      </tbody></table>"
            )
        field_blocks.append(
            f'    <section id="{_esc(sec["tab"])}">\n'
            f"      <h2>{_esc(sec['tabTitle'])}</h2>\n"
            + "\n".join(groups_html)
            + "\n    </section>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Custom Train Config Handbook</title>
<style>
  :root {{
    --bg: #f5f7fb;
    --surface: #ffffff;
    --surface-soft: #eef3fb;
    --border: #d5deeb;
    --text: #1e293b;
    --muted: #475569;
    --accent: #0f766e;
    --accent-soft: #e6f5f3;
    --code-bg: #0f172a;
    --code-text: #e2e8f0;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: radial-gradient(circle at top right, #e8f5f1 0%, var(--bg) 52%);
    color: var(--text);
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    line-height: 1.7;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 32px 18px 56px; }}
  .hero, .toc, section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
  }}
  .hero h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
  .hero p {{ margin: 0; color: var(--muted); }}
  .tip {{
    margin-top: 14px;
    background: var(--accent-soft);
    border-left: 4px solid var(--accent);
    padding: 10px 12px;
    border-radius: 8px;
    color: #0f4f4a;
  }}
  .toc h2, section h2 {{ margin: 0 0 10px; font-size: 1.15rem; }}
  h3 {{ margin: 16px 0 6px; font-size: 1rem; }}
  .toc ul {{ margin: 0; padding-left: 18px; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 0.95rem; }}
  th, td {{ border: 1px solid var(--border); padding: 8px; vertical-align: top; }}
  th {{ background: var(--surface-soft); text-align: left; }}
  .muted {{ color: var(--muted); font-size: 0.85rem; margin-top: 2px; }}
  code {{
    font-family: Consolas, "Cascadia Code", monospace;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 4px;
  }}
  pre {{
    background: var(--code-bg);
    color: var(--code-text);
    border-radius: 10px;
    padding: 12px;
    overflow: auto;
    font-size: 0.9rem;
  }}
  pre code {{ background: transparent; padding: 0; }}
</style>
</head>
<body>
  <main class="wrap">
    <header class="hero">
      <h1>Custom Train Config Handbook</h1>
      <p>由 <code>FIELD_TIPS</code> 与表单元数据自动生成，与 Web 控制台字段保持同步。</p>
      <div class="tip">建议流程：先跑默认配置，再每轮只改 1~2 个参数，并用不同 <code>run_name</code> 做对比。</div>
    </header>
    <nav class="toc">
      <h2>目录</h2>
      <ul>
{toc_html}
      </ul>
    </nav>
    <section id="files">
      <h2>关联文件</h2>
      <ul>
        <li><code>custom_train_config.json</code></li>
        <li><code>custom_train_config.schema.json</code></li>
        <li><code>snake_rl/config.py</code></li>
        <li><code>snake_rl/form_field_tips.py</code></li>
        <li><code>snake_rl/handbook.py</code></li>
      </ul>
    </section>
    <section id="impact">
      <h2>高影响参数</h2>
      <table>
        <thead><tr><th>字段</th><th>推荐起点</th></tr></thead>
        <tbody>
          <tr><td><code>episodes</code></td><td><code>30000</code></td></tr>
          <tr><td><code>model_type</code></td><td><code>adaptive_cnn</code></td></tr>
          <tr><td><code>learning_rate</code></td><td><code>1e-4</code></td></tr>
          <tr><td><code>n_step</code> / <code>per_enabled</code> / <code>dueling</code></td><td><code>3</code> / <code>true</code> / <code>true</code></td></tr>
          <tr><td><code>eval_episodes</code> / <code>eval_interval</code></td><td><code>10</code> / <code>200</code></td></tr>
          <tr><td><code>run_name</code></td><td>每次实验唯一</td></tr>
        </tbody>
      </table>
    </section>
{chr(10).join(field_blocks)}
    <section id="advanced-notes">
      <h2>课程 / 随机地图 / 并行</h2>
      <ul>
        <li><code>curriculum</code> 与 <code>random_board</code> 互斥。</li>
        <li>开启任一可变地图时，不要使用 <code>small_cnn</code>。</li>
        <li>并行建议先关闭，串行稳定后再开。</li>
      </ul>
      <pre><code>{{
  "parallel": {{
    "enabled": false,
    "num_workers": 4,
    "weight_sync_interval_steps": 512,
    "actor_device": "cpu"
  }}
}}</code></pre>
    </section>
    <section id="validate">
      <h2>校验与排错</h2>
      <pre><code>uv run snake-rl estimate --scheme custom --custom-config custom_train_config.json
python -m snake_rl.handbook</code></pre>
      <ul>
        <li>旧 checkpoint（schema &lt; 2）会被拒绝，需要按新 Rainbow-lite 结构重训。</li>
        <li>“可变尺寸不支持”：通常是 <code>small_cnn + curriculum/random_board</code>。</li>
        <li>曲线不涨：先看 <code>epsilon_decay_steps</code>、<code>learning_rate</code>、奖励权重。</li>
      </ul>
    </section>
  </main>
</body>
</html>
"""


def write_handbook(path: Path | None = None) -> Path:
    dest = path or HANDBOOK_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_handbook_html(), encoding="utf-8")
    return dest


def main() -> None:
    out = write_handbook()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
