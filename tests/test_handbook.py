from __future__ import annotations

from snake_rl.handbook import render_handbook_html, write_handbook


def test_handbook_contains_rainbow_and_curriculum_fields(tmp_path) -> None:
    html = render_handbook_html()
    for token in ("n_step", "per_enabled", "dueling", "eval_interval", "curriculum_enabled"):
        assert token in html
    out = write_handbook(tmp_path / "handbook.html")
    assert out.is_file()
    assert "n_step" in out.read_text(encoding="utf-8")
