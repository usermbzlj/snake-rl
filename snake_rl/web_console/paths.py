from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_DIR.parent
WEB_DIR = PROJECT_ROOT / "web"
DOCS_DIR = PROJECT_ROOT / "docs"
RUNS_DIR = PROJECT_ROOT / "runs"
GUI_STATE_PATH = PROJECT_ROOT / ".snake_gui_state.json"


def default_custom_path() -> str:
    return str(PROJECT_ROOT / "custom_train_config.json")
