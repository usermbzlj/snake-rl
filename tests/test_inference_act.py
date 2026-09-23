from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from snake_rl.inference_server import InferenceHandler, ModelRunner
from snake_rl.train import run_training
from tests.test_train_loop_smoke import _tiny_cfg


def test_v1_act_http_roundtrip(tmp_path: Path) -> None:
    summary = run_training(_tiny_cfg(tmp_path, run_name="act"))
    ckpt = Path(summary["run_dir"]) / "checkpoints" / "latest.pt"
    assert ckpt.is_file()

    runner = ModelRunner("cpu")
    runner.load_checkpoint(ckpt)
    server = ThreadingHTTPServer(("127.0.0.1", 0), InferenceHandler)
    server.runner = runner  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        status = json.loads(urlopen(f"http://127.0.0.1:{port}/v1/status", timeout=3).read())
        assert status["loaded"] is True
        body = {
            "state": {
                "envConfig": {
                    "difficulty": "normal",
                    "mode": "classic",
                    "boardSize": 8,
                    "enableBonusFood": False,
                    "enableObstacles": False,
                    "allowLeveling": False,
                    "maxStepsWithoutFood": 16,
                },
                "rewardWeights": {"food": 1.0},
                "seed": 3,
                "state": "running",
                "direction": "right",
                "snake": [{"x": 3, "y": 4}, {"x": 2, "y": 4}, {"x": 1, "y": 4}],
                "food": {"x": 6, "y": 4},
                "bonusFood": None,
                "obstacles": [],
                "score": 2,
                "level": 1,
                "foodsEaten": 1,
                "stepsSinceLastFood": 0,
                "episodeStats": {"episode": 1, "steps": 4, "totalReward": 0.2, "foods": 1},
                "lastTerminalReason": "",
            }
        }
        req = Request(
            f"http://127.0.0.1:{port}/v1/act",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = json.loads(urlopen(req, timeout=5).read())
        assert payload["action"] in (0, 1, 2)
        assert payload["modelType"] == "tiny"
    finally:
        server.shutdown()
        server.server_close()
