"""子进程温和停止：Windows 先发 CTRL_BREAK，等待，再 kill。"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def win_creation_flags() -> int:
    return subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0


def popen_service(
    args: list[str],
    *,
    cwd: str | Path,
    stdout: int | None = subprocess.DEVNULL,
    stderr: int | None = subprocess.PIPE,
) -> subprocess.Popen:
    return subprocess.Popen(
        args,
        cwd=str(cwd),
        stdout=stdout,
        stderr=stderr,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        creationflags=win_creation_flags(),
    )


def terminate_process(
    proc: subprocess.Popen | None,
    *,
    timeout_s: float = 5.0,
) -> int | None:
    if proc is None:
        return None
    if proc.poll() is not None:
        return proc.returncode
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
    except (OSError, ProcessLookupError):
        pass
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(0.05)
    try:
        proc.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    return proc.returncode
