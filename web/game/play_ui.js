"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const game = window.snakeGame;
  const ric = window.remoteInferenceController;
  if (!game || !ric) return;

  const playBtn = document.getElementById("overlayPlayBtn");
  const aiBtn = document.getElementById("overlayAiBtn");
  const oneClickBtn = document.getElementById("oneClickAiBtn");
  const runPicker = document.getElementById("runPicker");
  const runPickerBlock = document.getElementById("runPickerBlock");
  const coach = document.getElementById("coach");
  const coachDismiss = document.getElementById("coachDismiss");
  const canvas = document.getElementById("gameCanvas");
  const isHttp = window.location.protocol.startsWith("http");

  restoreSettings();
  bindSettingsPersist();
  bindOverlay();
  bindSwipe();
  bindCoach();
  bindTouchClass();
  if (isHttp) {
    void initHttpHelpers();
  }

  function restoreSettings() {
    try {
      const raw = localStorage.getItem("snake.playSettings");
      if (!raw) return;
      const saved = JSON.parse(raw);
      if (saved.difficulty && game.difficultySelect) game.difficultySelect.value = saved.difficulty;
      if (saved.mode && game.modeSelect) game.modeSelect.value = saved.mode;
      if (saved.boardSize && game.boardSizeSelect) game.boardSizeSelect.value = String(saved.boardSize);
      game.applySettingsFromUI();
      game.render(performance.now());
    } catch (_) {
      /* ignore */
    }
  }

  function persistSettings() {
    try {
      localStorage.setItem(
        "snake.playSettings",
        JSON.stringify({
          difficulty: game.difficultySelect && game.difficultySelect.value,
          mode: game.modeSelect && game.modeSelect.value,
          boardSize: game.boardSizeSelect && game.boardSizeSelect.value,
        })
      );
    } catch (_) {
      /* ignore */
    }
  }

  function bindSettingsPersist() {
    [game.difficultySelect, game.modeSelect, game.boardSizeSelect].forEach((el) => {
      if (!el) return;
      el.addEventListener("change", persistSettings);
    });
  }

  function syncOverlayLabels() {
    if (!playBtn) return;
    const title = (game.overlayTitle && game.overlayTitle.textContent) || "";
    if (game.state === "paused" || title === "已暂停") playBtn.textContent = "继续";
    else if (game.state === "over" || title === "本局结束" || title === "AI 对局结束") playBtn.textContent = "再来一局";
    else playBtn.textContent = "自己玩";
  }

  function bindOverlay() {
    if (playBtn) {
      playBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (game.state === "paused") {
          game.togglePause();
          return;
        }
        if (game.startBtn) {
          game.startBtn.click();
        } else {
          game.startNewGame();
        }
      });
    }
    if (aiBtn) {
      aiBtn.addEventListener("click", () => {
        void startAiFlow();
      });
    }
    if (oneClickBtn) {
      oneClickBtn.addEventListener("click", () => {
        void startAiFlow();
      });
    }

    const original = game.setOverlay.bind(game);
    game.setOverlay = function setOverlayPatched(visible, title, text) {
      original(visible, title, text);
      syncOverlayLabels();
    };
    syncOverlayLabels();
    setInterval(syncOverlayLabels, 400);
  }

  function bindSwipe() {
    if (!canvas) return;
    let startX = 0;
    let startY = 0;
    canvas.addEventListener(
      "touchstart",
      (event) => {
        if (!event.touches || !event.touches[0]) return;
        startX = event.touches[0].clientX;
        startY = event.touches[0].clientY;
      },
      { passive: true }
    );
    canvas.addEventListener(
      "touchmove",
      (event) => {
        event.preventDefault();
      },
      { passive: false }
    );
    canvas.addEventListener("touchend", (event) => {
      const touch = event.changedTouches && event.changedTouches[0];
      if (!touch) return;
      const dx = touch.clientX - startX;
      const dy = touch.clientY - startY;
      if (Math.abs(dx) < 24 && Math.abs(dy) < 24) return;
      if (Math.abs(dx) >= Math.abs(dy)) {
        game.queueRelativeTurn(dx > 0 ? "right" : "left");
      }
    });
  }

  function bindCoach() {
    if (!coach) return;
    try {
      if (localStorage.getItem("snake.coachDismissed")) return;
    } catch (_) {
      /* show anyway */
    }
    coach.hidden = false;
    if (coachDismiss) {
      coachDismiss.addEventListener("click", () => {
        coach.hidden = true;
        try {
          localStorage.setItem("snake.coachDismissed", "1");
        } catch (_) {
          /* ignore */
        }
      });
    }
  }

  function bindTouchClass() {
    window.addEventListener(
      "touchstart",
      () => {
        document.body.classList.add("has-touch");
      },
      { once: true, passive: true }
    );
  }

  async function initHttpHelpers() {
    try {
      const st = await fetch("/api/state").then((r) => (r.ok ? r.json() : null));
      if (ric.serverUrlInput) {
        ric.serverUrlInput.value = "/api/infer/proxy";
      }
    } catch (_) {
      /* console may be unavailable */
    }

    if (runPickerBlock) runPickerBlock.hidden = false;
    await refreshRuns();
    if (runPicker) {
      runPicker.addEventListener("change", () => {
        void selectRun(runPicker.value);
      });
    }

    const params = new URLSearchParams(window.location.search);
    if (params.get("autostart") === "1") {
      setTimeout(() => {
        void startAiFlow({ autostart: true });
      }, 400);
    }
  }

  async function refreshRuns() {
    if (!runPicker) return;
    try {
      const list = await fetch("/api/runs").then((r) => (r.ok ? r.json() : []));
      const playable = (list || []).filter((r) => r.can_demo || r.has_best_checkpoint || r.has_latest_checkpoint);
      const current = runPicker.value;
      runPicker.innerHTML = '<option value="">选择一次训练…</option>';
      playable.forEach((r) => {
        const opt = document.createElement("option");
        opt.value = r.name;
        opt.textContent = `${r.name} · ${r.model} · ${r.status} · ${r.episodes} 局`;
        runPicker.appendChild(opt);
      });
      if (current && playable.some((r) => r.name === current)) runPicker.value = current;
      if (!runPicker.value && playable.length === 1) {
        runPicker.value = playable[0].name;
        await selectRun(playable[0].name);
      }
    } catch (_) {
      /* ignore */
    }
  }

  async function selectRun(name) {
    if (!name) return;
    try {
      const art = await fetch("/api/runs/" + encodeURIComponent(name) + "/artifacts").then((r) => r.json());
      const best =
        (art.checkpoints || []).find((c) => c.name === "best.pt") ||
        (art.checkpoints || [])[0];
      if (best && ric.checkpointPathInput) {
        ric.checkpointPathInput.value = best.path;
        ric.inspectCheckpointPath();
      }
    } catch (err) {
      ric.setStatus("无法读取该训练的权重：" + (err.message || err), "error");
    }
  }

  async function startAiFlow(opts) {
    opts = opts || {};
    if (ric.running) return;

    if (ric.modelInfo && !opts.forceReload) {
      ric.start();
      return;
    }

    const runName = runPicker && runPicker.value;
    if (isHttp && runName) {
      ric.setStatus("正在启动推理服务并加载模型…", "running");
      try {
        const res = await fetch("/api/infer/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_name: runName }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || res.status);
        if (data.checkpoint && ric.checkpointPathInput) {
          ric.checkpointPathInput.value = data.checkpoint;
        }
        await waitForHealth(ric.serverUrlInput.value, 12000);
      } catch (err) {
        ric.setStatus("无法启动推理服务：" + (err.message || err), "error");
        return;
      }
    }

    if (!String(ric.checkpointPathInput && ric.checkpointPathInput.value || "").trim()) {
      ric.setStatus("请先选择一次训练，或填写模型路径。", "error");
      const card = document.querySelector(".card-ai");
      if (card) card.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }

    await ric.loadModel();
    if (ric.modelInfo) ric.start();
  }

  async function waitForHealth(rawUrl, timeoutMs) {
    const base = String(rawUrl || "").replace(/\/+$/, "");
    const deadline = Date.now() + (timeoutMs || 10000);
    while (Date.now() < deadline) {
      try {
        const res = await fetch(base + "/health");
        if (res.ok) return;
      } catch (_) {
        /* still starting */
      }
      await new Promise((resolve) => setTimeout(resolve, 400));
    }
    throw new Error("推理服务未就绪");
  }
});
