"use strict";

function pageIsHttp() {
  return window.location.protocol === "http:" || window.location.protocol === "https:";
}

function consoleInferProxyUrl() {
  return "/api/infer/proxy";
}

function isSameOriginUrl(raw) {
  try {
    const url = new URL(String(raw || "").trim(), window.location.href);
    return url.origin === window.location.origin;
  } catch {
    return false;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const game = new SnakeGame();
  const agentAPI = game.createAgentAPI();

  class RemoteInferenceController {
    constructor(gameInstance, api) {
      this.game = gameInstance;
      this.agentAPI = api;
      this.modelInfo = null;
      this.running = false;
      this.runToken = 0;

      this.serverUrlInput = document.getElementById("inferenceServerUrl");
      this.checkpointPathInput = document.getElementById("checkpointPath");
      this.checkpointHintEl = document.getElementById("checkpointHint");
      this.stepDelayInput = document.getElementById("aiStepDelay");
      this.autoLoopCheckbox = document.getElementById("aiAutoLoop");
      this.loadModelBtn = document.getElementById("loadModelBtn");
      this.startAiBtn = document.getElementById("startAiBtn");
      this.stopAiBtn = document.getElementById("stopAiBtn");
      this.aiStatus = document.getElementById("aiStatus");
      this.showPerceptionCheckbox = document.getElementById("showPerception");
      this.touchButtons = Array.from(document.querySelectorAll(".touch-controls button"));

      this.blockHumanKeys = (event) => {
        if (!this.running) {
          return;
        }
        if (isTypingInFormControl(event.target)) {
          return;
        }
        const blocked = new Set([
          "KeyA",
          "KeyD",
          "ArrowLeft",
          "ArrowRight",
          "Space",
          "KeyP",
          "Enter",
          "KeyR",
        ]);
        if (blocked.has(event.code)) {
          event.preventDefault();
          event.stopImmediatePropagation();
        }
      };

      document.addEventListener("keydown", this.blockHumanKeys, true);
      this.bindEvents();
      this.updateButtons();
      try {
        const params = new URLSearchParams(window.location.search);
        const qp = params.get("checkpoint");
        if (qp) {
          this.checkpointPathInput.value = decodeURIComponent(qp);
          // 延后一帧触发 inspect（等页面完全就绪）
          setTimeout(() => this.inspectCheckpointPath(), 300);
        }
      } catch {
        /* ignore */
      }
      void this.initServerUrl();
    }

    async initServerUrl() {
      if (pageIsHttp() && this.serverUrlInput) {
        try {
          const st = await fetch("/api/state").then((r) => (r.ok ? r.json() : null));
          if (st) {
            this.serverUrlInput.value = consoleInferProxyUrl();
            if (!st.infer) {
              return;
            }
          }
        } catch {
          /* standalone static host — keep the input value */
        }
      }
      await this.probeServerStatusIfConfigured();
    }

    async probeServerStatusIfConfigured() {
      try {
        const raw = String(this.serverUrlInput.value || "").trim();
        if (!raw) {
          return;
        }
        // Cross-origin probes (e.g. :7860 → :8765) spam CORS when another app owns 8765.
        if (!isSameOriginUrl(raw)) {
          return;
        }
        const serverUrl = raw.replace(/\/+$/, "");
        const response = await fetch(`${serverUrl}/v1/status`);
        const data = await response.json().catch(() => ({}));
        if (!response.ok || !data.loaded) {
          return;
        }
        this.modelInfo = {
          checkpoint: data.checkpoint,
          modelType: data.modelType,
          inputSize: data.inputSize,
          supportsVariableBoard: data.supportsVariableBoard,
          recommendedEnvConfig: data.recommendedEnvConfig,
        };
        this.applyRecommendedUI(this.modelInfo);
        if (data.checkpoint) {
          this.checkpointPathInput.value = data.checkpoint;
        }
        this.setStatus(
          `检测到服务已加载模型：${data.modelType}。可直接点击「AI 接管」。`,
          "success"
        );
        this.updateButtons();
      } catch {
        /* 服务未启动或网络不可达 */
      }
    }

    bindEvents() {
      this.loadModelBtn.addEventListener("click", () => {
        this.loadModel();
      });
      this.startAiBtn.addEventListener("click", () => {
        this.start();
      });
      this.stopAiBtn.addEventListener("click", () => {
        this.stop("已停止 AI，恢复人工接管。", "success");
      });

      // 路径输入框：失焦或粘贴后自动 inspect
      if (this.checkpointPathInput) {
        let inspectTimer = null;
        const scheduleInspect = () => {
          clearTimeout(inspectTimer);
          inspectTimer = setTimeout(() => this.inspectCheckpointPath(), 600);
        };
        this.checkpointPathInput.addEventListener("blur", () => this.inspectCheckpointPath());
        this.checkpointPathInput.addEventListener("input", scheduleInspect);
        this.checkpointPathInput.addEventListener("paste", () => {
          clearTimeout(inspectTimer);
          inspectTimer = setTimeout(() => this.inspectCheckpointPath(), 200);
        });
      }
    }

    setCheckpointHint(text, kind) {
      const el = this.checkpointHintEl;
      if (!el) return;
      if (!text) { el.style.display = "none"; el.textContent = ""; return; }
      el.textContent = text;
      el.className = "checkpoint-hint" + (kind ? " hint-" + kind : "");
      el.style.display = "";
    }

    async inspectCheckpointPath() {
      // 只在通过 HTTP 访问时才能调用同源 API
      if (!window.location.protocol.startsWith("http")) return;
      const raw = String(this.checkpointPathInput?.value || "").trim();
      if (!raw) { this.setCheckpointHint("", ""); return; }

      try {
        const res = await fetch("/api/checkpoint/inspect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: raw }),
          signal: AbortSignal.timeout ? AbortSignal.timeout(5000) : undefined,
        });
        if (!res.ok) { this.setCheckpointHint("", ""); return; }
        const data = await res.json();
        if (!data.exists) {
          this.setCheckpointHint("⚠ 文件不存在，请确认路径。", "err");
          return;
        }
        if (!data.is_pt) {
          this.setCheckpointHint("⚠ 不是 .pt 文件。", "warn");
          return;
        }
        if (data.hint) {
          const rec = data.recommendation || "";
          const kind = rec === "demo" || rec === "demo_external" ? "ok"
                     : rec === "resume" ? "warn"
                     : rec === "invalid" ? "err"
                     : "";
          this.setCheckpointHint(data.hint, kind);
        }
      } catch {
        // 静默失败，不影响正常使用
      }
    }

    normalizeServerUrl() {
      const raw = String(this.serverUrlInput.value || "").trim();
      if (!raw) {
        if (pageIsHttp()) {
          return consoleInferProxyUrl();
        }
        throw new Error("请先填写推理服务地址");
      }
      return raw.replace(/\/+$/, "");
    }

    getStepDelayMs() {
      const delay = Number(this.stepDelayInput.value);
      if (!Number.isFinite(delay) || delay < 0) {
        return 0;
      }
      return Math.round(delay);
    }

    setStatus(message, kind = "info") {
      this.aiStatus.textContent = message;
      this.aiStatus.dataset.kind = kind;
    }

    formatError(error) {
      if (error && typeof error === "object" && "message" in error) {
        return String(error.message);
      }
      return String(error);
    }

    updateButtons() {
      this.loadModelBtn.disabled = this.running;
      this.startAiBtn.disabled = this.running || !this.modelInfo;
      this.stopAiBtn.disabled = !this.running;
    }

    setHumanControlsDisabled(disabled) {
      this.game.startBtn.disabled = disabled;
      if (disabled) {
        this.game.pauseBtn.disabled = true;
      } else if (!this.running && !this.game.agentControlled) {
        this.game.pauseBtn.disabled = this.game.state !== "running" && this.game.state !== "paused";
      }
      this.touchButtons.forEach((button) => {
        button.disabled = disabled;
      });
    }

    wantsPerceptionDebug() {
      return Boolean(this.showPerceptionCheckbox && this.showPerceptionCheckbox.checked);
    }

    async postJson(path, payload) {
      const serverUrl = this.normalizeServerUrl();
      const response = await fetch(`${serverUrl}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = data.detail || data.error;
        throw new Error(typeof detail === "string" ? detail : `请求失败: ${response.status}`);
      }
      return data;
    }

    applyRecommendedUI(modelInfo) {
      const env = modelInfo.recommendedEnvConfig || {};
      const fixedBoardSize =
        modelInfo.modelType === "small_cnn"
          ? Number(env.board_size || modelInfo.inputSize)
          : Number(env.board_size || this.game.boardSizeSelect.value);

      if (env.difficulty && DIFFICULTY_CONFIG[env.difficulty]) {
        this.game.difficultySelect.value = env.difficulty;
      }
      if (env.mode === "classic" || env.mode === "wrap") {
        this.game.modeSelect.value = env.mode;
      }
      if (Array.from(this.game.boardSizeSelect.options).some((opt) => Number(opt.value) === fixedBoardSize)) {
        this.game.boardSizeSelect.value = String(fixedBoardSize);
      }
    }

    async loadModel() {
      const checkpoint = String(this.checkpointPathInput.value || "").trim();
      if (!checkpoint) {
        this.setStatus("请先填写模型路径。", "error");
        return;
      }

      this.setStatus("正在加载模型...", "running");
      try {
        const modelInfo = await this.postJson("/v1/load", { checkpoint });
        this.modelInfo = modelInfo;
        this.applyRecommendedUI(modelInfo);
        this.updateButtons();

        const sizeText = modelInfo.supportsVariableBoard
          ? "支持可变地图"
          : `固定地图 ${modelInfo.inputSize}x${modelInfo.inputSize}`;
        this.setStatus(
          `模型已加载：${modelInfo.modelType}，${sizeText}。现在可以点击“AI 接管”。`,
          "success"
        );
      } catch (error) {
        this.modelInfo = null;
        this.updateButtons();
        this.setStatus(`模型加载失败：${this.formatError(error)}`, "error");
      }
    }

    buildAgentResetOptions() {
      const recommended = this.modelInfo?.recommendedEnvConfig || {};
      const boardSize =
        this.modelInfo?.modelType === "small_cnn"
          ? Number(recommended.board_size || this.modelInfo.inputSize)
          : Number(this.game.boardSizeSelect.value);

      return {
        difficulty: this.game.difficultySelect.value,
        mode: this.game.modeSelect.value,
        boardSize,
        enableBonusFood:
          recommended.enable_bonus_food !== undefined ? Boolean(recommended.enable_bonus_food) : true,
        enableObstacles:
          recommended.enable_obstacles !== undefined ? Boolean(recommended.enable_obstacles) : true,
        allowLeveling:
          recommended.allow_leveling !== undefined ? Boolean(recommended.allow_leveling) : true,
        maxStepsWithoutFood:
          recommended.max_steps_without_food !== undefined
            ? Number(recommended.max_steps_without_food)
            : 0,
        renderEnabled: true,
      };
    }

    async requestAction() {
      const state = this.agentAPI.getState();
      const body = { state };
      if (this.wantsPerceptionDebug()) {
        body.include_debug = true;
      }
      const result = await this.postJson("/v1/act", body);
      const action = Number(result.action);
      if (this.wantsPerceptionDebug() && result.debug) {
        this.game.perceptionOverlay = {
          modelType: result.modelType || this.modelInfo?.modelType,
          features: result.debug.features,
          qValues: result.debug.q_values,
          action,
        };
        if (this.game.renderEnabled) {
          this.game.render(performance.now());
        }
      } else {
        this.game.perceptionOverlay = null;
      }
      return action;
    }

    restoreAfterEpisode(transition) {
      const terminalLabel =
        transition?.info?.terminalReasonLabel ||
        transition?.info?.terminalReason ||
        "本局结束";
      this.running = false;
      this.game.perceptionOverlay = null;
      this.game.agentControlled = false;
      this.game.setSettingsLocked(false);
      this.game.startBtn.disabled = false;
      this.game.pauseBtn.disabled = true;
      this.game.pauseBtn.textContent = "暂停";
      this.touchButtons.forEach((button) => {
        button.disabled = false;
      });
      this.game.setOverlay(true, "AI 对局结束", `${terminalLabel}。可重新开始，或再次让 AI 接管。`);
      this.updateButtons();
      this.setStatus(`AI 对局结束：${terminalLabel}`, "success");
    }

    stop(message = "AI 已停止。", kind = "success") {
      if (!this.running) {
        return;
      }
      this.running = false;
      this.runToken += 1;
      this.game.perceptionOverlay = null;
      this.setHumanControlsDisabled(false);
      this.agentAPI.resumeHumanControl();
      this.updateButtons();
      this.setStatus(message, kind);
    }

    async start() {
      if (this.running || !this.modelInfo) {
        return;
      }

      const token = this.runToken + 1;
      this.runToken = token;
      this.running = true;
      this.updateButtons();
      this.setHumanControlsDisabled(true);
      this.setStatus("AI 正在接管游戏...", "running");

      try {
        do {
          let transition = this.agentAPI.reset(this.buildAgentResetOptions());

          while (this.running && this.runToken === token && !transition.done) {
            const action = await this.requestAction();
            if (!this.running || this.runToken !== token) {
              return;
            }
            transition = this.agentAPI.step(action);
            const delayMs = this.getStepDelayMs();
            if (delayMs > 0) {
              await sleep(delayMs);
            }
          }

          if (!this.running || this.runToken !== token) {
            return;
          }

          if (!this.autoLoopCheckbox.checked) {
            this.restoreAfterEpisode(transition);
            return;
          }

          this.setStatus("本局结束，AI 即将自动重开...", "running");
          await sleep(800);
        } while (this.running && this.runToken === token);
      } catch (error) {
        this.running = false;
        this.setHumanControlsDisabled(false);
        this.agentAPI.resumeHumanControl();
        this.updateButtons();
        this.setStatus(`AI 运行失败：${this.formatError(error)}`, "error");
      }
    }
  }

  const remoteInferenceController = new RemoteInferenceController(game, agentAPI);

  window.snakeGame = game;
  window.snakeAgentAPI = agentAPI;
  window.remoteInferenceController = remoteInferenceController;
});
