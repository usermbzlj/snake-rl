"use strict";

// ─── 状态面板工具 ─────────────────────────────────────────────────────────────

const STEPS = ["engine", "train", "infer", "model", "play"];

function setStep(id, state) {
  // state: "idle" | "active" | "done" | "error"
  const el = document.getElementById("step-" + id);
  if (!el) return;
  el.className = "step " + (state === "idle" ? "" : state);
}

function setStatus(msg, kind) {
  // kind: "ok" | "err" | "info" | "" (default muted)
  const el = document.getElementById("previewStatus");
  if (!el) return;
  el.textContent = msg;
  el.className = kind || "";
}

function showAction(label, callback) {
  const area = document.getElementById("previewActions");
  if (!area) return;
  area.innerHTML = "";
  const btn = document.createElement("button");
  btn.className = "btn btn-sec";
  btn.textContent = label;
  btn.onclick = callback;
  area.appendChild(btn);
  area.style.display = "flex";
}

function hideActions() {
  const area = document.getElementById("previewActions");
  if (area) area.style.display = "none";
}

// ─── 网络工具 ─────────────────────────────────────────────────────────────────

function fetchWithTimeout(url, options, timeoutMs) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  return fetch(url, { ...options, signal: ctrl.signal }).finally(() =>
    clearTimeout(timer)
  );
}

async function apiGet(path, timeoutMs) {
  const res = await fetchWithTimeout(path, {}, timeoutMs || 8000);
  if (!res.ok) {
    let msg;
    try { msg = (await res.json()).detail || (await res.text()); } catch { msg = String(res.status); }
    throw new Error(msg);
  }
  return res.json();
}

async function apiPost(base, path, payload, timeoutMs) {
  const url = base.replace(/\/+$/, "") + path;
  const res = await fetchWithTimeout(
    url,
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) },
    timeoutMs || 10000
  );
  let data;
  try { data = await res.json(); } catch { data = {}; }
  if (!res.ok) {
    throw new Error(data.detail || data.error || `请求失败 (${res.status})`);
  }
  return data;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

// ─── 等待 game.js 全局对象（带超时）──────────────────────────────────────────

function waitForGlobals(timeoutMs) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + (timeoutMs || 8000);
    const tick = () => {
      if (window.snakeGame && window.remoteInferenceController) {
        resolve();
        return;
      }
      if (Date.now() >= deadline) {
        reject(new Error("游戏引擎初始化超时，请检查控制台错误或刷新页面"));
        return;
      }
      requestAnimationFrame(tick);
    };
    tick();
  });
}

// ─── 主流程 ───────────────────────────────────────────────────────────────────

window.addEventListener("load", async () => {
  STEPS.forEach((s) => setStep(s, "idle"));

  // 步骤 1：等待 game.js 全局对象
  setStep("engine", "active");
  setStatus("正在加载游戏引擎…");
  try {
    await waitForGlobals(8000);
  } catch (err) {
    setStep("engine", "error");
    setStatus("游戏引擎启动失败：" + err.message, "err");
    showAction("刷新重试", () => location.reload());
    return;
  }
  setStep("engine", "done");

  const game = window.snakeGame;
  const api = window.snakeAgentAPI;
  const ric = window.remoteInferenceController;

  // 主循环：反复拉取最新 checkpoint 并演示
  let lastMtime = 0;
  let loadedCheckpoint = "";
  let inferBase = "";

  const runLoop = async () => {
    while (true) {
      // 步骤 2：连接训练接口
      setStep("train", "active");
      setStatus("正在连接训练接口…");
      hideActions();
      let info;
      while (true) {
        try {
          info = await apiGet("/api/train/live-preview-info", 6000);
          break;
        } catch (err) {
          const msg = String(err.message || err);
          const isNoTrain = msg.includes("没有训练任务") || msg.includes("409");
          setStep("train", "error");
          if (isNoTrain) {
            setStatus("当前没有正在运行的训练任务。请先在控制台启动训练，然后点「训练实况」。", "err");
          } else {
            setStatus(`无法连接训练接口：${msg}。5 秒后重试…`, "err");
          }
          showAction("立刻重试", () => {
            hideActions();
            setStep("train", "active");
            setStatus("重试中…");
          });
          await sleep(5000);
          hideActions();
        }
      }
      setStep("train", "done");

      inferBase = `http://127.0.0.1:${info.inference_port}`;
      // 同步到隐藏 input（game.js 可能读取）
      const urlInput = document.getElementById("inferenceServerUrl");
      if (urlInput) urlInput.value = inferBase;

      // 步骤 3：确保推理服务运行
      setStep("infer", "active");
      setStatus("正在准备推理服务…");
      let inferReady = false;
      for (let attempt = 0; attempt < 3 && !inferReady; attempt++) {
        try {
          await apiPost(location.origin, "/api/infer/ensure-running", {}, 8000);
          inferReady = true;
        } catch (err) {
          if (attempt < 2) {
            setStatus(`推理服务启动中… (${err.message})，稍后重试`);
            await sleep(2000);
          } else {
            setStep("infer", "error");
            setStatus(`推理服务无法启动：${err.message}`, "err");
            showAction("重试", () => {
              hideActions();
              void runLoop();
            });
            return;
          }
        }
      }
      // 等待服务真正就绪（最多 12 秒）
      let inferHealthy = false;
      for (let i = 0; i < 24 && !inferHealthy; i++) {
        try {
          const h = await fetchWithTimeout(`${inferBase}/health`, {}, 2000);
          if (h.ok) { inferHealthy = true; }
        } catch { /* not ready yet */ }
        if (!inferHealthy) await sleep(500);
      }
      if (!inferHealthy) {
        setStep("infer", "error");
        setStatus(`推理服务未在端口 ${info.inference_port} 响应，请检查防火墙或端口占用。`, "err");
        showAction("重试", () => { hideActions(); void runLoop(); });
        return;
      }
      setStep("infer", "done");

      // 步骤 4：加载/更新模型
      const needLoad = info.checkpoint !== loadedCheckpoint || info.mtime > lastMtime;
      if (needLoad) {
        setStep("model", "active");
        setStatus(`加载模型：${info.run_name || ""}  ${info.model_type || ""}`);
        try {
          const modelInfo = await apiPost(inferBase, "/v1/load", { checkpoint: info.checkpoint }, 15000);
          ric.modelInfo = modelInfo;
          ric.applyRecommendedUI(modelInfo);
          loadedCheckpoint = info.checkpoint;
          lastMtime = info.mtime;
        } catch (err) {
          setStep("model", "error");
          setStatus(`模型加载失败：${err.message}`, "err");
          showAction("3 秒后重试", async () => {
            hideActions();
            await sleep(3000);
            void runLoop();
          });
          await sleep(3000);
          continue;
        }
      }
      setStep("model", "done");

      if (!ric.modelInfo) {
        await sleep(500);
        continue;
      }

      // 步骤 5：演示
      setStep("play", "active");
      setStatus(
        `演示中 · ${info.run_name || ""}  [${info.model_type || ric.modelInfo.modelType || "?"}]`,
        "ok"
      );

      game.agentControlled = true;
      game.setSettingsLocked(true);
      game.setOverlay(false);

      let transition = api.reset(ric.buildAgentResetOptions());

      while (!transition.done) {
        const state = api.getState();
        let result;
        try {
          result = await apiPost(inferBase, "/v1/act", { state, include_debug: true }, 4000);
        } catch (err) {
          setStep("play", "error");
          setStatus(`推理中断：${err.message}`, "err");
          game.agentControlled = false;
          game.setSettingsLocked(false);
          await sleep(1500);
          break;
        }
        const action = Number(result.action);
        if (result.debug) {
          game.perceptionOverlay = {
            modelType: result.modelType || ric.modelInfo?.modelType,
            features: result.debug.features,
            qValues: result.debug.q_values,
            action,
          };
        } else {
          game.perceptionOverlay = null;
        }
        if (game.renderEnabled) {
          game.render(performance.now());
        }
        transition = api.step(action);
        await sleep(45);
      }

      // 局结束
      game.perceptionOverlay = null;
      game.agentControlled = false;
      game.setSettingsLocked(false);
      if (game.renderEnabled) game.render(performance.now());

      setStep("play", "idle");
      setStep("model", "idle");
      setStatus("本局结束，检查是否有更新的 checkpoint…");
      await sleep(600);

      // 回到顶部重新拉取 live-preview-info
    }
  };

  void runLoop();
});
