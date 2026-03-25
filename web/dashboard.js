const state = {
  runs: [],
  selectedRun: "",
  timer: null,
  refreshInFlight: false,
  detailAbort: null,
  lastError: "",
  lastOkAt: null,
};

const el = {
  runsList: document.getElementById("runsList"),
  runsCount: document.getElementById("runsCount"),
  keywordInput: document.getElementById("keywordInput"),
  statusFilter: document.getElementById("statusFilter"),
  refreshBtn: document.getElementById("refreshBtn"),
  refreshInterval: document.getElementById("refreshInterval"),
  pointsLimit: document.getElementById("pointsLimit"),
  chartMetric: document.getElementById("chartMetric"),
  detailTitle: document.getElementById("detailTitle"),
  detailHint: document.getElementById("detailHint"),
  episodesValue: document.getElementById("episodesValue"),
  bestAvgValue: document.getElementById("bestAvgValue"),
  lastAvgValue: document.getElementById("lastAvgValue"),
  lastEpsValue: document.getElementById("lastEpsValue"),
  configPreview: document.getElementById("configPreview"),
  summaryBlock: document.getElementById("summaryBlock"),
  summaryPreview: document.getElementById("summaryPreview"),
  checkpointBadges: document.getElementById("checkpointBadges"),
  rewardChart: document.getElementById("rewardChart"),
  statusBar: document.getElementById("statusBar"),
  statusText: document.getElementById("statusText"),
  statusTime: document.getElementById("statusTime"),
  errorBanner: document.getElementById("errorBanner"),
  retryBtn: document.getElementById("retryBtn"),
  detailLoading: document.getElementById("detailLoading"),
  detailBody: document.getElementById("detailBody"),
};

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function setStatus(kind, text) {
  el.statusText.textContent = text;
  el.statusText.classList.remove("loading", "ok");
  if (kind === "loading") el.statusText.classList.add("loading");
  if (kind === "ok") el.statusText.classList.add("ok");
}

function setError(msg) {
  state.lastError = msg || "";
  if (msg) {
    el.errorBanner.textContent = msg;
    el.errorBanner.classList.remove("hidden");
    el.retryBtn.classList.remove("hidden");
    setStatus("", "加载失败");
  } else {
    el.errorBanner.classList.add("hidden");
    el.retryBtn.classList.add("hidden");
  }
}

function touchOk() {
  state.lastOkAt = new Date();
  el.statusTime.textContent = `上次成功：${state.lastOkAt.toLocaleTimeString()}`;
  setStatus("ok", "已连接");
  setError("");
}

function anyAbortSignal(signals) {
  const c = new AbortController();
  for (const s of signals) {
    if (!s) continue;
    if (s.aborted) {
      c.abort();
      return c.signal;
    }
    s.addEventListener("abort", () => c.abort(), { once: true });
  }
  return c.signal;
}

async function fetchJson(url, options = {}) {
  const { signal, timeoutMs = 30000 } = options;
  const timeoutCtrl = new AbortController();
  const t = setTimeout(() => timeoutCtrl.abort(), timeoutMs);
  const merged = anyAbortSignal([signal, timeoutCtrl.signal].filter(Boolean));
  try {
    const resp = await fetch(url, { signal: merged });
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    const ct = resp.headers.get("content-type") || "";
    if (!ct.includes("application/json")) {
      throw new Error("响应不是 JSON");
    }
    return await resp.json();
  } finally {
    clearTimeout(t);
  }
}

function getFilteredRuns() {
  const kw = el.keywordInput.value.trim().toLowerCase();
  const status = el.statusFilter.value;
  return state.runs.filter((run) => {
    if (status && run.status !== status) return false;
    if (!kw) return true;
    const text = `${run.name} ${run.model_type}`.toLowerCase();
    return text.includes(kw);
  });
}

function renderCheckpointBadges(run) {
  if (!run || !run.name) {
    el.checkpointBadges.innerHTML = "";
    return;
  }
  const parts = [
    ["best.pt", run.has_best_checkpoint],
    ["latest.pt", run.has_latest_checkpoint],
    ["training.pt", run.has_training_state],
  ];
  el.checkpointBadges.innerHTML = parts
    .map(
      ([label, on]) =>
        `<span class="badge ${on ? "on" : ""}">${escapeHtml(label)}${on ? " ✓" : ""}</span>`,
    )
    .join("");
}

function clearRunDetail(hint = "请选择左侧运行") {
  el.detailTitle.textContent = "运行详情";
  el.detailHint.textContent = hint;
  el.episodesValue.textContent = "-";
  el.bestAvgValue.textContent = "-";
  el.lastAvgValue.textContent = "-";
  el.lastEpsValue.textContent = "-";
  el.summaryBlock.classList.add("hidden");
  el.summaryPreview.textContent = "";
  el.configPreview.textContent = "";
  el.detailLoading.classList.add("hidden");
  renderCheckpointBadges(null);
  drawRewardChart([], el.chartMetric.value);
}

function renderRuns() {
  const runs = getFilteredRuns();
  el.runsCount.textContent = `${runs.length} 条`;
  el.runsList.textContent = "";
  if (!runs.length) {
    const empty = document.createElement("div");
    empty.className = "run-item";
    empty.textContent = "暂无运行数据";
    el.runsList.appendChild(empty);
    return;
  }
  runs.forEach((run) => {
    const item = document.createElement("div");
    item.className = `run-item ${run.name === state.selectedRun ? "active" : ""}`;
    const top = document.createElement("div");
    top.className = "run-top";
    const nameEl = document.createElement("span");
    nameEl.className = "run-name";
    nameEl.textContent = run.name;
    const stEl = document.createElement("span");
    stEl.textContent = run.status || "-";
    top.appendChild(nameEl);
    top.appendChild(stEl);
    const meta1 = document.createElement("div");
    meta1.className = "run-meta";
    meta1.innerHTML = [
      `<span>模型: ${escapeHtml(run.model_type || "-")}</span>`,
      `<span>局数: ${escapeHtml(String(run.episodes ?? "-"))}</span>`,
      `<span>best: ${escapeHtml(fmtNumber(run.best_avg_reward))}</span>`,
    ].join("");
    const meta2 = document.createElement("div");
    meta2.className = "run-meta";
    const span = document.createElement("span");
    span.textContent = run.updated_at || "-";
    meta2.appendChild(span);
    item.appendChild(top);
    item.appendChild(meta1);
    item.appendChild(meta2);
    item.addEventListener("click", () => {
      state.selectedRun = run.name;
      syncRunToUrl(run.name);
      renderRuns();
      void loadRunDetail();
    });
    el.runsList.appendChild(item);
  });
}

function syncRunToUrl(name) {
  const url = new URL(window.location.href);
  if (name) url.searchParams.set("run", name);
  else url.searchParams.delete("run");
  window.history.replaceState({}, "", url.toString());
}

function readRunFromUrl() {
  const q = new URLSearchParams(window.location.search).get("run");
  return q ? q.trim() : "";
}

function resizeCanvas(canvas) {
  const wrap = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const cssW = Math.max(320, wrap.clientWidth || 800);
  const cssH = 360;
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  const ctx = canvas.getContext("2d");
  if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: cssW, height: cssH };
}

function drawRewardChart(points, metricKey) {
  const canvas = el.rewardChart;
  const { ctx, width, height } = resizeCanvas(canvas);
  if (!ctx) return;

  const padLeft = 54;
  const padBottom = 36;
  const padTop = 20;
  const padRight = 18;
  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, width, height);

  if (!points.length) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px sans-serif";
    ctx.fillText("暂无曲线数据", padLeft, padTop + 26);
    return;
  }

  const ys = points.map((p) => Number(p[metricKey] ?? p.avg_reward));
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const yRange = Math.max(1e-6, maxY - minY);

  const ep0 = Number(points[0].episode);
  const ep1 = Number(points[points.length - 1].episode);
  const epSpan = Math.max(1, ep1 - ep0);

  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, padTop);
  ctx.lineTo(padLeft, padTop + chartH);
  ctx.lineTo(padLeft + chartW, padTop + chartH);
  ctx.stroke();

  let lastStage = points[0].stage_index;
  points.forEach((p) => {
    const st = p.stage_index;
    if (st !== undefined && st !== null && st !== lastStage) {
      const ep = Number(p.episode);
      const x = padLeft + ((ep - ep0) / epSpan) * chartW;
      ctx.strokeStyle = "#475569";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + chartH);
      ctx.stroke();
      ctx.setLineDash([]);
      lastStage = st;
    }
  });

  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((p, idx) => {
    const ep = Number(p.episode);
    const x = padLeft + ((ep - ep0) / epSpan) * chartW;
    const yv = Number(p[metricKey] ?? p.avg_reward);
    const yNorm = (yv - minY) / yRange;
    const y = padTop + (1 - yNorm) * chartH;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "#94a3b8";
  ctx.font = "12px sans-serif";
  ctx.fillText(`${metricKey} max ${maxY.toFixed(3)}`, padLeft, padTop - 4);
  ctx.fillText(`min ${minY.toFixed(3)}`, padLeft + chartW * 0.35, height - 10);
  ctx.fillText(`ep ${ep0} → ${ep1}`, padLeft, height - 10);
}

async function loadRuns(signal) {
  const data = await fetchJson("/api/runs", { signal });
  state.runs = data.runs || [];
  const fromUrl = readRunFromUrl();
  if (fromUrl && state.runs.some((r) => r.name === fromUrl)) {
    state.selectedRun = fromUrl;
  } else if (!state.selectedRun && state.runs.length) {
    state.selectedRun = state.runs[0].name;
  } else if (state.selectedRun && !state.runs.find((r) => r.name === state.selectedRun)) {
    state.selectedRun = state.runs.length ? state.runs[0].name : "";
  }
  syncRunToUrl(state.selectedRun);
  renderRuns();
}

async function loadRunDetail() {
  if (state.detailAbort) {
    state.detailAbort.abort();
    state.detailAbort = null;
  }
  if (!state.selectedRun) {
    clearRunDetail("请选择左侧运行");
    return;
  }

  const ctrl = new AbortController();
  state.detailAbort = ctrl;

  el.detailLoading.classList.remove("hidden");
  try {
    const limit = Number(el.pointsLimit.value) || 300;
    const data = await fetchJson(
      `/api/runs/${encodeURIComponent(state.selectedRun)}?limit=${limit}`,
      { signal: ctrl.signal },
    );
    if (ctrl.signal.aborted) return;

    const run = data.run || {};
    const latest = data.latest || {};
    const points = data.episodes || [];
    const summary = data.summary && typeof data.summary === "object" ? data.summary : null;

    el.detailTitle.textContent = `运行详情：${run.name || state.selectedRun}`;
    el.detailHint.textContent = `${run.status || "-"} | 模型 ${run.model_type || "-"}`;
    el.episodesValue.textContent = String(run.episodes ?? "-");
    el.bestAvgValue.textContent = fmtNumber(run.best_avg_reward);
    el.lastAvgValue.textContent = fmtNumber(latest.avg_reward);
    el.lastEpsValue.textContent = fmtNumber(latest.epsilon);

    if (summary && Object.keys(summary).length) {
      el.summaryBlock.classList.remove("hidden");
      el.summaryPreview.textContent = JSON.stringify(summary, null, 2);
    } else {
      el.summaryBlock.classList.add("hidden");
      el.summaryPreview.textContent = "";
    }

    el.configPreview.textContent = JSON.stringify(data.config || {}, null, 2);
    renderCheckpointBadges(run);
    drawRewardChart(points, el.chartMetric.value);
  } catch (err) {
    if (err.name === "AbortError") return;
    clearRunDetail("详情加载失败");
    throw err;
  } finally {
    if (!ctrl.signal.aborted) el.detailLoading.classList.add("hidden");
    if (state.detailAbort === ctrl) state.detailAbort = null;
  }
}

async function refreshAll() {
  if (state.refreshInFlight) return;
  state.refreshInFlight = true;
  setStatus("loading", "刷新中…");
  try {
    await loadRuns();
    await loadRunDetail();
    touchOk();
  } catch (err) {
    const msg = err.name === "AbortError" ? "请求已取消" : String(err.message || err);
    setError(`加载失败：${msg}`);
  } finally {
    state.refreshInFlight = false;
  }
}

function applyTimer() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  const ms = Number(el.refreshInterval.value);
  if (ms > 0) {
    state.timer = setInterval(() => {
      void refreshAll();
    }, ms);
  }
}

el.keywordInput.addEventListener("input", renderRuns);
el.statusFilter.addEventListener("change", renderRuns);
el.refreshBtn.addEventListener("click", () => {
  void refreshAll();
});
el.retryBtn.addEventListener("click", () => {
  void refreshAll();
});
el.refreshInterval.addEventListener("change", applyTimer);
el.pointsLimit.addEventListener("change", () => {
  void loadRunDetail();
});
el.chartMetric.addEventListener("change", () => {
  void loadRunDetail();
});

window.addEventListener("resize", () => {
  if (state.selectedRun) void loadRunDetail();
});

const initialRun = readRunFromUrl();
if (initialRun) state.selectedRun = initialRun;

void refreshAll();
applyTimer();
