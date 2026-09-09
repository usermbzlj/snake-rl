"use strict";

const { createApp, ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } = Vue;
const api = SnakeConsole.api;

createApp({
  setup() {
    const tab = ref("home");
    const subTab = ref("basic");
    const schemes = ref({});
    const formMeta = ref({ sections: [] });
    const config = reactive({});
    const configPath = ref("");
    const fields = SnakeConsole.createFieldAccessors(config);
    const ui = reactive({
      scheme: "custom",
      parallel: false,
      parallel_workers: 4,
      parallel_sync_interval: 512,
      custom_config_path: "",
      monitor_port: 6006,
      inference_port: 8765,
    });
    const status = reactive({
      training: false,
      monitor: false,
      infer: false,
      estimating: false,
      training_run: null,
    });
    const progress = reactive({
      episode: 0,
      total: 0,
      percent: 0,
      avg_reward: null,
      epsilon: null,
      stage: "-",
      started_at: null,
      training_run: null,
    });
    const logs = ref([]);
    const logEl = ref(null);
    const logFilter = ref("all");
    const logAutoScroll = ref(true);
    const toasts = ref([]);
    const confirmDlg = ref(null);
    const wsConnected = ref(false);
    const runs = ref([]);
    const selectedRun = ref(null);
    const runFilter = reactive({ keyword: "", status: "全部" });
    const trainMode = ref("new");
    const resumeRun = ref("");
    const resumeArtifacts = reactive({
      state_file: null,
      checkpoints: [],
      model_type: null,
      _loaded: false,
    });
    const warmCheckpoint = ref("");
    const warmInheritGlobalStep = ref(true);
    const warmLockedKeys = new Set(["model_type", "local_patch_size"]);
    const extraEpisodes = ref(10000);
    const lanIp = ref("");
    const configDirty = ref(false);
    const ignoreDirty = ref(true);
    const activePreset = ref("");
    const nowTick = ref(Date.now());
    const presets = SnakeConsole.PRESETS;
    let socketHandle = null;
    let toastSeq = 0;
    let clockId = 0;

    const formTabs = computed(() => {
      const seen = new Map();
      for (const sec of formMeta.value.sections || []) {
        if (!seen.has(sec.tab)) seen.set(sec.tab, sec.tabTitle);
      }
      return [...seen.entries()].map(([id, tabTitle]) => ({ tab: id, tabTitle }));
    });

    const activeFormSections = computed(() => {
      const out = [];
      let idx = 0;
      for (const sec of formMeta.value.sections || []) {
        if (sec.tab !== subTab.value) continue;
        for (const g of sec.groups || []) {
          out.push({ tab: sec.tab, groupIdx: idx++, group: g });
        }
      }
      return out;
    });

    const essentialFields = computed(() => {
      const out = [];
      const seen = new Set();
      for (const sec of formMeta.value.sections || []) {
        for (const g of sec.groups || []) {
          for (const f of g.fields || []) {
            if ((f.essential || SnakeConsole.ESSENTIAL_KEYS && SnakeConsole.ESSENTIAL_KEYS.has(f.key)) && !seen.has(f.key)) {
              seen.add(f.key);
              out.push(f);
            }
          }
        }
      }
      return out;
    });

    const filteredRuns = computed(() => {
      let list = runs.value;
      const kw = runFilter.keyword.trim().toLowerCase();
      if (kw) list = list.filter((r) => (r.name + " " + r.model).toLowerCase().includes(kw));
      if (runFilter.status !== "全部") list = list.filter((r) => r.status === runFilter.status);
      return list;
    });

    const recentRuns = computed(() => runs.value.slice(0, 5));
    const latestDemoRun = computed(() => runs.value.find((r) => SnakeConsole.canDemo(r)) || null);

    const filteredLogs = computed(() => {
      if (logFilter.value === "all") return logs.value;
      return logs.value.filter((line) => SnakeConsole.logKind(line) === logFilter.value);
    });

    const logHtml = computed(() =>
      filteredLogs.value
        .map((line) => {
          const kind = SnakeConsole.logKind(line);
          const cls = kind === "error" ? "log-error" : kind === "progress" ? "log-progress" : "";
          return `<div class="${cls}">${escapeHtml(line)}</div>`;
        })
        .join("")
    );

    const progressLabel = computed(() => {
      const total = progress.total || "?";
      return `${progress.episode} / ${total}（${Number(progress.percent || 0).toFixed(1)}%）`;
    });

    const etaLabel = computed(() => {
      nowTick.value;
      const started = Number(progress.started_at || 0);
      const pct = Number(progress.percent || 0);
      if (!status.training || !started || pct < 1.5) return "";
      const elapsed = Date.now() / 1000 - started;
      const remain = elapsed * (100 - pct) / pct;
      return "大约还要 " + SnakeConsole.formatDuration(remain);
    });

    const elapsedLabel = computed(() => {
      nowTick.value;
      const started = Number(progress.started_at || 0);
      if (!started) return "";
      return SnakeConsole.formatDuration(Date.now() / 1000 - started);
    });

    const playUrl = computed(() => location.origin + "/play/");
    const lanPlayUrl = computed(() => {
      const host = lanIp.value || location.hostname;
      return `http://${host}:${location.port || "7860"}/play/`;
    });
    const monitorUrl = computed(() => `http://127.0.0.1:${ui.monitor_port}/`);

    function escapeHtml(text) {
      return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function fmt(value) {
      return SnakeConsole.formatMetric(value);
    }

    function canDemo(run) {
      return SnakeConsole.canDemo(run);
    }

    function canResume(run) {
      return SnakeConsole.canResume(run);
    }

    function toast(msg, kind) {
      const id = ++toastSeq;
      toasts.value.push({ id, msg, kind: kind || "ok" });
      setTimeout(() => {
        toasts.value = toasts.value.filter((t) => t.id !== id);
      }, kind === "err" ? 7000 : 4000);
    }

    function askConfirm({ title, text, danger, okLabel }) {
      return new Promise((resolve) => {
        confirmDlg.value = {
          title,
          text,
          danger: !!danger,
          okLabel: okLabel || "确认",
          resolve(ok) {
            confirmDlg.value = null;
            resolve(!!ok);
          },
        };
      });
    }

    function applyHash() {
      const hash = (location.hash || "").replace("#", "");
      if (["home", "train", "runs", "svc", "doc"].includes(hash) && tab.value !== hash) {
        tab.value = hash;
        if (hash === "runs" || hash === "home") loadRuns();
      }
    }

    function onHashChange() {
      applyHash();
    }

    function setTab(next) {
      tab.value = next;
      if (next === "runs" || next === "home") loadRuns();
      if (location.hash !== "#" + next) history.replaceState(null, "", "#" + next);
    }

    async function copy(text) {
      try {
        await SnakeConsole.copyText(text);
        toast("已复制", "ok");
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    async function loadInitial() {
      const [sc, fm, st] = await Promise.all([
        api("/api/schemes"),
        api("/api/form-meta"),
        api("/api/state"),
      ]);
      schemes.value = sc;
      formMeta.value = fm;
      Object.assign(ui, {
        scheme: st.scheme,
        parallel: st.parallel,
        parallel_workers: st.parallel_workers,
        parallel_sync_interval: st.parallel_sync_interval,
        custom_config_path: st.custom_config_path,
        monitor_port: st.monitor_port,
        inference_port: st.inference_port,
      });
      Object.assign(status, {
        training: st.training,
        monitor: st.monitor,
        infer: st.infer,
        estimating: st.estimating,
        training_run: st.training_run || null,
      });
      if (st.progress) Object.assign(progress, st.progress);
      lanIp.value = st.lan_ip || "";
      await loadConfig();
      await loadRuns();
      ignoreDirty.value = false;
    }

    async function loadConfig() {
      try {
        ignoreDirty.value = true;
        const data = await api("/api/config");
        configPath.value = data.path;
        Object.keys(config).forEach((k) => delete config[k]);
        Object.assign(config, data.config);
        SnakeConsole.ensureConfigShape(config);
        ui.custom_config_path = data.path;
        configDirty.value = false;
      } catch (e) {
        toast(String(e.message || e), "err");
      } finally {
        nextTick(() => {
          ignoreDirty.value = false;
        });
      }
    }

    async function saveConfig() {
      try {
        SnakeConsole.ensureConfigShape(config);
        await api("/api/config", {
          method: "POST",
          body: JSON.stringify({ config, path: ui.custom_config_path || configPath.value }),
        });
        toast("已保存并校验通过", "ok");
        await loadConfig();
      } catch (e) {
        toast(String(e.message || e), "err");
        throw e;
      }
    }

    async function saveUi() {
      try {
        await api("/api/ui-settings", { method: "POST", body: JSON.stringify({ ...ui }) });
      } catch (_) {
        /* ignore */
      }
    }

    function applyPreset(id) {
      const preset = presets.find((p) => p.id === id);
      if (!preset) return;
      trainMode.value = "new";
      ui.scheme = preset.scheme;
      if (preset.scheme === "custom") {
        SnakeConsole.ensureConfigShape(config);
        preset.apply(config);
        configDirty.value = true;
      }
      activePreset.value = id;
      saveUi();
      toast("已套用「" + preset.title + "」，检查后即可开始", "ok");
      setTab("train");
    }

    const canStartTrain = computed(() => {
      if (trainMode.value === "new") return true;
      if (!resumeRun.value || !resumeArtifacts._loaded) return false;
      if (trainMode.value === "resume") return !!resumeArtifacts.state_file;
      if (trainMode.value === "warm") {
        return resumeArtifacts.checkpoints.length > 0 && !!warmCheckpoint.value;
      }
      return false;
    });

    async function loadArtifacts() {
      resumeArtifacts.state_file = null;
      resumeArtifacts.checkpoints = [];
      resumeArtifacts.model_type = null;
      resumeArtifacts._loaded = false;
      warmCheckpoint.value = "";
      if (!resumeRun.value) return;
      try {
        const data = await api("/api/runs/" + encodeURIComponent(resumeRun.value) + "/artifacts");
        resumeArtifacts.state_file = data.state_file;
        resumeArtifacts.checkpoints = data.checkpoints || [];
        resumeArtifacts.model_type = data.model_type;
        resumeArtifacts._loaded = true;
        if (resumeArtifacts.checkpoints.length) {
          const best = resumeArtifacts.checkpoints.find((c) => c.name === "best.pt");
          warmCheckpoint.value = best ? best.path : resumeArtifacts.checkpoints[0].path;
        }
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    async function quickResume(run) {
      if (!run) return;
      trainMode.value = "resume";
      resumeRun.value = run.name;
      await loadArtifacts();
      if (!resumeArtifacts.state_file) {
        toast(run.name + " 没有完整恢复文件，已自动切换为热加载模式", "err");
        trainMode.value = "warm";
      }
      setTab("train");
    }

    async function quickWarm(run) {
      if (!run) return;
      trainMode.value = "warm";
      resumeRun.value = run.name;
      await loadArtifacts();
      if (resumeArtifacts.checkpoints.length === 0) {
        toast(run.name + " 没有可用的权重文件", "err");
        return;
      }
      setTab("train");
    }

    async function startTrain() {
      try {
        await saveUi();
        if (ui.scheme === "custom" && trainMode.value !== "resume") await saveConfig();
        const body = {};
        if (trainMode.value === "resume" && resumeArtifacts.state_file) {
          body.resume_state = resumeArtifacts.state_file;
          if (extraEpisodes.value > 0) body.extra_episodes = extraEpisodes.value;
        } else if (trainMode.value === "warm" && warmCheckpoint.value) {
          body.warm_start = warmCheckpoint.value;
          if (!warmInheritGlobalStep.value) body.warm_start_inherit_global_step = false;
        }
        await api("/api/train/start", { method: "POST", body: JSON.stringify(body) });
        const modeLabel =
          trainMode.value === "resume" ? "恢复训练" : trainMode.value === "warm" ? "热加载训练" : "训练";
        toast(modeLabel + "已启动", "ok");
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    async function stopTrain() {
      const ok = await askConfirm({
        title: "停止训练？",
        text: "当前进度会保留在 runs/ 里，之后可以原样继续或热加载。",
        danger: true,
        okLabel: "停止",
      });
      if (!ok) return;
      await api("/api/train/stop", { method: "POST", body: "{}" });
    }

    async function startEstimate() {
      try {
        await saveUi();
        await api("/api/estimate/start", { method: "POST", body: "{}" });
        logFilter.value = "all";
        toast("估算已开始，结果写在下方日志", "ok");
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    async function stopEstimate() {
      try {
        await api("/api/estimate/stop", { method: "POST", body: "{}" });
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    async function loadRuns() {
      try {
        runs.value = await api("/api/runs");
        if (selectedRun.value) {
          selectedRun.value = runs.value.find((r) => r.name === selectedRun.value.name) || selectedRun.value;
        }
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    function openRun(run) {
      selectedRun.value = run;
      setTab("runs");
    }

    async function deleteSelected() {
      if (!selectedRun.value) return;
      const ok = await askConfirm({
        title: "删除这次训练？",
        text: "将删除 " + selectedRun.value.name + " 的全部文件，不可恢复。",
        danger: true,
        okLabel: "删除",
      });
      if (!ok) return;
      try {
        await api("/api/runs/" + encodeURIComponent(selectedRun.value.name), { method: "DELETE" });
        selectedRun.value = null;
        await loadRuns();
        toast("已删除", "ok");
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    async function clearAllRuns() {
      const ok = await askConfirm({
        title: "清空全部训练记录？",
        text: "会删掉整个 runs/ 目录，不可恢复。",
        danger: true,
        okLabel: "全部清空",
      });
      if (!ok) return;
      try {
        await api("/api/runs/clear-all", { method: "POST", body: JSON.stringify({ confirm: true }) });
        selectedRun.value = null;
        await loadRuns();
        toast("已清空", "ok");
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    async function launchDemo() {
      if (!selectedRun.value) return;
      try {
        const r = await api("/api/infer/start", {
          method: "POST",
          body: JSON.stringify({ run_name: selectedRun.value.name }),
        });
        if (r.already) {
          const go = await askConfirm({
            title: "推理服务已在运行",
            text: "端口上已有推理服务。要直接打开游戏页吗？",
            okLabel: "打开游戏",
          });
          if (!go) return;
        }
        const base = r.play_url || "/play/";
        const u = new URL(base, location.origin);
        if (r.checkpoint) u.searchParams.set("checkpoint", r.checkpoint);
        u.searchParams.set("autostart", "1");
        window.open(u.toString(), "_blank");
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    async function demoLatest() {
      if (!latestDemoRun.value) {
        toast("还没有可演示的模型", "err");
        return;
      }
      selectedRun.value = latestDemoRun.value;
      await launchDemo();
    }

    function openTrainPreview() {
      const name = (selectedRun.value && selectedRun.value.name) || progress.training_run || status.training_run || "";
      const q = name ? "?run=" + encodeURIComponent(name) : "";
      window.open(location.origin + "/play/train_preview.html" + q, "_blank");
    }

    async function openMonitorSelected() {
      if (!selectedRun.value) {
        toast("请先选择运行", "err");
        return;
      }
      try {
        const r = await api("/api/monitor/start", {
          method: "POST",
          body: JSON.stringify({ run: selectedRun.value.name }),
        });
        if (r.url) window.open(r.url, "_blank");
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    function openMonitorUrl() {
      window.open(monitorUrl.value, "_blank");
    }

    async function startMonitor() {
      try {
        const r = await api("/api/monitor/start", { method: "POST", body: "{}" });
        if (r.url) window.open(r.url, "_blank");
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    async function stopMonitor() {
      await api("/api/monitor/stop", { method: "POST", body: "{}" });
    }

    async function stopInfer() {
      await api("/api/infer/stop", { method: "POST", body: "{}" });
      toast("已请求停止推理服务", "ok");
    }

    function openGame() {
      window.open(playUrl.value, "_blank");
    }

    async function revealRun() {
      if (!selectedRun.value) return;
      try {
        await api("/api/runs/" + encodeURIComponent(selectedRun.value.name) + "/reveal", {
          method: "POST",
          body: "{}",
        });
        toast("已在文件管理器中打开", "ok");
      } catch (e) {
        toast(String(e.message || e), "err");
      }
    }

    function fieldLocked(key) {
      return trainMode.value === "resume" || (trainMode.value === "warm" && warmLockedKeys.has(key));
    }

    function onKeydown(event) {
      if (SnakeConsole.isTypingTarget(event.target) || confirmDlg.value) return;
      const map = { "1": "home", "2": "train", "3": "runs", "4": "svc", "5": "doc" };
      if (map[event.key]) {
        event.preventDefault();
        setTab(map[event.key]);
      }
    }

    onMounted(() => {
      const hash = (location.hash || "").replace("#", "");
      if (["home", "train", "runs", "svc", "doc"].includes(hash)) tab.value = hash;
      loadInitial().catch((e) => toast(String(e.message || e), "err"));
      clockId = setInterval(() => {
        nowTick.value = Date.now();
      }, 1000);
      document.addEventListener("keydown", onKeydown);
      window.addEventListener("hashchange", onHashChange);
      socketHandle = SnakeConsole.connectWs({
        onOpen() {
          wsConnected.value = true;
        },
        onClose() {
          wsConnected.value = false;
        },
        onMessage(msg) {
          if (msg.type === "log") {
            logs.value.push(msg.text);
            if (logs.value.length > 5000) logs.value.splice(0, logs.value.length - 5000);
            if (/训练完成/.test(msg.text)) toast("训练完成", "ok");
            nextTick(() => {
              if (logAutoScroll.value && logEl.value) logEl.value.scrollTop = logEl.value.scrollHeight;
            });
          } else if (msg.type === "progress") {
            Object.assign(progress, msg);
          } else if (msg.type === "status") {
            Object.assign(status, msg);
          } else if (msg.type === "runs_reload") {
            loadRuns();
          }
        },
      });
    });

    onUnmounted(() => {
      if (socketHandle) socketHandle.close();
      if (clockId) clearInterval(clockId);
      document.removeEventListener("keydown", onKeydown);
      window.removeEventListener("hashchange", onHashChange);
    });

    watch(() => ui.scheme, () => saveUi());
    watch(trainMode, (val) => {
      if (val !== "new") loadRuns();
    });
    watch(tab, (val) => {
      if (location.hash !== "#" + val) history.replaceState(null, "", "#" + val);
    });
    watch(config, () => {
      if (!ignoreDirty.value) configDirty.value = true;
    }, { deep: true });

    return {
      tab, subTab, schemes, formMeta, formTabs, activeFormSections, essentialFields, config, ui, status, progress,
      logs, logEl, logHtml, logFilter, logAutoScroll, toasts, confirmDlg, wsConnected, runs, selectedRun,
      runFilter, filteredRuns, recentRuns, latestDemoRun, trainMode, resumeRun, resumeArtifacts, warmCheckpoint,
      warmInheritGlobalStep, extraEpisodes, canStartTrain, progressLabel, etaLabel, elapsedLabel, playUrl,
      lanPlayUrl, monitorUrl, configDirty, activePreset, presets, lanIp,
      getField: fields.getField,
      setField: fields.setField,
      fieldStr: fields.fieldStr,
      setFieldCoerce: fields.setFieldCoerce,
      fieldLocked, setTab, loadConfig, saveConfig, saveUi, applyPreset, fmt, canDemo, canResume, copy,
      startTrain, stopTrain, startEstimate, stopEstimate, loadRuns, loadArtifacts, openRun,
      deleteSelected, clearAllRuns, launchDemo, demoLatest, openTrainPreview, quickResume, quickWarm,
      openMonitorSelected, openMonitorUrl, startMonitor, stopMonitor, stopInfer, openGame, revealRun,
    };
  },
}).mount("#app");
