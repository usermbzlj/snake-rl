"use strict";

window.SnakeConsole = window.SnakeConsole || {};

SnakeConsole.api = async function api(path, opts) {
  opts = opts || {};
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (_) {
    data = null;
  }
  if (!response.ok) {
    const detail = data && data.detail;
    const msg = detail
      ? typeof detail === "string"
        ? detail
        : JSON.stringify(detail)
      : text || String(response.status);
    throw new Error(msg);
  }
  return data;
};

SnakeConsole.connectWs = function connectWs(handlers) {
  let socket = null;
  let timer = null;
  const connect = () => {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(proto + "//" + location.host + "/ws");
    socket.onopen = () => handlers.onOpen && handlers.onOpen();
    socket.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch (_) {
        return;
      }
      handlers.onMessage && handlers.onMessage(msg);
    };
    socket.onclose = () => {
      handlers.onClose && handlers.onClose();
      timer = setTimeout(connect, 2000);
    };
  };
  connect();
  return {
    close() {
      if (timer) clearTimeout(timer);
      if (socket) socket.close();
    },
  };
};
