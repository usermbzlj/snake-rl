"use strict";

document.addEventListener("DOMContentLoaded", () => {
  if (!window.location.protocol.startsWith("http")) {
    return;
  }
  document.body.classList.add("is-http");
  const link = document.querySelector("[data-console-link]");
  if (link) {
    try {
      link.href = new URL("/", window.location.origin).toString();
    } catch (_) {
      link.href = "/";
    }
  }
});
