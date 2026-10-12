// Shared helpers of the ESCounting UI pages (plain script, Vue 3 global build, no Node build step).
"use strict";
const ESC = (() => {
  const KEY_NAME = "escount.apiKey";

  const apiKey = () => sessionStorage.getItem(KEY_NAME) || "";
  const setApiKey = (value) => sessionStorage.setItem(KEY_NAME, value || "");

  // fetch wrapper: adds X-API-Key when set, JSON bodies, readable errors ("<status>: <detail>")
  async function api(path, options = {}) {
    const headers = Object.assign({}, options.headers || {});
    if (apiKey()) headers["X-API-Key"] = apiKey();
    let body = options.body;
    if (options.json !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(options.json);
    }
    const response = await fetch(path, { method: options.method || (body ? "POST" : "GET"), headers, body });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const data = await response.json();
        detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
      } catch (ignored) { /* not JSON */ }
      throw new Error(`${response.status}: ${detail}`);
    }
    return options.raw ? response : response.json();
  }

  // images are fetched (not <img src>) so that the API key header is sent too
  async function bitmap(path) {
    const response = await api(path, { raw: true });
    return createImageBitmap(await response.blob());
  }

  function debounce(fn, ms) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), ms);
    };
  }

  return { apiKey, setApiKey, api, bitmap, debounce };
})();
