// Results page (PDF p13-14): signals + MACD + KD charts per ROI, live N_pred via POST /preview.
"use strict";
const C = { text: "#d1d4dc", muted: "#8a93a6", grid: "#2a3142", signal: "#4c8dff", high: "#ef5350", low: "#26a69a",
            dif: "#f0b90b", dea: "#4c8dff", up: "#26a69a", down: "#ef5350", rsv: "#8a93a6", k: "#f0b90b", d: "#d16ff0" };
const charts = {};                      // ECharts instances (kept out of Vue reactivity)

function baseOption(title, rows) {
  return {
    backgroundColor: "transparent", animation: false,
    title: { text: title, left: 8, top: 2, textStyle: { color: C.text, fontSize: 15 } },
    legend: { top: 2, right: 10, textStyle: { color: C.muted, fontSize: 13 } },
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    grid: { left: 56, right: 24, top: 34, bottom: 28 },
    xAxis: { type: "category", data: rows, boundaryGap: false, axisLabel: { color: C.muted },
             axisLine: { lineStyle: { color: C.grid } } },
    yAxis: { type: "value", scale: true, axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.grid } } },
    dataZoom: [{ type: "inside", zoomOnMouseWheel: true, moveOnMouseMove: true, throttle: 30 }],
  };
}

function marks(name, indices, values, color, symbol) {
  return { name, type: "scatter", symbol, symbolSize: 9, itemStyle: { color }, z: 5,
           data: indices.filter((i) => i < values.length).map((i) => [i, values[i]]) };
}

const line = (name, data, color, extra = {}) =>
  ({ name, type: "line", data, showSymbol: false, sampling: "lttb", lineStyle: { width: 1.4, color }, itemStyle: { color }, ...extra });

Vue.createApp({
  data() {
    return {
      apiKey: ESC.apiKey(), projects: [], caseId: "", models: [], model: "project", images: [], image: "",
      macd: { ema1: 2, ema2: 4, dea: 2 }, kd: { n: 4, k: 2, d: 2 }, overridden: false,
      result: null, roiIndex: 0, error: "", loading: false, requestId: 0, zoomKey: "",
    };
  },
  computed: {
    roi() { return this.result ? this.result.rois[this.roiIndex] : null; },
    affectNote() {
      const p = this.result.params;
      const off = ["macd", "kd"].filter((f) => !p[f].affects_prediction).map((f) => f.toUpperCase());
      return off.length ? `此模型的 N_pred 不使用 ${off.join(" / ")} 參數（僅更新圖表）` : "調整 MACD / KD 參數會即時重新預測 N_pred";
    },
  },
  async mounted() {
    this.schedule = ESC.debounce(() => this.preview(), 400);
    for (const key of ["signal", "macd", "kd"]) charts[key] = echarts.init(this.$refs[key], "dark");
    echarts.connect([charts.signal, charts.macd, charts.kd]);   // zoom / pan / tooltip in sync
    window.addEventListener("resize", () => Object.values(charts).forEach((c) => c.resize()));
    try {
      this.projects = (await ESC.api("/projects")).projects;
      const wanted = new URLSearchParams(location.search).get("case");      // deep link: ?case=<id>
      const first = this.projects.find((p) => p.has_model && (!wanted || p.case === wanted));
      if (first) { this.caseId = first.case; await this.loadCase(); }
    } catch (e) { this.error = e.message; }
  },
  methods: {
    saveKey() { ESC.setApiKey(this.apiKey); },

    async loadCase() {
      this.error = ""; this.result = null; this.overridden = false; this.roiIndex = 0;
      try {
        const info = await ESC.api(`/results/${encodeURIComponent(this.caseId)}/images`);
        this.images = info.images; this.models = info.models;
        if (!this.models.includes(this.model)) this.model = this.models[0] || "project";
        this.image = (info.images.find((i) => i.complete) || info.images[0] || {}).name || "";
        await this.preview();
      } catch (e) { this.error = e.message; }
    },
    modelChanged() { this.overridden = false; this.roiIndex = 0; this.preview(); },   // trained params of the model
    selectImage(name) { this.image = name; this.roiIndex = 0; this.preview(); },       // first ROI group by default
    edited() { this.overridden = true; this.schedule(); },
    resetParams() { this.overridden = false; this.preview(); },
    moveRoi(step) {
      this.roiIndex = Math.min(Math.max(0, this.roiIndex + step), this.result.rois.length - 1);
      this.render();
    },

    async preview() {
      if (!this.caseId || !this.image) return;
      const id = ++this.requestId;
      const body = { case: this.caseId, image: this.image, model: this.model };
      if (this.overridden) Object.assign(body, { macd: { ...this.macd }, kd: { ...this.kd } });
      this.loading = true;
      try {
        const result = await ESC.api("/preview", { json: body });
        if (id !== this.requestId) return;                       // a newer request superseded this one
        this.result = result; this.error = "";
        if (!this.overridden) {                                  // show the model's trained parameters
          const { ema1, ema2, dea } = result.params.macd; this.macd = { ema1, ema2, dea };
          const { n, k, d } = result.params.kd; this.kd = { n, k, d };
        }
        this.roiIndex = Math.min(this.roiIndex, Math.max(0, result.rois.length - 1));
        this.render();
      } catch (e) {
        if (id === this.requestId) this.error = e.message;
      } finally {
        if (id === this.requestId) this.loading = false;
      }
    },

    render() {
      const roi = this.roi;
      if (!roi) { Object.values(charts).forEach((c) => c.clear()); return; }
      const rows = roi.signal.map((_, i) => roi.y0 + i);
      const title = `N${roi.index}_pred = ${roi.prediction.toFixed(2)}`;
      const signal = baseOption(`${title} · 訊號（${this.result.marks.name} 高低點）`, rows);
      signal.series = [line("訊號", roi.signal, C.signal), marks("高點", roi.highs, roi.signal, C.high, "triangle"),
                       marks("低點", roi.lows, roi.signal, C.low, "circle")];
      const m = roi.macd;
      const macd = baseOption(`MACD (${this.macd.ema1}, ${this.macd.ema2}, ${this.macd.dea}) · 計數 ${m.count}`, rows);
      macd.series = [
        { name: "柱", type: "bar", data: m.hist.map((v) => ({ value: v, itemStyle: { color: v >= 0 ? C.up : C.down } })),
          barWidth: "60%", large: true },
        line("DIF", m.dif, C.dif), line("DEA", m.dea, C.dea),
        marks("黃金交叉", m.golden, m.dif, C.up, "triangle"), marks("死亡交叉", m.death, m.dif, C.down, "pin")];
      const k = roi.kd;
      const kd = baseOption(`KD (RSV ${this.kd.n}, K ${this.kd.k}, D ${this.kd.d}) · 計數 ${k.count}`, rows);
      kd.yAxis = { ...kd.yAxis, scale: false, min: 0, max: 100 };
      kd.grid = { ...kd.grid, bottom: 64 };
      kd.dataZoom = [kd.dataZoom[0], { type: "slider", bottom: 12, height: 22, textStyle: { color: C.muted } }];
      kd.series = [line("RSV", k.rsv, C.rsv, { lineStyle: { width: 1, color: C.rsv, opacity: 0.6 } }),
                   line("K", k.k, C.k), line("D", k.d, C.d),
                   marks("黃金交叉", k.golden, k.k, C.up, "triangle"), marks("死亡交叉", k.death, k.k, C.down, "pin")];
      const zoomKey = `${this.caseId}|${this.image}|${roi.key}`;
      const previous = zoomKey === this.zoomKey ? (charts.kd.getOption() || {}).dataZoom : null;
      this.zoomKey = zoomKey;
      for (const [key, option] of Object.entries({ signal, macd, kd })) {
        if (previous && previous[0]) {                            // keep the zoom window on live param edits
          option.dataZoom = option.dataZoom.map((z) => ({ ...z, start: previous[0].start, end: previous[0].end }));
        }
        charts[key].setOption(option, { notMerge: true });
      }
    },
  },
}).mount("#app");
