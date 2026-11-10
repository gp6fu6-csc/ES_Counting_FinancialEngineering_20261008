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
    grid: { left: 56, right: 24, top: 26, bottom: 20 },
    xAxis: { type: "category", data: rows, boundaryGap: false, axisLabel: { color: C.muted },
             axisLine: { lineStyle: { color: C.grid } } },
    yAxis: { type: "value", scale: true, axisLabel: { color: C.muted }, splitLine: { lineStyle: { color: C.grid } } },
    dataZoom: [
      { id: "insideZoom", type: "inside", zoomOnMouseWheel: true, moveOnMouseMove: true, throttle: 20 },
      { id: "sliderZoom", type: "slider", show: false, left: 56, right: 24, throttle: 20 }
    ],
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
      apiKey: ESC.apiKey(), projects: [], caseId: "", models: [], model: "project", roiMode: "minimal",
      recommendedEnsemble: "", methodChoice: "auto", images: [], image: "",
      macd: { ema1: 2, ema2: 4, dea: 2 }, kd: { n: 4, k: 2, d: 2 }, overridden: false,
      result: null, roiIndex: 0, error: "", loading: false, requestId: 0, zoomKey: "",
      zoomRange: { start: 0, end: 100 },
    };
  },
  computed: {
    roi() { return this.result ? this.result.rois[this.roiIndex] : null; },
    zoomRangeText() {
      if (this.zoomRange.start > 0.05 || this.zoomRange.end < 99.95) {
        return `【可視縮放區間: ${this.zoomRange.start.toFixed(1)}% – ${this.zoomRange.end.toFixed(1)}%】`;
      }
      return "";
    },
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
    
    // 頂部橫置光學切片即時更新指示線 (支援在任何縮放比例下 1:1 像素穿透對齊，P.10 安全夾逼防止右側跳回)
    const updateCursor = (params) => {
      const cursor = this.$refs.sliceCursor;
      const track = this.$refs.sliceTrack;
      if (!cursor || !track || !this.roi) return;
      if (!params || !params.axesInfo || !params.axesInfo.length) return;
      const axisInfo = params.axesInfo[0];
      const dataVal = axisInfo.value;
      const minX = 56;
      const maxX = Math.max(minX, track.clientWidth - 24);

      // 優先使用 ECharts convertToPixel 取得與畫布游標完全一致的像素位置並安全夾逼
      let pixel = null;
      try {
        pixel = charts.signal.convertToPixel({ xAxisIndex: 0 }, dataVal);
      } catch (err) {}

      if (typeof pixel === "number" && !isNaN(pixel)) {
        const clamped = Math.max(minX, Math.min(maxX, pixel));
        cursor.style.left = `${clamped}px`;
        cursor.style.display = "block";
        return;
      }

      // 備援計算：依當前 Zoom Window 比例換算
      const y0 = this.roi.y0, y1 = this.roi.y1;
      const total = Math.max(1, y1 - y0);
      const valRatio = (typeof dataVal === "number" && dataVal >= y0 && dataVal <= y1)
        ? (dataVal - y0) / total
        : (typeof dataVal === "number" ? Math.max(0, Math.min(1, dataVal / total)) : 0);
      const zS = (this.zoomRange.start || 0) / 100.0;
      const zE = (this.zoomRange.end || 100) / 100.0;
      const span = Math.max(0.0001, zE - zS);
      const ratioInView = Math.max(0, Math.min(1, (valRatio - zS) / span));
      const posX = minX + ratioInView * (maxX - minX);
      cursor.style.left = `${Math.max(minX, Math.min(maxX, posX))}px`;
      cursor.style.display = "block";
    };

    let isSyncingZoom = false;

    for (const key of ["signal", "macd", "kd"]) {
      charts[key].on("updateAxisPointer", updateCursor);
      charts[key].on("globalout", () => {
        if (this.$refs.sliceCursor) this.$refs.sliceCursor.style.display = "none";
      });

      // 雙向強同步 dataZoom：聯動所有圖表、最底部 Baseline 滑桿與頂部光學切片影像
      charts[key].on("dataZoom", (evt) => {
        const batch = evt.batch ? evt.batch[0] : evt;
        let start = batch.start !== undefined ? batch.start : undefined;
        let end = batch.end !== undefined ? batch.end : undefined;
        if ((start === undefined || end === undefined) && batch.startValue !== undefined && batch.endValue !== undefined && this.roi) {
          const y0 = this.roi.y0, y1 = this.roi.y1;
          const total = Math.max(1, y1 - y0);
          start = Math.max(0, Math.min(100, ((batch.startValue - y0) / total) * 100));
          end = Math.max(0, Math.min(100, ((batch.endValue - y0) / total) * 100));
        }

        if (start !== undefined && end !== undefined) {
          this.zoomRange = { start, end };
          this.updateSliceTransform(start, end);
        }

        if (isSyncingZoom) return;
        isSyncingZoom = true;
        try {
          for (const otherKey of ["signal", "macd", "kd"]) {
            if (otherKey !== key) {
              const action = { type: "dataZoom" };
              if (start !== undefined && end !== undefined) {
                action.start = start;
                action.end = end;
              } else if (batch.startValue !== undefined && batch.endValue !== undefined) {
                action.startValue = batch.startValue;
                action.endValue = batch.endValue;
              }
              charts[otherKey].dispatchAction(action);
            }
          }
        } finally {
          setTimeout(() => { isSyncingZoom = false; }, 20);
        }
      });
    }

    window.addEventListener("resize", () => Object.values(charts).forEach((c) => c.resize()));
    const observer = new ResizeObserver(() => Object.values(charts).forEach((c) => c.resize()));
    for (const key of ["signal", "macd", "kd"]) observer.observe(this.$refs[key]);
    try {
      this.projects = (await ESC.api("/projects")).projects;
      const wanted = new URLSearchParams(location.search).get("case");      // deep link: ?case=<id>
      const first = this.projects.find((p) => p.has_model && (!wanted || p.case === wanted));
      if (first) { this.caseId = first.case; await this.loadCase(); }
    } catch (e) { this.error = e.message; }
  },
  methods: {
    saveKey() { ESC.setApiKey(this.apiKey); },

    updateSliceTransform(startPct, endPct) {
      const img = this.$refs.sliceImg;
      if (!img) return;
      const s = Math.max(0, Math.min(100, typeof startPct === 'number' ? startPct : 0));
      const e = Math.max(s + 0.01, Math.min(100, typeof endPct === 'number' ? endPct : 100));
      const span = (e - s) / 100.0;
      const scaleX = 1.0 / span;
      const transX = -s;
      img.style.transformOrigin = "0 0";
      img.style.transform = `scaleX(${scaleX}) translateX(${transX}%)`;
    },

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
    roiModeChanged() { this.roiIndex = 0; this.preview(); },
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
      const body = { case: this.caseId, image: this.image, model: this.model, roi_mode: this.roiMode, method: this.methodChoice };
      if (this.overridden) Object.assign(body, { macd: { ...this.macd }, kd: { ...this.kd } });
      this.loading = true;
      try {
        const result = await ESC.api("/preview", { json: body });
        if (id !== this.requestId) return;                       // a newer request superseded this one
        this.result = result; this.error = "";
        if (result.all_rois_ensemble && result.all_rois_ensemble.recommended) {
          this.recommendedEnsemble = result.all_rois_ensemble.recommended;
        }
        if (!this.overridden) {                                  // show the model's trained parameters
          const { ema1, ema2, dea } = result.params.macd; this.macd = { ema1, ema2, dea };
          const { n, k, d } = result.params.kd; this.kd = { n, k, d };
        }
        this.roiIndex = Math.min(this.roiIndex, Math.max(0, result.rois.length - 1));
        this.render();
      } catch (e) {
        if (id === this.requestId) {                             // do not leave the previous image's curves
          this.error = e.message; this.result = null;
          Object.values(charts).forEach((c) => c.clear());
        }
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
      kd.grid = { ...kd.grid, bottom: 50 };
      kd.dataZoom = [
        { id: "insideZoom", type: "inside", zoomOnMouseWheel: true, moveOnMouseMove: true, throttle: 20 },
        { id: "sliderZoom", type: "slider", show: true, left: 56, right: 24, bottom: 8, height: 22, textStyle: { color: C.muted }, throttle: 20 }
      ];
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
        charts[key].resize();                                     // fit the current container width
      }
      const curZoom = (previous && previous[0] && previous[0].start !== undefined)
        ? { start: previous[0].start, end: previous[0].end }
        : { start: 0, end: 100 };
      this.zoomRange = curZoom;
      this.updateSliceTransform(curZoom.start, curZoom.end);
    },
  },
}).mount("#app");
