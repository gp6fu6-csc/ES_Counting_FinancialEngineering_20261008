// Training page: guided calibrate -> prepare -> train -> select-rois -> infer, run as server background jobs.
"use strict";
const STEPS = [
  { key: "calibrate", title: "0. 校正", desc: "在校正頁調整旋轉 ROI 並儲存 project.json（必須先完成）。" },
  { key: "height", title: "0-1. 像素-高度比例預測模型 (選用)", desc: "在校正頁「(c) 比例校正」點選 ratio.png 的左右比例尺建立模型；可直接跳過，不影響後續分析。" },
  { key: "prepare", title: "1. 建立特徵 (prepare)", desc: "依旋轉參數切出所有 ROI、投影並計算財務指標特徵，存成特徵快取。新增影像後需重新執行。" },
  { key: "train", title: "2. 訓練模型 (train)", desc: "以完全命中率比較 Ridge / SVR / XGBoost / LightGBM 與單一指標 baseline，保留最佳者。重新訓練後必須再做步驟 3。" },
  { key: "select", title: "3. 挑選最少 ROI 與建立全部 ROI 整合模型", desc: "挑選推論用最少 ROI，並訓練全部條帶之多 ROI 智能整合模型（預設同時建立，亦可單獨執行）。" },
  { key: "infer", title: "4. 分析全部影像 (infer)", desc: "載入模型並分析此專案所有影像，結果頁會標示 ✱ 已有完整分析結果。" },
  { key: "results", title: "5. 檢視結果", desc: "到結果頁查看 N_pred、ROI 訊號與 MACD / KD 曲線。" },
];

Vue.createApp({
  data() {
    return { cases: [], caseId: "", device: "cuda", steps: STEPS,
             status: {}, job: { log: [] }, error: "", timer: null,
             purgeOpts: { with_variants: false, research: false, all_files: false },
             variantId: "", overview: { projects: {}, variants: [] },
             batch: { cases: [], variants: [], max_rois: 25, max_rows_per_image: 1000, research: false, trials: 30, track: "default",
                      infer: true, note: "" } };
  },
  computed: {
    fullCase() { return this.variantId ? `${this.caseId}@${this.variantId}` : this.caseId; },   // <case>[@<variant>]
    variantNames() { return this.overview.projects[this.caseId] || []; },
    batchPairs() { return this.batch.cases.length * this.batch.variants.length; },
    nextStep() {
      return ["calibrate", "prepare", "train", "select", "infer"].find((k) => !this.status[k]) || "results";
    },
    statusText() {
      return { running: "執行中…", done: "完成", failed: "失敗（見下方記錄）", idle: "" }[this.job.status] || this.job.status;
    },
  },
  async mounted() {
    try {
      this.cases = (await ESC.api("/calibration/cases")).cases;
      const wanted = ESC.lastCase();
      this.caseId = this.cases.includes(wanted) ? wanted : (this.cases[0] || "");
      await this.loadOverview();
      if (this.caseId && !this.batch.cases.length) this.batch.cases = [this.caseId];
      await this.refresh();
    } catch (e) { this.error = e.message; }
  },
  methods: {
    async loadOverview() { this.overview = await ESC.api("/variants"); },
    caseChanged() { this.variantId = ""; this.refresh(); },
    async startBatch() {
      try {
        await ESC.api("/training/batch", { json: { ...this.batch, device: this.device } });
        this.error = "";
        await this.refresh();
      } catch (e) { this.error = e.message; }
    },
    canRun(key) {
      if (this.job.status === "running") return false;
      if (key === "all_rois") {
        return ["calibrate", "prepare", "train"].every((k) => this.status[k]);
      }
      const order = ["calibrate", "prepare", "train", "select", "infer"];
      return order.slice(0, order.indexOf(key)).every((k) => this.status[k]);
    },
    async refresh() {
      if (!this.caseId) return;
      ESC.setLastCase(this.caseId);
      try {
        const info = await ESC.api(`/training/${encodeURIComponent(this.fullCase)}/status`);
        this.status = info.steps; this.job = info.job; this.error = "";
        this.$nextTick(() => { if (this.$refs.log) this.$refs.log.scrollTop = this.$refs.log.scrollHeight; });
        clearTimeout(this.timer);
        if (info.job.status === "running") this.timer = setTimeout(() => this.refresh(), 1500);
        else await this.loadOverview();                       // a finished batch may have created variant folders
      } catch (e) { this.error = e.message; }
    },
    async purge() {
      const url = `/training/${encodeURIComponent(this.fullCase)}/purge`;
      const opts = { ...this.purgeOpts, with_variants: !this.variantId && this.purgeOpts.with_variants };
      try {
        const plan = await ESC.api(url, { json: { ...opts, dry_run: true } });
        if (!plan.items.length) { this.error = ""; alert(`${this.fullCase}：沒有可刪除的資料`); return; }
        const list = plan.items.slice(0, 15).map((i) => `${i.kind}: ${i.path}`).join("\n");
        const more = plan.items.length > 15 ? `\n…另 ${plan.items.length - 15} 項` : "";
        if (!confirm(`將刪除 ${this.fullCase} 的 ${plan.items.length} 項資料（${(plan.bytes / 1e6).toFixed(1)} MB），無法復原：\n${list}${more}`)) return;
        await ESC.api(url, { json: { ...opts, dry_run: false } });
        this.error = "";
        await this.loadOverview();
        if (this.variantId && !this.variantNames.includes(this.variantId)) this.variantId = "";
        await this.refresh();
      } catch (e) { this.error = e.message; }
    },
    async run(step) {
      try {
        await ESC.api(`/training/${encodeURIComponent(this.fullCase)}/run`, { json: { step, device: this.device } });
        this.error = "";
        await this.refresh();
      } catch (e) { this.error = e.message; }
    },
  },
}).mount("#app");
