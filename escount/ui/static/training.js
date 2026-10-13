// Training page: guided calibrate -> prepare -> train -> select-rois -> infer, run as server background jobs.
"use strict";
const STEPS = [
  { key: "calibrate", title: "0. 校正", desc: "在校正頁調整旋轉 ROI 並儲存 project.json（必須先完成）。" },
  { key: "prepare", title: "1. 建立特徵 (prepare)", desc: "依旋轉參數切出所有 ROI、投影並計算財務指標特徵，存成特徵快取。新增影像後需重新執行。" },
  { key: "train", title: "2. 訓練模型 (train)", desc: "以完全命中率比較 Ridge / SVR / XGBoost / LightGBM 與單一指標 baseline，保留最佳者。重新訓練後必須再做步驟 3。" },
  { key: "select", title: "3. 挑選最少 ROI (select-rois)", desc: "選出推論用的最少 ROI 組合；完成後模型即可載入使用。" },
  { key: "infer", title: "4. 分析全部影像 (infer)", desc: "載入模型並分析此專案所有影像，結果頁會標示 ✱ 已有完整分析結果。" },
  { key: "results", title: "5. 檢視結果", desc: "到結果頁查看 N_pred、ROI 訊號與 MACD / KD 曲線。" },
];

Vue.createApp({
  data() {
    return { apiKey: ESC.apiKey(), cases: [], caseId: "", device: "cuda", steps: STEPS,
             status: {}, job: { log: [] }, error: "", timer: null };
  },
  computed: {
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
      const wanted = new URLSearchParams(location.search).get("case");
      this.caseId = this.cases.includes(wanted) ? wanted : (this.cases[0] || "");
      await this.refresh();
    } catch (e) { this.error = e.message; }
  },
  methods: {
    saveKey() { ESC.setApiKey(this.apiKey); },
    canRun(key) {
      if (this.job.status === "running") return false;
      const order = ["calibrate", "prepare", "train", "select", "infer"];
      return order.slice(0, order.indexOf(key)).every((k) => this.status[k]);
    },
    async refresh() {
      if (!this.caseId) return;
      try {
        const info = await ESC.api(`/training/${encodeURIComponent(this.caseId)}/status`);
        this.status = info.steps; this.job = info.job; this.error = "";
        this.$nextTick(() => { if (this.$refs.log) this.$refs.log.scrollTop = this.$refs.log.scrollHeight; });
        clearTimeout(this.timer);
        if (info.job.status === "running") this.timer = setTimeout(() => this.refresh(), 1500);
      } catch (e) { this.error = e.message; }
    },
    async run(step) {
      try {
        await ESC.api(`/training/${encodeURIComponent(this.caseId)}/run`, { json: { step, device: this.device } });
        this.error = "";
        await this.refresh();
      } catch (e) { this.error = e.message; }
    },
  },
}).mount("#app");
