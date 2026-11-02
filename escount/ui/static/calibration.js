// Calibration page (PDF p12): rotation ROIs (tab a) and counting strips (tab b) -> /calibration/* API.
"use strict";
const MAX_WIDTH = 1224;           // downscaled preview width (px)
const COLORS = { valid: "#26a69a", centre: "#ff9f43", roi: "#4c8dff", selected: "#f0b90b",
                 strip: "rgba(240, 185, 11, 0.35)", stripLine: "#f0b90b", stack: "#d16ff0" };
const images = { original: null, rotated: null, rotatedAngle: null };   // bitmaps kept out of Vue reactivity

Vue.createApp({
  data() {
    return {
      apiKey: ESC.apiKey(), cases: [], caseId: "", images: [], reference: "", loadedReference: "",
      imageSize: { width: 0, height: 0 }, validRegion: null, defaultRois: [], minSide: 300, minMargin: 50,
      rois: [], selected: 1, side: 384, step: 10,
      strips: { n: 4, w: 16, pitch: 30, side_margin: 50, x_offset: 0 },
      tab: "rotation", result: null, error: "", message: "", busy: false, drag: null, requestId: 0,
    };
  },
  watch: {
    strips: { deep: true, handler() { this.schedulePreview(); } },
  },
  async mounted() {
    this.schedulePreview = ESC.debounce(() => this.preview(), 700);
    try {
      this.cases = (await ESC.api("/calibration/cases")).cases;
      const query = new URLSearchParams(location.search);              // deep link: ?case=<id>&tab=strips
      if (query.get("tab") === "strips") this.tab = "strips";
      const wanted = query.get("case");
      if (wanted && this.cases.includes(wanted)) { this.caseId = wanted; await this.loadCase(); }
    } catch (e) { this.error = e.message; }
  },
  methods: {
    fmt(value) { return Number(value).toFixed(4); },
    saveKey() { ESC.setApiKey(this.apiKey); },

    async loadCase() {
      this.error = this.message = ""; this.result = null; images.original = images.rotated = null;
      try {
        const info = await ESC.api(`/calibration/${encodeURIComponent(this.caseId)}/images`);
        Object.assign(this, { images: info.images, imageSize: info.image_size, validRegion: info.valid_region,
          defaultRois: info.default_rois, minSide: info.min_side, minMargin: info.min_side_margin,
          strips: Object.assign({}, info.strips) });
        const profile = info.profile;
        this.reference = profile && info.images.includes(profile.reference_image) ? profile.reference_image : info.reference;
        this.rois = (profile && profile.rois.length ? profile.rois : info.default_rois).map((r) => ({ ...r }));
        this.side = this.rois.length ? this.rois[0].side : info.default_side;
        this.selected = Math.floor(this.rois.length / 2);
        await this.loadReference();
      } catch (e) { this.error = e.message; }
    },

    async loadReference() {
      if (!this.reference) return;
      this.error = ""; this.busy = true;
      try {
        images.original = await ESC.bitmap(this.imageUrl(0));
        images.rotated = null; images.rotatedAngle = null;
        this.loadedReference = this.reference;
        this.draw();
        await this.preview();
      } catch (e) { this.error = e.message; } finally { this.busy = false; }
    },

    imageUrl(angle) {
      const name = encodeURIComponent(this.loadedReference || this.reference);
      return `/calibration/${encodeURIComponent(this.caseId)}/image/${name}?angle=${angle}&max_width=${MAX_WIDTH}`;
    },

    body() {
      return { reference: this.loadedReference, rois: this.rois.map(({ x, y, side }) => ({ x, y, side })),
               strips: { ...this.strips } };
    },

    async preview() {
      if (!this.loadedReference) return;
      const id = ++this.requestId;
      try {
        const result = await ESC.api("/calibration/preview", { json: { case: this.caseId, ...this.body() } });
        if (id !== this.requestId) return;                          // a newer request is pending
        this.result = result; this.error = "";
        if (images.rotatedAngle !== result.angle_deg) {
          images.rotated = await ESC.bitmap(this.imageUrl(result.angle_deg));
          images.rotatedAngle = result.angle_deg;
        }
      } catch (e) {
        if (id === this.requestId) { this.error = e.message; this.result = null; }
      }
      this.draw();
    },

    async save() {
      this.message = ""; this.busy = true;
      try {
        const saved = await ESC.api(`/calibration/${encodeURIComponent(this.caseId)}/save`, { json: this.body() });
        this.message = `已寫入 ${saved.path}\n品質：${saved.quality_status}\n${saved.note}`;
        this.error = "";
      } catch (e) { this.error = e.message; } finally { this.busy = false; }
    },

    // ---------------------------------------------------------------- ROI editing
    changed() { this.draw(); this.schedulePreview(); },
    setTab(tab) { this.tab = tab; this.draw(); },
    shiftAll(dx) { this.rois.forEach((r) => { r.x += dx; }); this.changed(); },
    middle() { return this.rois[Math.floor(this.rois.length / 2)]; },
    shiftMiddle(dy) { if (this.rois.length) { this.middle().y += dy; this.changed(); } },
    resizeMiddle(delta) {
      const roi = this.middle();
      if (!roi) return;
      const side = roi.side + 2 * Math.round(delta / 2);
      roi.x -= (side - roi.side) / 2; roi.y -= (side - roi.side) / 2; roi.side = side;   // keep the centre
      this.changed();
    },
    applySide() {
      const side = Math.round(this.side);
      this.rois.forEach((r) => {
        const offset = Math.floor((r.side - side) / 2);
        r.x += offset; r.y += offset; r.side = side;
      });
      this.changed();
    },
    resetRois() {
      this.rois = this.defaultRois.map((r) => ({ ...r }));
      this.side = this.rois.length ? this.rois[0].side : this.side;
      this.selected = Math.floor(this.rois.length / 2);
      this.changed();
    },

    // ---------------------------------------------------------------- mouse / keyboard
    toImage(event) {
      const rect = this.$refs.canvas.getBoundingClientRect();
      return { x: (event.clientX - rect.left) / rect.width * this.imageSize.width,
               y: (event.clientY - rect.top) / rect.height * this.imageSize.height };
    },
    onDown(event) {
      this.$refs.canvas.focus();
      if (this.tab !== "rotation" || !images.original) return;
      const p = this.toImage(event);
      const index = this.rois.findIndex((r) => p.x >= r.x && p.x <= r.x + r.side && p.y >= r.y && p.y <= r.y + r.side);
      if (index < 0) return;
      this.selected = index;
      this.drag = { index, start: p, x: this.rois[index].x, y: this.rois[index].y };
    },
    onMove(event) {
      if (!this.drag) return;
      const p = this.toImage(event);
      const roi = this.rois[this.drag.index];
      roi.x = Math.round(this.drag.x + p.x - this.drag.start.x);
      roi.y = Math.round(this.drag.y + p.y - this.drag.start.y);
      this.draw();
    },
    onUp() {
      if (this.drag) { this.drag = null; this.schedulePreview(); }
    },
    onKey(event) {
      const moves = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
      const move = moves[event.key];
      if (!move) return;
      event.preventDefault();
      if (this.tab === "rotation") {
        const roi = this.rois[this.selected];
        if (!roi) return;
        roi.x += move[0] * this.step; roi.y += move[1] * this.step;
        this.changed();
      } else if (move[0]) {
        this.strips.x_offset = Math.max(0, this.strips.x_offset + move[0] * this.step);   // watcher -> preview
      }
    },

    // ---------------------------------------------------------------- drawing
    draw() {
      const canvas = this.$refs.canvas;
      const bitmap = this.tab === "strips" && images.rotated ? images.rotated : images.original;
      const ctx = canvas.getContext("2d");
      if (!bitmap) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
      canvas.width = bitmap.width; canvas.height = bitmap.height;
      const s = bitmap.width / this.imageSize.width;
      ctx.drawImage(bitmap, 0, 0);
      ctx.lineWidth = 2; ctx.font = "bold 18px sans-serif";
      if (this.tab === "rotation") this.drawRotation(ctx, s, canvas);
      else this.drawStrips(ctx, s, canvas);
    },
    drawRotation(ctx, s, canvas) {
      if (this.validRegion) {
        const [x0, y0, x1, y1] = this.validRegion.bbox;
        ctx.setLineDash([8, 6]); ctx.strokeStyle = COLORS.valid;
        ctx.strokeRect(x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s);
        ctx.strokeStyle = COLORS.centre; ctx.beginPath();
        ctx.moveTo(this.validRegion.centre_x * s, 0); ctx.lineTo(this.validRegion.centre_x * s, canvas.height); ctx.stroke();
        ctx.setLineDash([]);
      }
      const angles = this.result ? this.result.roi_angles_deg : [];
      this.rois.forEach((r, i) => {
        ctx.strokeStyle = i === this.selected ? COLORS.selected : COLORS.roi;
        ctx.lineWidth = i === this.selected ? 3 : 2;
        ctx.strokeRect(r.x * s, r.y * s, r.side * s, r.side * s);
        ctx.fillStyle = ctx.strokeStyle;
        const label = angles[i] !== undefined ? `${i + 1}: ${angles[i].toFixed(3)}°` : `${i + 1}`;
        ctx.fillText(label, r.x * s + 6, r.y * s + 22);
      });
    },
    drawStrips(ctx, s, canvas) {
      if (!this.result || !images.rotated) {
        ctx.fillStyle = COLORS.selected; ctx.fillText("請先預覽（需有效的旋轉校正 ROI 與條帶參數）", 16, 32);
        return;
      }
      const region = this.result.rotated_valid_region;
      const [x0, y0, x1, y1] = region.bbox;
      ctx.setLineDash([8, 6]); ctx.strokeStyle = COLORS.valid;
      ctx.strokeRect(x0 * s, y0 * s, (x1 - x0) * s, (y1 - y0) * s);
      ctx.setLineDash([]);
      // PDF P.8: 取消電磁鋼上源切線，條帶由非 supporter 有效全區間繪製
      const strips = this.result.strips;
      const roomy = strips.length < 2 || (strips[1].x - strips[0].x) * s >= 34;   // room for one label per strip
      strips.forEach((strip) => {
        ctx.fillStyle = COLORS.strip;
        ctx.fillRect(strip.x * s, strip.y0 * s, Math.max(1, strip.w * s), (strip.y1 - strip.y0) * s);
        ctx.strokeStyle = COLORS.stripLine; ctx.lineWidth = 1;
        ctx.strokeRect(strip.x * s, strip.y0 * s, Math.max(1, strip.w * s), (strip.y1 - strip.y0) * s);
        if (roomy) { ctx.fillStyle = COLORS.stripLine; ctx.fillText(`N${strip.index}`, strip.x * s, strip.y0 * s + 40); }
      });
      if (!roomy) {
        const last = strips[strips.length - 1];
        ctx.fillStyle = COLORS.stripLine;
        ctx.fillText(`N1–N${strips.length}`, (last.x + last.w) * s + 8, strips[0].y0 * s + 40);
      }
    },
  },
}).mount("#app");
