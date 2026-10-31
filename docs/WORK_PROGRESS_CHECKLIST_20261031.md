# ESCounting V4 系統改進與優化工作進度核對清單 (Checklist & Handoff)

> 更新時間：2026-10-31 22:40  
> 遵循準則：Karpathy 12 條黃金法則、自主全速推進、嚴格 Token 控管與檢查點機制。

---

## 總覽與進度狀態看板

- [x] **階段一：光學視覺對齊與邊界緩衝增強 (Optical Alignment & Margins)**
  - [x] **任務 1.1**：堆疊邊界上下緣緩衝安全區（Stack Extent Margins/Padding），防止頂底片漏計。
    - *成果*：`roi_bank.py` 實作 `top_margin` / `bottom_margin`，`test_signals_v4.py` 9/9 通過。
  - [x] **任務 1.2**：後端 ROI 橫置影像切片模組（嚴格轉置，方向與訊號完全一致）。
    - *成果*：`projection.py` 新增 `crop_horizontal_slice()`（`block.T` 轉置），數值斷言 Column 0 100% 吻合訊號起點 $y_0$（上緣），並於 `/preview` 回傳 Base64 PNG。
  - [x] **任務 1.3**：前端結果頁（results.html/js）加入橫置影像切片條，實現與 ECharts 橫軸 1:1 像素對齊與十字游標貫穿。
    - *成果*：`results.html` 與 `app.css` 建立切片軌道，`results.js` 實作 `updateAxisPointer`，游標即時標定鋼片真實紋理層位。
  - [x] **任務 1.4**：修正 ECharts 跨圖層縮放（DataZoom）同步與局部放大電磁鋼功能。
    - *成果*：雙向 `dataZoom` 事件分發，解決不同步問題。

- [x] **階段二：指標參數彈性化與量化敏感度研究 (Indicators & Research)**
  - [x] **任務 2.1**：抽離與建立指標參數設定檔，支援靈活自訂網格。
    - *成果*：新增 `configs/indicators.json`，`extract.py` 實作 `load_indicator_configs()`。
  - [x] **任務 2.2**：擴充 research 敏感度分析圖表（1D 參數曲線、2D 敏感度矩陣熱力圖）。
    - *成果*：`figures.py` 新增 `plot_param_sensitivity_matrix()`，實測產出 2D 熱力圖與數值標籤。
  - [x] **任務 2.3**：自動化研究報告升級（HTML/Markdown 圖表整合）。

- [x] **階段三：多方案模型解鎖與動態切換 (Multi-Model Architecture & Dynamic Switch)**
  - [x] **任務 3.1 & 3.2**：推論引擎（InferenceEngine/benchmark.py）與 Server API 支援指定預測方案（`method`）。
    - *成果*：支援 `auto`、`baseline`、`kd`、`macd`、`xgboost` 動態切換與覆蓋，`/preview` 全面接通。
  - [x] **任務 3.3**：前端結果頁上方新增「★ 功能拓展需求！」模型/指標切換下拉選單，並支援即時動態預測。
    - *成果*：`results.html` 加入醒目金黃色選單，`results.js` 支援切換時即刻觸發預測。

- [x] **階段四：AutoResearch 自動優化迴圈與全系統回歸 (AutoResearch & Final Verification)**
  - [x] **任務 4.1**：整合本機 GPU/CPU 之 AutoResearch 演化迭代迴圈，打通 Web/CMD/Server/LabVIEW。
  - [x] **任務 4.2**：全系統測試回歸、Windows XGBoost 暫存檔案鎖相容性修復、55 項依賴許可證審查全數通過。
  - [x] **任務 4.3**：清理暫存垃圾檔，撰寫專案接力交接總結文檔。

---

## AI Agent 接力與技術檔案索引

1. **核心修改清單**：
   - `escount/signals/roi_bank.py`：邊界安全緩衝擴展。
   - `escount/signals/projection.py`：幾何轉置與切片 Base64 編碼。
   - `escount/models/store.py`：Windows XGBoost 載入自我修復（Bytearray fallback）。
   - `escount/inference/benchmark.py`：多方案預測分流器（`forced_method`）。
   - `escount/server/app.py`：`/preview` 注入切片影像與方法覆蓋。
   - `escount/ui/static/results.html`：新增橫置切片軌道與「★ 功能拓展需求！」選單。
   - `escount/ui/static/results.js`：十字游標貫穿連動與雙向 DataZoom 同步。
   - `escount/ui/static/app.css`：切片樣式與選單視覺高亮。
   - `configs/indicators.json`：指標參數搜尋網格。
   - `escount/research/figures.py`：2D 敏感度熱力圖繪製函式。

2. **驗證狀態**：
   - `tests/test_signals_v4.py`：PASS
   - `tests/test_ui_results_api.py`：PASS
   - `tests/test_research.py`：PASS
   - `tests/test_models.py` & `test_es_counting.py`：PASS
   - `scripts/check_licenses.py`：PASS（55 packages OK）
