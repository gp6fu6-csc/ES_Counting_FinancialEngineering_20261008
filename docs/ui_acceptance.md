# Web UI 手動驗收清單

啟動：`uv run escount serve`（預設 http://127.0.0.1:8000），瀏覽器開啟下列頁面。整個 UI 離線運作：只載入 `/ui/` 下的檔案（Vue 3 與 ECharts 已內含於 `/ui/vendor/`，版本與授權見 `escount/ui/static/vendor/VERSIONS.md`），不需網路、不需 Node。若 Server 有設定 `--api-key`，先在右上角「API Key」欄輸入金鑰（只存在該分頁的 sessionStorage）。

## 校正頁 `/ui/calibration.html`（PDF p12）

可用 `?case=<專案ID>&tab=strips` 直接開啟某專案與分頁。

- [ ] 拔掉網路（或開發者工具 Network 設為 Offline）後重新整理，頁面與圖片仍正常顯示；Network 面板只有 `127.0.0.1:8000` 的請求。
- [ ] 「選擇資料夾」下拉列出 `Data/` 下所有專案；選擇後影像清單只列出數字影像，**不含** `background.png`、`background_BW.png`。
- [ ] 預設選取參考影像（檔名數字最大者，或已存的 `project.json` 的 reference）；點其他影像再按「載入參考影像」會換圖，清單上「參考」標記跟著換。
- [ ] (a) 旋轉校正 ROI：預設 3 個方形 ROI（邊長 384）位於有效區域（綠色虛線框）正中、沿中線（橘色虛線）上下等距排列。
- [ ] 滑鼠拖曳任一 ROI 可移動；放開後約 0.7 秒自動預覽，右側顯示補償角度、各 ROI 角度、轉正前／後殘差與品質（ok 綠色／quality_failure 紅色）。
- [ ] 點選 ROI（框變黃色或按「選取 ROI」按鈕）後，按方向鍵 ←→↑↓ 依「位移量 (Pixel)」移動該 ROI。
- [ ] 「左右控制 · 中線位置」◀ ▶ 同時移動全部 ROI；「上下控制 · 中框高低」▲ ▼ 移動中間 ROI，「− 尺寸／＋ 尺寸」以中心為準縮放中間 ROI。
- [ ] 「ROI 邊長」改成 299 → 預覽顯示錯誤（邊長小於 300）；ROI 拖出有效區 → 錯誤（不在有效區內）；改回 ≥ 300 且在區內 → 恢復正常。
- [ ] 「恢復預設 ROI」回到有效區正中 3 組。
- [ ] (b) 計片條帶：顯示以預覽角度轉正後的影像，黃色條帶 N1…Nn（數量 N、寬度 w、間距 D）位於有效區中央、紫色線為堆疊上下範圍；改 N / w / D / 水平位移後自動更新。
- [ ] 條帶分頁按 ← → 依「位移量」改變水平位移 x。
- [ ] 「距有效區左右」設 49 → 錯誤（< 50 px）；w=30、D=20 → 錯誤（重疊 ≥ w/3，即 D ≤ 2w/3）；D 欄位標籤即時顯示 2w/3 的下限。
- [ ] 「儲存 project.json」後顯示寫入路徑與品質；重新選擇該專案時載入剛存的 ROI 與條帶參數（`strip_params`）。儲存後須重跑 `escount prepare / train / select-rois`（畫面有提示）。
- [ ] `Data/` 內檔案不被修改（`uv run python scripts/check_data_sha.py`）。
- [ ] 選項與清單文字明顯大於一般網頁（≥ 16 px），深色主題，鍵盤 Tab 可走訪所有控制項且有明顯焦點框。

## 結果頁 `/ui/results.html`（PDF p13–14）

前置：專案已 `escount calibrate / prepare / train / select-rois`；想看到「已有完整分析結果」標記，先執行 `uv run escount infer --project <專案ID> --input Data/<專案ID>`（含 analysis.png，不要加 `--no-plots` 等關閉選項）。可用 `?case=<專案ID>` 直接開啟。

- [ ] 離線（Network 設為 Offline）重新整理仍可操作；只載入 `/ui/` 與 `/ui/vendor/` 的檔案（Vue、ECharts 皆為本機檔）。
- [ ] 「選擇資料夾」列出已校正專案，未訓練者標示「（未訓練）」且不可選；「模型」下拉有「專案客製化模型 (Project)」與「General 模型」（General 已訓練時）。
- [ ] 影像清單：`outputs/infer/<專案>/<影像>/` 四個輸出皆存在的影像以「✱ N」標記（N = 已存的預測值），預設選第一張有完整結果的影像。
- [ ] 上方顯示 `N_pred`、GT、命中／未命中、模型（專案或 general）、方法與耗時；與 `escount infer` 該影像的 `result.json` 的 `N_pred` 相同。
- [ ] 預設顯示第一組 ROI（N1）；◀ ▶ 切換 N1…Nn，標題 `N{i}_pred = 值` 跟著換；換影像或換模型時回到 N1。
- [ ] 三張圖（訊號＋高低點、MACD：DIF / DEA / 柱 + 黃金／死亡交叉、KD：RSV / K / D + 交叉）；滑鼠滾輪縮放、拖曳平移流暢，三張圖同步；最下方滑桿可選區間。
- [ ] MACD 的 EMA1 / EMA2 / DEA 與 KD 的 RSV / K / D 預載模型訓練參數；在欄位輸入新值約 0.4 秒後自動重新預測，`N_pred`、圖形與「計數」即時更新，縮放區間保持不變。
- [ ] 模型不使用某組參數時（例如 baseline 指標是 KD），提示「此模型的 N_pred 不使用 MACD 參數（僅更新圖表）」；ML 模型兩組參數都會影響 `N_pred`。
- [ ] 「恢復模型訓練參數」回到預載值且 `N_pred` 回到原預測。
- [ ] 下拉切換 General 模型後 `N_pred` 以 General 模型重算（標示 model_case = general；Ring-type 已排除，不再有 general_no_ring），參數重新預載。
- [ ] 輸入 0 或空白參數時顯示錯誤訊息（422），修正後恢復。
- [ ] `Data/` 內檔案不被修改（`uv run python scripts/check_data_sha.py`）。
