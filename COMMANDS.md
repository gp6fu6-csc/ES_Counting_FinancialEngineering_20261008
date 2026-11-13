# ESCounting V4 指令速查手冊 (Command Reference Guide)

本手冊彙整 ESCounting V4 系統之完整操作指令、參數說明、網頁界面操作、推論 Sample Code 與 AutoResearch 專題研究腳本。

---

## 1. 網頁 UI 界面操作 (Web UI)

### 啟動服務
```powershell
uv run escount serve --host 127.0.0.1 --port 8000
```

### 頁面網址與操作流程
開啟瀏覽器訪問以下頁面：
* **訓練與部署引導頁**：`http://localhost:8000/ui/training.html`
  * 選擇資料夾（如 `x05_unpress`）與訓練裝置（CUDA 或 CPU）。
  * 依照畫面提示依序按下按鈕：**0. 校正 $\to$ 1. 建立特徵 $\to$ 2. 訓練模型 $\to$ 3. 挑選 ROI 與建立全部 ROI 整合模型 $\to$ 4. 分析全部**。
  * **步驟 3 支援雙軌模式**：可一鍵同時完成「最少 ROI 挑選」與「全部 ROI 機器學習整合模型」，亦提供獨立子按鈕「僅建立全部 ROI 整合模型 (all-rois)」。
* **旋轉校正頁**：`http://localhost:8000/ui/calibration.html`
  * 檢視 3 個定位方塊與 FFT 旋轉角度補償，調整計片條帶參數（如條帶數量 $N=10$、寬度 $w$、間距 $D$），點選「儲存 project.json」。
* **推論結果檢視頁**：`http://localhost:8000/ui/results.html`
  * 點擊影像清單切換檢視；
  * **ROI 方案下拉選單**：可即時切換「最少 ROI」與「全部 ROI」方案。
    * **最少 ROI (Minimal)**：載入演算法挑選之精簡組合（如 2 組，顯示 `ROI N1 / 2`），兼顧極速推論與高精度。
    * **全部 ROI (All Strips)**：載入校正頁所設計之全部 $N$ 組條帶（如 10 組，顯示 `ROI N1 / 10`），支援點擊 ◀ ▶ 巡覽 $N_1 \sim N_{10}$ 每一條帶之切片、波形與個別預測值。
  * **★ 功能拓展需求選單**：
    * 最少 ROI 模式：可切換 `Auto`、`Baseline`、`KD`、`MACD`、`XGBoost`；
    * 全部 ROI 模式：導入第二階段多條帶 ML 整合技術（Stacking Ensemble），可切換 `Auto (系統推薦最佳方案)`、`XGBoost 多 ROI 智能整合`、`Ridge 空間加權整合`、`去極值平均 (Trimmed Mean)` 與各單項指標；
  * **切片與訊號對齊**：上方顯示旋轉後橫置真實鋼片切片，垂直貫穿十字游標與下方 ECharts 訊號 1:1 像素對齊，支援滾輪同步局部縮放。

---

## 2. 產線標準命令行流程 (CLI Pipeline)

### 步驟 0：旋轉校正 (Calibrate)
```powershell
uv run escount calibrate --project x05_unpress
# 參數：
#   --project <名稱|路徑>  : 專案目錄（如 Data/x05_unpress）
#   --roi-side 384        : 方塊 ROI 邊長像素（預設 384）
```

#### 步驟 0 前置：background_BW.png 確認／圈選（校正頁 `(0) 背景 BW 確認`）
* 已有 `background_BW.png`：以原圖疊加透明紅色顯示有效區域；無誤可直接進入 `(a) 旋轉校正 ROI`。
* 無檔案或要修正：按「開始圈選／重新圈選」。左鍵 = 加節點；右鍵 = 取消上一個節點；左鍵連點兩下 = 確定範圍（至少 3 點）；再按「儲存 background_BW.png」。多邊形內 = 有效區（黑 0），外側 = 白 255。
* 覆寫前舊檔會備份到 `outputs/projects/<專案>/background_BW.prev.png`。遮罩改變後需重跑 校正 → prepare → train → select-rois。

#### 跨專案轉移設定（有效區域 / 旋轉校正 ROI / 計片條帶）
`project.json` 內的 `valid_region` 是由 `background_BW.png` 算出的衍生值，不能單獨貼上；實際可轉移的是三項：遮罩檔、旋轉 ROI、計片條帶 `strip_params`。旋轉角度一律用目標專案自己的影像重算。
```powershell
# 全部轉移（遮罩 + ROI + 條帶）；兩專案影像尺寸須相同
uv run escount copy-params --from 20261020_TH01mm_03x_non-press --to 20261020_TH01mm_05x_non-press
# 只轉移條帶參數（目標保留自己的遮罩與 ROI）
uv run escount copy-params --from <來源> --to <目標> --what strips
# --what 可選 mask rois strips，可複選
```
手動複製貼上：複製 `Data/<來源>/background_BW.png` 到目標資料夾，並把 `project.json` 的 `strip_params` 區塊貼到目標；之後仍需執行 `escount calibrate --project <目標>` 重算角度。轉移後目標的 prepare / train / select-rois / all-rois 要重跑。

#### 一鍵批次：範本參數 → 全部專案訓練 → 通用模型 → AutoResearch 四軌 → 報告 (`escount batch`)
```powershell
uv run escount batch --template press-Red-70 --dry-run      # 先預演，不寫檔
uv run escount batch --template press-Red-70                # 預設 --data-root Data/Case_Study
uv run escount batch --template press-Red-70 --data-root Data\Case_Study D:\其他資料集 --trials 10 --device cuda
```
* `--template`：只提供幾何設定（旋轉 ROI、計片條帶；目標沒有遮罩時連 `background_BW.png` 一起複製），旋轉角度用各專案自己的影像重算；每個專案用自己的影像訓練。
* 只用「檔名 = 片數」的影像；`testX.png`、`test.txt`、子資料夾（如 `test\`）、以 `test` 開頭的資料夾一律不用；可訓練影像少於 3 張的資料夾略過。
* 輸出全部在 `--output`（預設 `outputs/case_study/`），不會動到 `outputs/projects/`。已有的 `project.json` 先改名為 `project_YYYYMMDD_N.json` 再套用（`--keep-existing` 則保留）。
* 結果：`summary.md` / `summary.csv`（各專案 × 四軌交叉對比）、`mask_overlays/*.png`（遮罩疊圖，請抽查）、`report*/latest/report.html`、`projects/<專案>/model/`。
* 推論時加 `--profiles-root outputs\case_study\projects`（Python：`InferenceEngine.load(..., profiles_root="outputs/case_study/projects")`）。
* 完整教學範例：`scripts/batch_sample.ps1`。

### 步驟 1：建立多特徵快取 (Prepare)
```powershell
# 單一專案
uv run escount prepare --project x05_unpress

# 全數專案（多執行緒加速）
uv run escount prepare --all --workers 8
```

### 步驟 2：模型訓練與基準競賽 (Train)
```powershell
# 單一專案模型（自動競賽：Baseline vs XGBoost/LightGBM/SVR/Ridge）
uv run escount train --project x05_unpress --device cuda

# 跨專案通用模型（包含尺度歸一化與 LOPO 驗證）
uv run escount train --general --device cuda
```

### 步驟 3：最小 ROI 集合挑選與全 ROI 整合模型訓練 (Select-ROIs & All-ROIs)
```powershell
# 同時執行最小 ROI 貪婪挑選與全部條帶 ML 整合模型訓練（推薦）：
uv run escount select-rois --project x05_unpress --max-rois 25
# 參數：
#   --max-rois 25         : 最大挑選條帶數量上限（通常選出 2~8 條）

# 若只需單獨更新全部條帶之多 ROI ML 整合模型（All-ROIs Stacking）：
uv run escount all-rois --project x05_unpress
```

### 步驟 4：正式推論 (Infer CLI)
```powershell
# 單張影像推論（預設最小 ROI 集合）：
uv run escount infer --project x05_unpress --input Data/x05_unpress/test2.png --method auto --output outputs/infer

# 指定使用「全部 ROI 模式」並透過 ML 整合輸出最終片數：
uv run escount infer --project x05_unpress --input Data/x05_unpress/test2.png --roi-mode all --method auto --output outputs/infer

# 常用參數：
#   --project <專案名>     : 專案名稱 (如 x05_unpress)
#   --input <檔案或目錄>   : 輸入待分析影像或資料夾
#   --roi-mode <方案>      : minimal (最少 ROI，預設) 或 all (校正條帶全部 ROI)
#   --method <方案>        : 預測/整合方案: auto / baseline / kd / macd / xgboost / ridge / trimmed_mean (預設 auto)
#   --model <模型範疇>     : project (專案模型) 或 general (跨專案聯合通用模型)
#   --output <輸出目錄>    : 結果輸出資料夾 (輸出 predictions.csv, analysis.png, result.json 等)
```

### 步驟 5：推論速度測試 (Benchmark)
```powershell
uv run escount benchmark --project x05_unpress
# 驗證單執行緒 CPU 推論時間（確保產線邊緣端 < 1 秒）
```

---

## 3. 推論調用與模型切換範例 (Sample Code)

本章節提供 Python 純代碼調用範例，可供第三方程式、LabVIEW、PLC 整合或研發人員直接複製引用：

### 3.1 基礎直接調用標準範例 (Quickstart Sample)
```python
from escount.inference.engine import InferenceEngine

# 1. 指定並載入專案模型（耗時僅載入一次，長駐記憶體）
engine = InferenceEngine.load("x05_unpress")

# 2. 傳入影像路徑進行推論（亦可直接傳入 OpenCV uint8 灰階矩陣）
result = engine.infer("Data/x05_unpress/test2.png")

# 3. 取得預測片數與耗時資訊
pred_count = result["N_pred"]               # 預測片數 (例如: 102)
total_ms = result["timings_ms"]["total"]     # 總推論耗時 (例如: 593.8 ms)

print(f"預測鋼片數量：{pred_count} 片")
print(f"單張推論耗時：{total_ms:.1f} 毫秒")
```

### 3.2 自由切換預測計算方案 (Method Switching)
呼叫 `engine.infer(..., method="...")` 時傳入 `method` 參數，即可自由選擇計算方式：
```python
from escount.inference.engine import InferenceEngine

engine = InferenceEngine.load("x05_unpress")
image_path = "Data/x05_unpress/test2.png"

# 【方案 1：Auto 自動最佳】(預設：若 ML 準確率優於 Baseline 則用 ML，否則退回 Baseline)
res_auto = engine.infer(image_path, method="auto")

# 【方案 2：Baseline 基準線】(使用訓練時選出的最佳單一指標參考線)
res_base = engine.infer(image_path, method="baseline")

# 【方案 3：KD 指標】(強制使用 KD 隨機指標黃金交叉/波峰計數)
res_kd = engine.infer(image_path, method="kd")

# 【方案 4：MACD 指標】(強制使用 MACD 柱狀線/差離值交叉計數)
res_macd = engine.infer(image_path, method="macd")

# 【方案 5：XGBoost 機器學習】(強制使用 XGBoost 物理殘差迴歸模型)
res_xgb = engine.infer(image_path, method="xgboost")

print(f"Auto    預測片數: {res_auto['N_pred']}")
print(f"Baseline預測片數: {res_base['N_pred']}")
print(f"KD 指標 預測片數: {res_kd['N_pred']}")
print(f"MACD指標預測片數: {res_macd['N_pred']}")
print(f"XGBoost 預測片數: {res_xgb['N_pred']}")
```

### 3.3 切換「最少 ROI」vs「全部 ROI」雙軌方案 (Dual ROI Modes & ML Ensembles)
推論時可直接透過 `roi_mode="all"` 與指定整合演算法進行推論：
```python
from escount.inference.engine import InferenceEngine

engine = InferenceEngine.load("x05_unpress")
image_path = "Data/x05_unpress/test2.png"

# 【方案 A：最少 ROI 模式】(Minimal ROIs: 部署由演算法精選的少數互補條帶，推論速度極快)
res_min = engine.infer(image_path, roi_mode="minimal", method="auto")
print(f"最少 ROI 預測片數: {res_min['N_pred']} (使用 {len(res_min['rois'])} 組精選條帶)")

# 【方案 B：全部 ROI 模式 + ML 智能整合】(All ROIs: 涵蓋校正頁全部 N 組條帶，並以第二階段 ML 整合各條帶結果)
res_all_auto = engine.infer(image_path, roi_mode="all", method="auto")
print(f"全部 ROI 自動最佳預測: {res_all_auto['N_pred']} (分析全部 {len(res_all_auto['rois'])} 組條帶)")

# 【方案 C：全部 ROI 模式 + 強制使用 XGBoost 智能整合】
res_all_xgb = engine.infer(image_path, roi_mode="all", method="xgboost")
print(f"全部 ROI XGBoost 整合: {res_all_xgb['N_pred']}")

# 【方案 D：全部 ROI 模式 + 強制使用 Ridge 空間加權整合】
res_all_ridge = engine.infer(image_path, roi_mode="all", method="ridge")
print(f"全部 ROI Ridge 加權整合: {res_all_ridge['N_pred']}")

# 【方案 E：全部 ROI 模式 + 統計中位數整合】
res_all_med = engine.infer(image_path, roi_mode="all", method="baseline")
print(f"全部 ROI 中位數整合:   {res_all_med['N_pred']}")

# 可巡覽各個條帶獨立算出的預測值：
for roi in res_all_auto["rois"]:
    print(f"  條帶 N{roi['index']} ({roi['key']}): 預測 {roi['prediction']:.2f} 片")
```

### 3.4 切換「特化專案模型」vs「跨專案聯合通用模型 (General Model)」
在 `InferenceEngine.load()` 時指定 `model` 參數：
```python
from escount.inference.engine import InferenceEngine

# 1. 專案特化模型 (Project Model: 針對 x05_unpress 幾何規格最佳化)
engine_project = InferenceEngine.load("x05_unpress", model="project")
res_proj = engine_project.infer("Data/x05_unpress/test2.png")

# 2. 跨專案聯合預測通用模型 (General Model: 聯合多個不同鋼片厚度/專案訓練，採尺度歸一化)
engine_general = InferenceEngine.load("x05_unpress", model="general")
res_gen = engine_general.infer("Data/x05_unpress/test2.png")

print(f"專案特化預測: {res_proj['N_pred']} 片")
print(f"跨專案通用預測: {res_gen['N_pred']} 片")
```

### 3.5 動態即時調整技術指標參數（無需重新訓練）
推論時可直接透過 `overrides` 參數動態覆蓋 KD 或 MACD 的計算週期：
```python
from escount.indicators.core import KDParams, MACDParams
from escount.inference.engine import InferenceEngine

engine = InferenceEngine.load("x05_unpress")

# 動態設定 KD 參數為 (n=9, k=3, d=3) 以及 MACD 參數為 (ema1=12, ema2=26, dea=9)
res_custom = engine.infer(
    "Data/x05_unpress/test2.png",
    method="kd",
    overrides={
        "kd": KDParams(n=9, k=3, d=3),
        "macd": MACDParams(ema1=12, ema2=26, dea=9)
    }
)
print(f"自訂指標參數預測結果: {res_custom['N_pred']} 片 (耗時: {res_custom['timings_ms']['total']:.1f} ms)")
```

### 3.6 批量推論並輸出檔案 (Run Batch)
```python
from escount.inference.engine import InferenceEngine, OutputOptions

engine = InferenceEngine.load("x05_unpress")

# 批量分析目錄下所有影像，並輸出至指定資料夾
results = engine.run(
    source="Data/x05_unpress",
    out_dir="outputs/infer_custom",
    options=OutputOptions(corrected=True, analysis_plot=True, signals_csv=True, result_json=True),
    method="xgboost"
)
print(f"批量分析完成，共處理 {len(results)} 張影像。")
```

### 3.6 PowerShell 快速調用與 `--method` 切換
在終端機亦可直接透過 CLI 加入 `--method` 參數進行相同切換：
```powershell
# 使用預設 Auto 方案
uv run escount infer --project x05_unpress --input Data/x05_unpress/test2.png

# 強制指定用 kd 計算
uv run escount infer --project x05_unpress --input Data/x05_unpress/test2.png --method kd

# 強制指定用 macd 計算
uv run escount infer --project x05_unpress --input Data/x05_unpress/test2.png --method macd

# 強制指定用 xgboost 計算
uv run escount infer --project x05_unpress --input Data/x05_unpress/test2.png --method xgboost

# 強制指定用 baseline 計算
uv run escount infer --project x05_unpress --input Data/x05_unpress/test2.png --method baseline
```

---

## 4. AutoResearch 自動研究專題指令

### 軌道一：全域 MAE 誤差最小化研究 (Track 1: MAE)
專注於宏觀誤差收斂，消除粗大誤判：
```powershell
uv run python scripts/run_research_mae.py --trials 10 --device cpu
# 參數：
#   --trials <N>   : 執行試驗次數（預設 5）
#   --device <cpu|cuda> : 運算裝置
#   --cases <名稱> : 指定案例（預設自動探索所有有效專案）
```

### 軌道二：工業級絕對精確命中研究 (Track 2: Exact Hit)
引入客製化軟命中損失函數 (Soft-Hit Loss) 與離散殘差多分類器，直攻 100% 片數命中：
```powershell
uv run python scripts/run_research_exact_hit.py --trials 10 --device cpu
```

### 軌道三：輕量級低算力 ML 快速聯合運算研究 (Track 3: Ensemble)
探索 XGBoost + LightGBM + Ridge 之 Stacking 與投票融合，推升命中率極限：
```powershell
uv run python scripts/run_research_ensemble.py --trials 10 --device cpu
```

### 統一 CLI 呼叫方式
```powershell
uv run escount research --track mae --trials 10
uv run escount research --track exact --trials 10
uv run escount research --track ensemble --trials 10
```

### 編譯整合研究報告 (Report)
匯整所有研究結果，產出包含 2D 敏感度矩陣、fANOVA 重要度與對照表之獨立 HTML 與 Markdown 報告：
```powershell
uv run escount report --out outputs/research_report
# 產出位置：outputs/research_report/latest/report.html 與 summary.md
```

---

## 5. 系統測試與授權合規檢查

```powershell
# 執行全單元測試 (126 項驗證)
```powershell
uv run escount report --out outputs/research_report
# 產出位置：outputs/research_report/latest/report.html 與 summary.md
#   --results-root <資料夾> : escount infer 的輸出（正式模型 GT vs Predict 圖用；預設 outputs/infer）
```
報告內容（2026-11-13 新增）：
* **GT vs Predict 點陣圖**：橫軸 GT、縱軸 Predict；虛線 y = x = 完全命中，淺綠帶 ±0.5 = 四捨五入後命中；● 命中、× 未命中；標題 = 資料名稱 + 完全命中 hits/total (%)，第二行標註預測來源。
  * `<案例>/gt_vs_pred.png`：AutoResearch 最佳設定的 **LOIO** 預測（影像未參與訓練，原始預測值）→ 代表沒看過的影像。
  * `<案例>/gt_vs_pred_model.png`：**正式推論模型** `escount infer` 的結果（訓練影像自我預測，偏樂觀，整數值）；需先執行 infer（batch 加 `--infer`），否則顯示「尚無預測資料」。
  * `comparison/gt_vs_pred_all.png`、`comparison/gt_vs_pred_all_model.png`：全部資料集合併，顏色區分資料集，圖例列出各組 hits/total。
* **正式推論模型表**新增「全部 ROI 整合（方法 / 條數）」「全部 ROI 命中」兩欄，與「部署 ROI（最少 ROI）」直接對比。

### AutoResearch 名詞說明
* **trial**：試一組完整設定並評分一次（條帶 w / D / 位移、投影、指標族、模型與超參數、ROI 上限、整合方式）。取樣用 Optuna **TPE 貝氏最佳化**：前 5 次隨機探索，之後依結果往表現好的區域取樣；評分 = LOIO 完全命中。`--trials N` = 每個專案、每一軌各 N 次。
* **以「完全正確」為目標**：default 軌（排序 = 命中率 → ROI 少 → 速度）與 exact_hit 軌（命中數優先，另含 Soft Exact-Hit Loss、離散殘差分類器）；mae 軌則以平均誤差最小為目標。
* 同一個輸出資料夾再跑會**接續累積** trial；換新資料夾則從零開始。

### Case_Study 深度研究（cmd 指令，`escount batch`）
以指定專案的參數為預設條件，對 `Data/Case_Study` 全部資料集：套用範本 → 各專案訓練 1~4 → 通用模型 → AutoResearch 四軌 → 報告。
```cmd
cd /d F:\Antigravity_2_0\20261102_ESCountingFinace
:: 先預演（不寫檔）
uv run escount batch --template press-Red-70 --dry-run
:: 正式執行（輸出到新資料夾；資料夾要先建立，cmd 才能寫 log）
mkdir outputs\case_study_20261113
uv run escount batch --template press-Red-70 --data-root Data/Case_Study --output outputs/case_study_20261113 --trials 30 --device cuda --infer > outputs\case_study_20261113\batch_log_deep.txt 2>&1
:: 另開視窗看進度
powershell -Command "Get-Content outputs\case_study_20261113\batch_log_deep.txt -Tail 20 -Wait"
```
* 兩個路徑意義不同：`--output ...` = batch 所有結果的根目錄（`projects/`、`features/`、`research/`、`candidates/`、`infer/`、`report*/`、`summary.*`、`mask_overlays/`）；`> ...\batch_log_deep.txt` = cmd 把畫面輸出存成 log 檔。改輸出位置時兩處都要改。
* 已有 `project.json` 會先改名為 `project_YYYYMMDD_N.json` 再套用範本；要保留手動調整用 `--keep-existing`。
* 執行時間約與 trials 成正比（trials=2 的研究部分即 > 1 小時）；電腦勿睡眠（`powercfg /change standby-timeout-ac 0`），斷網不影響。
* 網頁 / 推論指向該次輸出：
```cmd
uv run escount serve --root Data/Case_Study --profiles-root outputs/case_study_20261113/projects --features-root outputs/case_study_20261113/features --results-root outputs/case_study_20261113/infer
uv run escount infer --project press-Red-20 --profiles-root outputs/case_study_20261113/projects --input <影像或資料夾> --output outputs/case_study_20261113/infer_new
```
* 只更新報告（不重跑研究），例如 default 軌：
```cmd
uv run escount report --out outputs/case_study_20261113/report --cases press-Red-20 press-Red-40 press-Red-70 press-Red-max press-White-18 press-White-28 press-White-45 press-White-60 unpress-Red-70 unpress-White-45 general --research-root outputs/case_study_20261113/research/default --candidates-root outputs/case_study_20261113/candidates/default --storage "sqlite:///outputs/case_study_20261113/research/study_default.db" --profiles-root outputs/case_study_20261113/projects --results-root outputs/case_study_20261113/infer
```

### Data 底下單一專案：用指定專案參數完整訓練（網頁預設即可顯示）
```cmd
:: 套用 press-Red-70 參數 → prepare / train / select-rois / all-rois → 分析全部，結果寫入 outputs/projects 等預設位置
uv run escount batch --template press-Red-70 --data-root Data --output outputs --projects press-Red-max --skip-research --no-general --infer
:: 之後用預設啟動網頁（已開著請 Ctrl+C 重啟，讓伺服器載入新模型）
uv run escount serve
```
* 一定要加 `--projects`，否則 `Data/` 底下全部專案（含範本本身）都會被重新校正與重訓；目標不要寫成範本自己。
* 已校正過、只重跑 1~4（全部步驟用同一組路徑）：
```cmd
uv run escount prepare --project press-Red-max
uv run escount train --project press-Red-max --device cuda
uv run escount select-rois --project press-Red-max
uv run escount all-rois --project press-Red-max --device cuda
uv run escount infer --project press-Red-max --input Data/press-Red-max --output outputs/infer/press-Red-max
```
注意 `escount calibrate` 會重寫 `project.json` 並清空 `strip_params`；要保留條帶請用 `copy-params` 或網頁校正頁「儲存」。run python scripts/check_licenses.py
```
