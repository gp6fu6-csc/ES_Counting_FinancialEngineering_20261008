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
  * 依照畫面提示依序按下按鈕：**0. 校正 $\to$ 1. 建立特徵 $\to$ 2. 訓練模型 $\to$ 3. 挑選 ROI $\to$ 4. 分析全部**。
* **旋轉校正頁**：`http://localhost:8000/ui/calibration.html`
  * 檢視 3 個定位方塊與 FFT 旋轉角度補償，微調後點選「儲存校正結果」。
* **推論結果檢視頁**：`http://localhost:8000/ui/results.html`
  * 點擊影像清單切換檢視；
  * **★ 功能拓展需求選單**：右上角可即時切換預測模型（`baseline`、`kd`、`macd`、`xgboost`、`auto`），立即動態更新片數；
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

### 步驟 3：最小 ROI 集合挑選 (Select-ROIs)
```powershell
uv run escount select-rois --project x05_unpress --max-rois 25
# 參數：
#   --max-rois 25         : 最大挑選條帶數量上限（通常選出 3~8 條）
```

### 步驟 4：正式推論 (Infer CLI)
```powershell
# 單張影像推論（指定專案與預測方案）
uv run escount infer --project x05_unpress --input Data/x05_unpress/test2.png --method auto --output outputs/infer

# 常用參數：
#   --project <專案名>     : 專案名稱 (如 x05_unpress)
#   --input <檔案或目錄>   : 輸入待分析影像或資料夾
#   --method <方案>        : 自由切換預測方案: auto / baseline / kd / macd / xgboost (預設 auto)
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

### 3.3 切換「特化專案模型」vs「跨專案聯合通用模型 (General Model)」
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

### 3.4 動態即時調整技術指標參數（無需重新訓練）
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

### 3.5 批量推論並輸出檔案 (Run Batch)
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
uv run pytest

# 執行 55 項依賴套件開源授權合規審查（確保無 GPL 傳染性與商業版稅）
uv run python scripts/check_licenses.py
```
