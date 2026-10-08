# ESCounting V4（電磁鋼片計數）

以影像計算電磁鋼片真實堆疊數量：**校正 → 訓練資料準備 → 模型訓練 / 參數研究 → 推論**，
並提供 CLI、推論 Server（供 LabVIEW 等系統呼叫）與網頁 UI。完整規劃見
[docs/實作計畫_ESCounting_V4.md](docs/實作計畫_ESCounting_V4.md)。

> 狀態：草稿（Task 1：專案骨架與既有程式遷移）。後續功能依計畫逐步加入。

## 環境建置（uv + .venv，不影響既有環境）

需求：[uv](https://docs.astral.sh/uv/)（會自動取得 Python 3.12）。

```powershell
uv sync                 # 依 uv.lock 建立 .venv 並安裝鎖定版本
uv run pytest           # 執行測試（需於 repo 根目錄；部分測試會使用本機 Data/）
uv run python scripts/check_licenses.py   # 檢查所有依賴授權
```

## 使用（目前可用：遷移自 V3 的流程）

```powershell
uv run escount --version
uv run escount legacy-count run-all --data-root Data --output-root outputs/es_counting
uv run escount legacy-count calibrate --scenario Data/T-type
uv run escount legacy-count predict --params outputs/es_counting/T-type --input <資料夾或檔案> --output outputs/pred
uv run escount legacy-rotate --input Data/T-type/160.png --output-dir outputs/rotation_160
```

## 專案結構

```
escount/
  calibration/   旋轉校正（rotation_correction.py：3 ROI FFT 估角）
  signals/       ROI bank 與寬度方向投影
  indicators/    財務指標（MA / EMA / MACD / KD / 布林 / 峰值 / 週期）
  features/      特徵擷取與快取
  models/        Project / General 模型
  inference/     推論引擎
  research/      autoresearch 參數研究與報告
  server/        推論 Web Service
  ui/            校正與結果網頁
  es_counting.py V3 計片流程（KD / MACD + LOO + XGBoost），行為不變
  cli.py         `escount` 指令入口
scripts/check_licenses.py   依賴授權檢查
tests/                       單元測試
```

`Data/`、`outputs/`、`*.7z`、PDF 不納入版本控制。

## 授權

本專案為 Proprietary（All rights reserved），見 [LICENSE](LICENSE)。
所有第三方依賴僅允許 MIT / BSD / Apache-2.0 / PSF / ISC / LGPL（動態連結），
由 `scripts/check_licenses.py` 檢查，GPL / AGPL / SSPL / 商業授權一律拒絕；影像使用 `opencv-python-headless`。
