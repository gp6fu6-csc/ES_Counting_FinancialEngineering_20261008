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

## 排除的專案（Ring-type 例外）

Ring-type 分析結果不可靠（Project 模型 LOIO 完全命中僅 3/6，legacy 1/6），依使用者決定視為例外，
以設定排除於 `prepare --all`、`train`、`research`、`report` 與 General 模型之外；`Data/Ring-type` 原始檔不變。
設定位於 `configs/general.json`：

```json
{"excluded_projects": ["Ring-type"], "exclude_from_general": ["Ring-type"]}
```

重新納入：自 `excluded_projects` 移除 `Ring-type`（若也自 `exclude_from_general` 移除，General 將包含 Ring-type；
保留時會同時訓練 `general` 與 `general_no_ring` 供比較），再重跑 `prepare` / `train` / `research` / `report`。
研究結果與專案組合綁定：更改排除清單後請使用新的 `--storage` / `--research-root` / `--candidates-root`。

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
SPDX `A AND B` 授權須每一部分都在允許清單內。

明確列出的例外（`scripts/check_licenses.py` 的 `ACCEPTED_EXCEPTIONS`）：

| 套件 | 授權 | 原因 |
|---|---|---|
| tqdm 4.70.1 | MPL-2.0 AND MIT | optuna 的必要依賴，未修改原始碼直接使用；MPL-2.0 僅為檔案層級 copyleft（修改 tqdm 本身的檔案才須公開該檔），不收費、不限制使用 |

httpx（certifi 為 MPL-2.0）未列為例外，測試改用 `tests/asgi_client.py`。

## 研究成果與正式推論

`escount research` / `escount report` 的命中率屬於研究程序（`candidates/<case>/best_params.json`）。
正式推論（`escount infer`、Server、網頁）只使用 `escount train` + `escount select-rois` 產生的
`outputs/projects/<case>/model/`；`escount adopt <case>` 只把候選設定複製到
`outputs/projects/<case>/research/` 作為紀錄，推論不會讀取。`select-rois` 沿貪婪路徑（最多 `--max-rois`）
以 nested 曲線（每張影像都用不含它的選擇結果評分）決定 ROI 數：命中數優先、同分取最少 ROI；baseline 模型的
完整 ROI 組（單一 bank）誠實命中數較多時改部署完整組。簡報請引用 manifest `selection.deployed`
（`escount report` 的「正式推論模型」表），部署後以 `escount benchmark` 確認 < 1 s。
校正（CLI 或網頁）重新儲存後，舊模型會被拒絕載入，需重跑 `prepare` / `train` / `select-rois`。
