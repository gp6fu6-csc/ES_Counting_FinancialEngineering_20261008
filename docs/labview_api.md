# ESCounting V4 推論 Server：LabVIEW 呼叫說明

`escount serve` 啟動 FastAPI 推論服務，LabVIEW（或任何能送 HTTP 的語言）可用兩種方式送圖：
**檔案路徑**（Server 自己讀圖）或**影像資料**（PNG/BMP 位元組直接上傳）。互動式 API 文件（OpenAPI / Swagger UI）在 `http://<host>:<port>/docs`。

## 1. 啟動

```powershell
# 本機使用（預設 127.0.0.1:8000，不需 API Key）
uv run escount serve

# 開放給其他電腦（非 127.0.0.1 一律需要 API Key，否則拒絕啟動）
$env:ESCOUNT_API_KEY = "<自訂金鑰>"   # 建議：用環境變數，金鑰不會出現在行程清單
uv run escount serve --host 0.0.0.0 --port 8000 --root Data --root D:\Images
# 或 uv run escount serve --host 0.0.0.0 --api-key <自訂金鑰> ...（金鑰會出現在行程命令列）
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--host` | `127.0.0.1` | 綁定位址；非 loopback（如 `0.0.0.0`）必須同時給 API Key（`--api-key` 或 `ESCOUNT_API_KEY`） |
| `--port` | `8000` | 連接埠 |
| `--api-key` | 環境變數 `ESCOUNT_API_KEY` | 設定後，除 `/health`、`/docs`、`/openapi.json` 與 `/ui/` 靜態頁外，每個請求都要帶 `X-API-Key` 標頭，否則回 401 |
| `--root` | `Data` | `/infer/path` 只允許讀取這些資料夾內的檔案（可重複指定） |
| `--profiles-root` | `outputs/projects` | 專案設定檔（`project.json`、`model/`）所在處 |
| `--config` | `configs/general.json` | General 模型設定（決定正式 General 模型） |

推論採單一行程、單執行緒 CPU，依序處理（同時只跑一張），不寫任何輸出檔。

## 2. 呼叫流程

1. `GET /health` 確認服務存活。
2. `POST /projects/{專案ID}/load` 指定專案情境並一次載入所有參數與模型（之後的推論都不含載入時間）。
3. 重複呼叫 `POST /infer/path` 或 `POST /infer/image` 取得片數 `N_pred`。

未先 load 就推論 → **409**。可同時載入多個專案；未指定 `project` 時使用最後一次 load 的專案。

## 3. 端點

### `GET /health`
回應：`{"status": "ok", "version": "0.1.0"}`

### `GET /projects`
列出已校正的專案、可用的 General 模型與目前載入的引擎：
```json
{"projects": [{"case": "T-type", "quality_status": "ok", "angle_deg": -1.0875, "reference_image": "171.png",
               "has_model": true, "selected_rois": 1, "loaded_models": ["project"]}],
 "general_models": ["general"],
 "current": {"case": "T-type", "model": "project", "model_case": "T-type", "method": "baseline",
             "model_type": "ridge", "roi_count": 1, "image_size": {"width": 2448, "height": 2048}}}
```

### `POST /projects/{case}/load`
Body（可省略）：`{"model": "project"}` 或 `{"model": "general"}`（以該專案幾何套用正式 General 模型）。
回應：`{"loaded": true, "case": "T-type", "model": "project", "model_case": "T-type", "method": "baseline", "model_type": "ridge", "roi_count": 1, "image_size": {...}}`
錯誤：404 找不到 `project.json`；422 專案未通過校正品質或尚未訓練 / 選 ROI。

### `POST /infer/path`（模式 A：傳檔案路徑）
Header `Content-Type: application/json`；Body：
```json
{"path": "F:\\Kiro\\20261005_ESCounting_V4\\Data\\T-type\\160.png", "project": "T-type", "model": "project"}
```
`project`、`model` 可省略（使用目前載入的專案）。路徑經 `resolve()` 後必須位於 `--root` 指定的資料夾內；含 `..` 或在範圍外 → **403**；檔案不存在 → 404；副檔名非 `.png/.bmp` → 415。相對路徑以 Server 啟動目錄為基準，建議一律用絕對路徑（JSON 字串中的 `\` 要寫成 `\\`）。

### `POST /infer/image`（模式 B：直接傳影像資料）
二擇一：
- **原始位元組**：`Content-Type: image/png`（或 `image/bmp`、`application/octet-stream`），Body 為整個 PNG/BMP 檔內容。
- **multipart**：`Content-Type: multipart/form-data`，欄位名稱 `file`。

Query 參數（皆可省略）：`project`、`model`、`name`（檔名，如 `160.png`；檔名為數字時會回傳 GT 與是否命中）。
例：`POST /infer/image?project=T-type&name=160.png`。
影像須與專案影像同尺寸（2448×2048）；彩色圖會先轉灰階。非 PNG/BMP → 415；無法解碼 → 400；尺寸不符 → 422。

### 推論回應（兩種模式相同）
```json
{"image": "160.png", "source": "F:\\...\\Data\\T-type\\160.png", "case": "T-type", "model": "project",
 "model_case": "T-type", "method": "baseline", "model_type": "ridge",
 "N_pred": 160, "gt": 160, "hit": true,
 "rois": [{"index": 1, "key": "mean|34|1805", "projection": "mean", "w": 34, "x": 1805, "y0": 180, "y1": 1890,
           "sigma": null, "period_count": 159, "prediction": 159.5}],
 "timings_ms": {"read": 43.0, "rotate": 24.1, "extent": 13.2, "features": 7.1, "model": 0.1, "aggregate": 0.1, "total": 87.6},
 "source_sha256_before": "..."}
```
`N_pred` 為最終整數片數；`timings_ms` 為各階段耗時（不含模型載入；模式 B 無 `read`，`source` 為 `null`）。錯誤時回應 `{"detail": "說明文字"}`。

## 4. LabVIEW 實作（HTTP Client VIs）

LabVIEW 內建 **Data Communication » Protocols » HTTP Client** 函式庫：

1. `OpenHandle.vi` 取得 client handle。
2. 若 Server 有設定 API Key：`AddHeader.vi`，header = `X-API-Key`，value = 金鑰。
3. 載入專案：`AddHeader.vi`（`Content-Type` = `application/json`）→ `POST.vi`，URL = `http://<host>:8000/projects/T-type/load`，buffer = `{"model":"project"}`。
4. 推論：
   - 模式 A：`POST.vi`，URL = `http://<host>:8000/infer/path`，buffer = 以 `Flatten To JSON` 或字串組出的 `{"path":"D:\\Images\\160.png"}`。
   - 模式 B：先將影像存成 PNG 位元組（IMAQ `IMAQ Write String`／或讀取 PNG 檔為 string），`AddHeader.vi`（`Content-Type` = `image/png`）→ `POST.vi`，URL = `http://<host>:8000/infer/image?name=160.png`，buffer = PNG 位元組字串。
5. `Unflatten From JSON`（或 JSONtext 等函式庫）解析回應，取 `N_pred`；`GetHeader.vi`／狀態碼判斷 200 以外的錯誤並顯示 `detail`。
6. 結束時 `CloseHandle.vi`。

建議：一個 handle 重複使用以減少連線開銷；載入（步驟 3）只需在切換專案時做一次。
