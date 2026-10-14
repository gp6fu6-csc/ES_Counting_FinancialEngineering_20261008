# ESCounting V4 交接文件（人與 AI Agent 皆可閱讀）

> 更新：2026-10（Task 1–12 完成並通過 3 輪審查；Task 13 release 未完成）
> 分支：`feat/escount-v4`（main 只有 Task 1 骨架）｜Repo：`gp6fu6-csc/ES_Counting_FinancialEngineering_20261008`（private）

## 1. 一句話現況

程式**可以使用**：校正、訓練、推論（CLI / Server / 網頁）都能在真實資料上運作，單張 CPU 推論約 0.4 秒。尚未完成的是「長時間正式研究 + 最終報告 + PR」。

## 1.1 進度紀錄（收尾工作，依序執行，每項完成後暫停確認）

| # | 項目 | 狀態 | 說明 |
|---|---|---|---|
| 0 | 網頁「訓練」頁（prepare → train → select-rois → infer 引導） | ✅ 完成 | `escount/ui/static/training.html/.js`；API `GET /training/{case}/status`、`POST /training/{case}/run`；commit `201bf18` |
| 1 | 根網址 `http://127.0.0.1:8000/` 404 | ✅ 完成 | 根網址自動導向 `/ui/training.html`（設定 API key 時也可開啟）；`tests/test_server.py::test_root_redirects_to_ui` |
| 2 | 訓練頁自動化測試 | ✅ 完成 | `tests/test_training_page.py`（6 tests，以假的子行程取代真實 CLI）：步驟狀態、順序檢查、成功/失敗記錄、同時只跑一個工作、train 後卸載模型、infer 參數；全部 118 tests 通過。ServerConfig 新增 `features_root` |
| 3 | 開 PR 並合併 `feat/escount-v4` → `main` | ⏳ 未開始 | 無 `gh`，用 GitHub REST API |
| 4 | 正式研究與簡報報告 | ⏳ 未開始 | `escount research --trials 50` → `escount report`，檢查 `outputs/research_report/latest/` 每個 case 檔案齊全 |

網頁入口：啟動 `uv run escount serve` 後直接開 `http://127.0.0.1:8000/`（導向訓練頁，上方可切換 校正 / 訓練 / 結果）。

## 2. 快速開始（Windows PowerShell，於專案根目錄）

```powershell
uv sync                                   # 建立 .venv（不影響系統 Python）
uv run pytest -q                          # 111 tests
# 推論：單檔或資料夾
uv run escount infer --project T-type --input Data/T-type --output outputs/infer_T
uv run escount infer --project 20260904-thinkness02_postition2 --input Data/20260904-thinkness02_postition2/098.png --output outputs/one
#   --no-plots 可關閉 analysis.png；--model general 改用 General 模型
# Server + 網頁 UI（預設 127.0.0.1:8000，API 文件 /docs）
uv run escount serve
```

新專案的完整流程：
```powershell
uv run escount calibrate --project Data/<專案>      # position2 需加 --roi-side 512
uv run escount prepare   --project <專案>
uv run escount train     --project <專案>           # 或 --general；--device cuda|cpu
uv run escount select-rois --project <專案>         # 必須執行，否則 infer 會報 no selected_rois
uv run escount benchmark --project <專案>
uv run escount infer     --project <專案> --input <檔案或資料夾>
```
研究與報告：`escount research --trials N | --hours H`、`escount report`（輸出 `outputs/research_report/latest/report.html`）。

## 3. 目前實測結果（2026-10 本次驗證，in-sample：模型訓練時看過這些影像）

| 專案 | ROI 數 | 推論命中 | 誠實估計（LOIO，未見過的影像） | 單張耗時 |
|---|---|---|---|---|
| position1（19 張） | 19 | 19/19 | 15/19（全 ROI 16/19） | ~0.4 s |
| position2（14 張） | 8 | 14/14 | 13/14 | ~0.4 s |
| T-type（4 張） | 75 | 3/4（160 預測 157） | 3/4 | ~0.4 s |
| Ring-type | — | 已排除 | 舊版 1/6、新版 3/6 | — |

舊版（V3）LOO 命中：8/19、10/14、1/6、3/4。**評估請以 LOIO 誠實估計為準**，in-sample 數字偏樂觀。

## 4. 已確定的需求與設計（變更前請先與使用者確認）

- 唯一準確度指標：**完全命中率**；同分比 ROI 數少、延遲短。驗證以影像為單位留出（LOIO），General 另做 LOPO。
- ROI：每列距 background_BW 有效區左右 ≥ 50 px；相鄰重疊 < w/3（D > 2w/3）；垂直涵蓋整個堆疊；**垂直方向不平滑**。
- 投影：寬度方向 mean / gaussian / median。指標：MA、EMA、MACD、KD、布林、find_peaks、FFT 週期（皆以 px 為單位、參數可調）。
- 模型：Ridge、SVR、XGBoost（CUDA，自動退回 CPU）、LightGBM；ML 需勝過最佳單一指標 baseline 才採用。
- Ring-type 以 `configs/general.json`（excluded_projects）排除；`Data/` 不可修改（`scripts/check_data_sha.py` 驗證）。
- 授權：只用 MIT/BSD/Apache/PSF/ISC；`uv run python scripts/check_licenses.py` 檢查。httpx 因 certifi（MPL-2.0）未採用。本專案 Proprietary。
- Server：預設 127.0.0.1；對外開放（--host 0.0.0.0）必須設 API key（`--api-key` 或環境變數 `ESCOUNT_API_KEY`）；`/infer/path` 限制在 `--root`。
- GitHub token 只放在 `%USERPROFILE%\.escount_github_token`，絕不可 commit；commit 前用 `git grep` 搜尋 GitHub token 前綴（g-h-p-底線），必須無輸出。

## 5. 程式結構

```
escount/
  calibration/   方形 ROI FFT 旋轉校正 → project.json（含 legacy rotation_correction）
  signals/       ROI bank（幾何規則強制）與寬度方向投影
  indicators/    財務指標與計數器
  features/      特徵快取（parquet，多行程）
  models/        訓練、模型儲存、最少 ROI 選擇、延遲基準
  inference/     InferenceEngine（單檔 / 資料夾，輸出開關、dpi）
  research/      Optuna 研究器、固定評估器、報告產生
  server/        FastAPI（/health /projects /projects/{id}/load /infer/path /infer/image /preview）
  ui/            Vue3 + ECharts（vendored，離線）校正頁、結果頁
  es_counting.py legacy V3（escount legacy-count）
configs/general.json  General 設定（排除清單）
outputs/projects/<專案>/  project.json、model/、manifest.json（不進 git）
```
其他文件：`docs/實作計畫_ESCounting_V4.md`（完整計畫）、`docs/ui_acceptance.md`（UI 手動驗收）、README、LabVIEW API 說明（server 文件）。

## 6. 尚未完成（後續工作，依優先序）

1. **推送與 PR**：`feat/escount-v4` 已推送到遠端（若未推送，見下方指令），需開 PR 合併到 main（無 `gh`，用 GitHub REST API）。
2. **正式研究**：每個 case（position1、position2、T-type、general）至少 `--trials 50`，接著 `escount report`；確認 `outputs/research_report/latest/` 下每個 case 都有 best_params.json、results.tsv、progress.png、param_importance.png、param_sensitivity.png、per_image.csv，comparison/ 檔案齊全，report.html 已嵌入圖片。
3. **position1 與 general 的最少 ROI** 誠實命中略低於全 ROI（全 ROI 違反重疊規則無法部署），可增加資料或擴大搜尋空間改善。
4. **T-type 160.png 少算 3 片**；T-type 只有 4 張，需補資料。
5. **Ring-type** 需更多資料或專屬方法，之後從 `configs/general.json` 排除清單移除即可重新納入。
6. 審查留下的 LOW 等級建議：見 `.agents/tasks/` 下 review.md。
7. `escount adopt` 目前只做紀錄，推論仍使用 train + select-rois 的模型（設計如此，可再討論是否改為直接部署）。

推送指令（token 不寫入 git 設定）：
```powershell
$tok=[IO.File]::ReadAllText("$env:USERPROFILE\.escount_github_token")
$b=[Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("x-access-token:$tok"))
git -c http.extraheader="AUTHORIZATION: basic $b" push -u https://github.com/gp6fu6-csc/ES_Counting_FinancialEngineering_20261008.git feat/escount-v4
```

## 7. 給 AI Agent 的規則

- 先讀本檔與 `.agents/tasks/escount-v4-implementation-plan.md`。
- Shell 指令前綴 `rtk`；用 uv；不修改 `Data/`；每次變更後跑 `uv run pytest -q`、授權檢查、Data SHA 檢查。
- 需求不明確時先問使用者，不要自行假設。
