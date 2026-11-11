# ============================================================================
# escount batch 教學範例 (Sample usage of `escount batch`)
#
# 用途：指定一個「已校正好的範本專案」，把它的旋轉校正 ROI、計片條帶參數（以及
#       background_BW.png，若目標資料夾沒有）套用到資料集群中的每個子資料夾，
#       接著自動完成：
#         1. 每個專案：旋轉角度重算 -> prepare -> train -> select-rois -> all-rois
#         2. 通用模型（所有專案合併訓練）
#         3. AutoResearch 四軌：default / mae / exact_hit / ensemble
#         4. 跨專案研究報告 + summary.md / summary.csv 交叉對比表
#
# 資料規則：
#   - 只有「檔名 = 片數」的影像（例如 190.png、083_2.png）會拿來訓練與研究。
#   - test1.png、test.txt、子資料夾（例如 test\）、名稱以 test 開頭的資料夾 一律不使用。
#   - 少於 3 張可訓練影像的資料夾會被略過。
#
# 所有輸出都寫到 --output（預設 outputs\case_study\），不會動到 outputs\projects\。
# 執行方式：在專案根目錄開 PowerShell，複製下面任一段指令執行。
# ============================================================================

# ---------------------------------------------------------------------------
# 範例 0：先「預演」，只列出會處理哪些專案、不寫入任何檔案（建議每次先跑）
# ---------------------------------------------------------------------------
uv run escount batch --template press-Red-70 --dry-run

# ---------------------------------------------------------------------------
# 範例 1：最常用。Data\Case_Study 底下全部專案，套用 Data\press-Red-70 的參數
#         （預設 --data-root = Data\Case_Study，--trials 10，--device cuda）
# ---------------------------------------------------------------------------
uv run escount batch --template press-Red-70

# ---------------------------------------------------------------------------
# 範例 2：沒有 GPU、先小規模試跑（每軌只跑 2 個 trials）
# ---------------------------------------------------------------------------
uv run escount batch --template press-Red-70 --device cpu --trials 2

# ---------------------------------------------------------------------------
# 範例 3：只處理其中幾個專案
# ---------------------------------------------------------------------------
uv run escount batch --template press-Red-70 --projects press-Red-20 press-White-18

# ---------------------------------------------------------------------------
# 範例 4：多組資料集群（可放在其他磁碟），用空格隔開多個路徑
#         注意：不同路徑底下的子資料夾名稱不能重複，重複時程式會直接停止並提示。
# ---------------------------------------------------------------------------
uv run escount batch --template press-Red-70 --data-root Data\Case_Study D:\新資料\批次A

# ---------------------------------------------------------------------------
# 範例 5：範本專案放在別處（直接給資料夾路徑；其 project.json 由 --template-profiles-root 指定）
# ---------------------------------------------------------------------------
uv run escount batch --template D:\舊資料\press-Red-70 --template-profiles-root D:\舊輸出\projects

# ---------------------------------------------------------------------------
# 範例 6：每次研究輸出到不同資料夾，方便保留歷次結果互相比較
# ---------------------------------------------------------------------------
uv run escount batch --template press-Red-70 --output outputs\case_study_20261111

# ---------------------------------------------------------------------------
# 範例 7：只要訓練、不跑 AutoResearch 與報告（快速建立各專案模型）
# ---------------------------------------------------------------------------
uv run escount batch --template press-Red-70 --skip-research

# ---------------------------------------------------------------------------
# 範例 8：重跑時保留已手動調整過的 project.json（不備份、不覆蓋）
#         預設行為：已存在的 project.json 會先改名為 project_YYYYMMDD_N.json 再套用範本。
# ---------------------------------------------------------------------------
uv run escount batch --template press-Red-70 --keep-existing

# ---------------------------------------------------------------------------
# 完成後要看的檔案（以預設 --output outputs\case_study 為例）
# ---------------------------------------------------------------------------
#   outputs\case_study\summary.md              各專案各步驟狀態 + 四軌最佳結果交叉對比表
#   outputs\case_study\summary.csv             同上（Excel 可開）
#   outputs\case_study\mask_overlays\*.png      每個專案的有效區域紅色疊圖（請抽查遮罩是否正確）
#   outputs\case_study\report\latest\report.html        跨專案研究報告（default 軌 + 通用模型）
#   outputs\case_study\report_mae\latest\report.html    mae 軌報告
#   outputs\case_study\report_exact_hit\latest\report.html
#   outputs\case_study\report_ensemble\latest\report.html
#   outputs\case_study\projects\<專案>\model\  各專案推論用模型
#
# 遮罩不對時：到校正頁 (0) 背景 BW 確認 重新圈選該專案的 background_BW.png
#             （Data\Case_Study\<專案>\），再用 --projects <專案> 重跑該專案即可。
#
# 用 batch 訓練好的模型做推論（Python）：
#   from escount.inference.engine import InferenceEngine
#   engine = InferenceEngine.load("press-Red-20", profiles_root="outputs/case_study/projects")
#   print(engine.infer("Data/Case_Study/press-Red-20/test1.png")["N_pred"])
#
# 用 batch 訓練好的模型做推論（CLI）：
#   uv run escount infer --project press-Red-20 --profiles-root outputs\case_study\projects `
#       --input Data\Case_Study\press-Red-20\test1.png --output outputs\case_study\infer
