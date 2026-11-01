"""Generate comprehensive Word (.docx), PDF (.pdf), and PowerPoint (.pptx) documents
for ESCounting V4 Training and AutoResearch pipelines.
"""
import os
import subprocess
import sys
from pathlib import Path

# Ephemeral imports handled via `uv run --with python-docx --with python-pptx`
import docx
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

import pptx
from pptx import Presentation
from pptx.dml.color import RGBColor as PPTColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches as PPTInches, Pt as PPTPt

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

DOCX_PATH = DOCS_DIR / "ESCounting_V4_Training_and_AutoResearch_Guide.docx"
PDF_PATH = DOCS_DIR / "ESCounting_V4_Training_and_AutoResearch_Guide.pdf"
PPTX_PATH = DOCS_DIR / "ESCounting_V4_Training_and_AutoResearch_Slides.pptx"
HTML_TEMP_PATH = DOCS_DIR / "guide_temp.html"

# Color Palette
NAVY = RGBColor(26, 54, 93)       # #1A365D Primary
SLATE = RGBColor(45, 55, 72)      # #2D3748 Text
TEAL = RGBColor(13, 148, 136)     # #0D9488 Accent
LIGHT_BG = RGBColor(247, 250, 252)# #F7FAFC Shading
BORDER_GRAY = RGBColor(226, 232, 240)


def set_cell_background(cell, fill_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:val="clear" w:color="auto" w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    # Correct OpenXML sequence: top, left, bottom, right
    for m, val in [('top', top), ('left', left), ('bottom', bottom), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def build_docx():
    print(f"Building Word document: {DOCX_PATH}...")
    doc = Document()

    # Page setup - 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Document Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("ESCounting V4 電磁鋼片微米級計數系統")
    run_title.font.name = "Microsoft JhengHei"
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = NAVY

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(18)
    run_sub = p_sub.add_run("模型訓練 (Training) 與自動研究演化 (AutoResearch) 工作流程完整技術手冊")
    run_sub.font.name = "Microsoft JhengHei"
    run_sub.font.size = Pt(14)
    run_sub.font.bold = True
    run_sub.font.color.rgb = TEAL

    # Meta Info Table
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    col_widths = [Inches(3.2), Inches(3.3)]
    for row in meta_table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = col_widths[i]
            set_cell_background(cell, "F7FAFC")
            set_cell_margins(cell, 80, 80, 120, 120)

    r0c0 = meta_table.cell(0, 0).paragraphs[0].add_run("文件版本：v4.2.0 (正式發行版)")
    r0c1 = meta_table.cell(0, 1).paragraphs[0].add_run("發布日期：2026 年 11 月")
    r1c0 = meta_table.cell(1, 0).paragraphs[0].add_run("適用對象：產線演算法工程師、研究人員、專案主管")
    r1c1 = meta_table.cell(1, 1).paragraphs[0].add_run("智慧財產權：Proprietary (All Rights Reserved)")
    for r in [r0c0, r0c1, r1c0, r1c1]:
        r.font.name = "Microsoft JhengHei"
        r.font.size = Pt(9.5)
        r.font.color.rgb = SLATE

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Microsoft JhengHei"
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = NAVY
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Microsoft JhengHei"
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = TEAL
        return p

    def add_body(text, bold_prefix=None):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Microsoft JhengHei"
            r_pre.font.size = Pt(10.5)
            r_pre.font.bold = True
            r_pre.font.color.rgb = NAVY
        run = p.add_run(text)
        run.font.name = "Microsoft JhengHei"
        run.font.size = Pt(10.5)
        run.font.color.rgb = SLATE
        return p

    def add_callout(title, text):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        cell.width = Inches(6.5)
        set_cell_background(cell, "F0FDF4")
        set_cell_margins(cell, 120, 120, 180, 180)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        r_title = p.add_run(f"★ 核心重點：{title}\n")
        r_title.font.name = "Microsoft JhengHei"
        r_title.font.size = Pt(10.5)
        r_title.font.bold = True
        r_title.font.color.rgb = TEAL
        r_text = p.add_run(text)
        r_text.font.name = "Microsoft JhengHei"
        r_text.font.size = Pt(10)
        r_text.font.color.rgb = SLATE
        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # --- Section 1 ---
    add_h1("一、 系統全貌與核心架構理念")
    add_body("電磁鋼片堆疊計數在精密馬達、變壓器與發電機製造中是決定電磁性能與工安的關鍵指標。微米級（0.2mm ~ 0.5mm）鋼片在真實產線光學環境下，極易受到光影反射不均、邊界氧化、毛邊及輕微傾斜干擾。傳統固定閥值邊緣計數極不穩定，而單純的深度學習端到端回歸則缺乏物理可解釋性且容易過擬合。")
    add_body("ESCounting V4 結合了「光學物理幾何校正」與「金融工程量化技術指標（Technical Indicators）」，並透過「多 ROI 條帶融合」與「機器學習物理殘差回歸」，建立高強健度的完全精確計數流程。")
    add_callout("V4 的核心突破", "本系統定義的唯一成功指標為「完全精確命中率 (Exact Hit Rate: round(pred) == GT)」，並採用物理週期殘差預測 (Delta = GT - period_count)。只有當機器學習模型的命中率嚴格高於單一指標 Baseline 時才會被採用，兼顧物理確定性與模型適應性。")

    # --- Section 2 ---
    add_h1("二、 訓練工作流程 (Training Pipeline 詳解)")
    add_body("訓練流程由 5 個確定性環節構成（對應 Web UI 訓練導引之步驟 0 至 4）：")

    add_h2("步驟 0：光學旋轉校正 (Calibration)")
    add_body("藉由三個定位方塊的 FFT 頻譜分析，精準估計鋼片堆疊的微小旋轉傾角（補償至 0.01 度內），並自動生成堆疊區域的外接工作遮罩（Mask），儲存為專案基準檔 project.json。")

    add_h2("步驟 1：建立多特徵快取 (Prepare)")
    add_body("在校正遮罩內部，沿橫向均勻取樣多個具有代表性的垂直條帶（ROI）：")
    add_body("1. 多尺寸幾何條帶：支援寬度 w (12, 16, 20 px)、間距 D (30 px) 及橫向偏移 x_offset，並具備上下邊界安全緩衝（Padding），避免邊緣漏計。\n"
             "2. 三大投影信號：支援均值投影 (Mean)、高斯加權投影 (Gaussian)、中位數濾波投影 (Median)，有效壓抑單點髒污雜訊。\n"
             "3. 物理主頻率估算：利用 FFT 擷取主頻率，計算基底週期 period 與基本計數 period_count = round(L / period)。\n"
             "4. 18 組金融指標計數：套用 KD（隨機指標）、MACD（指數平滑異同移動平均線）、布林通道（Bollinger Bands）與波峰偵測，計算黃金交叉（波峰）、死亡交叉（波谷）及雙向平均計數。\n"
             "5. 產出特徵矩陣：將所有影像之幾何、SNR 信噪比、週期穩定度與各指標計數存為結構化 Parquet 快取。")

    add_h2("步驟 2：模型訓練與基準競賽 (Train)")
    add_body("採用嚴格的「留一影像交叉驗證 (Grouped-LOO / LOIO)」，並實施兩階段競賽評選機制：")
    add_body("• Baseline 最佳參考線自動評選 (BaselineProcedure)：遍歷資料集內所有單一指標與純幾何計數，在訓練折上以完全命中率排名，挑選表現最佳、所需條帶最少、寬度最小者作為專案 Baseline。\n"
             "• 機器學習殘差回歸 (ModelProcedure)：回歸器預測殘差 Delta = GT - period_count。候選模型包含 Ridge、SVR、XGBoost（支援 NVIDIA GPU CUDA 加速）與 LightGBM。\n"
             "• 嚴格採用法則 (Adoption Rule)：唯有當最佳 ML 模型的完全命中率「嚴格超越」Baseline 時，才將專案方法宣告為 'ml'；否則退回採用 'baseline'，杜絕過擬合。")

    add_h2("步驟 3：最小 ROI 集合精選 (Select-ROIs)")
    add_body("為達到產線邊緣端 < 1 秒的極速推論，採用貪婪前向選擇算法（Greedy Forward Selection），在 Nested LOIO 曲線上挑選互斥度高且彼此重疊小於 w/3 的 3~8 條最少 ROI 組合。")

    add_h2("步驟 4 & 5：推論部署與視覺化驗證 (Infer & Web UI)")
    add_body("推論引擎提供 FastAPI REST 服務，並在 Web 結果頁提供橫置切片影像軌道與 ECharts 訊號線 1:1 橫軸對齊、貫穿十字游標與雙向 DataZoom。同時支援即時動態切換 baseline / kd / macd / xgboost 預測方案。")

    # Table of Training Pipeline
    train_tbl = doc.add_table(rows=6, cols=3)
    train_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["階段步驟", "核心輸入 / 運算", "輸出產出物 / 驗證標準"]
    for i, h in enumerate(headers):
        cell = train_tbl.cell(0, i)
        set_cell_background(cell, "1A365D")
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.name = "Microsoft JhengHei"
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_margins(cell, 100, 100, 120, 120)

    rows_data = [
        ("0. 校正 (Calibrate)", "3 ROI 方塊 FFT 估角與遮罩邊界偵測", "project.json (旋轉傾角殘差 < 0.05°)"),
        ("1. 特徵工程 (Prepare)", "多 ROI 投影、FFT 週期、18 組指標交叉計數", "features.parquet (結構化快取)"),
        ("2. 競賽訓練 (Train)", "LOIO 完全命中率比較：Baseline vs ML (XGB/SVR)", "model/manifest.json (Adoption 判定)"),
        ("3. 挑選 ROI (Select)", "貪婪前向路徑，挑選無重疊最優子集", "project.json 更新 selected_rois (3~8 條)"),
        ("4. 產線推論 (Infer)", "單執行緒 CPU/GPU 高速推論、Web UI 即時對齊", "predictions.csv、推論延遲 < 1 秒")
    ]
    for row_idx, data in enumerate(rows_data, start=1):
        for col_idx, text in enumerate(data):
            cell = train_tbl.cell(row_idx, col_idx)
            set_cell_background(cell, "FFFFFF" if row_idx % 2 == 1 else "F7FAFC")
            set_cell_margins(cell, 80, 80, 120, 120)
            p = cell.paragraphs[0]
            r = p.add_run(text)
            r.font.name = "Microsoft JhengHei"
            r.font.size = Pt(9.5)
            r.font.color.rgb = SLATE

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # --- Section 3 ---
    add_h1("三、 AutoResearch（自動研究演化）工作流程詳解")
    add_body("AutoResearch 模組源自頂尖 AI 研究實驗室的自動化實作哲學，旨在完全免除人工盲目調參的低效與偏見，針對鋼片計數建構一套自主探索、閉環評估、自我修正的研究引擎。")

    add_h2("1. 龐大多維搜尋空間 (Search Space)")
    add_body("在每輪演化 Trial 中，Optuna TPE 取樣器會自動探索以下核心維度：")
    add_body("• 幾何參數空間：條帶寬度 w (6 ~ 40 px)、間距 D (6 ~ 90 px)、橫向偏移比例 x_frac。\n"
             "• 投影與濾波：均值、中位數、高斯濾波及其平滑尺度 sigma_ratio (0.1 ~ 0.6)。\n"
             "• 參考線內部參數網格：自動調配 KD 之 (n, k, d, gate)、MACD 之 (ema1, ema2, dea, gate)、布林通道寬度與波峰顯著度（配置於 configs/indicators.json）。\n"
             "• 回歸器超參數：Ridge 正則化係數 alpha、SVR 的 C 與 epsilon、XGBoost 的樹深/學習率/子取樣比，以及 LightGBM 葉節點數。\n"
             "• 決策聚合與條帶上限：搜尋 ROI 上限 roi_cap (1 ~ 8 條) 以及群組中位數或截尾平均聚合。")

    add_h2("2. 雙重研究範式：專案特化 (Project) vs 多組數據聯合 (General)")
    add_body("AutoResearch 原生支援兩大研究範式，並深入解答「是否聯合多組數據共同預測」之效益：")
    add_body("• 單一專案特化 (Project Study)：針對單一規格（如 0.2mm 厚度、T-type 機型）進行過擬合防護下的極致優化。\n"
             "• 多組數據聯合預測 (General Study)：\n"
             "  - 尺度歸一化特徵 (Scale-Normalised)：消除不同鏡頭距離或鋼片規格造成的絕對像素偏差，將特徵轉化為無因次相對比例 (cnt / period_count, w / period, x_norm)。\n"
             "  - 留一專案交叉驗證 (LOPO, Leave-One-Project-Out)：嚴格驗證跨專案泛化能力。在完全不提供目標專案任何訓練樣本的條件下，用其他專案聯合預測未見專案，量化多數據聯合的實際增益。\n"
             "  - 異常數據排除 (Excluded Projects)：支援於 configs/general.json 排除品質受損資料集（如 Ring-type），並可一鍵訓練 general vs general_no_ring 進行成效對照。")

    add_h2("3. 閉環評估指標與狀態判定 (Rank Key & Status)")
    add_body("每個 Trial 均由獨立固定評估器評分，其排序規則嚴格依據字典序 Tuple：")
    add_body("Rank Key = (-Exact_Hit_Rate, ROI_Count, Latency_ms)\n"
             "• keep（保留）：若當前 Trial 的評分嚴格優於目前候選者，即刻判定為 keep，自動生成候選檔案 candidates/<case>/best_params.json 與逐張影像預測表記錄 per_image.csv。\n"
             "• discard（捨棄）：若表現未超越歷史最佳，標記為 discard，並將試驗紀錄追加至 research/<case>/results.tsv。")

    add_h2("4. 自動化簡報級研究報告 (Report Generation)")
    add_body("執行指令 uv run escount report 即可一鍵編譯完整圖文報告（位於 outputs/research_report/）：")
    add_body("1. progress.png：展示 Optuna 隨試驗次數推進的完全命中率收斂軌跡。\n"
             "2. param_importance.png：利用 fANOVA 演算法計算各參數（w, D, 模型種類）對命中率的貢獻權重。\n"
             "3. param_sensitivity.png：2D 參數敏感度矩陣熱力圖，清晰視覺化參數高原區與危險邊界。\n"
             "4. project_vs_general.csv & .png：量化呈現各專案獨立預測 vs 多專案聯合預測在 LOIO / LOPO 下的命中率對比表。\n"
             "5. report.html & summary.md：完全獨立、內建 Base64 圖表與逐張影像檢驗表的離線網頁與 Markdown 摘要。")

    # --- Section 4 ---
    add_h1("四、 訓練 (Training) 與自動研究 (AutoResearch) 之協同分工")
    add_body("本專案將「研究探索」與「產線部署」進行了優雅的關注點分離：")
    add_body("• AutoResearch 扮演「開拓者 (Explorer)」：透過非線性隨機採樣與多維網格，尋求跨參數與跨專案聯合預測的最優前沿邊界，輸出位於 candidates/。\n"
             "• Training 扮演「部署者 (Deployer)」：依據確定性工程規則進行快速重現、防護校驗（Adoption Rule）、最小 ROI 提取與模型指紋簽署，產出正式推論 model/。\n"
             "• escount adopt 指令作為兩者的銜接橋樑，只複製候選參數作為歷史參照，而正式推論嚴格防守版本一致性，若校正檔更動則自動要求重新訓練，杜絕產線事故。")

    doc.save(str(DOCX_PATH))
    print(f"Successfully generated Word document: {DOCX_PATH}")


def build_pdf_via_edge():
    print(f"Building PDF document: {PDF_PATH} via Microsoft Edge headless...")
    html_content = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<title>ESCounting V4 電磁鋼片計數技術手冊</title>
<style>
  @page {
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-right {
      content: counter(page);
    }
  }
  body {
    font-family: 'Segoe UI', 'Microsoft JhengHei', -apple-system, sans-serif;
    color: #2D3748;
    line-height: 1.6;
    font-size: 10.5pt;
  }
  h1 {
    color: #1A365D;
    font-size: 18pt;
    border-bottom: 2px solid #1A365D;
    padding-bottom: 6px;
    margin-top: 24px;
    margin-bottom: 12px;
  }
  h2 {
    color: #0D9488;
    font-size: 13pt;
    margin-top: 18px;
    margin-bottom: 8px;
  }
  .doc-header {
    text-align: center;
    margin-bottom: 24px;
    border-bottom: 3px double #CBD5E0;
    padding-bottom: 16px;
  }
  .doc-title {
    font-size: 22pt;
    font-weight: bold;
    color: #1A365D;
    margin-bottom: 6px;
  }
  .doc-subtitle {
    font-size: 13pt;
    font-weight: 600;
    color: #0D9488;
    margin-bottom: 16px;
  }
  .meta-grid {
    display: table;
    width: 100%;
    background-color: #F7FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 10px;
    font-size: 9.5pt;
  }
  .meta-row { display: table-row; }
  .meta-cell { display: table-cell; padding: 4px 12px; }
  .callout {
    background-color: #F0FDF4;
    border-left: 4px solid #0D9488;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 16px 0;
  }
  .callout-title {
    font-weight: bold;
    color: #0D9488;
    margin-bottom: 4px;
  }
  table.data-table {
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 9.5pt;
  }
  table.data-table th, table.data-table td {
    border: 1px solid #CBD5E0;
    padding: 8px 12px;
    text-align: left;
  }
  table.data-table th {
    background-color: #1A365D;
    color: #FFFFFF;
    font-weight: bold;
  }
  table.data-table tr:nth-child(even) {
    background-color: #F7FAFC;
  }
  ul, ol {
    margin-top: 4px;
    margin-bottom: 8px;
    padding-left: 20px;
  }
  li {
    margin-bottom: 4px;
  }
  .formula {
    background-color: #F1F5F9;
    padding: 6px 12px;
    border-radius: 4px;
    font-family: Consolas, monospace;
    font-size: 10pt;
    display: inline-block;
  }
</style>
</head>
<body>

<div class="doc-header">
  <div class="doc-title">ESCounting V4 電磁鋼片微米級計數系統</div>
  <div class="doc-subtitle">模型訓練 (Training) 與自動研究演化 (AutoResearch) 工作流程完整技術手冊</div>
  <div class="meta-grid">
    <div class="meta-row">
      <div class="meta-cell"><strong>文件版本：</strong>v4.2.0 (正式發行版)</div>
      <div class="meta-cell"><strong>發布日期：</strong>2026 年 11 月</div>
    </div>
    <div class="meta-row">
      <div class="meta-cell"><strong>適用對象：</strong>產線演算法工程師、研究員、專案主管</div>
      <div class="meta-cell"><strong>智慧財產權：</strong>Proprietary (All Rights Reserved)</div>
    </div>
  </div>
</div>

<h1>一、 系統全貌與核心架構理念</h1>
<p>電磁鋼片堆疊計數在精密馬達、變壓器與發電機製造中是決定電磁性能與工安的關鍵指標。微米級（0.2mm ~ 0.5mm）鋼片在真實產線光學環境下，極易受到光影反射不均、邊界氧化、毛邊及輕微傾斜干擾。傳統固定閥值邊緣計數極不穩定，而單純的深度學習端到端回歸則缺乏物理可解釋性且容易過擬合。</p>
<p>ESCounting V4 結合了「光學物理幾何校正」與「金融工程量化技術指標（Technical Indicators）」，並透過「多 ROI 條帶融合」與「機器學習物理殘差回歸」，建立高強健度的完全精確計數流程。</p>

<div class="callout">
  <div class="callout-title">★ 核心突破與成功標準</div>
  本系統定義的唯一成功指標為「完全精確命中率 (Exact Hit Rate: round(pred) == GT)」，並採用物理週期殘差預測 (<span class="formula">&Delta; = GT - period_count</span>)。只有當機器學習模型的命中率嚴格高於單一指標 Baseline 時才會被採用，兼顧物理確定性與模型適應性。
</div>

<h1>二、 訓練工作流程 (Training Pipeline 詳解)</h1>
<p>訓練流程由 5 個確定性環節構成（對應 Web UI 訓練導引之步驟 0 至 4）：</p>

<h2>步驟 0：光學旋轉校正 (Calibration)</h2>
<p>藉由三個定位方塊的 FFT 頻譜分析，精準估計鋼片堆疊的微小旋轉傾角（補償至 0.01 度內），並自動生成堆疊區域的外接工作遮罩（Mask），儲存為專案基準檔 <code>project.json</code>。</p>

<h2>步驟 1：建立多特徵快取 (Prepare)</h2>
<ul>
  <li><strong>多尺寸幾何條帶：</strong>支援寬度 <code>w &isin; {12, 16, 20} px</code>、間距 <code>D (30 px)</code> 及橫向偏移 <code>x_offset</code>，並具備上下邊界安全緩衝（Padding），避免邊緣漏計。</li>
  <li><strong>三大投影信號：</strong>支援均值投影 (Mean)、高斯加權投影 (Gaussian)、中位數濾波投影 (Median)，有效壓抑單點髒污雜訊。</li>
  <li><strong>物理主頻率估算：</strong>利用 FFT 擷取主頻率，計算基底週期 <code>period</code> 與基本計數 <span class="formula">period_count = round(L / period)</span>。</li>
  <li><strong>18 組金融指標計數：</strong>套用 KD（隨機指標）、MACD（指數平滑異同移動平均線）、布林通道（Bollinger Bands）與波峰偵測，計算金叉（波峰）、死叉（波谷）及雙向平均計數。</li>
  <li><strong>產出特徵矩陣：</strong>將所有影像之幾何、SNR 信噪比、週期穩定度與各指標計數存為結構化 <code>features.parquet</code> 快取。</li>
</ul>

<h2>步驟 2：模型訓練與基準競賽 (Train)</h2>
<ul>
  <li><strong>Baseline 最佳參考線自動評選 (BaselineProcedure)：</strong>遍歷資料集內所有單一指標與純幾何計數，在訓練折上以完全命中率排名，挑選表現最佳、所需條帶最少、寬度最小者作為專案 Baseline。</li>
  <li><strong>機器學習殘差回歸 (ModelProcedure)：</strong>回歸器預測殘差 <span class="formula">&Delta; = GT - period_count</span>。候選模型包含 Ridge、SVR、XGBoost（支援 NVIDIA GPU CUDA 加速）與 LightGBM。</li>
  <li><strong>嚴格採用法則 (Adoption Rule)：</strong>唯有當最佳 ML 模型的完全命中率「嚴格超越」Baseline 時，才將專案方法宣告為 <code>method = 'ml'</code>；否則退回採用 <code>'baseline'</code>，杜絕過擬合。</li>
</ul>

<h2>步驟 3：最小 ROI 集合精選 (Select-ROIs)</h2>
<p>為達到產線邊緣端 &lt; 1 秒的極速推論，採用貪婪前向選擇算法（Greedy Forward Selection），在 Nested LOIO 曲線上挑選互斥度高且彼此重疊小於 <code>w/3</code> 的 3~8 條最少 ROI 組合。</p>

<h2>步驟 4 &amp; 5：推論部署與視覺化驗證 (Infer &amp; Web UI)</h2>
<p>推論引擎提供 FastAPI REST 服務，並在 Web 結果頁提供橫置切片影像軌道與 ECharts 訊號線 1:1 橫軸對齊、貫穿十字游標與雙向 DataZoom。同時支援即時動態切換 <code>baseline / kd / macd / xgboost</code> 預測方案。</p>

<table class="data-table">
  <thead>
    <tr>
      <th>階段步驟</th>
      <th>核心輸入 / 運算</th>
      <th>輸出產出物 / 驗證標準</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>0. 校正 (Calibrate)</strong></td>
      <td>3 ROI 方塊 FFT 估角與遮罩邊界偵測</td>
      <td>project.json (旋轉傾角殘差 &lt; 0.05°)</td>
    </tr>
    <tr>
      <td><strong>1. 特徵工程 (Prepare)</strong></td>
      <td>多 ROI 投影、FFT 週期、18 組指標交叉計數</td>
      <td>features.parquet (結構化快取)</td>
    </tr>
    <tr>
      <td><strong>2. 競賽訓練 (Train)</strong></td>
      <td>LOIO 完全命中率比較：Baseline vs ML (XGB/SVR)</td>
      <td>model/manifest.json (Adoption 判定)</td>
    </tr>
    <tr>
      <td><strong>3. 挑選 ROI (Select)</strong></td>
      <td>貪婪前向路徑，挑選無重疊最優子集</td>
      <td>project.json 更新 selected_rois (3~8 條)</td>
    </tr>
    <tr>
      <td><strong>4. 產線推論 (Infer)</strong></td>
      <td>單執行緒 CPU/GPU 高速推論、Web UI 即時對齊</td>
      <td>predictions.csv、推論延遲 &lt; 1 秒</td>
    </tr>
  </tbody>
</table>

<h1>三、 AutoResearch（自動研究演化）工作流程詳解</h1>
<p>AutoResearch 模組源自頂尖 AI 研究實驗室的自動化實作哲學，旨在完全免除人工盲目調參的低效與偏見，針對鋼片計數建構一套自主探索、閉環評估、自我修正的研究引擎。</p>

<h2>1. 龐大多維搜尋空間 (Search Space)</h2>
<ul>
  <li><strong>幾何參數空間：</strong>條帶寬度 <code>w (6 ~ 40 px)</code>、間距 <code>D (6 ~ 90 px)</code>、橫向偏移比例 <code>x_frac</code>。</li>
  <li><strong>投影與濾波：</strong>均值、中位數、高斯濾波及其平滑尺度 <code>sigma_ratio (0.1 ~ 0.6)</code>。</li>
  <li><strong>參考線內部參數網格：</strong>自動調配 KD 之 (n, k, d, gate)、MACD 之 (ema1, ema2, dea, gate)、布林通道寬度與波峰顯著度（配置於 <code>configs/indicators.json</code>）。</li>
  <li><strong>回歸器超參數：</strong>Ridge 正則化係數 alpha、SVR 的 C 與 epsilon、XGBoost 的樹深/學習率/子取樣比，以及 LightGBM 葉節點數。</li>
  <li><strong>決策聚合與條帶上限：</strong>搜尋 ROI 上限 <code>roi_cap (1 ~ 8 條)</code> 以及群組中位數或截尾平均聚合。</li>
</ul>

<h2>2. 雙重研究範式：專案特化 (Project) vs 多組數據聯合 (General)</h2>
<p>AutoResearch 原生支援兩大研究範式，並深入解答「是否聯合多組數據共同預測」之效益：</p>
<ul>
  <li><strong>單一專案特化 (Project Study)：</strong>針對單一規格（如 0.2mm 厚度、T-type 機型）進行過擬合防護下的極致優化。</li>
  <li><strong>多組數據聯合預測 (General Study)：</strong>
    <ul>
      <li><strong>尺度歸一化特徵 (Scale-Normalised)：</strong>消除不同鏡頭距離或鋼片規格造成的絕對像素偏差，將特徵轉化為無因次相對比例 (<span class="formula">cnt / period_count</span>, <span class="formula">w / period</span>, <span class="formula">x_norm</span>)。</li>
      <li><strong>留一專案交叉驗證 (LOPO, Leave-One-Project-Out)：</strong>嚴格驗證跨專案泛化能力。在完全不提供目標專案任何訓練樣本的條件下，用其他專案聯合預測未見專案，量化多數據聯合的實際增益。</li>
      <li><strong>異常數據排除 (Excluded Projects)：</strong>支援於 <code>configs/general.json</code> 排除品質受損資料集（如 Ring-type），並可一鍵訓練 general vs general_no_ring 進行成效對照。</li>
    </ul>
  </li>
</ul>

<h2>3. 閉環評估指標與狀態判定 (Rank Key &amp; Status)</h2>
<p>每個 Trial 均由獨立固定評估器評分，其排序規則嚴格依據字典序 Tuple：</p>
<div class="formula">Rank Key = (-Exact_Hit_Rate, ROI_Count, Latency_ms)</div>
<ul>
  <li><strong>keep（保留）：</strong>若當前 Trial 的評分嚴格優於目前候選者，即刻判定為 keep，自動生成候選檔案 <code>candidates/&lt;case&gt;/best_params.json</code> 與逐張影像預測表記錄 <code>per_image.csv</code>。</li>
  <li><strong>discard（捨棄）：</strong>若表現未超越歷史最佳，標記為 discard，並將試驗紀錄追加至 <code>research/&lt;case&gt;/results.tsv</code>。</li>
</ul>

<h2>4. 自動化簡報級研究報告 (Report Generation)</h2>
<p>執行指令 <code>uv run escount report</code> 即可一鍵編譯完整圖文報告（位於 <code>outputs/research_report/</code>）：</p>
<ol>
  <li><strong>progress.png：</strong>展示 Optuna 隨試驗次數推進的完全命中率收斂軌跡。</li>
  <li><strong>param_importance.png：</strong>利用 fANOVA 演算法計算各參數（w, D, 模型種類）對命中率的貢獻權重。</li>
  <li><strong>param_sensitivity.png：</strong>2D 參數敏感度矩陣熱力圖，清晰視覺化參數高原區與危險邊界。</li>
  <li><strong>project_vs_general.csv &amp; .png：</strong>量化呈現各專案獨立預測 vs 多專案聯合預測在 LOIO / LOPO 下的命中率對比表。</li>
  <li><strong>report.html &amp; summary.md：</strong>完全獨立、內建 Base64 圖表與逐張影像檢驗表的離線網頁與 Markdown 摘要。</li>
</ol>

<h1>四、 訓練 (Training) 與自動研究 (AutoResearch) 之協同分工</h1>
<p>本專案將「研究探索」與「產線部署」進行了優雅的關注點分離：</p>
<ul>
  <li><strong>AutoResearch 扮演「開拓者 (Explorer)」：</strong>透過非線性隨機採樣與多維網格，尋求跨參數與跨專案聯合預測的最優前沿邊界，輸出位於 <code>candidates/</code>。</li>
  <li><strong>Training 扮演「部署者 (Deployer)」：</strong>依據確定性工程規則進行快速重現、防護校驗（Adoption Rule）、最小 ROI 提取與模型指紋簽署，產出正式推論 <code>model/</code>。</li>
  <li><code>escount adopt</code> 指令作為兩者的銜接橋樑，只複製候選參數作為歷史參照，而正式推論嚴格防守版本一致性，若校正檔更動則自動要求重新訓練，杜絕產線事故。</li>
</ul>

</body>
</html>
"""
    HTML_TEMP_PATH.write_text(html_content, encoding="utf-8")
    edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    cmd = [
        edge_path,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={PDF_PATH.resolve()}",
        str(HTML_TEMP_PATH.resolve()),
    ]
    subprocess.run(cmd, check=True)
    if HTML_TEMP_PATH.exists():
        HTML_TEMP_PATH.unlink()
    print(f"Successfully generated PDF document: {PDF_PATH}")


def build_pptx():
    print(f"Building PowerPoint presentation: {PPTX_PATH}...")
    prs = Presentation()
    prs.slide_width = PPTInches(13.333)  # 16:9 widescreen
    prs.slide_height = PPTInches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Colors
    C_NAVY = PPTColor(26, 54, 93)       # #1A365D
    C_TEAL = PPTColor(13, 148, 136)     # #0D9488
    C_SLATE = PPTColor(45, 55, 72)      # #2D3748
    C_MUTED = PPTColor(100, 116, 139)   # #64748B
    C_BG_CARD = PPTColor(247, 250, 252) # #F7FAFC
    C_WHITE = PPTColor(255, 255, 255)
    C_GOLD = PPTColor(217, 119, 6)

    def add_header(slide, title_text, category="ESCounting V4 技術專題"):
        # Top Category Accent
        cat_box = slide.shapes.add_textbox(PPTInches(0.8), PPTInches(0.5), PPTInches(11.5), PPTInches(0.4))
        p_cat = cat_box.text_frame.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.name = "Microsoft JhengHei"
        p_cat.font.size = PPTPt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_TEAL

        # Slide Main Title
        title_box = slide.shapes.add_textbox(PPTInches(0.8), PPTInches(0.85), PPTInches(11.5), PPTInches(0.8))
        p_title = title_box.text_frame.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = "Microsoft JhengHei"
        p_title.font.size = PPTPt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = C_NAVY

    def add_card(slide, left, top, width, height, title, items, header_color=C_NAVY, bg_color=C_BG_CARD):
        # Card Background Shape
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = PPTColor(226, 232, 240)
        shape.line.width = PPTPt(1)

        # Content Text Box
        tb = slide.shapes.add_textbox(left + PPTInches(0.2), top + PPTInches(0.2), width - PPTInches(0.4), height - PPTInches(0.4))
        tf = tb.text_frame
        tf.word_wrap = True

        p_head = tf.paragraphs[0]
        p_head.text = title
        p_head.font.name = "Microsoft JhengHei"
        p_head.font.size = PPTPt(14)
        p_head.font.bold = True
        p_head.font.color.rgb = header_color
        p_head.space_after = PPTPt(8)

        for item in items:
            p = tf.add_paragraph()
            p.text = f"•  {item}"
            p.font.name = "Microsoft JhengHei"
            p.font.size = PPTPt(11)
            p.font.color.rgb = C_SLATE
            p.space_after = PPTPt(5)

    # ---------------- SLIDE 1: Title Slide ----------------
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, PPTInches(13.333), PPTInches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = C_NAVY
    bg1.line.fill.background()

    tb1 = s1.shapes.add_textbox(PPTInches(1.2), PPTInches(2.0), PPTInches(11.0), PPTInches(3.5))
    tf1 = tb1.text_frame
    p_tag = tf1.paragraphs[0]
    p_tag.text = "INDUSTRIAL PRECISION METROLOGY & AI SYSTEM"
    p_tag.font.name = "Microsoft JhengHei"
    p_tag.font.size = PPTPt(13)
    p_tag.font.bold = True
    p_tag.font.color.rgb = C_TEAL
    p_tag.space_after = PPTPt(10)

    p_main = tf1.add_paragraph()
    p_main.text = "ESCounting V4 電磁鋼片計數"
    p_main.font.name = "Microsoft JhengHei"
    p_main.font.size = PPTPt(38)
    p_main.font.bold = True
    p_main.font.color.rgb = C_WHITE
    p_main.space_after = PPTPt(8)

    p_sub = tf1.add_paragraph()
    p_sub.text = "模型訓練 (Training) 與自動研究演化 (AutoResearch) 工作流程完整解析"
    p_sub.font.name = "Microsoft JhengHei"
    p_sub.font.size = PPTPt(18)
    p_sub.font.color.rgb = PPTColor(203, 213, 225)
    p_sub.space_after = PPTPt(20)

    p_foot = tf1.add_paragraph()
    p_foot.text = "全可編輯投影片組 · 金融工程指標量化 · 物理殘差迴歸 · 多資料集聯合預測"
    p_foot.font.name = "Microsoft JhengHei"
    p_foot.font.size = PPTPt(12)
    p_foot.font.color.rgb = C_TEAL

    # ---------------- SLIDE 2: 系統全貌與痛點解決 ----------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "工業現場痛點與 V4 破局核心理念")

    add_card(s2, PPTInches(0.8), PPTInches(1.8), PPTInches(5.6), PPTInches(5.0),
             "真實產線光學量測痛點",
             [
                 "微米級鋼片（0.2mm~0.5mm）反射不均與氧化雜訊。",
                 "鋼片衝壓邊界存在微小毛邊、陰影與堆疊傾斜。",
                 "傳統邊緣閥值（Thresholding）易受照度影響導致誤判。",
                 "黑盒子端到端深度學習參數量大，容易過擬合且缺乏物理可解釋性。"
             ],
             header_color=PPTColor(185, 28, 28))

    add_card(s2, PPTInches(6.8), PPTInches(1.8), PPTInches(5.7), PPTInches(5.0),
             "ESCounting V4 創新作法",
             [
                 "光學幾何校正：3 ROI 方塊 FFT 估角，補償至 0.01 度內。",
                 "量化指標交叉驗證：借鑑金融量化指標（KD / MACD / 布林 / 波峰）。",
                 "物理週期殘差學習：目標回歸 Delta = GT - period_count。",
                 "硬核命中標準：以完全精確命中率 (Exact Hit Rate) 作為唯一決策依據。",
                 "嚴格 Adoption 防護：ML 表現必須嚴格超越 Baseline 始予採用。"
             ],
             header_color=C_TEAL)

    # ---------------- SLIDE 3: 訓練 5 步鏈條架構 ----------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "模型訓練 (Training) 五大確定性步驟鏈條")

    step_data = [
        ("0. 校正 Calibrate", ["3 定位方塊 FFT 估角", "旋轉傾角殘差 < 0.05°", "產出 project.json 遮罩"]),
        ("1. 特徵 Prepare", ["多條帶 (w/D/x) 投影", "FFT 提取主頻率 period", "18 組 KD/MACD 指標計數"]),
        ("2. 訓練 Train", ["LOIO 留一影像交叉評估", "Baseline 最佳參考線競賽", "XGBoost 殘差迴歸學習"]),
        ("3. 挑選 Select", ["貪婪前向選擇算法", "重疊 < w/3 空間互斥", "鎖定 3~8 條最少 ROI"]),
        ("4. 推論 Infer", ["單執行緒極速 (< 1 秒)", "Web UI 十字游標貫穿", "支援方案動態切換"])
    ]

    card_w = PPTInches(2.2)
    gap = PPTInches(0.2)
    left_start = PPTInches(0.8)
    for idx, (title, items) in enumerate(step_data):
        c_left = left_start + idx * (card_w + gap)
        add_card(s3, c_left, PPTInches(1.9), card_w, PPTInches(4.8), title, items, header_color=C_NAVY)

    # ---------------- SLIDE 4: 特徵工程與物理週期殘差 ----------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "特徵工程：多重投影與物理週期殘差機制")

    add_card(s4, PPTInches(0.8), PPTInches(1.8), PPTInches(5.6), PPTInches(5.0),
             "三大投影方式與幾何緩衝",
             [
                 "橫向投影 (Projection)：沿鋼片寬度垂直壓縮為 1D 條帶訊號。",
                 "均值 (Mean)：平均壓抑高斯噪聲，忠實保留邊緣階躍。",
                 "高斯 (Gaussian)：以條帶中心為核 (sigma = ratio * w)，排除邊緣刮痕。",
                 "中位數 (Median)：非線性濾波，對偶發亮點毛邊具備免疫力。",
                 "邊界安全緩衝 (Padding)：上下邊緣預留 Padding，杜絕頂底漏計。"
             ],
             header_color=C_NAVY)

    add_card(s4, PPTInches(6.8), PPTInches(1.8), PPTInches(5.7), PPTInches(5.0),
             "物理週期殘差學習 (Delta Regression)",
             [
                 "傳統直接預測：直接預測 160 片，誤差難以約束且依賴大樣本。",
                 "物理基底：period_count = round(L / dominant_period)。",
                 "殘差目標：模型僅學習微小偏差 Delta = GT - period_count。",
                 "單 ROI 預測：pred_row = period_count + Model(Features)。",
                 "整張影像聚合：以多條 ROI 之 Median（中位數）抗雜訊輸出。"
             ],
             header_color=C_TEAL)

    # ---------------- SLIDE 5: Baseline 自動評選與嚴格 Adoption ----------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "模型評選機制：Baseline 競賽與 Adoption 法則")

    add_card(s5, PPTInches(0.8), PPTInches(1.8), PPTInches(5.6), PPTInches(5.0),
             "Baseline 最佳參考線自動評選",
             [
                 "全指標搜尋：遍歷所有 KD、MACD、布林、波峰與純幾何計數。",
                 "嚴格分組：在訓練折以 (w, D, x_offset, 投影法) 佈局組分群。",
                 "命中率優先：以 LOIO 完全精確命中數 (Exact Hits) 作為首要準則。",
                 "次級指標仲裁：命中數相同時，取所需 ROI 數量最少者。",
                 "成本仲裁：若仍平手，取條帶寬度 w 最小（計算成本最低）者。"
             ],
             header_color=C_NAVY)

    add_card(s5, PPTInches(6.8), PPTInches(1.8), PPTInches(5.7), PPTInches(5.0),
             "Adoption 採用法則 (守護產線確定性)",
             [
                 "評估架構：LOIO 留一影像驗證，確保影像未參與訓練。",
                 "ML 候選群：Ridge、SVR、XGBoost (GPU+CPU)、LightGBM。",
                 "採用門檻：Best_ML_Hits 必須「嚴格大於」Baseline_Hits。",
                 "自動退回防護：若 ML 未超越 Baseline，自動標記採用 Baseline！",
                 "模型指紋保護：若旋轉校正重新儲存，舊模型自動失效阻擋載入。"
             ],
             header_color=PPTColor(180, 83, 9))

    # ---------------- SLIDE 6: AutoResearch 自動研究演化架構 ----------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "AutoResearch 自動研究：免人工閉環最佳化")

    add_card(s6, PPTInches(0.8), PPTInches(1.8), PPTInches(3.7), PPTInches(5.0),
             "1. 多維空間取樣 (Suggest)",
             [
                 "條帶幾何：w (6~40)、D (6~90)、橫向偏移比例。",
                 "投影平滑：Mean / Gaussian / Median 及 sigma_ratio。",
                 "指標參數：KD / MACD / 波峰內部參數自訂網格。",
                 "模型超參：SVR C/eps、XGB 深度/學習率/採樣比。",
                 "ROI 上限：roi_cap (1~8 條) 與聚合方式。"
             ],
             header_color=C_NAVY)

    add_card(s6, PPTInches(4.8), PPTInches(1.8), PPTInches(3.7), PPTInches(5.0),
             "2. 閉環目標評估 (Evaluate)",
             [
                 "專屬特徵快取：依幾何 Hash 自動多程序生成特徵。",
                 "固定運算預算：每張影像採樣列數與樹輪數固定。",
                 "固定評估器：LOIO 交叉驗證計算完全命中率。",
                 "延遲監控：單張推論延遲精準測量至毫秒。",
                 "日誌追加：即時寫入 results.tsv 留存全軌跡。"
             ],
             header_color=C_TEAL)

    add_card(s6, PPTInches(8.8), PPTInches(1.8), PPTInches(3.7), PPTInches(5.0),
             "3. 狀態決策與保留 (Keep)",
             [
                 "排序依據：Rank Key = (-Hit_Rate, ROI_Count, Latency)。",
                 "優化判定：嚴格優於當前最優解時標記為 keep。",
                 "產出候選：自動持久化 candidates/<case>/best_params.json。",
                 "逐張驗證：匯出 per_image.csv 記錄每張影像預測值。",
                 "歷史沉澱：未超越者標記 discard 供重要度分析。"
             ],
             header_color=PPTColor(21, 128, 61))

    # ---------------- SLIDE 7: 單一專案特化 vs 多組數據聯合 (General) ----------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "研究範式：單一專案特化 vs 多組數據聯合預測")

    add_card(s7, PPTInches(0.8), PPTInches(1.8), PPTInches(5.6), PPTInches(5.0),
             "單一專案特化研究 (Project Study)",
             [
                 "適用場景：特定產品線（如 0.2mm 厚度、專用 T-type）。",
                 "特徵空間：使用絕對物理特徵（像素寬度 w、絕對位置 x、L）。",
                 "評估方法：LOIO（留一影像交叉驗證）。",
                 "優勢：針對單一幾何構型具備極致的針對性與高命中率。",
                 "限制：跨厚度或鏡頭距離變更時需個別校正適應。"
             ],
             header_color=C_NAVY)

    add_card(s7, PPTInches(6.8), PPTInches(1.8), PPTInches(5.7), PPTInches(5.0),
             "多組數據聯合共同預測 (General Study)",
             [
                 "尺度歸一化特徵 (Scale-Normalised)：消除絕對像素差異：\n  cnt / period_count,  w / period,  x_norm",
                 "雙重驗證標準：\n  • LOIO：同專案影像之泛化能力。\n  • LOPO (Leave-One-Project-Out)：留一專案驗證，完全不見目標專案進行預測，檢驗真實泛化性！",
                 "靈活排除 (configs/general.json)：排除雜訊異常專案（如 Ring-type），並支援 general vs general_no_ring 對比。"
             ],
             header_color=C_TEAL)

    # ---------------- SLIDE 8: 自動化研究報告與產出物 ----------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "自動化簡報級研究報告 (AutoResearch Report)")

    add_card(s8, PPTInches(0.8), PPTInches(1.8), PPTInches(5.6), PPTInches(5.0),
             "一鍵產出五大高解析視覺化圖表 (200 DPI)",
             [
                 "1. progress.png：展示命中率隨試驗進展之階梯收斂軌跡。",
                 "2. param_importance.png：fANOVA 演算法解析 w、D、模型重要度。",
                 "3. param_sensitivity.png：2D 參數敏感度熱力圖（定位性能高原區）。",
                 "4. project_vs_general.csv & .png：量化各專案獨立 vs 聯合預測差異。",
                 "5. latency.png：各候選模型單張推論毫秒耗時分佈。"
             ],
             header_color=C_NAVY)

    add_card(s8, PPTInches(6.8), PPTInches(1.8), PPTInches(5.7), PPTInches(5.0),
             "離線獨立整合報告 (HTML & Markdown)",
             [
                 "report.html：零外部依賴、純離線 Base64 圖表整合報告。",
                 "summary.md：GitHub 與主管匯報友善之 Markdown 摘要。",
                 "候選規格檔：candidates/<case>/best_params.json 完整記錄最優解。",
                 "逐張誤差表記錄：per_image.csv 精確記錄每張鋼片之 GT 與預測值。",
                 "正式推論隔離：escount adopt 僅作封存記錄，推論維持 model/ 穩定。"
             ],
             header_color=C_TEAL)

    # ---------------- SLIDE 9: Training 與 AutoResearch 協同矩陣 ----------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "兩大模組協同架構：探索 (Research) 與部署 (Deploy)")

    # Comparison Table
    rows = 5
    cols = 3
    left = PPTInches(0.8)
    top = PPTInches(1.8)
    width = PPTInches(11.7)
    height = PPTInches(4.8)

    table_shape = s9.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table
    table.columns[0].width = PPTInches(2.5)
    table.columns[1].width = PPTInches(4.6)
    table.columns[2].width = PPTInches(4.6)

    t_headers = ["比較維度", "AutoResearch (自動研究探索)", "Training Pipeline (產線正式部署)"]
    for i, h in enumerate(t_headers):
        cell = table.cell(0, i)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Microsoft JhengHei"
        p.font.size = PPTPt(12)
        p.font.bold = True
        p.font.color.rgb = C_WHITE

    t_data = [
        ("核心目標", "開拓探索 (Exploration)：搜尋最優幾何、指標組合與模型", "穩定利用 (Exploitation)：快速訓練、確定性防護與邊緣推論"),
        ("運作模式", "非同步、多輪 Optuna 隨機取樣與 fANOVA 重要度分析", "同步 5 步鏈條 (0~4 步)，Web UI 一鍵引導執行"),
        ("產出位置", "candidates/<case>/ (候選解) + research/ (日誌/圖表)", "outputs/projects/<case>/model/ (正式推論二進制與 manifest)"),
        ("銜接機制", "產出最優幾何與指標超參，作為演算法演進指引", "escount adopt 複製候選作為備份；推論透過指紋校驗確保安全")
    ]

    for r_idx, row_values in enumerate(t_data, start=1):
        for c_idx, val in enumerate(row_values):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PPTColor(255, 255, 255) if r_idx % 2 == 1 else PPTColor(247, 250, 252)
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = "Microsoft JhengHei"
            p.font.size = PPTPt(11)
            p.font.color.rgb = C_SLATE

    # ---------------- SLIDE 10: 總結與交付 ----------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "結論與後續成果應用建議")

    add_card(s10, PPTInches(0.8), PPTInches(1.8), PPTInches(11.7), PPTInches(5.0),
             "技術突破與應用價值總覽",
             [
                 "✔ 完全精確命中率第一：打破傳統估計片數容許 ±1 片的妥協，直接鎖定 100% 精準計數。",
                 "✔ 金融工程技術指標賦能：KD 與 MACD 等指標在光學條帶訊號中展現極高穩健度，抗陰影能力強大。",
                 "✔ 物理殘差迴歸杜絕過擬合：只學習週期微小殘差，兼顧模型靈活性與物理基底可靠性。",
                 "✔ AutoResearch 實現無人值守進化：自動探索多專案聯合泛化能力，並提供 LOPO 跨場景嚴格檢驗。",
                 "✔ 多格式交付便於引用：本套件已同步輸出 Word 說明書、向量 PDF 與原生可編輯 PowerPoint 投影片，方便向研發團隊及高層管理層進行專題簡報。"
             ],
             header_color=C_TEAL)

    prs.save(str(PPTX_PATH))
    print(f"Successfully generated PowerPoint presentation: {PPTX_PATH}")


def main():
    build_docx()
    build_pdf_via_edge()
    build_pptx()
    print("All artifacts successfully created!")


if __name__ == "__main__":
    main()
