from escount.inference.engine import InferenceEngine

# 1. 指定並載入專案模型（耗時僅載入一次）
engine = InferenceEngine.load("x05_unpress")

# 2. 傳入影像路徑進行推論（或直接傳入 OpenCV uint8 灰階矩陣）
result = engine.infer("Data/x05_unpress/test2.png")

# 3. 取得預測片數與耗時資訊
pred_count = result["N_pred"]               # 預測片數 (例如: 105)
total_ms = result["timings_ms"]["total"]     # 總推論耗時 (例如: 325.5 ms)

print(f"預測鋼片數量：{pred_count} 片")
print(f"單張推論耗時：{total_ms:.1f} 毫秒")