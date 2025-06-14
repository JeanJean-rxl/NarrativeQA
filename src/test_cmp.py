import pandas as pd
import glob
import re

# 所有结果文件（你可以根据实际目录调整）
files = glob.glob("data/rag_results_*.csv")

summary_rows = []

for file in files:
    print(f"📊 处理文件: {file}")
    df = pd.read_csv(file)

    # 提取模型名和chunk大小
    match = re.search(r"data/rag_results_(.+)_chunk(\d+)\.csv", file)
    if match:
        model = match.group(1)
        chunk_words = int(match.group(2))
    else:
        print(f"⚠️ 文件名解析失败: {file}")
        continue

    # 统计 fuzzy_score ≥ 0.8 和 exact_match 的样本数量
    total = len(df)
    fuzzy_80 = len(df[df["fuzzy_score"] >= 0.8])
    em_true = len(df[df["exact_match"] == True])

    summary_rows.append({
        "model": model,
        "chunk_words": chunk_words,
        "scope_tokens": chunk_words * 3,  # 假设 top_k = 3
        "fuzzy_rate_80": fuzzy_80 / total,
        "em_rate": em_true / total,
        "total": total
    })

# 生成汇总表格
summary_df = pd.DataFrame(summary_rows)
# print(summary_df)
summary_df = summary_df.sort_values(by=["model", "chunk_words"])
summary_df.to_csv("summary_results.csv", index=False)
print(summary_df)