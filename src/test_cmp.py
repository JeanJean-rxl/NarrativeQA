import pandas as pd
import glob
import re

files = glob.glob("data/rag_results_*.csv")

summary_rows = []

for file in files:
    print(f"📊 processing: {file}")
    df = pd.read_csv(file)

    match = re.search(r"data/rag_results_(.+)_chunk(\d+)\.csv", file)
    if match:
        model = match.group(1)
        chunk_words = int(match.group(2))
    else:
        print(f"⚠️ filename unmatch: {file}")
        continue

    total = len(df)
    fuzzy_50 = len(df[df["fuzzy_score"] >= 0.5])
    fuzzy_80 = len(df[df["fuzzy_score"] >= 0.8])
    em_true = len(df[df["exact_match"] == True])

    summary_rows.append({
        "model": model,
        "chunk_words": chunk_words,
        "scope_tokens": chunk_words * 3,  
        "fuzzy_rate_50": fuzzy_50 / total,
        "fuzzy_rate_80": fuzzy_80 / total,
        "em_rate": em_true / total,
        "total": total
    })


summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(by=["model", "chunk_words"])
summary_df.to_csv("summary_results.csv", index=False)
print(summary_df)