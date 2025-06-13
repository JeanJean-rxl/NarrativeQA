from datasets import load_dataset
from rag_pipeline import process_sample
import pandas as pd
from tqdm import tqdm

# ✅ 直接加载 narrativeqa 数据集（包括 test split）
ds = load_dataset("deepmind/narrativeqa")
test_data = ds["test"]

# ✅ 选择前 100 条样本（已是预对齐结构）
samples = test_data.select(range(100))

# ✅ 调用 RAG pipeline 逐条处理
results = []
for i, sample in enumerate(tqdm(samples)):
    # 保证字段结构和 rag_pipeline 兼容（有些 Dataset 项不是纯 dict）
    sample_dict = {
        "document": {"text": sample["document"]["text"]},
        "question": {"text": sample["question"]["text"]},
        "answers": sample["answers"]
    }
    result = process_sample(sample_dict)
    result["sample_index"] = i
    results.append(result)

# ✅ 存储或查看结果
df = pd.DataFrame(results)
df.to_csv("narrativeqa_test100_results_flan-t5-large.csv", index=False)
print(df[["question", "prediction", "answers", "fuzzy_score"]])

# 你已有的 DataFrame df
high_fuzzy = df[df["fuzzy_score"] >= 0.9]
high_em = df[df["exact_match"] == True]

print(f"🔹 fuzzy_score ≥ 0.9 的样本占比: {len(high_fuzzy) / len(df):.2%}")
print(f"🔹 exact_match == True 的样本占比: {len(high_em) / len(df):.2%}")