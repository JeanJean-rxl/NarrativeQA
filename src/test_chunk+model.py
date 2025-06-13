from datasets import load_dataset
from rag_pipeline import process_sample
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import time

def get_llm_pipeline(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# 加载 test 数据（预对齐）
ds = load_dataset("deepmind/narrativeqa")
test_data = ds["test"]
samples = test_data.select(range(100))

# 实验配置：chunk × model
configs = [
    ("google/flan-t5-base", 64),
    ("google/flan-t5-base", 128),
    ("google/flan-t5-base", 256),
    ("google/flan-t5-base", 512),
    ("google/flan-t5-large", 64),
    ("google/flan-t5-large", 128),
    ("google/flan-t5-large", 256),
    ("google/flan-t5-large", 512),
]

summary = []
all_results = []

for model_name, chunk_size in configs:
    print(f"\n🚀 Running {model_name} | chunk_words={chunk_size}")
    llm = get_llm_pipeline(model_name)

    results = []
    for i, sample in enumerate(tqdm(samples)):
        sample_dict = {
            "document": {"text": sample["document"]["text"]},
            "question": {"text": sample["question"]["text"]},
            "answers": sample["answers"]
        }
        try:
            result = process_sample(sample_dict, chunk_words=chunk_size, model=llm)
            result.update({
                "sample_index": i,
                "model": model_name,
                "chunk_words": chunk_size
            })
            results.append(result)
        except Exception as e:
            print(f"⚠️ Error on sample {i}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(f"data/rag_results_{model_name.split('/')[-1]}_chunk{chunk_size}.csv", index=False)
    all_results.extend(results)

    # 计算 summary
    fuzzy_over_90 = df[df["fuzzy_score"] >= 0.9]
    em_exact = df[df["exact_match"] == True]
    summary.append({
        "model": model_name,
        "chunk_words": chunk_size,
        "fuzzy>=0.9": len(fuzzy_over_90),
        "EM==True": len(em_exact),
        "total": len(df),
        "fuzzy_rate": len(fuzzy_over_90) / len(df),
        "em_rate": len(em_exact) / len(df)
    })

# 汇总 summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv("summary_results.csv", index=False)
print("\n📊 Summary:")
print(summary_df[["model", "chunk_words", "fuzzy_rate", "em_rate"]])