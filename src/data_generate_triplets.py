import re
import random
from datasets import load_dataset
import json

def split_into_passages(text, n=100):
    # 简单将长文划分为大约 n 字的段落
    sentences = re.split(r'(?<=[.?!])\s+', text)
    passages, current = [], []
    length = 0
    for s in sentences:
        current.append(s)
        length += len(s.split())
        if length >= n:
            passages.append(" ".join(current))
            current, length = [], 0
    if current:
        passages.append(" ".join(current))
    return passages

def construct_triplet(example, negatives_pool=10):
    question = example["question"]["text"]
    answers = [a["text"] for a in example["answers"]]
    doc_text = example["document"]["text"]
    passages = split_into_passages(doc_text)

    # 1. 找到 positive passage（包含任一答案 substring）
    positive = None
    for p in passages:
        if any(a.lower() in p.lower() for a in answers):
            positive = p
            break
    if not positive:
        return None  # 无有效positive，跳过

    # 2. 筛选 negative passages（不含答案）
    negatives = [p for p in passages if not any(a.lower() in p.lower() for a in answers)]
    if len(negatives) < 1:
        return None  # 无负例，跳过

    selected_negatives = random.sample(negatives, k=min(3, len(negatives)))

    return {
        "query": question,
        "positive": positive,
        "negatives": selected_negatives
    }

ds = load_dataset("deepmind/narrativeqa")
# samples = ds["train"].select(range(5000))
samples = ds["train"]

triplets = []
cnt = 0

for i, sample in enumerate(samples):
    triplet = construct_triplet(sample)
    if triplet:
        triplets.append(triplet)
    else:
        cnt += 1

# Save as JSON Lines
with open("data/triplets_trainall.jsonl", "w", encoding="utf-8") as f:
    for t in triplets:
        f.write(json.dumps(t, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(triplets)} triplets to triplets.jsonl")
print(f"❌ Total samples with no valid triplet: {cnt}")