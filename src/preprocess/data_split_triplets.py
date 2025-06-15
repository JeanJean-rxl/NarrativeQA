import json
import random

input_path = "data/triplets_trainall.jsonl"
train_path = "data/triplets_trainall_train.jsonl"
dev_path = "data/triplets_trainall_dev.jsonl"

# 加载所有 triplets
with open(input_path, "r") as f:
    triplets = [json.loads(line) for line in f]

# 打乱顺序
random.shuffle(triplets)

# 划分 500 条为 dev
dev_size = 1000
dev_triplets = triplets[:dev_size]
train_triplets = triplets[dev_size:]

# 保存两个文件
with open(train_path, "w") as f:
    for item in train_triplets:
        f.write(json.dumps(item) + "\n")

with open(dev_path, "w") as f:
    for item in dev_triplets:
        f.write(json.dumps(item) + "\n")

print(f"✅ Saved {len(train_triplets)} train and {len(dev_triplets)} dev triplets.")