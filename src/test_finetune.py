from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 加载 NarrativeQA 训练集（建议先取部分调试）
# ds = load_dataset("deepmind/narrativeqa", split="train[:1000]")
ds = load_dataset("deepmind/narrativeqa", split="train[:1000]", download_mode="force_redownload")

# 构造训练样本
def preprocess(example):
    q = example["question"]["text"]
    ctx = example["document"]["text"][:1500]
    a = example["answers"][0]["text"] if example["answers"] else ""
    return {"input_text": f"question: {q} context: {ctx}", "target_text": a}

ds = ds.map(preprocess)

# Tokenize
def tokenize(example):
    input = tokenizer(example["input_text"], max_length=512, truncation=True, padding="max_length")
    target = tokenizer(example["target_text"], max_length=64, truncation=True, padding="max_length")
    input["labels"] = target["input_ids"]
    return input

ds = ds.map(tokenize, remove_columns=ds.column_names)
train_valid = ds.train_test_split(test_size=0.1)

# 训练参数
args = TrainingArguments(
    output_dir="./flan-t5-finetune",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    fp16=True,  # when using GPU with mixed precision
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_valid["train"],
    eval_dataset=train_valid["test"],
)

trainer.train()