from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType

# initialize model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8, lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

# load dataset
ds = load_dataset("deepmind/narrativeqa", split="train[:1000]")

# preprocess function to create input-output pairs
def preprocess(example):
    question = example["question"]["text"]
    context = example["document"]["text"][:1500]
    answer = example["answers"][0]["text"] if example["answers"] else ""

    input_text = f"question: {question} context: {context}"
    target_text = answer

    model_inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    labels = tokenizer(
        target_text,
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# tokenize the dataset
tokenized = ds.map(preprocess, remove_columns=ds.column_names)

# set training arguments
args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="logs",
    save_total_limit=2,
    save_steps=10,
    logging_steps=5,
    evaluation_strategy="no"
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()
model.save_pretrained("checkpoints/last")
tokenizer.save_pretrained("checkpoints/last")