import argparse
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType


def preprocess(example, tokenizer):
    question = example["question"]["text"]
    context = example["document"]["text"][:1500]
    answer = example["answers"][0]["text"] if example["answers"] else ""

    input_text = f"question: {question} context: {context}"
    target_text = answer

    model_inputs = tokenizer(
        input_text, max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        target_text, max_length=64, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_model(
    model_name,
    dataset_split,
    num_train_epochs,
    batch_size,
    lora_r,
    lora_alpha,
    lora_dropout
):
    # Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none"
    )
    model = get_peft_model(model, peft_config)

    # Load and preprocess dataset
    ds = load_dataset("deepmind/narrativeqa", split=dataset_split)
    # ds = load_dataset("deepmind/narrativeqa", split="train[:1000]")
    tokenized = ds.map(lambda ex: preprocess(ex, tokenizer), remove_columns=ds.column_names)

    # Output directory
    if dataset_split.endswith("[:5000]"):
        num_dataset_split = "train5000"
    output_dir = f'checkpoints_{model_name.split("/")[1]}_{num_dataset_split}_{num_train_epochs}_{batch_size}_{lora_r}_{lora_alpha}_{lora_dropout}'
    os.makedirs(output_dir, exist_ok=True)

    # Training args
    args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    evaluation_strategy="no",  
    report_to="tensorboard"
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
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Training complete. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on NarrativeQA")

    parser.add_argument("--model_name", type=str, default="google/flan-t5-base",
                        help="Base model to fine-tune")
    parser.add_argument("--dataset_split", type=str, default="train[:5000]",
                        help="Subset of the dataset to use")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout")

    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        dataset_split=args.dataset_split,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )