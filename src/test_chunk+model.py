from datasets import load_dataset
from rag_pipeline import process_sample
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import torch
from rag_pipeline import process_sample
import torch
torch.set_num_threads(1)


def get_llm_pipeline(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def get_llm_from_pretrained(model_path):
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def llm_infer(prompt, max_new_tokens=64):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

        return [
            {"generated_text": tokenizer.decode(output, skip_special_tokens=True)}
            for output in outputs
        ]

    return llm_infer  # ⚠️ 返回的是函数（行为类似 pipeline）

# 加载 test 数据（预对齐）
ds = load_dataset("deepmind/narrativeqa")
test_data = ds["test"]
samples = test_data.select(range(100))

# 实验配置：chunk × model
configs = [
    ("checkpoints/last", 64),
    # ("google/flan-t5-base", 128),
    # ("google/flan-t5-base", 256),
    # ("google/flan-t5-base", 512),
    # ("google/flan-t5-large", 64),
    # ("google/flan-t5-large", 128),
    # ("google/flan-t5-large", 256),
    # ("google/flan-t5-large", 512),
]


for model_name, chunk_size in configs:
    print(f"\n🚀 Running {model_name} | chunk_words={chunk_size}")
    llm = get_llm_from_pretrained(model_name)

    results = []
    for i, sample in enumerate(tqdm(samples)):
        sample_dict = {
            "document": {"text": sample["document"]["text"]},
            "question": {"text": sample["question"]["text"]},
            "answers": sample["answers"]
        }

        try:
            result = process_sample(sample_dict, chunk_words=chunk_size, model=llm)
            # print(result)
            # print(result.keys())
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
