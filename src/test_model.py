import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
from rag_pipeline import process_sample

def load_model(model_path):
    is_lora = "checkpoints" in model_path
    if is_lora:
        try:
            peft_config = PeftConfig.from_pretrained(model_path)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(peft_config.base_model_name_or_path)
            model = PeftModel.from_pretrained(base_model, model_path).eval()
        except Exception as e:
            raise ValueError(f"❌ load LoRA fail: {e}")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def llm_infer(prompt, max_new_tokens=64):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return [{"generated_text": tokenizer.decode(output, skip_special_tokens=True)} for output in outputs]

    return llm_infer

def main(args):
    ds = load_dataset("deepmind/narrativeqa")
    samples = ds["test"].select(range(args.num_samples))

    for chunk_size in args.chunk_sizes:
        print(f"\n🚀 Running {args.model_path} | chunk_words={chunk_size}")
        llm = load_model(args.model_path)

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
                    "model": args.model_path,
                    "chunk_words": chunk_size
                })
                results.append(result)
            except Exception as e:
                print(f"⚠️ Error on sample {i}: {e}")

        os.makedirs(args.output_dir, exist_ok=True)
        model_id = Path(args.model_path).name
        out_file = os.path.join(args.output_dir, f"rag_results_{model_id}_chunk{chunk_size}_sample{args.num_samples}.csv")
        pd.DataFrame(results).to_csv(out_file, index=False)
        print(f"✅ Saved results to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG-style evaluation on NarrativeQA with chunked context")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF or LoRA model")
    parser.add_argument("--chunk_sizes", type=int, nargs="+", default=[64], help="List of chunk sizes to evaluate")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of test samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save CSV results")
    args = parser.parse_args()

    main(args)
