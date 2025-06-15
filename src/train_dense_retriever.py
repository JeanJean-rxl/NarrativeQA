from sentence_transformers import SentenceTransformer, losses, models, InputExample
from torch.utils.data import DataLoader
import json
from tqdm import tqdm

# ✅ 加载 triplet jsonl 文件
def load_triplets(jsonl_path):
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            query = data['query']
            pos = data['positive']
            for neg in data['negatives']:
                samples.append(InputExample(texts=[query, pos, neg]))
    return samples

# ✅ 主训练逻辑
def train_retriever(model_name, triplet_file, output_path, batch_size=16, epochs=3):
    # Load model
    model = SentenceTransformer(model_name)

    # Load triplet data
    train_samples = load_triplets(triplet_file)
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

    # Triplet loss
    
    train_loss = losses.TripletLoss(model=model, triplet_margin=0.2)  

    # Fit
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=output_path
    )

    print(f"✅ Model fine-tuned and saved to {output_path}")


if __name__ == "__main__":
    train_retriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        triplet_file="data/triplets_trainall.jsonl", 
        output_path="dense_encoder_finetuned_trainall_epoch3_margin0.2",
        batch_size=16,
        epochs=3
    )