from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import numpy as np
import faiss

def load_triplets(jsonl_path):
    queries, positives, all_passages = [], [], []
    with open(jsonl_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            queries.append(d['query'])
            positives.append(d['positive'])
            all_passages.append(d['positive'])
            all_passages.extend(d['negatives'])
    return queries, positives, list(set(all_passages)) 

def build_faiss_index(passages, model):
    passage_embeddings = model.encode(passages, show_progress_bar=True, convert_to_numpy=True)
    dim = passage_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(passage_embeddings)
    return index, passage_embeddings

def evaluate_retrieval(model_path, queries, positives, passages):
    model = SentenceTransformer(model_path)
    query_embeddings = model.encode(queries, convert_to_numpy=True)
    index, passage_embeddings = build_faiss_index(passages, model)

    pid_map = {i: p for i, p in enumerate(passages)}

    hits = 0
    for i, qvec in enumerate(tqdm(query_embeddings)):
        _, idxs = index.search(np.array([qvec]), k=1)
        top_passage = pid_map[idxs[0][0]]
        if positives[i].strip() == top_passage.strip():
            hits += 1

    acc = hits / len(queries)
    print(f"🎯 Retrieval@1 accuracy: {acc:.3f} on {len(queries)} samples")
    return acc

if __name__ == "__main__":
    triplet_file = "data/triplets_test5000.jsonl"
    queries, positives, passages = load_triplets(triplet_file)

    print("\n🔹 Evaluating original model:")
    evaluate_retrieval("sentence-transformers/all-MiniLM-L6-v2", queries, positives, passages)

    print("\n🔹 Evaluating fine-tuned model:")
    evaluate_retrieval("dense_encoder_finetuned_train1950_epoch3_margin0.5", queries, positives, passages)
    # evaluate_retrieval("dense_encoder_finetuned_train3769_epoch1", queries, positives, passages)
    # evaluate_retrieval("dense_encoder_finetuned_train3769_epoch1", queries, positives, passages)
