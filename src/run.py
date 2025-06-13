import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from difflib import SequenceMatcher
import re
from datasets import load_dataset

# 🔸 1. 加载嵌入模型 & LLM
embedder = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# 🔸 2. 文本预处理与切分
def chunk_text(text, max_words=150):
    sents = re.split(r'[.?!]\s+', text)
    chunks, chunk = [], ""
    for sent in sents:
        if len((chunk + sent).split()) < max_words:
            chunk += sent + " "
        else:
            chunks.append(chunk.strip())
            chunk = sent + " "
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# 🔸 3. 构建检索索引
def build_faiss_index(chunks):
    vectors = embedder.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors

# 🔸 4. 执行检索
def retrieve(query, chunks, index, top_k=3):
    q_vec = embedder.encode([query])
    scores, ids = index.search(q_vec, top_k)
    return [chunks[i] for i in ids[0]]

# 🔸 5. LLM 回答
def generate_answer(context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    out = generator(prompt, max_new_tokens=50)[0]["generated_text"]
    return out.strip()

# 🔸 6. 评估
def exact_match(pred, golds):
    pred = pred.lower().strip()
    return any(pred == g["text"].lower().strip() for g in golds)

# 🔸 7. 测试样本（你给的格式）
sample = {
    "document": {
        "text": "Joe Bloggs lives in a small town. He is a quiet man. One day, something changes his life."
    },
    "question": {
        "text": "Where does Joe Bloggs live?"
    },
    "answers": [
        {"text": "Joe Bloggs lives in a small town"},
        {"text": "a small town"}
    ]
}

# 🔸 8. 执行流程
chunks = chunk_text(sample["document"]["text"])
index, _ = build_faiss_index(chunks)
retrieved = retrieve(sample["question"]["text"], chunks, index)
context = "\n".join(retrieved)

prediction = generate_answer(context, sample["question"]["text"])
is_em = exact_match(prediction, sample["answers"])

# 🔸 9. 输出结果
print(f"Question: {sample['question']['text']}")
print(f"Retrieved Context:\n{context}")
print(f"Predicted Answer: {prediction}")
print(f"Exact Match: {is_em}")