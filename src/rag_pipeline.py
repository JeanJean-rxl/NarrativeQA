import re
import faiss
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 📌 初始化模型
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ 切 chunk
def chunk_text(text, max_words=150):
    sents = re.split(r'[.?!]\s+', text)
    chunks, cur = [], ""
    for s in sents:
        if len((cur + s).split()) < max_words:
            cur += s + " "
        else:
            chunks.append(cur.strip())
            cur = s + " "
    if cur:
        chunks.append(cur.strip())
    return chunks

# ✅ 构建 FAISS 索引
def build_index(chunks):
    vectors = embedder.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors

# ✅ 检索 top-k chunks
def retrieve_chunks(query, chunks, index, top_k=3):
    q_vec = embedder.encode([query])
    _, ids = index.search(q_vec, top_k)
    return [chunks[i] for i in ids[0]]

# ✅ 构造 prompt 并生成回答
def generate_answer(question, context, model):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result = model(prompt, max_new_tokens=50)[0]["generated_text"]
    return result.strip()

# ✅ 评估函数
def exact_match(pred, golds):
    pred = pred.lower().strip()
    return any(pred == g["text"].lower().strip() for g in golds)

def fuzzy_match(pred, golds):
    return max(SequenceMatcher(None, pred.lower(), g["text"].lower()).ratio() for g in golds)

# ✅ 主流程：处理单条样本
def process_sample(sample, chunk_words=150, top_k=3, model=None):
    text = sample["document"]["text"]
    question = sample["question"]["text"]
    answers = sample["answers"]

    chunks = chunk_text(text, max_words=chunk_words)
    index, _ = build_index(chunks)
    retrieved = retrieve_chunks(question, chunks, index, top_k=top_k)
    context = "\n".join(retrieved)

    prediction = generate_answer(question, context, model) ##TODO
    em = exact_match(prediction, answers)
    fuzzy = fuzzy_match(prediction, answers)

    return {
        "question": question,
        "prediction": prediction,
        "answers": [a["text"] for a in answers],
        "exact_match": em,
        "fuzzy_score": fuzzy,
        "retrieved_context": context
    }