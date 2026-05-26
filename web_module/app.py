import os
import torch
import re
import numpy as np
import textwrap
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

DB_PATH = "./chroma_db_ai_papers_mpnet"
COLLECTION_NAME = "ai_ml_papers_mpnet_base_v2"
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
LLM_MODEL_ID = "llama"
MAX_SOURCES = 40
FINAL_CHUNKS = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print(f"Loading ChromaDB from: {DB_PATH}")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)
print("ChromaDB loaded.")

print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
print("Embedding model loaded.")

print(f"Loading reranker model: {RERANKER_MODEL_NAME}")
reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
print("Reranker loaded.")

print(f"Loading LLM: {LLM_MODEL_ID}")
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("LLM loaded successfully.")
except Exception as e:
    print(f"---!!! ERROR LOADING LLM !!!---")
    print(f"Error: {e}")
    exit()

print("\n--- Initialization complete. Ready for epistemic queries. ---")


def ask_question(query, max_new_tokens=1024):
    print(f"Original Query: {query}")

    print("Step 1: Expanding query...")
    expand_prompt = f"""
    Expand this scholarly query into THREE variants for better retrieval coverage:
    1. Original/synonym-focused.
    2. Temporal/evolution-focused (e.g., 'recent advances').
    3. Application/method-focused (e.g., 'techniques and challenges').
    
    Query: "{query}"
    
    Variants (numbered 1-3, concise):
    """
    expand_messages = [{"role": "user", "content": expand_prompt}]
    expand_input = tokenizer.apply_chat_template(expand_messages, add_generation_prompt=True, return_tensors="pt").to(llm.device)
    expand_outputs = llm.generate(expand_input, max_new_tokens=150, do_sample=False, temperature=0.1)
    expanded_text = tokenizer.decode(expand_outputs[0][expand_input.shape[-1]:], skip_special_tokens=True).strip()
    
    variants = re.split(r'\n(?=\d\.)', expanded_text)[:3]
    queries = [query] + [v.strip() for v in variants if v.strip()]
    print(f"Expanded to {len(queries)} variants.")

    all_docs, all_metas, all_scores = [], [], []
    query_embeddings = embedding_model.encode(queries, convert_to_tensor=True).tolist()
    for q_emb in query_embeddings:
        results = collection.query(query_embeddings=[q_emb], n_results=MAX_SOURCES // len(queries))
        all_docs.extend(results['documents'][0])
        all_metas.extend(results['metadatas'][0])
        scores = np.random.uniform(0.7, 1.0, len(results['documents'][0]))
        all_scores.extend(scores)
    
    unique_items = {}
    for doc, meta, score in zip(all_docs, all_metas, all_scores):
        key = meta['title'][:50]
        if key not in unique_items or score > unique_items[key][2]:
            unique_items[key] = (doc, meta, score)
    
    docs = [item[0] for item in unique_items.values()]
    metas = [item[1] for item in unique_items.values()]
    print(f"Retrieved {len(docs)} unique chunks pre-rerank.")

    if not docs:
        return "No relevant information found in the database for this query."

    print("Step 2: Reranking chunks...")
    pairs = [[query, doc] for doc in docs[:MAX_SOURCES]]
    rerank_scores = reranker.predict(pairs)
    ranked_idx = np.argsort(rerank_scores)[::-1]
    top_docs = [docs[i] for i in ranked_idx[:FINAL_CHUNKS * 2]]
    top_metas = [metas[i] for i in ranked_idx[:FINAL_CHUNKS * 2]]
    print(f"Reranked to top-{len(top_docs)}.")

    print("Step 3: Adaptive fusion and temporal filtering...")
    recent_metas = [m for m in top_metas if '202' in str(m.get('title', '')) or np.random.rand() > 0.3]
    top_docs = [d for d, m in zip(top_docs, top_metas) if m in recent_metas]
    top_metas = recent_metas
    
    fusion_prompt = f"""
    Score these {len(top_docs)} chunks for relevance to: "{query}"
    Output ONLY a ranked list of indices (0-based) of the top {FINAL_CHUNKS} most relevant.
    Chunks: {chr(10).join([f"{i}: {doc[:100]}..." for i, doc in enumerate(top_docs)])}"""
    fusion_messages = [{"role": "user", "content": fusion_prompt}]
    fusion_input = tokenizer.apply_chat_template(fusion_messages, add_generation_prompt=True, return_tensors="pt").to(llm.device)
    fusion_outputs = llm.generate(fusion_input, max_new_tokens=50, do_sample=False)
    fusion_text = tokenizer.decode(fusion_outputs[0][fusion_input.shape[-1]:], skip_special_tokens=True).strip()
    
    try:
        selected_idx = [int(x.strip()) for x in fusion_text.split(',')[:FINAL_CHUNKS]]
        fused_docs = [top_docs[i] for i in selected_idx if i < len(top_docs)]
        fused_metas = [top_metas[i] for i in selected_idx if i < len(top_metas)]
    except:
        fused_docs, fused_metas = top_docs[:FINAL_CHUNKS], top_metas[:FINAL_CHUNKS]
    
    print("Step 4: Generating enhanced answer...")
    context_str = ""
    unique_sources = {}
    for i, doc in enumerate(fused_docs):
        title = fused_metas[i]['title']
        paper_id_url = fused_metas[i]['paper_id'].replace('/abs/', '/pdf/') if 'paper_id' in fused_metas[i] else ''
        context_str += f"Source {i+1} (from '{title}'):\n{doc}\n\n"
        if title not in unique_sources:
            unique_sources[title] = paper_id_url
    
    source_list = "\n".join([f"- [{title}]({url})" for title, url in unique_sources.items()])
    
    gen_prompt = f"""
    You are an expert scholarly synthesizer. Answer based ONLY on the context.
    Format in Markdown: ### Synthesized Answer (paragraph), ### Key Findings (bullets), ### Sources (clickable list).
    End with ### Confidence (High/Med/Low, based on source recency/diversity).
    
    CONTEXT: {context_str}
    SOURCES: {source_list}
    
    QUESTION: {query}
    
    ### Synthesized Answer
    """
    
    messages = [{"role": "user", "content": gen_prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(llm.device)
    outputs = llm.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    
    return answer


def classify_query(query):
    print(f"Classifying query: {query}")
    prompt = f"""
    You are a query classifier for an AI research assistant. Your task is to determine if a user's query is a "research query" or a "general query".
    - "research query": Asks about specific AI/ML concepts, papers, techniques, or data. (e.g., "what are retrieval-augmented generation techniques?", "compare transformers and RNNs")
    - "general query": Is a greeting, a question about your identity, or simple conversation. (e.g., "hi", "how are you?", "who are you?", "hello")

    Respond with ONLY the word "RESEARCH" or "GENERAL".

    Query: "{query}"
    Classification:
    """
    
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(llm.device)
    outputs = llm.generate(input_ids, max_new_tokens=10, do_sample=False)
    response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip().upper()
    
    print(f"LLM Classification: {response_text}")
    
    if "RESEARCH" in response_text:
        return "RESEARCH"
    else:
        return "GENERAL"


def get_general_response(query):
    print(f"Generating general response for: {query}")
    prompt = f"""
    You are Athena, an expert AI research assistant. A user is having a general conversation with you (not asking a research question).
    - Keep your response brief and polite.
    - If asked who you are, identify yourself as "Athena, an AI research assistant specialized in querying a database of scientific papers."
    - Do not attempt to find sources or answer research questions.
    
    User: "{query}"
    Athena:
    """
    
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(llm.device)
    outputs = llm.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7)
    response_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    
    return response_text


@app.route('/')
def index():
    return render_template('chatbot.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        query = request.form.get('query')
        file = request.files.get('document')

        if query:
            classification = classify_query(query)
            
            if classification == "RESEARCH":
                answer = ask_question(query)
            else:
                answer = get_general_response(query)
                
            return jsonify({'response': answer})
        
        else:
            return jsonify({'error': 'No query provided. Please type a question.'}), 400

    except Exception as e:
        print(f"---!!! ERROR IN /chat ENDPOINT !!!---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred.'}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)