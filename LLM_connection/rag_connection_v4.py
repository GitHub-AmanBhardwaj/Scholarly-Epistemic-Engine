import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import textwrap
import re
import numpy as np

# --- CONFIGURATION ---
DB_PATH = "./chroma_db_ai_papers_mpnet"
COLLECTION_NAME = "ai_ml_papers_mpnet_base_v2"
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-12-v2'  # Lightweight reranker
LLM_MODEL_ID = "llama"  # Explicit for clarity
MAX_SOURCES = 40  # Initial retrieval (pre-rerank)
FINAL_CHUNKS = 10  # Post-rerank + fusion

# --- 1. INITIALIZE MODELS AND DATABASE ---
print("Initializing the Advanced RAG-v4 query engine...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load vector DB
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

# Load reranker
reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)

# Load LLM
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Initialization complete. Ready for epistemic queries.")

# --- 2. THE ADVANCED RAG-v4 FUNCTION ---
def ask_question(query, max_new_tokens=1024):
    """
    Advanced RAG-v4 Pipeline:
    1. Multi-query expansion (original + 2 LLM variants).
    2. Ensemble retrieval (top-40 from all queries).
    3. Reranking (cross-encoder scores).
    4. Adaptive fusion (LLM selects top-8 relevant chunks).
    5. Temporal filtering (prioritize recent papers).
    6. Enhanced generation with confidence scoring.
    """
    print("\n--- New Query (v4: Advanced Rerank + Fusion) ---")
    print(f"Original Query: {query}")

    # Step 1: Multi-Query Expansion
    print("Step 1: Expanding query into multi-variants...")
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
    
    # Parse variants (simple regex; improve as needed)
    variants = re.split(r'\n(?=\d\.)', expanded_text)[:3]  # First 3 numbered lines
    queries = [query] + [v.strip() for v in variants if v.strip()]
    print(f"Expanded to {len(queries)} variants.")

    # Step 2: Ensemble Retrieval
    all_docs, all_metas, all_scores = [], [], []
    query_embeddings = embedding_model.encode(queries, convert_to_tensor=True).tolist()
    for q_emb in query_embeddings:
        results = collection.query(query_embeddings=[q_emb], n_results=MAX_SOURCES // len(queries))
        all_docs.extend(results['documents'][0])
        all_metas.extend(results['metadatas'][0])
        # Mock scores for now (use distances if available)
        scores = np.random.uniform(0.7, 1.0, len(results['documents'][0]))  # Placeholder; use real distances
        all_scores.extend(scores)
    
    # Dedup by unique IDs (simple title hash)
    unique_items = {}
    for doc, meta, score in zip(all_docs, all_metas, all_scores):
        key = meta['title'][:50]  # Crude dedup
        if key not in unique_items or score > unique_items[key][2]:
            unique_items[key] = (doc, meta, score)
    
    docs = [item[0] for item in unique_items.values()]
    metas = [item[1] for item in unique_items.values()]
    scores = [item[2] for item in unique_items.values()]
    print(f"Retrieved {len(docs)} unique chunks pre-rerank.")

    if not docs:
        return "No relevant information found."

    # Step 3: Reranking
    print("Step 2: Reranking chunks...")
    pairs = [[query, doc] for doc in docs[:MAX_SOURCES]]  # Limit for efficiency
    rerank_scores = reranker.predict(pairs)
    # Sort by rerank score (higher = better)
    ranked_idx = np.argsort(rerank_scores)[::-1]
    top_docs = [docs[i] for i in ranked_idx[:FINAL_CHUNKS * 2]]  # Over-fetch for fusion
    top_metas = [metas[i] for i in ranked_idx[:FINAL_CHUNKS * 2]]
    print(f"Reranked to top-{len(top_docs)}.")

    # Step 4: Adaptive Fusion + Temporal Filter
    print("Step 3: Adaptive fusion and temporal filtering...")
    # Temporal filter: Prioritize recent papers (assume year in metadata; mock here)
    recent_metas = [m for m in top_metas if '202' in str(m.get('title', '')) or np.random.rand() > 0.3]  # Mock; parse real year from title/ID
    top_docs = [d for d, m in zip(top_docs, top_metas) if m in recent_metas]
    top_metas = recent_metas
    
    # LLM-based fusion: Score relevance
    fusion_prompt = f"""
    Score these {len(top_docs)} chunks for relevance to: "{query}"
    Output ONLY a ranked list of indices (0-based) of the top {FINAL_CHUNKS} most relevant.
    Chunks: {chr(10).join([f"{i}: {doc[:100]}..." for i, doc in enumerate(top_docs)])}"""
    fusion_messages = [{"role": "user", "content": fusion_prompt}]
    fusion_input = tokenizer.apply_chat_template(fusion_messages, add_generation_prompt=True, return_tensors="pt").to(llm.device)
    fusion_outputs = llm.generate(fusion_input, max_new_tokens=50, do_sample=False)
    fusion_text = tokenizer.decode(fusion_outputs[0][fusion_input.shape[-1]:], skip_special_tokens=True).strip()
    
    # Parse top indices (simple; e.g., "1, 3, 0, 5")
    try:
        selected_idx = [int(x.strip()) for x in fusion_text.split(',')[:FINAL_CHUNKS]]
        fused_docs = [top_docs[i] for i in selected_idx if i < len(top_docs)]
        fused_metas = [top_metas[i] for i in selected_idx if i < len(top_metas)]
    except:
        fused_docs, fused_metas = top_docs[:FINAL_CHUNKS], top_metas[:FINAL_CHUNKS]
    
    # Step 5: Enhanced Generation with Confidence
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

# --- 3. INTERACTIVE LOOP ---
if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a research question (or 'quit'): ")
        if user_query.lower() == 'quit':
            break
        answer = ask_question(user_query)
        print("\n--- v4 Generated Answer ---")
        print(answer)