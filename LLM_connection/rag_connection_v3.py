import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import textwrap

# --- CONFIGURATION ---
DB_PATH = "./chroma_db_ai_papers_mpnet"
COLLECTION_NAME = "ai_ml_papers_mpnet_base_v2"
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
LLM_MODEL_ID = "llama"

# New variable to control the maximum number of chunks to fetch
MAX_SOURCES = 20

# --- 1. INITIALIZE MODELS AND DATABASE ---
print("Initializing the RAG query engine...")

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the vector database
print(f"Loading vector database from: {DB_PATH}")
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Load the embedding model (for embedding the user's query)
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

# Load the LLM and its tokenizer
print(f"Loading LLM for generation: {LLM_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
print("Initialization complete. You can now ask questions.")

# --- 2. THE RAG FUNCTION ---
def ask_question(query, max_new_tokens=4096, num_sources=MAX_SOURCES):
    """
    Performs an advanced RAG pipeline with Query Rewriting:
    1. Rewrites the user query with an LLM for better search.
    2. Embeds the rewritten query.
    3. Retrieves relevant chunks from the database.
    4. Constructs a prompt for the LLM using the original query.
    5. Generates and returns a sourced answer with clickable links.
    """
    print("\n--- New Query ---")
    print(f"Original Query: {query}")

    # --- Step 1: Query Rewriting ---
    print("Step 1: Rewriting query with LLM for better retrieval...")
    rewrite_prompt = f"""
Given the following user question, transform it into a descriptive statement that is optimized for semantic search in a vector database of research papers.

User Question: "{query}"

Optimized Search Statement:
"""
    rewrite_messages = [{"role": "user", "content": rewrite_prompt}]
    rewrite_input_ids = tokenizer.apply_chat_template(
        rewrite_messages, add_generation_prompt=True, return_tensors="pt"
    ).to(llm.device)
    
    rewrite_outputs = llm.generate(
        rewrite_input_ids, max_new_tokens=100, do_sample=False
    )
    rewritten_query = tokenizer.decode(rewrite_outputs[0][rewrite_input_ids.shape[-1]:], skip_special_tokens=True).strip()
    print(f"Rewritten Query for Search: {rewritten_query}")

    # --- Step 2: Retrieve Documents ---
    query_embedding = embedding_model.encode(rewritten_query, convert_to_tensor=True).tolist()
    print("Step 2: Retrieving relevant documents from the database...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_sources
    )
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    if not documents:
        return "I could not find any relevant information in the database to answer your question."

    # --- Step 3: Construct the Prompt ---
    print("Step 3: Constructing prompt for the LLM...")
    context_str = ""
    unique_sources = {}
    for i, doc in enumerate(documents):
        title = metadatas[i]['title']
        paper_id_url = metadatas[i]['paper_id']
        
        context_str += f"Source {i+1} (from paper titled '{title}'):\n"
        context_str += f"{doc}\n\n"
        
        # Store unique titles and their PDF URLs for the final prompt
        if title not in unique_sources:
            # The paper_id is the abstract URL; we convert it to the PDF URL
            pdf_url = paper_id_url.replace('/abs/', '/pdf/')
            unique_sources[title] = pdf_url
    
    source_list_for_prompt = "\nList of available sources with their links:\n"
    for title, url in unique_sources.items():
        source_list_for_prompt += f"- Title: {title}\n- Link: {url}\n"
        
    full_context = context_str + source_list_for_prompt
        
    # --- NEW, ENHANCED PROMPT TEMPLATE ---
    prompt_template = f"""
You are an expert research assistant. Your task is to answer the user's question based ONLY on the following context from scientific papers.
Format your entire response in Markdown.

First, provide a comprehensive, synthesized answer in a paragraph under a '### Synthesized Answer' heading.

Then, under a '### Key Findings' heading, summarize the main points from the context as a bulleted list.

Finally, under a '### Sources' heading, list the titles of the papers you used. Each source must be a clickable Markdown link pointing to its corresponding URL from the 'List of available sources' provided in the context. For example: `* [Paper Title](URL)`

CONTEXT:
{full_context}

QUESTION:
{query}

### Synthesized Answer
"""
    # --- END OF NEW PROMPT ---
    
    # --- Step 4: Generate the Answer ---
    print("Step 4: Generating answer with the LLM...")
    messages = [{"role": "user", "content": prompt_template}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(llm.device)
    
    outputs = llm.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    answer = tokenizer.decode(response, skip_special_tokens=True)
    
    return answer

# --- 3. INTERACTIVE LOOP ---
if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a research question (or type 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        answer = ask_question(user_query)
        
        print("\n--- Generated Answer ---")
        print(answer)

