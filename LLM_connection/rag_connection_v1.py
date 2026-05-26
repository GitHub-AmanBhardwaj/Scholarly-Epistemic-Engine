import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import textwrap

# --- CONFIGURATION ---
DB_PATH = "./chroma_db_ai_papers_mpnet"
COLLECTION_NAME = "ai_ml_papers_mpnet_base_v2"
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

# We'll use a powerful but manageable LLM from Hugging Face
# This requires a good amount of VRAM (~18-20 GB)
LLM_MODEL_ID = "llama"

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
def ask_question(query, max_new_tokens=512, num_sources=5):
    """
    Performs the full RAG pipeline:
    1. Embeds the query.
    2. Retrieves relevant chunks from the database.
    3. Constructs a prompt for the LLM.
    4. Generates and returns a sourced answer.
    """
    print("\n--- New Query ---")
    print(f"Query: {query}")

    # 1. Embed the user's query
    query_embedding = embedding_model.encode(query, convert_to_tensor=True).tolist()

    # 2. Retrieve relevant chunks from ChromaDB
    print("Step 1: Retrieving relevant documents from the database...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=num_sources
    )
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    if not documents:
        return "I could not find any relevant information in the database to answer your question.", []

    # 3. Construct the prompt for the LLM
    print("Step 2: Constructing prompt for the LLM...")
    context_str = ""
    for i, doc in enumerate(documents):
        context_str += f"Source {i+1} (from paper titled '{metadatas[i]['title']}'):\n"
        context_str += f"{doc}\n\n"
        
    prompt_template = f"""
You are an expert research assistant. Your task is to answer the user's question based ONLY on the following context from scientific papers.
Synthesize the information from the provided sources and give a clear, concise answer.
After your answer, list the titles of the papers you used as sources.

CONTEXT:
{context_str}

QUESTION:
{query}

ANSWER:
"""
    
    # 4. Generate the answer using the LLM
    print("Step 3: Generating answer with the LLM...")
    messages = [
        {"role": "user", "content": prompt_template},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
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

    # Extract unique source titles
    source_titles = sorted(list(set([meta['title'] for meta in metadatas])))
    
    return answer, source_titles

# --- 3. INTERACTIVE LOOP ---
if __name__ == "__main__":
    while True:
        user_query = input("\nAsk a research question (or type 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        
        answer, sources = ask_question(user_query)
        
        print("\n--- Generated Answer ---")
        print(textwrap.fill(answer, width=100))
        
        if sources:
            print("\n--- Sources Used ---")
            for title in sources:
                print(f"- {title}")