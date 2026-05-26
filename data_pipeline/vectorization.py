import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm
import nltk
import torch
import os

def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except Exception as e:
        print("Downloading NLTK's 'punkt' sentence tokenizer...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("Download complete.")

def main():
    INPUT_CSV = 'final_data.csv' 
    DB_PATH = "./chroma_db_ai_papers_mpnet"
    COLLECTION_NAME = "ai_ml_papers_mpnet_base_v2"
    MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
    BATCH_SIZE = 4096

    print("Loading data and embedding model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version PyTorch was compiled with: {torch.version.cuda}")

    try:
        df = pd.read_csv(INPUT_CSV, low_memory=False)
        df.dropna(subset=['full_text', 'id'], inplace=True)
        print(f"Loaded {len(df)} papers successfully.")
    except FileNotFoundError:
        print(f"Error: The input file '{INPUT_CSV}' was not found.")
        return 

    print(f"Loading sentence transformer model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    print("Database and collection are ready.")

    print("Checking for already processed documents to resume...")
    existing_ids_full = collection.get(include=[])['ids']
    existing_paper_ids = set([doc_id.split('_')[0] for doc_id in existing_ids_full])
    df_unprocessed = df[~df['id'].isin(existing_paper_ids)]

    if df_unprocessed.empty:
        print("All papers have already been processed and are in the database.")
        return

    print(f"Found {len(df_unprocessed)} new papers to process.")

    print("Processing papers and adding to the vector database...")
    batch_docs, batch_metadatas, batch_ids = [], [], []

    for index, row in tqdm(df_unprocessed.iterrows(), total=df_unprocessed.shape[0]):
        full_text = str(row.get('full_text', ''))
        paper_id = str(row['id'])
        
        if not full_text.strip() or len(full_text) < 100:
            continue
        
        sentences = nltk.sent_tokenize(full_text)
        chunk_size = 5
        chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

        for i, chunk in enumerate(chunks):
            cleaned_chunk = chunk.strip()
            if len(cleaned_chunk) > 50:
                batch_docs.append(cleaned_chunk)
                batch_metadatas.append({"paper_id": paper_id, "title": str(row.get('title', ''))})
                batch_ids.append(f"{paper_id}_{i}")

                if len(batch_docs) >= BATCH_SIZE:
                    embeddings = model.encode(batch_docs, show_progress_bar=False, device=device)
                    collection.add(
                        embeddings=embeddings.tolist(),
                        documents=batch_docs,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    batch_docs, batch_metadatas, batch_ids = [], [], []

    if batch_docs:
        embeddings = model.encode(batch_docs, show_progress_bar=False, device=device)
        collection.add(
            embeddings=embeddings.tolist(),
            documents=batch_docs,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

    print("\n--- Database Creation Complete! ---")
    total_docs = collection.count()
    print(f"The collection '{COLLECTION_NAME}' now contains {total_docs} document chunks.")

if __name__ == '__main__':
    setup_nltk()
    main()