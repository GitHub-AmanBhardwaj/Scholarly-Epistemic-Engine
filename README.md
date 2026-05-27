# Scholarly-Epistemic-Engine: A Four-Stage RAG System for Semantic Knowledge Elicitation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Data: Hugging Face](https://img.shields.io/badge/Data-Hugging%20Face-orange)](https://huggingface.co/datasets/whyamanbhardwaj/Scholarly-Epistemic-Engine)

## Overview

The **Scholarly-Epistemic-Engine** is an advanced Retrieval-Augmented Generation (RAG) framework designed to address information overload in scientific literature. By orchestrating a vast corpus of nearly 90,000 arXiv cs.AI papers (1993–2024), this system transitions from a baseline naive RAG (V1) to a highly optimized, faithfully grounded synthesis engine (V4). The final pipeline utilizes query expansion, cross-encoder reranking, and Large Language Model (LLM) fusion to deliver highly accurate, verifiable research answers.

> **Note on Data Storage:** This GitHub repository contains only the code (execution scripts, UI, and framework). Due to size constraints, the heavy data assets—including the processed csv files and pre-computed ChromaDB embeddings (total ~56 GB) are hosted entirely on Hugging Face. See the [Data Availability](#data-availability) section for access.

---

## Repository Structure

The repository is organized to reflect the discrete phases of the methodology:

```text
.
├── LICENSE
├── README.md
├── requirements.txt                 # Project dependencies
├── data_pipeline/                   # Phase 1-3: Data Preparation
│   ├── data_processing.ipynb        # Data cleaning, formatting, and final dataset consolidation
│   ├── full_text_data_creation.ipynb # Stateless PDF downloading and PyMuPDF full-text extraction
│   ├── meta_data_fetch.py           # Resumable arXiv API metadata scraping
│   ├── meta_data_fetch_logs.txt     # Output logs for metadata operations
│   ├── vectorization.py             # Embedding generation and ChromaDB indexing
│   └── vectorization_log.txt        # Execution logs for vectorization
├── LLM_connection/                  # Phase 4: The 4-Stage RAG Engine
│   ├── rag_connection_v1.py         # Naive RAG implementation
│   ├── rag_connection_v2.py         # Query rewriting pipeline
│   ├── rag_connection_v3.py         # Sourced synthesis pipeline
│   ├── rag_connection_v4.py         # Full cross-encoder reranking & fusion pipeline
|   ├── logv1.txt                    # Execution logs for each RAG variant
|   ├── logv2.txt
|   ├── logv3.txt
│   └── logv4.txt
├── results_and_analysis/            # Output Artifacts of final system
│   ├── sample_llm_result.md         # Markdown output sample
│   └── sample_llm_result.pdf        # PDF output sample
└── web_module/                      # User Interface
    ├── app.py                       # Web application routing
    ├── static/                      # Static assets
    └── templates/                   
        └── chatbot.html             # Front-end GUI for the RAG system

```

---

## Data Availability

Due to repository size constraints, the complete ~56 GB data ecosystem is hosted externally on Hugging Face. This dataset provides every artifact from our pipeline to ensure full reproducibility: from the initial master catalog of 89,375 scraped papers (`metadata.csv`) and the raw extraction checkpoints (`v1.csv` through `v4.csv`), to the finalized, cleaned text corpus of 87,984 manuscripts (`final_data.csv`). 

Additionally, we provide the fully compiled vector database (`chroma_db_ai_papers_mpnet.rar`). Downloading and extracting this archive provides immediate access to the 7.49 million pre-computed text chunks and their embeddings, allowing you to bypass the computationally expensive vectorization phase entirely.

**Access the dataset here:** 

[🤗 Hugging Face Dataset – Scholarly-Epistemic-Engine](https://huggingface.co/datasets/whyamanbhardwaj/Scholarly-Epistemic-Engine)

---

## Installation and Setup

**1. Clone the repository:**

```bash
git clone https://github.com/GitHub-AmanBhardwaj/Scholarly-Epistemic-Engine.git
cd Scholarly-Epistemic-Engine

```

**2. Set up the environment:**
This project requires Python 3.9+. It is highly recommended to use a virtual environment.

```bash
python -m venv env
env\Scripts\activate
pip install -r requirements.txt

```

---

## Usage Guide

### 1. Running the RAG Pipeline Locally

You can test the individual RAG versions using the scripts in the `LLM_connection` directory. Ensure your local GPU environment is properly configured for Llama-3 inference.

```bash
cd LLM_connection
python rag_connection_v4.py

```

### 2. Launching the Web Interface

To interact with the system via the graphical user interface developed for this study:

```bash
cd web_module
python app.py

```

*(The application will typically host on `http://127.0.0.1:5000/`. Check your terminal output for the exact local address).*

---

## License

* **Code:** The source code in this repository is licensed under the **MIT License**.
* **Data:** The dataset and vector embeddings hosted on Hugging Face are licensed under **CC-BY-NC-SA 4.0** for academic and non-commercial research purposes.
