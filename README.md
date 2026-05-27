# Scholarly-Epistemic-Engine: A Four-Stage RAG System for Semantic Knowledge Elicitation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Data: Hugging Face](https://img.shields.io/badge/Data-Hugging%20Face-orange)](https://huggingface.co/datasets/whyamanbhardwaj/Scholarly-Epistemic-Engine)

## Overview

The **Scholarly-Epistemic-Engine** is an advanced Retrieval-Augmented Generation (RAG) framework designed to address information overload in scientific literature. By orchestrating a vast corpus of nearly 90,000 arXiv cs.AI papers (1993–2024), this system transitions from a baseline naive RAG (V1) to a highly optimized, faithfully grounded synthesis engine (V4). The final pipeline utilizes query expansion, cross-encoder reranking, and Large Language Model (LLM) fusion to deliver highly accurate, verifiable research answers.

> **Note on Data Availability:** This GitHub repository hosts the execution scripts, UI, and framework infrastructure. Due to size constraints, the full processed dataset (~56 GB) and pre-computed ChromaDB vector embeddings are hosted separately on Hugging Face. See the [Data Availability](#data-availability) section for access.

---

## Repository Structure

The repository is organized to reflect the discrete phases of the methodology:

```text
.
├── LICENSE
├── README.md
├── requirements.txt             # Project dependencies
├── data_pipeline/               # Phase 1-3: Offline Data Preparation
│   ├── meta_data_fetch.py       # Resumable arXiv API metadata scraping
│   ├── data_processing.ipynb    # Stateless PDF extraction and text cleaning
│   ├── full_text_data_creation.ipynb 
│   ├── vectorization.py         # Embedding generation and ChromaDB indexing
│   └── *_logs.txt               # Output logs for data operations
├── LLM_connection/              # Phase 4: The 4-Stage RAG Engine
│   ├── rag_connection_v1.py     # Naive RAG implementation
│   ├── rag_connection_v2.py     # Query rewriting pipeline
│   ├── rag_connection_v3.py     # Sourced synthesis pipeline
│   ├── rag_connection_v4.py     # Full cross-encoder reranking & fusion pipeline
│   └── logv*.txt                # Execution logs for each RAG variant
├── results_and_analysis/        # Output Artifacts and Metrics
│   ├── sample_llm_result.md     # Markdown output sample
│   ├── sample_llm_result.pdf    
│   └── yearly_publications_plot.png # Exploratory Data Analysis visuals
└── web_module/                  # User Interface
    ├── app.py                   # Web application routing
    ├── static/                  # Static assets (CSS)
    └── templates/               
        └── chatbot.html         # Front-end GUI for the RAG system

```

---

## Data Availability

To ensure complete reproducibility, the resources for this study are openly available across two platforms:

* **Code Repository (GitHub):** Contains the RAG evaluation framework, vectorization scripts, UI module, and the final RAG generation logs (`logv*.txt`).
* **Data Repository (Hugging Face):** Due to size constraints (~56 GB), the core data assets are hosted externally. The Hugging Face repository contains:
* `metadata.csv`: The master catalog for all 89,375 scraped papers (Phase 1).
* `v1.csv` to `v4.csv`: Iterative checkpoints of the full-text extraction process.
* `final_data.csv`: The consolidated, cleaned, statelessly extracted full text of 87,984 processed PDFs (Phase 2 output).
* `chroma_db_ai_papers_mpnet.rar`: The compressed ChromaDB vector database containing 7,496,671 chunks and their corresponding embeddings. Download and extract this archive to bypass the vectorization phase.



**Access the dataset here:** 

[🤗 Hugging Face Dataset – Scholarly-Epistemic-Engine](https://huggingface.co/datasets/whyamanbhardwaj/Scholarly-Epistemic-Engine)

---

## Installation and Setup

**1. Clone the repository:**

```bash
git clone [https://github.com/GitHub-AmanBhardwaj/Scholarly-Epistemic-Engine.git](https://github.com/GitHub-AmanBhardwaj/Scholarly-Epistemic-Engine.git)
cd Scholarly-Epistemic-Engine

```

**2. Set up the environment:**
This project requires Python 3.9+. It is highly recommended to use a virtual environment.

```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
pip install -r requirements.txt

```

**3. Download the Vector Database:**
To bypass the 40+ hour vectorization process, download the `chroma_db_ai_papers_mpnet.rar` file from the Hugging Face dataset link. Extract it and place the resulting `chroma_db` directory in the root folder of this repository.

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

## Evaluation Results

The system was evaluated using an information-theoretic framework measuring **Relevance** (Mean Similarity), **Diversity** (Shannon Entropy), and **Semantic Faithfulness** (LLM-as-a-Judge).

Moving from standard retrieval (V1) to an advanced multi-step pipeline (V4), the system demonstrated a significant reduction in hallucinations. The final **V4 pipeline** achieved a Semantic Faithfulness score of **0.732** while maintaining a robust Relevance score of **0.668**. This successfully validates the "Filter-then-Generate" paradigm, proving that employing a cross-encoder as a semantic buffer allows the LLM to synthesize reliable, verifiable academic answers from large context windows without suffering from diversity saturation.

---

## License

* **Code:** The source code in this repository is licensed under the **MIT License**.
* **Data:** The dataset and vector embeddings hosted on Hugging Face are licensed under **CC-BY-NC-SA 4.0** for academic and non-commercial research purposes.
