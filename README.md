# A Scalable Engineering Methodology for the Diachronic Construction of a Semantically-Vectorized Epistemic Repository from Unstructured Scholarly Corpora

The research paper titled "A Scalable Engineering Methodology for the Diachronic Construction of a Semantically-Vectorized Epistemic Repository from Unstructured Scholarly Corpora" is made public through this repository, which serves as the abstract and access point.

The project is based on a well-structured 4-stage framework for the complete ingestion, processing, and vectorization of a vast, longitudinal, and unstructured document corpus (the 30-year arXiv `cs.AI` collection). The outcome high-dimensional knowledge base (7.5 million vectors) acts as the retrieval backend for an innovative Retrieval-Augmented Generation (RAG) query engine, thus being the proof of a scalable alternative for the revelation of complex, domain-specific knowledge.

## Access to Project Artifacts & Code

> [!IMPORTANT]
> **This repository contains the project `README.md` only.**
>
> All project artifacts—including the full Python and Jupyter Notebook source code, experimental logs, generated data, and final vector database—are hosted on a private Google Drive.
>
> This approach has been taken for the following reasons:
>
> 1.  **Data Scale:** The generated data artifacts (`arxiv_ai_metadata.csv`, `final_data.csv`) and the final vector database (`chroma_db/`) are in excess of 50GB, making them unsuitable for a Git repository.
>
> 2.  **Experimental Integrity:** To ensure the integrity and reproducibility of the experimental setup, all code and logs are versioned and stored alongside the data they generated.
>
> 3.  **Resource Management:** The models and data pipelines require specific hardware configurations (e.g., high-VRAM GPUs), and providing access via a managed repository ensures that reviewers or researchers can inspect the exact environment that produced the results.

### How to Request Access

Got it 👍 — here’s a clean, polite, and journal-appropriate version **without email contact**, where access requests are handled directly through the Google Drive link itself:

---

### 🔹 **Data and Code Availability**

Access to the Google Drive repository is available **upon request** for research verification and academic purposes.

Interested researchers may **request access directly through the following link**:
<a href="https://drive.google.com/drive/folders/1UlyTblrofWos_URo3QGEPGyI-CKJchdh?usp=drive_link" target="_blank" style="background:linear-gradient(90deg,#0073e6,#00c6ff); color:white; padding:8px 14px; border-radius:8px; font-weight:bold; text-decoration:none;">
🚀 View the Google Drive Repository
</a>


Requests will be reviewed, and approved users will receive **view-only access** to preserve the integrity of the original datasets and experimental artifacts.

## Framework Overview

The project methodology is divided into a 4-stage framework:

1.  **(Phase 1) Metadata Acquisition:** A fault-tolerant script iteratively scrapes the arXiv API to build a complete metadata catalog of the `cs.AI` domain from 1993-2024.
2.  **(Phase 2) Full-Text Extraction:** A high-performance, stateless notebook script downloads and parses nearly 90,000 PDFs, extracting their full text in a resource-constrained cloud environment.
3.  **(Phase 3) Knowledge Base Vectorization:** A GPU-accelerated script chunks the processed text into 7.5 million semantic segments, generates 768-dimensional embeddings using `all-mpnet-base-v2`, and ingests them into a persistent `ChromaDB`.
4.  **(Phase 4) RAG Query Engine:** An advanced orchestration script (`rag_connection_v3.py`) that accepts a user query, employs an LLM (`Llama-3-8B`) for query rewriting, retrieves relevant context from the vector database, and synthesizes a final, sourced answer.

## System Requirements (for Full Project)

* **Python 3.10+**
* **Hardware:**
    * **Phases 1-2:** Standard workstation.
    * **Phases 3-4:** A CUDA-enabled GPU with significant VRAM (e.g., NVIDIA RTX 30/40/50 series) is required for efficient embedding generation and for loading the LLM.
* **Key Python Libraries:** `arxiv`, `pymupdf`, `pandas`, `nltk`, `torch`, `transformers`, `sentence-transformers`, `chromadb`, `jupyter`

## Citation

```bibtex
@article{your_name_2025,
  title   = {A Scalable Engineering Methodology for the Diachronic Construction of a Semantically-Vectorized Epistemic Repository from Unstructured Scholarly Corpora},
  author  = {Author, One and Author, Two},
  journal = {Knowledge-Based Systems},
  year    = {2025}
}
