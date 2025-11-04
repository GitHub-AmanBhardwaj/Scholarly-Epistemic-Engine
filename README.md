# A Scalable Engineering Methodology for the Diachronic Construction of a Semantically-Vectorized Epistemic Repository from Unstructured Scholarly Corpora

This repository serves as the public abstract and access point for the research paper, "A Scalable Engineering Methodology for the Diachronic Construction of a Semantically-Vectorized Epistemic Repository from Unstructured Scholarly Corpora."

The project implements a robust 4-stage framework for the end-to-end ingestion, processing, and vectorization of a large, longitudinal, and unstructured document corpus (the 30-year arXiv `cs.AI` collection). The resulting high-dimensional knowledge base (7.5 million vectors) serves as the retrieval backend for an advanced Retrieval-Augmented Generation (RAG) query engine, demonstrating a scalable solution for complex, domain-specific knowledge discovery.

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

Access to the Google Drive folder is provided upon reasonable request for research verification and academic purposes.

* **Contact:** Please email the corresponding author (e.g., `author.one@email.com`) with your institutional affiliation and a brief statement explaining the purpose of your request.
* **Access Level:** Approved requests will be granted **view-only access**. This policy is in place to protect the integrity of the original experimental artifacts and datasets.

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
