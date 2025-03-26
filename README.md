# Information Retrieval Project: BM25 + ColBERT Fusion

This project implements a framework for combining traditional retrieval methods (BM25) with dense neural methods (BERT) using PyTerrier and Hugging Face, and evaluates different fusion techniques.

## Setup

1. Activate conda environment or create a python3 virtual env then activate it.

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the main script:
   ```
   python src/main.py
   ```

## Framework Components

- **Retrieval Methods**:
  - BM25: Traditional lexical retrieval
  - TAS-B: Dense neural retrieval
  
- **Fusion Methods**:
  - Linear Fusion (with various α values)
  - Reciprocal Rank Fusion (RRF)
  - Similarity RRF (SRRF)
  - TM2C2

- **Evaluation**:
  - MAP@100
  - NDCG@10
  - MRR@10

## Results

The framework evaluates each fusion method and compares their performance using MAP@100 as the primary metric. The results show which fusion method most effectively combines the strengths of traditional and neural retrieval methods.