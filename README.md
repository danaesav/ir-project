# Information Retrieval Project: BM25 + ColBERT Fusion

This project implements a framework for combining traditional retrieval methods (BM25) with dense neural methods using PyTerrier and Hugging Face, and evaluates different fusion techniques.

## Hardware Requirement
The experiment indexes the datasets into RAM, therefore it is recommended to have a pc with at least 32GB RAM. 

## Software Requirement
The experiments are run in a **Linux** environment. There are some problems with running it on a Windows system. The recommended Python version is 3.12 if using virtual environments, dependencies are described in `requirements.txt`, including specific versions required.


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
  - BM25:  Lexical retriever for the initial retrieval
  - TAS-B: Neural re-ranker 1
  - Facebook Contriever: Neural re-ranker 2
  
- **Fusion Methods**:
  - Reciprocal Rank Fusion (RRF)
  - Convex Combination with different normalization tactics
    - Min-max
    - Theoretical Min-max
    - z-score
    - Lexical min-max
    - Lexical Theoretical Min-max
    - Lexical z-score
    - Identity(unnormalized)
- **Evaluation**:
  - recall@10, @100
  - MAP@100
  - NDCG@10
  - MRR@10

## Results

The framework evaluates each fusion method and compares their performance using recall@100 and nDCG@100 as the primary metric, some other metrics such as AP@100 are also listed but not mentioned in the paper. The results show which fusion method most effectively combines the strengths of traditional and neural retrieval methods, different baselines are used. 

See `src/results_baseline_xx`, where `xx` represents the fusion method used as baselines. The names of .csv files suggest what dataset is run on.
