import time
import pandas as pd
import pyterrier as pt
from datasets import load_dataset
import re
import os
import shutil

# ensure Java is running in the backend
if not pt.started():
    pt.init()

# import BERT reranker extension
try:
    from pyterrier_bert import MonoBERT
    has_bert = True
    print("Successfully imported pyterrier_bert")
except ImportError:
    has_bert = False
    print("pyterrier_bert not available - this is expected as it requires special installation.")
    print("Using alternative retrieval approaches based on built-in PyTerrier features.")

#######################################
# 1. Load and Preprocess the Dataset
#######################################
print("Loading BEIR dataset...")
try:
    # https://huggingface.co/BeIR
    corpus_dataset = load_dataset("BeIR/trec-covid", "corpus")
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")
    
    corpus_split = list(corpus_dataset.keys())[0]
    queries_split = list(queries_dataset.keys())[0]
    
    print(f"Corpus split: {corpus_split}, Queries split: {queries_split}")
    print(f"Loaded {len(corpus_dataset[corpus_split])} documents and {len(queries_dataset[queries_split])} queries")
except Exception as e:
    print("Error loading dataset:", e)
    exit(1)

# include only non-empty texts
docs_list = []
for item in corpus_dataset[corpus_split]:
    if item["text"] and item["text"].strip():
        docs_list.append({
            "docno": item["_id"],
            "text": item["text"],
            "title": item["title"]
        })
docs = pd.DataFrame(docs_list)
print(f"Filtered corpus to {len(docs)} non-empty documents")

def sanitize_query(query_text):
    sanitized = query_text.replace('?', '')
    sanitized = sanitized.replace('"', ' ')
    sanitized = sanitized.replace(':', ' ')
    sanitized = sanitized.replace('(', ' ')
    sanitized = sanitized.replace(')', ' ')
    sanitized = re.sub(r'\b(AND|OR|NOT)\b', '', sanitized)
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized

# process queries and create a df
queries_df = pd.DataFrame([
    {"qid": item["_id"], "query": sanitize_query(item["text"])} 
    for item in queries_dataset[queries_split]
])
print("Sample queries:")
print(queries_df.head(3))

#######################################
# 2. Build the Index and Retrieval Pipelines
#######################################
index_path = "./index"
if os.path.exists(index_path):
    print("Index already exists. Removing the old index...")
    shutil.rmtree(index_path)

# Create an index from the document records.
indexer = pt.index.IterDictIndexer(index_path, meta=['docno'])
indexref = indexer.index(docs.to_dict(orient='records'))
print("Index built successfully.")

# Build the BM25 retriever.
bm25 = pt.terrier.Retriever(indexref, wmodel="BM25")

# Create a second retriever using a different weighting model
print("Creating second retriever...")
# Use a different model from BM25 for comparison
retriever2 = pt.terrier.Retriever(indexref, wmodel="DirichletLM")
print("Retrieving results with DirichletLM...")
retriever2_results = retriever2.transform(queries_df)
if 'qid' in retriever2_results.columns and 'query_id' not in retriever2_results.columns:
    retriever2_results = retriever2_results.rename(columns={"qid": "query_id"})

print("Using DirichletLM as the second retrieval model")
bert_results = retriever2_results
second_retriever_name = "DirichletLM"

print("Retrieving BM25 results...")
bm25_results = bm25.transform(queries_df)
# Rename column for consistency.
if 'qid' in bm25_results.columns and 'query_id' not in bm25_results.columns:
    bm25_results = bm25_results.rename(columns={"qid": "query_id"})

#######################################
# 3. Define Fusion Functions
#######################################
def linear_fusion_function(results1, results2, alpha=0.5):
    """
    Combine two ranked lists with a weighted linear combination.
    """
    merged = pd.merge(results1, results2, on=["query_id", "docno"], 
                      suffixes=("_1", "_2"), how="outer").fillna(0)
    merged['score'] = alpha * merged['score_1'] + (1 - alpha) * merged['score_2']
    return merged[["query_id", "docno", "score"]].sort_values(["query_id", "score"], ascending=[True, False])

def rrf_fusion(results1, results2, k=60):
    """
    Reciprocal Rank Fusion (RRF) combines rankings from two methods.
    """
    r1 = results1.copy()
    r2 = results2.copy()
    r1['rank'] = r1.groupby("query_id")["score"].rank(ascending=False, method="first")
    r2['rank'] = r2.groupby("query_id")["score"].rank(ascending=False, method="first")
    merged = pd.merge(r1[['query_id', 'docno', 'rank']],
                      r2[['query_id', 'docno', 'rank']],
                      on=["query_id", "docno"], suffixes=("_1", "_2"), how="outer")
    merged['score'] = (1 / (k + merged['rank_1'].fillna(1000))) + (1 / (k + merged['rank_2'].fillna(1000)))
    return merged[["query_id", "docno", "score"]].sort_values(["query_id", "score"], ascending=[True, False])

def srrf_fusion(results1, results2, k=60):
    """
    Similarity Reciprocal Rank Fusion (SRRF) uses original scores weighted by reciprocal rank.
    """
    r1 = results1.copy()
    r2 = results2.copy()
    r1['rank'] = r1.groupby("query_id")["score"].rank(ascending=False, method="first")
    r2['rank'] = r2.groupby("query_id")["score"].rank(ascending=False, method="first")
    merged = pd.merge(r1, r2, on=["query_id", "docno"], suffixes=("_1", "_2"), how="outer")
    merged['score'] = (merged['score_1'].fillna(0) / (k + merged['rank_1'].fillna(1000))) + \
                      (merged['score_2'].fillna(0) / (k + merged['rank_2'].fillna(1000)))
    return merged[["query_id", "docno", "score"]].sort_values(["query_id", "score"], ascending=[True, False])

def tm2c2_fusion(results1, results2):
    """
    TM2C2 fusion uses rank information to simulate a fusion strategy.
    """
    r1 = results1.copy()
    r2 = results2.copy()
    r1['rank'] = r1.groupby("query_id")["score"].rank(ascending=False, method="first")
    r2['rank'] = r2.groupby("query_id")["score"].rank(ascending=False, method="first")
    merged = pd.merge(r1[['query_id', 'docno', 'rank']],
                      r2[['query_id', 'docno', 'rank']],
                      on=["query_id", "docno"], suffixes=("_1", "_2"), how="outer")
    merged['score'] = 1 / (merged['rank_1'].fillna(1000) + merged['rank_2'].fillna(1000))
    return merged[["query_id", "docno", "score"]].sort_values(["query_id", "score"], ascending=[True, False])

#######################################
# 4. Apply Fusion Functions
#######################################
fusion_results = {}
print("Applying fusion functions...")
fusion_results["BM25"] = bm25_results
fusion_results[second_retriever_name] = bert_results  # Use the name of the second retriever we actually used
fusion_results["Linear (α=0.5)"] = linear_fusion_function(bm25_results, bert_results, alpha=0.5)
fusion_results["RRF"] = rrf_fusion(bm25_results, bert_results)
fusion_results["SRRF"] = srrf_fusion(bm25_results, bert_results)
fusion_results["TM2C2"] = tm2c2_fusion(bm25_results, bert_results)

#######################################
# 5. Evaluation of Fusion Results
#######################################
try:
    # https://huggingface.co/BeIR
    print("Attempting to load qrels with datasets...")
    try:
        import datasets
        dataset = datasets.load("BeIR/trec-covid")
        # Convert qrels to the expected format
        qrels_data = [
            {"qid": str(qrel.query_id), "docno": str(qrel.doc_id), "relevance": qrel.relevance}
            for qrel in dataset.qrels_iter()
        ]
        qrels_df = pd.DataFrame(qrels_data)
        print(f"Loaded {len(qrels_df)} relevance judgments from ir_datasets")
    except ImportError:
        print("ir_datasets not available. Please install with: pip install ir_datasets")
        raise Exception("No qrels available")
    except Exception as e:
        print(f"Error loading from ir_datasets: {e}")
        
        # Second attempt: Try to directly download qrels from a known location
        print("Attempting to download qrels directly...")
        import requests
        from io import StringIO
        
        # URL for TREC-COVID qrels
        qrels_url = "https://ir.nist.gov/covidSubmit/data/qrels-covid_d5_j0.5-5.txt"
        try:
            response = requests.get(qrels_url)
            response.raise_for_status()  # Raise exception for bad status codes
            
            # Parse the TREC format qrels (topic_id 0 doc_id relevance)
            qrels_data = []
            for line in StringIO(response.text):
                parts = line.strip().split()
                if len(parts) >= 4:
                    qrels_data.append({
                        "qid": parts[0],
                        "docno": parts[2],
                        "relevance": int(parts[3])
                    })
            
            qrels_df = pd.DataFrame(qrels_data)
            print(f"Loaded {len(qrels_df)} relevance judgments from NIST website")
        except Exception as e:
            print(f"Error downloading qrels: {e}")
            
            # Third attempt: Create a simplified/approximate evaluation without qrels
            print("Creating approximate evaluation using top-k overlap...")
            print("\n===== SIMPLIFIED EVALUATION (NO QRELS) =====")
            print("Using reference method: BM25")
            
            # Function to calculate overlap with reference method
            def top_k_overlap(results1, results2, k=10):
                """Calculate percentage overlap in top k results across all queries"""
                overlap_sum = 0
                query_count = 0
                
                for qid in set(results1["query_id"].unique()) & set(results2["query_id"].unique()):
                    r1_top = results1[results1["query_id"] == qid].head(k)["docno"].tolist()
                    r2_top = results2[results2["query_id"] == qid].head(k)["docno"].tolist()
                    overlap = len(set(r1_top) & set(r2_top))
                    overlap_sum += overlap / k
                    query_count += 1
                    
                return (overlap_sum / query_count) * 100 if query_count > 0 else 0
            
            # Calculate overlap with the reference method (BM25)
            reference = fusion_results["BM25"]
            overlap_scores = {}
            for method, results in fusion_results.items():
                if method != "BM25":
                    overlap = top_k_overlap(reference, results, k=10)
                    print(f"{method} overlap with BM25 (top-10): {overlap:.1f}%")
                    overlap_scores[method] = overlap
            
            print("\nNote: This is not a true effectiveness evaluation, just a comparison of")
            print("result similarity with the BM25 baseline.")
            
            # Skip the rest of the evaluation
            raise Exception("No suitable qrels found - using overlap metrics instead")

    metrics = ['ndcg_cut.100', 'recip_rank', 'map']
    
    print("\n===== EFFECTIVENESS EVALUATION =====")
    
    # Try different ways to access PyTerrier's evaluation functionality
    try:

        print("Attempting to use PyTerrier's evaluation...")
        
        # Make sure qrels_df has the right column names
        if 'qid' not in qrels_df.columns or 'docno' not in qrels_df.columns or 'relevance' not in qrels_df.columns:
            print("Renaming qrels columns to match PyTerrier's expected format")
            qrels_columns_map = {}
            for column in qrels_df.columns:
                if column.lower() in ['qid', 'query', 'query_id', 'query-id']:
                    qrels_columns_map[column] = 'qid'
                elif column.lower() in ['docno', 'docid', 'doc_id', 'corpus-id']:
                    qrels_columns_map[column] = 'docno'
                elif column.lower() in ['rel', 'relevance', 'label', 'score']:
                    qrels_columns_map[column] = 'relevance'
            
            qrels_df = qrels_df.rename(columns=qrels_columns_map)
        evaluator = pt.Evaluate(qrels=qrels_df, metrics=metrics, perquery=False)
            
        # Evaluate each fusion method
        eval_results = {}
        for method, result in fusion_results.items():
            result_copy = result.copy()
            # Ensure the column for query identifier is named 'qid'
            if 'query_id' in result_copy.columns and 'qid' not in result_copy.columns:
                result_copy = result_copy.rename(columns={"query_id": "qid"})
                
            # Evaluate the results
            eval_result = evaluator.evaluate(result_copy)
            eval_results[method] = eval_result
            
            print(f"\nEvaluation for {method}:")
            for metric in metrics:
                if metric in eval_result:
                    print(f"  {metric}: {eval_result[metric]:.4f}")
                else:
                    print(f"  {metric}: Not available")
    
    except (ImportError, AttributeError, TypeError, ValueError) as e:
        print(f"PyTerrier evaluation failed: {e}")
        
        # Second attempt: use pytrec_eval directly with corrected metric names
        try:
            import pytrec_eval
            
            # Map PyTerrier metrics to pytrec_eval metrics
            pytrec_metrics = []
            for metric in metrics:
                if metric == 'ndcg_cut.100':
                    pytrec_metrics.append('ndcg_cut.100')
                elif metric == 'recip_rank':
                    pytrec_metrics.append('recip_rank')
                elif metric == 'map':
                    pytrec_metrics.append('map')
                else:
                    pytrec_metrics.append(metric)
            
            # Convert qrels to pytrec_eval format: {qid: {docno: relevance}}
            qrels_dict = {}
            for _, row in qrels_df.iterrows():
                qid = str(row['qid'])
                if qid not in qrels_dict:
                    qrels_dict[qid] = {}
                qrels_dict[qid][row['docno']] = int(row['relevance'])
            
            # Create evaluator
            evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, pytrec_metrics)
            
            # Evaluate each fusion method
            eval_results = {}
            for method, result in fusion_results.items():
                # Convert results to pytrec_eval format: {qid: {docno: score}}
                run_dict = {}
                for _, row in result.iterrows():
                    qid = str(row['query_id']) if 'query_id' in row else str(row['qid'])
                    if qid not in run_dict:
                        run_dict[qid] = {}
                    run_dict[qid][row['docno']] = float(row['score'])
                
                # Evaluate
                method_results = evaluator.evaluate(run_dict)
                
                # Average the results across all queries
                avg_results = {}
                for query_id, query_results in method_results.items():
                    for measure, value in query_results.items():
                        if measure not in avg_results:
                            avg_results[measure] = 0.0
                        avg_results[measure] += value
                
                # Divide by number of queries to get average
                num_queries = len(method_results)
                for measure in avg_results:
                    avg_results[measure] /= num_queries if num_queries > 0 else 1
                
                eval_results[method] = avg_results
                
                print(f"\nEvaluation for {method}:")
                for metric in avg_results:
                    print(f"  {metric}: {avg_results[metric]:.4f}")
            
        except ImportError:
            print("pytrec_eval not installed. Please install with: pip install pytrec_eval")
            raise
    
    # Summarize evaluation results in a DataFrame if we have them
    if eval_results:
        # Find common metrics across all methods
        common_metrics = set(eval_results[list(eval_results.keys())[0]].keys())
        for method in eval_results:
            common_metrics &= set(eval_results[method].keys())
        
        # Create summary DataFrame
        summary_data = {}
        for method, results in eval_results.items():
            summary_data[method] = {metric: results[metric] for metric in common_metrics}
        
        summary = pd.DataFrame(summary_data)
        print("\n===== EVALUATION SUMMARY =====")
        print(summary)
    
except Exception as e:
    print(f"Qrels evaluation skipped: {e}")
    print("Proceeding with efficiency evaluation only")
    
    # Add efficiency comparison
    print("\n===== EFFICIENCY EVALUATION =====")
    _ = bm25.transform(queries_df)
    
    # Measure second retriever time (BERT or DirichletLM)
    if has_bert:
        start = time.time()
        _ = bert_pipeline.transform(queries_df)

    else:
        start = time.time()
        _ = retriever2.transform(queries_df)

    
    # Measure fusion times
    for fusion_name, fusion_func in [
        ("Linear (α=0.5)", lambda r1, r2: linear_fusion_function(r1, r2, alpha=0.5)),
        ("RRF", rrf_fusion),
        ("SRRF", srrf_fusion),
        ("TM2C2", tm2c2_fusion)
    ]:
        start = time.time()
        _ = fusion_func(bm25_results, bert_results)

    
    # Calculate and display result set sizes
    print("\n===== RESULT SET SIZES =====")
    for method, results in fusion_results.items():
        total_results = len(results)
        unique_docs = results["docno"].nunique()
        avg_per_query = total_results / results["query_id"].nunique()
        print(f"{method}: {total_results} total results, {unique_docs} unique documents, {avg_per_query:.1f} avg per query")

print("\nScript complete.")
