import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import pyterrier as pt
from pathlib import Path
from pyterrier.measures import *
from fast_forward.encoder import TASBEncoder
import torch
from fast_forward.index import OnDiskIndex, Mode
from fast_forward.util import Indexer
from fast_forward.util.pyterrier import FFInterpolate
from fast_forward.util import Indexer

from fusions.FFTM2C2 import FFTM2C2
from fast_forward.util.pyterrier import FFScore


# Implement fusion functions reference: 
# https://github.com/mrjleo/fast-forward-indexes/blob/main/src/fast_forward/util/pyterrier.py
class FFRRF(pt.Transformer):
    """
    Fusion function implementing Reciprocal Rank Fusion (RRF):
    - Computes hard ranks for BM25 and neural scores.
    - Final score is the sum of reciprocals: 1/(k + rank) for each.
    """
    def __init__(self, k=60):
        self.k = k
        super().__init__()
        
    def transform(self, df):
        """Transform using the RRF fusion method."""
        new_df = df[["qid", "docno", "query"]].copy()
        bm25_rank = df['score_0'].rank(method='min', ascending=False)
        neural_rank = df['score'].rank(method='min', ascending=False)
        new_df['score'] = 1 / (self.k + bm25_rank) + 1 / (self.k + neural_rank)
        return pt.model.add_ranks(new_df, single_query=False)


class FFSRRF(pt.Transformer):
    """
    Fusion function implementing Soft Reciprocal Rank Fusion (SRRF):
    - Computes a soft rank for BM25 and neural scores using a logistic function.
    - Final score is computed similarly to RRF, but using the soft ranks.
    """
    def __init__(self, k=60, beta=1.0):
        self.k = k
        self.beta = beta
        super().__init__()

    def transform(self, df):
        """Transform using the SRRF fusion method."""
        new_df = df[["qid", "docno", "query"]].copy()

        def compute_soft_rank(scores):
            n = len(scores)
            soft_ranks = np.ones(n)
            for i in range(n):
                soft_ranks[i] += np.sum(1 / (1 + np.exp(self.beta * (scores[i] - scores)))) - 1
            return soft_ranks

        bm25_scores = df['score_0'].values.astype(np.float32)
        neural_scores = df['score'].values.astype(np.float32)
        sr_bm25 = compute_soft_rank(bm25_scores)
        sr_neural = compute_soft_rank(neural_scores)
        new_df['score'] = 1 / (self.k + sr_bm25) + 1 / (self.k + sr_neural)
        return pt.model.add_ranks(new_df, single_query=False)

# choose datasets: https://pyterrier.readthedocs.io/en/latest/datasets.html
dataset_name = "irds:beir/fiqa"

safe_dataset_name = dataset_name.replace(":", "_").replace("/", "_")
dataset = pt.get_dataset(dataset_name)
testset = pt.get_dataset(dataset_name + "/test")

indexer = pt.IterDictIndexer(
    str(Path.cwd()),  # this will be ignored
    type=pt.index.IndexingType.MEMORY,
)
index_ref = indexer.index(dataset.get_corpus_iter(), fields=["text"])
bm25 = pt.terrier.Retriever(index_ref, wmodel="BM25")

# To change the encoder, consult:
# https://github.com/mrjleo/fast-forward-indexes/blob/main/src/fast_forward/encoder/transformer.py
q_encoder = d_encoder = TASBEncoder(
    device="cuda:0" if torch.cuda.is_available() else "cpu"
)
ff_index = None
ff_index_path = Path.cwd() / "indexes" / f"ffindex_{safe_dataset_name}_tasb.h5"


# Create parent directory if it doesn't exist.
try: 
    ff_index = OnDiskIndex.load(
        ff_index_path,
        query_encoder=q_encoder,
        mode=Mode.MAXP,
    )
except FileNotFoundError:
    ff_index_path.parent.mkdir(exist_ok=True, parents=True)
    ff_index = OnDiskIndex(
        ff_index_path,
        query_encoder=q_encoder,
        mode=Mode.MAXP,
    )
    from fast_forward.util import Indexer
    def docs_iter():
        for d in dataset.get_corpus_iter():
            yield {"doc_id": d["docno"], "text": d["text"]}

    Indexer(ff_index, d_encoder, batch_size=8).from_dicts(docs_iter())


ff_index = ff_index.to_memory()

ff_score = FFScore(ff_index)
candidates = (bm25 % 5)(testset.get_topics())
re_ranked = ff_score(candidates)

hybrid = bm25 % 1000 >> ff_score
ff_int = FFInterpolate(alpha=0.5)
ff_int(re_ranked)
ff_tm2c2 = FFTM2C2()
ff_tm2c2(re_ranked)
ff_rrf = FFRRF()
ff_srrf = FFSRRF()
ff_rrf(re_ranked)
ff_srrf(re_ranked)

result = pt.Experiment(
    [bm25, hybrid >> ff_int,  hybrid >> ff_tm2c2],
    testset.get_topics(),
    testset.get_qrels(),
    eval_metrics=[RR @ 10, nDCG @ 10, MAP @ 100],
    names=["BM25", "linear(alpha = 0.5)", "TM2C2"],
    baseline=0,
    correction="bonferroni"
)

print(result)