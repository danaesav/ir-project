
import numpy as np
import pandas as pd
import pyterrier as pt
from torch.utils.flop_counter import suffixes


def convex_precondition(*weights):
    # Check if all weights are >= 0
    if any(weight < 0 for weight in weights):
        raise ValueError("All weights must be greater than or equal to 0.")

    # Check if the sum of weights is 1
    if sum(weights) != 1:
        raise ValueError("The sum of weights must be equal to 1.")

    return True  # If all conditions are met, return True

def z_score_normalization(arr: np.ndarray) -> np.ndarray:
    mean = np.mean(arr)
    std_dev = np.std(arr)
    if std_dev == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std_dev

def min_max_normalization(arr: np.ndarray) -> np.ndarray:
    data_min = np.min(arr)
    data_max = np.max(arr)
    if data_max > data_min:
        return (arr - data_min) / (data_max - data_min)
    else:
        return np.ones_like(arr) * 0.5

def theoretical_min_max_normalization(arr: np.ndarray, theoretical_min: int) -> np.ndarray:
    data_max = np.max(arr)
    assert np.min(arr) >= theoretical_min
    if data_max > theoretical_min:
        return (arr - theoretical_min) / (data_max - theoretical_min)
    else:
        return np.ones_like(arr) * 0.5


def normalize(arr: np.ndarray, method: str, theoretical_min: int = None) -> np.ndarray:
    if method == "z_score":
        return z_score_normalization(arr)
    elif method == "min_max":
        return min_max_normalization(arr)
    elif method == "theoretical_min_max":
        if theoretical_min is None:
            raise ValueError("theoretical_min must be provided for theoretical_min_max normalization")
        return theoretical_min_max_normalization(arr, theoretical_min)
    elif method == "unnormalized":
        return arr
    else:
        raise ValueError("Invalid normalization method")


def fuse_convex_norm(df1, df2, df3, w1, w2, w3,
                     normalization_method_1,
                     normalization_method_2,
                     normalization_method_3,
                     theoretical_min=-1):

    # ================== PRECONDITION CHECKS ==========================
    convex_precondition(w1, w2, w3)
    assert df1.shape[0] == df2.shape[0] == df3.shape[0], "DataFrames must have the same number of rows"
    # Extract (qid, docno) pairs as sets
    pairs1 = set(df1[['qid', 'docno']].itertuples(index=False, name=None))
    pairs2 = set(df2[['qid', 'docno']].itertuples(index=False, name=None))
    pairs3 = set(df3[['qid', 'docno']].itertuples(index=False, name=None))
    # Assert all sets are identical
    assert pairs1 == pairs2 == pairs3, "Mismatch in (qid, docno) pairs across DataFrames They should all be the same length with the exact same (qid,docno) pairs (in any order)"
    # ==================== END PRECONDITION CHECKS ========================

    df1 = df1[['qid', 'docno', 'score','query']]
    df2 = df2[['qid', 'docno', 'score','query']]
    df3 = df3[['qid', 'docno', 'score','query']]





    merged_df = df1.merge(df2, on=['qid', 'docno'], how='inner', suffixes=('_1','_2'))
    # Merge the result with df3, renaming df3's score column explicitly
    merged_df = merged_df.merge(df3[['qid', 'docno', 'score']], on=['qid', 'docno'], how='inner')
    merged_df = merged_df.rename(columns={'score': 'score_3'})
    merged_df['query'] = merged_df['query_1']

    scores1 = normalize(merged_df['score_1'].values.astype(np.float32), normalization_method_1, 0)
    scores2 = normalize(merged_df['score_2'].values.astype(np.float32), normalization_method_2, theoretical_min)
    scores3 = normalize(merged_df['score_3'].values.astype(np.float32), normalization_method_3, theoretical_min)

    merged_df['score'] = w1 * scores1 + w2 * scores2 + w3 * scores3
    new_df = merged_df[['qid', 'docno', 'query', 'score']]
    return pt.model.add_ranks(new_df, single_query=False)