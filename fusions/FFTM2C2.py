import numpy as np
import pyterrier as pt

class FFTM2C2(pt.Transformer):
    """
    Fusion function implementing TM2C2 (Tobin's Method 2 with Convex Combination):
    - Uses mathematical reasoning to combine BM25 and neural scores
    - Designed to leverage strengths of both retrieval methods
    """
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        super().__init__()
        
    def transform(self, df):
        """Transform using the TM2C2 fusion method."""
        new_df = df[["qid", "docno", "query"]].copy()
        
        # Get BM25 and neural scores
        bm25_scores = df['score_0'].values.astype(np.float32)
        neural_scores = df['score'].values.astype(np.float32)
        
        # Normalize BM25 scores to [0,1] range
        if np.max(bm25_scores) > np.min(bm25_scores):
            bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        else:
            bm25_norm = np.ones_like(bm25_scores) * 0.5
            
        # Normalize neural scores to [0,1] range
        if np.max(neural_scores) > np.min(neural_scores):
            neural_norm = (neural_scores - np.min(neural_scores)) / (np.max(neural_scores) - np.min(neural_scores))
        else:
            neural_norm = np.ones_like(neural_scores) * 0.5
        
        # Apply TM2C2 fusion - using both scores with non-linear combination
        combined_scores = np.sqrt(bm25_norm * neural_norm) + self.alpha * (bm25_norm + neural_norm) / 2
        
        new_df['score'] = combined_scores
        return pt.model.add_ranks(new_df, single_query=False)