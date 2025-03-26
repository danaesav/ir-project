import pyterrier as pt
from pyterrier.measures import RR, nDCG, MAP

dataset = pt.get_dataset("vaswani")
index = dataset.get_index(variant="terrier_stemmed")
bm25 = pt.terrier.Retriever(index, wmodel="BM25")

result = pt.Experiment(
    [bm25],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=[RR @ 10, nDCG @ 20, MAP],
)
print(result)