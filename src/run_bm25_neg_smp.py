import tqdm
from beir.datasets.data_loader import GenericDataLoader

from src.model.bm25 import BM25Okapi
from src.utils import *

set_seed(seed=42)

code = 'CCS_CODE'

mimic_data_path = os.path.join(data_path, f'mimic3/processed/mimic3_{code}')
tr_corpus, tr_queries, tr_qrels = GenericDataLoader(mimic_data_path).load(split="train")

model = BM25Okapi(tr_corpus)

tr_qrels_w_neg = {}

for q_id, q in tqdm.tqdm(tr_queries.items()):
    d_ids = [d_id for d_id in tr_qrels[q_id] if tr_qrels[q_id][d_id] > 0]
    ds = [tr_corpus[d_id]["text"] for d_id in d_ids]
    for d_id, d in zip(d_ids, ds):
        scores = model.get_scores(d)
        for (ned_d_id, neg_s) in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if ned_d_id != d_id:
                tr_qrels_w_neg[q_id] = {d_id: 1, ned_d_id: -1}
                break

with open(os.path.join(mimic_data_path, 'qrels/train_w_neg.tsv'), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['query-id', 'corpus-id', 'score'])
    qids = list(tr_qrels_w_neg.keys())
    qids = sorted(qids)
    for qid in qids:
        for cid, score in tr_qrels_w_neg[qid].items():
            tsv_writer.writerow([qid, cid, score])
