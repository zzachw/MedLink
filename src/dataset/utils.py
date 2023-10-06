from typing import Dict

from torch.utils.data import DataLoader


def get_train_dataloader(
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        qrels: Dict[str, Dict[str, int]],
        batch_size: int,
        shuffle: bool = True
):
    query_ids = list(queries.keys())
    train_samples = []
    for query_id in query_ids:
        s_q = queries[query_id]
        id_p, s_p, s_n = None, None, None
        assert len(qrels[query_id]) <= 2
        for corpus_id, score in qrels[query_id].items():
            if score == 1:
                id_p = corpus_id
                s_p = corpus[corpus_id].get("text")
            if score == -1:
                s_n = corpus[corpus_id].get("text")
        if s_n is not None:
            train_samples.append((query_id, id_p, s_q, s_p, s_n))
        else:
            train_samples.append((query_id, id_p, s_q, s_p))
    print("Loaded {} training pairs.".format(len(train_samples)))
    train_dataloader = DataLoader(train_samples, shuffle=shuffle, batch_size=batch_size)
    return train_dataloader


def get_eval_dataloader(
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        batch_size: int
):
    corpus_ids = list(corpus.keys())
    eval_samples = []
    for corpus_id in corpus_ids:
        s = corpus[corpus_id].get("text")
        eval_samples.append((corpus_id, s))
    print("Loaded {} eval corpus.".format(len(eval_samples)))
    eval_corpus_dataloader = DataLoader(eval_samples, shuffle=False, batch_size=batch_size)

    query_ids = list(queries.keys())
    eval_samples = []
    for query_id in query_ids:
        s = queries[query_id]
        eval_samples.append((query_id, s))
    print("Loaded {} eval queries.".format(len(eval_samples)))
    eval_queries_dataloader = DataLoader(eval_samples, shuffle=False, batch_size=batch_size)
    return eval_corpus_dataloader, eval_queries_dataloader


if __name__ == '__main__':
    from beir.datasets.data_loader import GenericDataLoader
    from src.utils import *

    mimic_data_path = os.path.join(data_path, f'mimic3/processed/mimic3_CCS_CODE')
    corpus, queries, qrels = GenericDataLoader(mimic_data_path).load(split="train_w_neg")
    train_dataloader = get_train_dataloader(corpus, queries, qrels, batch_size=16, shuffle=False)
    batch_tr = next(iter(train_dataloader))

    eval_corpus_dataloader, eval_queries_dataloader = get_eval_dataloader(corpus, queries, batch_size=16)
    batch_ev_c = next(iter(eval_corpus_dataloader))
    batch_ev_q = next(iter(eval_queries_dataloader))
    pass
