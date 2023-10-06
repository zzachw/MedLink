import random
from collections import OrderedDict
from typing import List, Dict

import numpy as np
import pytrec_eval


def update_dict(d, k, v):
    if k not in d:
        d[k] = []
    d[k].append(v)
    return d


def sample_from_dict(d, num_samples=10):
    keys = np.random.choice(list(d.keys()), num_samples)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))


def bootstrapping(qrels, results, k_values, num_iterations, num_samples):
    if num_samples is None:
        num_samples = len(qrels)
    outputs = {}
    for _ in range(num_iterations):
        qrels_sample = sample_from_dict(qrels, num_samples)
        results_sample = {k: results[k] for k in qrels_sample}
        output = metrics_ir(qrels_sample, results_sample, k_values)
        for k, v in output.items():
            outputs = update_dict(outputs, k, v)
    statistics = {}
    for k, v in outputs.items():
        statistics[k] = np.mean(v)
        statistics[k + '_std'] = np.std(v)
    return statistics


def get_metrics_ir(qrels, results, k_values, candidate, bootstrap, num_iterations=1000, num_samples=None):
    results = {q_id: candidate_sample(q_id, scores, qrels, candidate) for q_id, scores in results.items()}
    if bootstrap:
        ret = bootstrapping(qrels, results, k_values, num_iterations, num_samples)
    else:
        ret = metrics_ir(qrels, results, k_values)
    return OrderedDict(sorted(ret.items()))


def candidate_sample(q_id, scores, qrels, candidate, size=None):
    c_ids = list(qrels[q_id].keys())
    candidate_ids = candidate[q_id]
    for c_id in c_ids:
        assert c_id not in candidate_ids
    if size is not None:
        candidate_ids = random.sample(candidate_ids, size - len(c_ids))
    final_ids = candidate_ids + c_ids
    return {c_id: scores[c_id] for c_id in final_ids}


def metrics_ir(qrels: Dict[str, Dict[str, int]],
               results: Dict[str, Dict[str, float]],
               k_values: List[int]) -> Dict[str, float]:
    ret = {}

    for k in k_values:
        ret[f"NDCG@{k}"] = 0.0
        ret[f"MAP@{k}"] = 0.0
        ret[f"Recall@{k}"] = 0.0
        ret[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ret[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            ret[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            ret[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            ret[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ret[f"NDCG@{k}"] = round(ret[f"NDCG@{k}"] / len(scores), 5)
        ret[f"MAP@{k}"] = round(ret[f"MAP@{k}"] / len(scores), 5)
        ret[f"Recall@{k}"] = round(ret[f"Recall@{k}"] / len(scores), 5)
        ret[f"P@{k}"] = round(ret[f"P@{k}"] / len(scores), 5)

    return ret
