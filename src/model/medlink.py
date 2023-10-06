from typing import List

import torch.nn as nn
import torch.nn.functional as F

from src.utils import *


def batch_to_one_hot(label_batch, num_class):
    """ convert to one hot label """
    label_batch_onehot = []
    for label in label_batch:
        label_batch_onehot.append(F.one_hot(label, num_class).sum(dim=0))
    label_batch_onehot = torch.stack(label_batch_onehot, dim=0)
    return label_batch_onehot


class AdmissionPrediction(nn.Module):
    def __init__(self, tokenizer, encoder, device='cpu'):
        super(AdmissionPrediction, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.criterion = nn.BCEWithLogitsLoss()
        self.device = device

    def encode_one_hot(self, input: List[str]):
        input_batch = self.tokenizer(input, padding=False, prefix='', suffix='')
        input_onehot = batch_to_one_hot(input_batch, self.encoder.vocabs_size)
        input_onehot = input_onehot.float().to(self.device)
        return input_onehot

    def encode_dense(self, input: List[str]):
        input_batch = self.tokenizer(input, padding=True, prefix='<cls>', suffix='').to(self.device)
        mask = input_batch != 0
        input_embeddings = self.encoder(input_batch)
        return input_embeddings, mask

    def get_loss(self, logits, target_onehot):
        true_batch_size = min(logits.shape[0], target_onehot.shape[0])
        loss = self.criterion(logits[:true_batch_size], target_onehot[:true_batch_size])
        return loss

    def forward(self, input, vocab_emb):
        input_dense, mask = self.encode_dense(input)
        input_one_hot = self.encode_one_hot(input)
        logits = torch.matmul(input_dense, vocab_emb.T)
        logits[~mask] = -1e9
        logits = logits.max(dim=1)[0]
        return logits, input_one_hot


class MedLink(nn.Module):
    def __init__(self, tokenizer, queries_model, corpus_model, device='cpu'):
        super(MedLink, self).__init__()
        self.fwd_adm_pred = AdmissionPrediction(tokenizer, queries_model, device=device)
        self.bwd_adm_pred = AdmissionPrediction(tokenizer, corpus_model, device=device)
        self.criterion = nn.CrossEntropyLoss()
        self.vocabs_size = queries_model.vocabs_size
        self.device = device

    def encode_queries(self, queries: List[str]):
        all_vocab = torch.tensor(list(range(self.vocabs_size)), device=self.device)
        bwd_vocab_emb = self.bwd_adm_pred.encoder.embedding(all_vocab)
        pred_corpus, queries_one_hot = self.bwd_adm_pred(queries, bwd_vocab_emb)
        pred_corpus = torch.log(1 + torch.relu(pred_corpus))
        queries_emb = pred_corpus + queries_one_hot
        return queries_emb

    def encode_corpus(self, corpus: List[str]):
        all_vocab = torch.tensor(list(range(self.vocabs_size)), device=self.device)
        fwd_vocab_emb = self.fwd_adm_pred.encoder.embedding(all_vocab)
        pred_queries, corpus_one_hot = self.fwd_adm_pred(corpus, fwd_vocab_emb)
        pred_queries = torch.log(1 + torch.relu(pred_queries))
        corpus_emb = corpus_one_hot + pred_queries
        return corpus_emb

    def compute_scores(self, queries_emb, corpus_emb):
        n = torch.tensor(corpus_emb.shape[0]).to(queries_emb.device)
        df = (corpus_emb > 0).sum(dim=0)
        idf = torch.log(1 + n) - torch.log(1 + df)

        tf = torch.einsum('ac,bc->abc', queries_emb, corpus_emb)

        tf_idf = tf * idf
        final_scores = tf_idf.sum(dim=-1)
        return final_scores

    def get_loss(self, scores):
        label = torch.tensor(list(range(scores.shape[0])), device=scores.device)
        loss = self.criterion(scores, label)
        return loss

    def forward(self, queries, corpus):
        all_vocab = torch.tensor(list(range(self.vocabs_size)), device=self.device)
        fwd_vocab_emb = self.fwd_adm_pred.encoder.embedding(all_vocab)
        bwd_vocab_emb = self.bwd_adm_pred.encoder.embedding(all_vocab)
        pred_queries, corpus_one_hot = self.fwd_adm_pred(corpus, fwd_vocab_emb)
        pred_corpus, queries_one_hot = self.bwd_adm_pred(queries, bwd_vocab_emb)

        fwd_cls_loss = self.fwd_adm_pred.get_loss(pred_queries, queries_one_hot)
        bwd_cls_loss = self.bwd_adm_pred.get_loss(pred_corpus, corpus_one_hot)

        pred_queries = torch.log(1 + torch.relu(pred_queries))
        pred_corpus = torch.log(1 + torch.relu(pred_corpus))

        corpus_emb = corpus_one_hot + pred_queries
        queries_emb = pred_corpus + queries_one_hot

        scores = self.compute_scores(queries_emb, corpus_emb)
        ret_loss = self.get_loss(scores)
        return fwd_cls_loss, bwd_cls_loss, ret_loss

    def search(self, queries_ids, queries_embeddings, corpus_ids, corpus_embeddings):
        scores = self.compute_scores(queries_embeddings, corpus_embeddings)
        results = {}
        for q_idx, q_id in enumerate(queries_ids):
            results[q_id] = {}
            for c_idx, c_id in enumerate(corpus_ids):
                results[q_id][c_id] = scores[q_idx, c_idx].item()
        return results


if __name__ == '__main__':
    from beir.datasets.data_loader import GenericDataLoader
    from src.dataset.utils import get_train_dataloader
    from src.dataset.tokenizer import CustomTokenizer
    from src.model.bert import BERT

    dataset = 'mimic3'
    code = 'CCS_CODE'
    mimic_data_path = os.path.join(data_path, f'{dataset}/processed/{dataset}_{code}')
    corpus, queries, qrels = GenericDataLoader(mimic_data_path).load(split="train")
    train_dataloader = get_train_dataloader(corpus, queries, qrels, batch_size=16)
    batch = next(iter(train_dataloader))
    tokenizer = CustomTokenizer(dataset, code)
    corpus_encoder = BERT(vocabs_size=tokenizer.get_vocabs_size(), embedding_size=128, dropout=0.5, layers=2, heads=2)
    queries_encoder = BERT(vocabs_size=tokenizer.get_vocabs_size(), embedding_size=128, dropout=0.5, layers=2, heads=2)
    model = MedLink(tokenizer, corpus_encoder, queries_encoder)
    print(model)
    with torch.autograd.detect_anomaly():
        fwd_cls_loss, bwd_cls_loss, ret_loss = model(batch[2], batch[3])
        print("fwd_cls_loss:", fwd_cls_loss)
        print("bwd_cls_loss:", bwd_cls_loss)
        print("ret_loss:", ret_loss)
        (fwd_cls_loss + bwd_cls_loss + ret_loss).backward()
