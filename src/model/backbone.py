import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset.tokenizer import CustomTokenizer
from src.metrics import get_metrics_ir
from src.model.bert import BERT
from src.model.medlink import MedLink


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()
        self.args = args

        if args.model == "medlink":
            tokenizer = CustomTokenizer(args.dataset, args.code)
            queries_model = BERT(vocabs_size=tokenizer.get_vocabs_size(),
                                 embedding_size=tokenizer.get_embedding_size(),
                                 dropout=args.dropout,
                                 layers=args.layers,
                                 heads=args.heads)
            corpus_model = BERT(vocabs_size=tokenizer.get_vocabs_size(),
                                embedding_size=tokenizer.get_embedding_size(),
                                dropout=args.dropout,
                                layers=args.layers,
                                heads=args.heads)
            # load GloVe emb
            emb = tokenizer.get_embedding()
            queries_model.embedding.weight.data.copy_(torch.from_numpy(emb))
            corpus_model.embedding.weight.data.copy_(torch.from_numpy(emb))
            # retrieval model
            self.model = MedLink(tokenizer, queries_model, corpus_model, device=args.device)
        else:
            raise NotImplementedError

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        return

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = []
        for i, batch in enumerate(tqdm(data_loader)):
            if len(batch) == 4:
                query_ids, corpus_ids, queries, corpus = batch
                fwd_cls_loss, bwd_cls_loss, ret_loss = self.model(queries, corpus)
            if len(batch) == 5:
                query_ids, corpus_ids, queries, corpus, neg_corpus = batch
                fwd_cls_loss, bwd_cls_loss, ret_loss = self.model(queries, corpus + neg_corpus)
            loss = self.args.alpha * fwd_cls_loss + \
                   self.args.beta * bwd_cls_loss + \
                   self.args.gamma * ret_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())
        return {"loss": np.mean(total_loss)}

    def eval_epoch(self, corpus_dataloader, queries_dataloader, qrels, candidate, bootstrap):
        self.model.eval()
        all_corpus_ids, all_corpus_embeddings = [], []
        all_queries_ids, all_queries_embeddings = [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(corpus_dataloader)):
                corpus_ids, corpus = batch
                corpus_embeddings = self.model.encode_corpus(corpus)
                all_corpus_ids.extend(corpus_ids)
                all_corpus_embeddings.append(corpus_embeddings)
            for i, batch in enumerate(tqdm(queries_dataloader)):
                queries_ids, queries = batch
                queries_embeddings = self.model.encode_queries(queries)
                all_queries_ids.extend(queries_ids)
                all_queries_embeddings.append(queries_embeddings)
            all_corpus_embeddings = torch.cat(all_corpus_embeddings)
            all_queries_embeddings = torch.cat(all_queries_embeddings)
            results = self.model.search(all_queries_ids, all_queries_embeddings, all_corpus_ids, all_corpus_embeddings)
        ret = get_metrics_ir(qrels, results, k_values=[1, 5], candidate=candidate, bootstrap=bootstrap)
        return ret
