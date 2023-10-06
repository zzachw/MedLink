import os
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from src.dataset.vocab import build_vocab
from src.utils import data_path, load_pickle


def to_index(sequence, vocab, prefix='', suffix=''):
    """ convert code to index """
    prefix = [vocab(prefix)] if prefix else []
    suffix = [vocab(suffix)] if suffix else []
    sequence = prefix + [vocab(token) for token in sequence] + suffix
    return sequence


class CustomTokenizer:
    def __init__(self, dataset, code):
        self.dataset = dataset
        self.code = code
        self.vocabs, self.vocabs_size, self.emb, self.emb_size = self._load_vocab()

    def _load_vocab(self):
        vocab_dir = os.path.join(data_path, f'{self.dataset}/processed/{self.code}')
        build_vocab(self.dataset, self.code, vocab_dir)
        vocabs = load_pickle(os.path.join(vocab_dir, "vocab.pickle"))
        vocabs_size = len(vocabs)
        emb = load_pickle(os.path.join(vocab_dir, "emb.pickle"))
        emb_size = emb.shape[1]
        return vocabs, vocabs_size, emb, emb_size

    def get_vocabs_size(self):
        return self.vocabs_size

    def get_embedding_size(self):
        return self.emb_size

    def get_embedding(self):
        return self.emb

    def __call__(self, text: List[str], padding=True, prefix='<cls>', suffix=''):
        text_tokenized = []
        for sent in text:
            text_tokenized.append(torch.tensor(to_index(sent.split(' '), self.vocabs, prefix=prefix, suffix=suffix),
                                               dtype=torch.long))
        if padding:
            text_tokenized = pad_sequence(text_tokenized, batch_first=True)
        return text_tokenized


if __name__ == '__main__':
    from beir.datasets.data_loader import GenericDataLoader
    from src.dataset.utils import get_train_dataloader

    dataset = 'mimic3'
    code = 'CCS_CODE'
    mimic_data_path = os.path.join(data_path, f'{dataset}/processed/{dataset}_{code}')
    corpus, queries, qrels = GenericDataLoader(mimic_data_path).load(split="train")
    train_dataloader = get_train_dataloader(corpus, queries, qrels, batch_size=16)
    batch = next(iter(train_dataloader))
    tokenizer = CustomTokenizer(dataset, code)
    queries_tokenized = tokenizer(batch[2], padding=True, prefix='<cls>')
    print(queries_tokenized.shape)
