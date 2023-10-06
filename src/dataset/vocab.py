# This file is partly adapted from VecMap (https://github.com/artetxem/vecmap) developed by Mikel Artetxe.

import os

import numpy as np

from src.utils import data_path, dump_pickle


def read(file, threshold=0, vocabulary=None, dtype='float'):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    matrix = np.empty((count, dim), dtype=dtype) if vocabulary is None else []
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            matrix[i] = np.fromstring(vec, sep=' ', dtype=dtype)
        elif word in vocabulary:
            words.append(word)
            matrix.append(np.fromstring(vec, sep=' ', dtype=dtype))
    return (words, matrix) if vocabulary is None else (words, np.array(matrix, dtype=dtype))


class Vocabulary(object):

    def __init__(self):
        self.word2idx = {'<pad>': 0, '<cls>': 1}
        self.idx2word = {0: '<pad>', 1: '<cls>'}
        assert len(self.word2idx) == len(self.idx2word)
        self.idx = len(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(dataset, code, output_folder):
    filename = os.path.join(data_path, f'{dataset}/processed/{code}/word2vec.txt')
    print(f"Building vocab for {dataset}-{code}")

    # read word2vec
    with open(filename) as file:
        words, matrix = read(file)
    # vocab
    vocab = Vocabulary()
    init_size = len(vocab)
    for word in words:
        vocab.add_word(word)
    print(f"Vocab len:", len(vocab))
    # emb
    new_matrix = np.zeros((matrix.shape[0] + init_size, matrix.shape[1]))
    new_matrix[init_size:] = matrix

    # sanity check
    assert set(vocab.word2idx.keys()) == set(vocab.idx2word.values())
    assert set(vocab.word2idx.values()) == set(vocab.idx2word.keys())
    for word in vocab.word2idx.keys():
        assert word == vocab.idx2word[vocab(word)]
    assert len(vocab) == new_matrix.shape[0]

    dump_pickle(vocab, os.path.join(output_folder, 'vocab.pickle'))
    dump_pickle(new_matrix, os.path.join(output_folder, 'emb.pickle'))


if __name__ == '__main__':
    build_vocab('mimic3', 'CCS_CODE', os.path.join(data_path, f'mimic3/processed/CCS_CODE'))
