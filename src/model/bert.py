import math
import torch.nn as nn

from src.utils import *


class Attention(nn.Module):
    def forward(self, query, key, value, mask, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        """
        :param query, key, value: [batch_size, seq_len, d_model]
        :param mask: [batch_size, seq_len, seq_len]
        :return: [batch_size, seq_len, d_model]
        """

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask.unsqueeze(1), dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        mask = mask.sum(dim=-1) > 0
        x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """ Apply residual connection to any sublayer with the same size. """
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param dropout: dropout rate
        """

        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=4 * hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        print(f"TransformerBlock added with hid-{hidden}, head-{attn_heads}, in_hid-{2 * hidden}, drop-{dropout}")

    def forward(self, x, mask):
        """
        :param x: [batch_size, seq_len, hidden]
        :param mask: [batch_size, seq_len, seq_len]
        :return: batch_size, seq_len, hidden]
        """

        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return x


class BERT(nn.Module):
    def __init__(self, vocabs_size: int, embedding_size: int, dropout: float, layers: int, heads: int):
        super(BERT, self).__init__()
        self.vocabs_size = vocabs_size
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.layers = layers
        self.heads = heads

        self.embedding = nn.Embedding(vocabs_size, embedding_size, padding_idx=0)
        self.transformer = nn.ModuleList([TransformerBlock(embedding_size, heads, dropout) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

        print(f"# of params: {count_parameters(self)}")

    def forward(self, codes: torch.tensor):
        """
        :param codes: [# admissions, # codes]
        """

        """ embedding """
        x = self.embedding(codes)  # [# admissions, # codes, embedding_size]

        """ transformer """
        mask = (codes != 0)
        mask = torch.einsum('ab,ac->abc', mask, mask)
        for transformer in self.transformer:
            x = transformer(x, mask)  # [# admissions, # codes, embedding_size]
        x = self.dropout(x)

        return x


if __name__ == '__main__':
    from src.dataset.tokenizer import CustomTokenizer
    from beir.datasets.data_loader import GenericDataLoader
    from src.dataset.utils import get_train_dataloader

    dataset = 'mimic3'
    code = 'CCS_CODE'
    mimic_data_path = os.path.join(data_path, f'{dataset}/processed/{dataset}_{code}')
    corpus, queries, qrels = GenericDataLoader(mimic_data_path).load(split="train")
    train_dataloader = get_train_dataloader(corpus, queries, qrels, batch_size=16)
    batch = next(iter(train_dataloader))
    tokenizer = CustomTokenizer(dataset, code)
    batch_tokenized = tokenizer(batch[2], padding=True, prefix='<cls>')
    model = BERT(vocabs_size=tokenizer.get_vocabs_size(),
                 embedding_size=128,
                 dropout=0.5,
                 layers=2,
                 heads=2)
    print(model)
    with torch.autograd.detect_anomaly():
        x = model(batch_tokenized)
        print("x:", x.shape)
        x.sum().backward()
