import math
import numpy as np

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from kt_util import *
import random


def get_format_list(tri_s, n_question, seq_len):
    q_lst, qa_lst, p_lst = [], [], []
    for pro_s, cons_s, ans_s in tri_s:
        q_seq, a_seq = [], []
        for i in range(len(ans_s)):
            question = [0] * n_question
            answer = [0] * (2 * n_question)
            for q in cons_s[i]:
                question[int(q)] = 1
                answer[int(q) + int(ans_s[i]) * n_question] = int(ans_s[i])
            q_seq.append(question)
            a_seq.append(answer)
        q_lst.append(q_seq[:seq_len])
        qa_lst.append(a_seq[:seq_len])
        p_lst.append(pro_s[:seq_len])
    return q_lst, qa_lst, p_lst


class KTData(object):
    def __init__(self, n_question, seq_len):
        self.seq_len = seq_len
        self.n_question = n_question
        self.q_train, self.qa_train, self.p_train = [], [], []
        self.q_test, self.qa_test, self.p_test = [], [], []

    def load_data(self, path, test_rate=0.3, train_path=None, test_path=None):
        if train_path is not None and test_path is not None:
            train_data = get_triple_list(train_path)
            test_data = get_triple_list(test_path)
        else:
            data = get_triple_list(path)
            random.shuffle(data)
            test = int(len(data) * test_rate)
            test_data, train_data = data[:test], data[test:]
        self.q_train, self.qa_train, self.p_train = get_format_list(train_data, self.n_question, self.seq_len)
        self.q_test, self.qa_test, self.p_test = get_format_list(test_data, self.n_question, self.seq_len)


class ClsDataset(Dataset):

    def __init__(self, q_data, qa_data, pid_data):
        self.q_data = q_data
        self.qa_data = qa_data
        self.pid_data = pid_data

    def __getitem__(self, item):
        return self.q_data[item], self.qa_data[item], self.pid_data[item]

    def __len__(self):
        return len(self.q_data)


def collate_fn(batch_data):
    q_data_, qa_data_, pid_data_ = zip(*batch_data)
    q_data = [torch.FloatTensor(item) for item in q_data_]
    padded_q = pad_sequence(q_data, batch_first=True, padding_value=0)  # auto-padding
    qa_data = [torch.FloatTensor(item) for item in qa_data_]
    padded_qa = pad_sequence(qa_data, batch_first=True, padding_value=0)  # auto-padding
    pid_data = [torch.LongTensor(item) for item in pid_data_]
    padded_pid = pad_sequence(pid_data, batch_first=True, padding_value=0)  # auto-padding
    return torch.FloatTensor(padded_q), torch.FloatTensor(padded_qa), torch.LongTensor(padded_pid)


def get_kt_batch(q, qa, pid, batch_size):
    cls_dataset = ClsDataset(q, qa, pid)
    data_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return data_loader


class KT(nn.Module):
    def __init__(self, n_question, n_pid, n_blocks, d_model, dropout, kq_same, final_fc_dim=512, n_heads=8, d_ff=2048):
        nn.Module.__init__(self)
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid, 1).cuda()
            self.q_embed_diff = nn.Linear(self.n_question, embed_l).cuda()
            self.qa_embed_diff = nn.Linear(2 * self.n_question, embed_l).cuda()
        self.q_embed = nn.Linear(self.n_question, embed_l).cuda()
        self.qa_embed = nn.Linear(2 * self.n_question, embed_l).cuda()

        self.model = TransformerBlock(n_blocks, d_model, d_model // n_heads, d_ff, n_heads, dropout, kq_same).cuda()

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).cuda()
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                constant_(p, 0.)

    def forward(self, q_data, qa_data, pid_data):
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        q_embed_diff_data = self.q_embed_diff(q_data)
        qa_embed_diff_data = self.qa_embed_diff(qa_data)
        pid_embed_data = self.difficult_param(pid_data)

        q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
        qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data

        d_output = self.model(q_embed_data, qa_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, n_blocks, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model, d_feature, d_ff, n_heads, dropout, kq_same)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model, d_feature, d_ff, n_heads, dropout, kq_same)
            for _ in range(n_blocks * 2)
        ])

    def forward(self, q_embed_data, qa_embed_data):
        x = q_embed_data
        y = qa_embed_data

        # encoder
        for block in self.blocks_1:  # encode qas
            y = block(mask=1, query=y, key=y, values=y)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False)
                flag_first = False
            else:  # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        nn.Module.__init__(self)
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        Input:
            block : object of type BasicBlock(nn.Module).
                    It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            values: In transformer paper,
                    it is the input for encoder and encoded output for decoder (in masked attention part)
        Output:
            query: Input gets changed over the layer and returned.
        """
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        # print(torch.from_numpy(nopeek_mask))
        src_mask = (torch.from_numpy(nopeek_mask) == 0).cuda()
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1(query2)
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, self.gammas)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).cuda()
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().cuda()
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).cuda()
        dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)
    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
    total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill(mask == 0, -1e23)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).cuda()
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

