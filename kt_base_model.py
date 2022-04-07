import json
import torch
import torch.nn as nn
from torch.nn.init import constant_
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from kt_util import *
import random


class KT(nn.Module):
    def __init__(self, n_question, n_pid, d_model, dropout, n_blocks, final_fc_dim=512, n_heads=8, d_ff=2048):
        nn.Module.__init__(self)
        self.n_question = n_question
        self.dropout = dropout
        self.n_pid = n_pid
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid, 1).cuda()
            self.q_embed_diff = nn.Linear(self.n_question, embed_l).cuda()
            self.qa_embed_diff = nn.Linear(2 * self.n_question, embed_l).cuda()
        self.q_embed = nn.Linear(self.n_question, embed_l).cuda()
        self.qa_embed = nn.Linear(2 * self.n_question, embed_l).cuda()

        self.model = TransformerBlock(n_blocks, d_model, d_ff, n_heads, dropout).cuda()

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Tanh()
        ).cuda()
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                constant_(p, 0.)

    def forward(self, q_data, qa_data, pid_data, need=False):
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

        return output if not need else (output, d_output)


class TransformerBlock(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout):
        nn.Module.__init__(self)
        self.d_model = d_model
        self.n_heads = n_heads
        self.blocks_1 = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout)
            for _ in range(n_blocks * 2)
        ])

    def forward(self, q_embed_data, qa_embed_data):
        x = q_embed_data
        y = qa_embed_data
        flg = True

        # encoder
        for block in self.blocks_1:  # encode qas
            lx, ly = x.size(0), y.size(0)
            mask = (torch.triu(torch.ones(ly, ly), 1) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, -1e32).masked_fill(mask == 1, float(0.0)).cuda()
            y = block(src=y, src_mask=mask)

        # decoder
        for block in self.blocks_2:
            lx, ly = x.size(0), y.size(0)
            if flg:  # peek current question
                mask = (torch.triu(torch.ones(lx, lx), 0) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, -1e32).masked_fill(mask == 1, float(1.0)).cuda()
                x = block(tgt=x, memory=x, tgt_mask=mask)
                flg = False
            else:  # dont peek current response
                mask = (torch.triu(torch.ones(lx, lx), 0) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, -1e32).masked_fill(mask == 1, float(1.0)).cuda()
                x = block(tgt=x, memory=y, tgt_mask=mask)
                flg = True
        return x


def get_format_list(tri_s, n_question, seq_len):
    q_lst, qa_lst, p_lst = [], [], []
    for pro_s, cons_s, ans_s in tri_s:
        q_seq, a_seq = [], []
        for i in range(len(ans_s)):
            question = [0] * n_question
            answer = [0] * (2 * n_question)
            for q in cons_s[i]:
                question[int(q)] = 1
                answer[int(q) + int(ans_s[i]) * n_question] = int(ans_s[i]) * 2 - 1
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
