import json
import copy
import torch
import torch.nn as nn
from torch.nn.init import constant_
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from kt_util import *
import random
import numpy as np


def get_kt_batch(qa, pid, batch_size):
    cls_dataset = KTDataset(qa, pid)
    data_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_kt)
    return data_loader


def collate_kt(batch_data):
    a_data_, p_data_ = zip(*batch_data)
    a_data = [torch.FloatTensor(item) for item in a_data_]
    padded_a = pad_sequence(a_data, batch_first=True, padding_value=0)  # auto-padding
    p_data = [torch.LongTensor(item) for item in p_data_]
    padded_p = pad_sequence(p_data, batch_first=True, padding_value=0)  # auto-padding
    return torch.FloatTensor(padded_a), torch.LongTensor(padded_p)


class KTDataset(Dataset):

    def __init__(self, a_data, p_data):
        self.a_data = a_data
        self.p_data = p_data

    def __getitem__(self, item):
        return self.a_data[item], self.p_data[item]

    def __len__(self):
        return len(self.a_data)


def get_format_list(tri_s, n_question, seq_len):
    a_lst, p_lst = [], []
    for pro_s, cons_s, ans_s in tri_s:
        q_seq, a_seq = [], []
        for i in range(len(ans_s)):
            answer = [0] * (2 * n_question)
            stt = n_question * ans_s[i]
            end = n_question * (ans_s[i] + 1)
            answer[stt: end] = [ans_s[i] * 2 - 1] * n_question
            a_seq.append(answer)
        a_lst.append(a_seq[:seq_len])
        p_lst.append(pro_s[:seq_len])
    return a_lst, p_lst


class KTData(object):
    def __init__(self, n_question, seq_len):
        self.seq_len = seq_len
        self.n_question = n_question
        self.a_train, self.p_train = [], []
        self.a_test, self.p_test = [], []

    def load_data(self, path, test_rate=0.3, train_path=None, test_path=None):
        if train_path is not None and test_path is not None:
            train_data = get_triple_list(train_path)
            test_data = get_triple_list(test_path)
        else:
            data = get_triple_list(path)
            random.shuffle(data)
            test = int(len(data) * test_rate)
            test_data, train_data = data[:test], data[test:]
        self.a_train, self.p_train = get_format_list(train_data, self.n_question, self.seq_len)
        self.a_test, self.p_test = get_format_list(test_data, self.n_question, self.seq_len)


class KT(nn.Module):
    def __init__(self, ec_data, n_question, n_pid, mid_size, dropout, n_blocks, final_fc_dim=512, n_heads=8, d_ff=2048):
        nn.Module.__init__(self)
        self.n_question = n_question
        self.dropout = dropout
        self.n_pid = n_pid
        d_model = mid_size * 2
        embed_l = d_model

        self.difficult_param = nn.Embedding(self.n_pid, 1).cuda()
        self.q_embed_diff = nn.Linear(self.n_question, embed_l).cuda()
        self.qa_embed_diff = nn.Linear(2 * self.n_question, embed_l).cuda()
        self.q_embed = nn.Linear(self.n_question, embed_l).cuda()
        self.qa_embed = nn.Linear(2 * self.n_question, embed_l).cuda()

        self.ec_net = EC(ec_data=ec_data, inp_size=50, mid_size=mid_size, num_types=n_question, num_layers=1)
        self.ec_net.k_net.weight = nn.Parameter(copy.copy(self.q_embed.weight.t())).cuda()

        self.att = TransformerBlock(n_blocks, d_model, d_ff, n_heads, dropout).cuda()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Tanh()
        ).cuda()
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                constant_(p, 0.)

    def forward(self, qa_data, pid_data, need=False):
        q_data = self.ec_net(pid_data)
        qa_data = qa_data * torch.cat([q_data, q_data], dim=-1)

        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        q_embed_diff_data = self.q_embed_diff(q_data)
        qa_embed_diff_data = self.qa_embed_diff(qa_data)
        pid_embed_data = self.difficult_param(pid_data)

        q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
        qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data

        d_output = self.att(q_embed_data, qa_embed_data)
        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.mlp(concat_q)

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
            mask = (torch.triu(torch.ones(ly, ly)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda()
            y = block(src=y, src_mask=mask)

        # decoder
        for block in self.blocks_2:
            lx, ly = x.size(0), y.size(0)
            if flg:  # peek current question
                mask = (torch.triu(torch.ones(lx, lx)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda()
                x = block(tgt=x, memory=x, tgt_mask=mask)
                flg = False
            else:  # dont peek current response
                mask = (torch.triu(torch.ones(lx, lx)) == 1).transpose(0, 1)
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).cuda()
                x = block(tgt=x, memory=y, tgt_mask=mask)
                flg = True
        return x


class EC(nn.Module):

    def __init__(self, ec_data, inp_size, mid_size, num_types=8, num_layers=1):
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(num_embeddings=ec_data.word_num, embedding_dim=inp_size,
                                      _weight=torch.tensor(ec_data.embedding, dtype=torch.float)).cuda()
        self.q_net = nn.LSTM(input_size=inp_size, hidden_size=mid_size, num_layers=num_layers,
                             bidirectional=True, batch_first=True, dropout=0.5).cuda()
        self.k_net = nn.Linear(in_features=2 * mid_size, out_features=num_types, bias=False).cuda()
        self.block = nn.TransformerEncoderLayer(d_model=num_types, nhead=num_types, activation='gelu').cuda()
        self.layer_norm = torch.nn.LayerNorm(num_types).cuda()
        self.sigmoid = torch.nn.Sigmoid().cuda()
        self.num_types = num_types
        self.ec_data = ec_data

    def forward_func(self, x):
        x = [self.ec_data.data_content[i] for i in x.tolist()]
        # pid.shape = [batch, un-padded]
        x = [torch.LongTensor(item) for item in x]
        # pid.shape = [batch, un-padded]
        x = pad_sequence(x, batch_first=True, padding_value=0)
        # pid.shape = [batch, length]
        x = self.embedding(torch.LongTensor(x).cuda())
        # x.shape = (batch, length, feature)
        ques, _ = self.q_net(x)
        # ques.shape = (batch, length, feature)
        ques = self.k_net(ques)
        # pred.shape = (batch, length, type)
        pred = torch.max(self.block(ques), dim=1)[0].squeeze(1)
        # pred.shape = (batch, type)
        pred = self.sigmoid(pred + self.layer_norm(pred))
        # pred.shape = (batch, type)
        return pred

    def forward(self, pid, ec=False):
        if ec:
            return self.forward_func(pid)
        bs, lr = pid.shape
        pid_ = pid.reshape((bs * lr))
        pred = self.forward_func(pid_)
        pred = pred.reshape((bs, lr, self.num_types))
        return pred


def pre_process(data, name_dict):
    dist = list()
    for problem in data:
        term = dict()
        term['class'] = [0] * len(name_dict)
        for tag in problem['topicTags']:
            term['class'][name_dict[tag]] = 1
        term['content'] = problem['content']
        term['questionId'] = problem['questionId']
        term['questionTitle'] = problem['questionTitle']
        dist.append(term)
    return dist


class ECData(object):
    def __init__(self, data, name_dict, trained_dict, test_rate=0.1, train_path=None, test_path=None):
        self.word_num = 0
        self.word_idx = dict()
        self.data_idx = dict()
        self.data = pre_process(data, name_dict)
        if train_path is not None and test_path is not None:
            with open(train_path, 'r', encoding='utf8') as train_data:
                self.train = json.load(train_data)
            with open(test_path, 'r', encoding='utf8') as test_data:
                self.test = json.load(test_data)
        else:
            self.train, self.test = data_split(data, name_dict, test_rate=test_rate)
        self.train_pid = [term['questionId'] for term in self.train]
        self.test_pid = [term['questionId'] for term in self.test]
        self.train_y = [term['class'] for term in self.train]
        self.test_y = [term['class'] for term in self.test]
        self.data_y = [term['class'] for term in self.data]
        self.trained_dict = trained_dict
        self.data_content = list()
        self.embedding = list()
        self.longest = 0

    def get_words(self):
        self.embedding.append([0] * 50)
        for term in self.data:
            s = term['content'].upper()
            words = s.split()
            for word in words:
                if word not in self.word_idx:
                    self.word_idx[word] = len(self.word_idx) + 1
                    if word in self.trained_dict:
                        self.embedding.append(self.trained_dict[word])
                    else:  # for unknown words
                        self.embedding.append([0] * 50)
        self.word_num = len(self.word_idx) + 1  # padding is No.0

    def get_id(self):
        for term in self.data:
            s = term['content'].upper()
            words = s.split()
            item = [self.word_idx[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.data_content.append(item)


fre_dct = {'array': 0.5633333333333334, 'hash-table': 0.18733333333333332, 'math': 0.19266666666666668,
           'string': 0.278, 'dynamic-programming': 0.202, 'greedy': 0.126, 'sorting': 0.12866666666666668,
           'depth-first-search': 0.11666666666666667}
fre_med = 0.19579240027727424


def data_split(data, name_dict, test_rate=0.1):
    test, train = list(), list()
    random.shuffle(data)
    for problem in data:
        term = dict()
        term['class'] = [0] * len(name_dict)
        for tag in problem['topicTags']:
            term['class'][name_dict[tag]] = 1
        term['content'] = problem['content']
        term['questionId'] = problem['questionId']
        term_rate = 1.0
        for tag in problem['topicTags']:
            term_rate *= fre_med / fre_dct[tag]
        if len(problem['topicTags']) != 0:
            term_rate = np.power(term_rate, 1 / len(problem['topicTags']))
        term_rate *= random.random()
        if term_rate > test_rate:
            train.append(term)
        else:
            test.append(term)
    train.sort(key=lambda x: len(x['content'].split()))
    test.sort(key=lambda x: len(x['content'].split()))
    return train, test


class ECDataset(Dataset):

    def __init__(self, pid_s, class_s):
        self.pid_s = pid_s
        self.class_s = class_s

    def __getitem__(self, item):
        return self.pid_s[item], self.class_s[item]

    def __len__(self):
        return len(self.pid_s)


def collate_ec(batch_data):
    pid_, class_ = zip(*batch_data)
    return torch.LongTensor(pid_), torch.LongTensor(class_)


def get_ec_batch(pid, y, batch_size):
    cls_dataset = ECDataset(pid, y)
    data_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_ec)
    return data_loader

