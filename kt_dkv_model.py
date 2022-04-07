import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from kt_util import *
import random


class KT(nn.Module):
    """
    Dynamic Key-Value Memory Networks for Knowledge Tracing at WWW'2017
    """

    # @staticmethod
    # def add_arguments(parser):
    #     RNN.add_arguments(parser)
    #     parser.add_argument('-k', type=int, default=10, help='use top k similar topics to predict')
    #     parser.add_argument('--knows', default='data/know_list.txt', help='numbers of knowledge concepts')
    #     parser.add_argument('-ks', '--knowledge_hidden_size', type=int, default=25, help='knowledge emb size')
    #     parser.add_argument('-l', '--num_layers', type=int, default=2, help='#topic rnn layers')
    #     # parser.add_argument('-es', '--erase_vector_size', type=float, default=25, help='erase vector emb size')
    #     # parser.add_argument('-as', '--add_vector_size', type=float, default=25, help='add vector emb size')

    def __init__(self, n_question, mem_size, hid_size):
        nn.Module.__init__(self)

        self.n_question = n_question
        self.mem_size = mem_size
        self.hid_size = hid_size
        self.value_size = hid_size * 2

        # knowledge embedding module
        self.knowledge_model = nn.Linear(self.n_question, self.hid_size).cuda()

        # knowledge memory matrix
        self.memory = nn.Parameter(torch.zeros(self.n_question, self.mem_size)).cuda()
        self.memory.data.uniform_(-1, 1)

        # read process embedding module
        self.ft_embedding = nn.Linear(self.hid_size + self.mem_size, 50).cuda()
        self.score_layer = nn.Linear(50, 1).cuda()

        # write process embedding module
        self.cks_embedding = nn.Linear(self.n_question * 2, self.value_size).cuda()
        self.erase_embedding = nn.Linear(self.value_size, self.hid_size).cuda()
        self.add_embedding = nn.Linear(self.value_size, self.hid_size).cuda()

        # the first student state
        self.h_initial = nn.Parameter(torch.zeros(self.n_question, self.hid_size)).cuda()
        self.h_initial.data.uniform_(-1, 1)

    def forward(self, knowledge, score):
        hidden = self.h_initial

        expand_vec = knowledge.float() * score.unsqueeze(-1)
        # expand_vec.shape = (batch, length, type)
        cks = torch.cat([knowledge.float(), expand_vec], dim=-1)
        # cks.shape = (batch, length, type * 2)
        hidden_vec = self.knowledge_model(knowledge.float())
        # hidden_vec.shape = (batch, length, hid_size)

        '''calculate alpha weights of knowledge using dot product'''
        # hidden_vec.shape = (batch, length, hid_size)
        alpha = torch.matmul(hidden_vec, hidden.t())
        # hidden.shape = (type, hid_size)
        alpha = torch.softmax(alpha, dim=-1)
        # alpha.shape = (batch, length, type)

        '''read process'''
        # memory.shape = (type, mem_size)
        rt = torch.matmul(alpha, self.memory)
        # rt.shape = (batch, length, mem_size)
        com_r_k = torch.cat([rt, hidden_vec], dim=-1)
        # com_r_k.shape = (batch, length, mem_size + hid_size)
        ft = torch.tanh(self.ft_embedding(com_r_k))
        # com_r_k.shape = (batch, length, 50)
        predict_score = torch.sigmoid(self.score_layer(ft))
        # predict_score.shape = (batch, length, 1)

        # '''write process'''
        # # cks.shape = (batch, length, type * 2)
        # vt = self.cks_embedding(cks)
        # # vt.shape = (batch, length, hid_size * 2)
        # et = torch.sigmoid(self.erase_embedding(vt))
        # # et.shape = (batch, length, hid_size)
        # at = torch.tanh(self.add_embedding(vt))
        # # at.shape = (batch, length, hid_size)
        # ht = hidden * (1 - (alpha.reshape(-1, 1) * et))
        # hidden = ht + (alpha.reshape(-1, 1) * at)
        # print(hidden.shape)

        return predict_score  # .reshape(1), hidden


def get_format_list(tri_s, n_question, seq_len):
    q_lst, a_lst = [], []
    for pro_s, cons_s, ans_s in tri_s:
        q_seq, a_seq = [], []
        for i in range(len(ans_s)):
            question = [0] * n_question
            for q in cons_s[i]:
                question[int(q)] = 1
            q_seq.append(question)
            a_seq.append(ans_s[i])
        q_lst.append(q_seq[:seq_len])
        a_lst.append(a_seq[:seq_len])
    return q_lst, a_lst


class KTData(object):
    def __init__(self, n_question, seq_len):
        self.seq_len = seq_len
        self.n_question = n_question
        self.q_train, self.a_train = [], []
        self.q_test, self.a_test = [], []

    def load_data(self, path, test_rate=0.3, train_path=None, test_path=None):
        if train_path is not None and test_path is not None:
            train_data = get_triple_list(train_path)
            test_data = get_triple_list(test_path)
        else:
            data = get_triple_list(path)
            random.shuffle(data)
            test = int(len(data) * test_rate)
            test_data, train_data = data[:test], data[test:]
        self.q_train, self.a_train = get_format_list(train_data, self.n_question, self.seq_len)
        self.q_test, self.a_test = get_format_list(test_data, self.n_question, self.seq_len)


class ClsDataset(Dataset):

    def __init__(self, q_data, a_data):
        self.q_data = q_data
        self.a_data = a_data

    def __getitem__(self, item):
        return self.q_data[item], self.a_data[item]

    def __len__(self):
        return len(self.q_data)


def collate_fn(batch_data):
    q_data_, a_data_ = zip(*batch_data)
    q_data = [torch.FloatTensor(item) for item in q_data_]
    padded_q = pad_sequence(q_data, batch_first=True, padding_value=0)  # auto-padding
    a_data = [torch.FloatTensor(item) for item in a_data_]
    padded_a = pad_sequence(a_data, batch_first=True, padding_value=0)  # auto-padding
    return torch.FloatTensor(padded_q), torch.FloatTensor(padded_a)


def get_kt_batch(q, qa, batch_size):
    cls_dataset = ClsDataset(q, qa)
    data_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return data_loader
