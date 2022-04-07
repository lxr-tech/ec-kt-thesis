import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from kt_util import *
import random


class KT(nn.Module):
    def __init__(self, n_question, hidden_size, num_layer):
        nn.Module.__init__(self)
        self.n_question = n_question
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.rnn = nn.RNN(input_size=n_question * 2, hidden_size=hidden_size, num_layers=num_layer, batch_first=True).cuda()
        self.fc = nn.Linear(hidden_size, n_question).cuda()

    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.shape[0], self.hidden_size).cuda()
        # x.shape = (batch, length, type * 2)
        dist, _ = self.rnn(x, h0)
        # dist.shape = (batch, length, mid)
        pred = torch.sigmoid(self.fc(dist))
        # pred.shape = (batch, length, type)
        return pred


def get_format_list(tri_s, n_question, seq_len):
    q_lst, qa_lst = [], []
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
    return q_lst, qa_lst


class KTData(object):
    def __init__(self, n_question, seq_len):
        self.seq_len = seq_len
        self.n_question = n_question
        self.q_train, self.qa_train = [], []
        self.q_test, self.qa_test = [], []

    def load_data(self, path, test_rate=0.3, train_path=None, test_path=None):
        if train_path is not None and test_path is not None:
            train_data = get_triple_list(train_path)
            test_data = get_triple_list(test_path)
        else:
            data = get_triple_list(path)
            random.shuffle(data)
            test = int(len(data) * test_rate)
            test_data, train_data = data[:test], data[test:]
        self.q_train, self.qa_train = get_format_list(train_data, self.n_question, self.seq_len)
        self.q_test, self.qa_test = get_format_list(test_data, self.n_question, self.seq_len)


class ClsDataset(Dataset):

    def __init__(self, q_data, qa_data):
        self.q_data = q_data
        self.qa_data = qa_data

    def __getitem__(self, item):
        return self.q_data[item], self.qa_data[item]

    def __len__(self):
        return len(self.q_data)


def collate_fn(batch_data):
    q_data_, qa_data_ = zip(*batch_data)
    q_data = [torch.FloatTensor(item) for item in q_data_]
    padded_q = pad_sequence(q_data, batch_first=True, padding_value=0)  # auto-padding
    qa_data = [torch.FloatTensor(item) for item in qa_data_]
    padded_qa = pad_sequence(qa_data, batch_first=True, padding_value=0)  # auto-padding
    return torch.FloatTensor(padded_q), torch.FloatTensor(padded_qa)


def get_kt_batch(q, qa, batch_size):
    cls_dataset = ClsDataset(q, qa)
    data_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return data_loader
