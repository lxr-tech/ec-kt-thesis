import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random


class EC(nn.Module):

    def __init__(self, inp_size, mid_size, num_words, num_types=8, num_layers=1, weight=None):
        nn.Module.__init__(self)
        self.embedding = nn.Embedding(num_embeddings=num_words, embedding_dim=inp_size, _weight=weight).cuda()
        self.q_net = nn.LSTM(input_size=inp_size, hidden_size=mid_size, num_layers=num_layers,
                             bidirectional=True, batch_first=True, dropout=0.5).cuda()
        self.k_net = nn.Linear(in_features=2 * mid_size, out_features=num_types, bias=False).cuda()
        self.block = nn.TransformerEncoderLayer(d_model=num_types, nhead=num_types, activation='gelu').cuda()
        self.layer_norm = torch.nn.LayerNorm(num_types).cuda()
        self.sigmoid = torch.nn.Sigmoid().cuda()

    def forward(self, x):
        x = self.embedding(torch.LongTensor(x).cuda())
        # x.shape = (batch, length, feature)
        ques, _ = self.q_net(x)
        # ques.shape = (batch, length, feature)
        out = self.k_net(ques)
        # out.shape = (batch, length, type)
        out = torch.max(self.block(out), dim=1)[0].squeeze(1)
        # out.shape = (batch, type)
        out = self.sigmoid(out + self.layer_norm(out))
        # out.shape = (batch, type)
        return out


def data_split(data, name_dict, test_rate=0.2):
    test = list()
    train = list()
    for i, problem in enumerate(data):
        term = dict()
        term['class'] = [0] * len(name_dict)
        for tag in problem['topicTags']:
            term['class'][name_dict[tag]] = 1
        term['content'] = problem['content']
        term['title'] = problem['questionTitle']
        if random.random() > test_rate:
            train.append(term)
        else:
            test.append(term)
    train.sort(key=lambda x: len(x['content'].split()))
    test.sort(key=lambda x: len(x['content'].split()))
    return train, test


class ECData(object):
    def __init__(self, data, name_dict, trained_dict, test_rate=0.2, train_path=None, test_path=None):
        self.data = data
        self.word_num = 0
        self.word_idx = dict()
        if train_path is not None and test_path is not None:
            with open(train_path, 'r', encoding='utf8') as train_data:
                self.train = json.load(train_data)
            with open(test_path, 'r', encoding='utf8') as test_data:
                self.test = json.load(test_data)
        else:
            self.train, self.test = data_split(data, name_dict, test_rate=test_rate)
        self.train_y = [term['class'] for term in self.train]
        self.test_y = [term['class'] for term in self.test]
        self.trained_dict = trained_dict
        self.train_content = list()
        self.test_content = list()
        self.embedding = list()
        self.longest = 0

    def save_data(self, train_path, test_path):
        with open(train_path, 'w') as json_file:
            json_file.write(json.dumps(self.train, indent=4))
        with open(test_path, 'w') as json_file:
            json_file.write(json.dumps(self.test, indent=4))

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
        for term in self.train:
            s = term['content'].upper()
            words = s.split()
            item = [self.word_idx[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_content.append(item)
        for term in self.test:
            s = term['content'].upper()
            words = s.split()
            item = [self.word_idx[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_content.append(item)


class ClsDataset(Dataset):

    def __init__(self, contents, classes):
        self.contents = contents
        self.classes = classes

    def __getitem__(self, item):
        return self.contents[item], self.classes[item]

    def __len__(self):
        return len(self.classes)


def collate_fn(batch_data):
    content_, class_ = zip(*batch_data)
    contents = [torch.LongTensor(item) for item in content_]
    padded_contents = pad_sequence(contents, batch_first=True, padding_value=0)  # auto-padding
    return torch.LongTensor(padded_contents), torch.LongTensor(class_)


def get_ec_batch(x, y, batch_size):
    cls_dataset = ClsDataset(x, y)
    data_loader = DataLoader(cls_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn)
    return data_loader
