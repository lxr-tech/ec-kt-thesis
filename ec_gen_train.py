import json
import torch
import torch.nn.functional as F
from torch import optim

idx = [0, 2, 3, 4]
name_list = ['array', 'hash-table', 'math', 'string', 'dynamic-programming', 'greedy', 'sorting', 'depth-first-search']


class Trainer(object):

    def __init__(self, model, train_data, epoch_num, alpha, batch_size, **kwargs):
        self.model = model
        self.alpha = alpha
        self.epoch_num = epoch_num
        self.train_data = train_data
        self.batch_size = batch_size
        self.max_grad_norm = kwargs['max_grad_norm'] if 'max_grad_norm' in kwargs else -1
        self.weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        self.optimizer = optim.Adam(model.parameters(), lr=alpha, weight_decay=self.weight_decay)
        self.loss_fun = F.cosine_similarity

    def model_save(self, epoch, path):
        state = {'ec_net': self.model.state_dict(),
                 'ec_opt': self.optimizer.state_dict(),
                 'lr_ec': self.alpha,
                 'epoch_num': epoch,
                 'bs_ec': self.batch_size,
                 'l2_ec': self.weight_decay,
                 'max_grad_norm': self.max_grad_norm,
                 }
        torch.save(state, path)

    def train_func(self):
        self.model.train()
        loss_tra = torch.Tensor([0]).cuda()
        for i, batch in enumerate(self.train_data):
            x, y = batch
            y = y.cuda()
            pred = self.model(x).cuda()
            self.optimizer.zero_grad()
            loss = 1 - self.loss_fun(pred, y).cuda()
            loss = torch.sum(loss, dim=0)
            loss_tra += loss
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
        return loss_tra

    def model_eval(self, batch_data):
        self.model.eval()
        loss_tot, acc1, acc2, acc3 = 0, [], [], []
        hit1, hit2 = [], []
        for i, batch in enumerate(batch_data):
            x, y = batch
            y = y.cuda()
            pred = self.model(x).cuda()
            loss = 1 - self.loss_fun(pred, y).cuda()
            loss = torch.sum(loss, dim=0)
            loss_tot += loss.item()
            pred = torch.round(pred)
            acc = torch.mean(torch.tensor([pred[k].equal(y[k]) for k in range(pred.shape[0])], dtype=torch.float))
            acc1.append(acc.item())
            acc = torch.mean(torch.tensor([pred[k][idx].equal(y[k][idx]) for k in range(pred.shape[0])], dtype=torch.float))
            acc2.append(acc.item())
            acc = torch.mean((torch.tensor(pred == y, dtype=torch.float)))
            acc3.append(acc.item())
            hit = torch.mean(torch.tensor([(torch.sum(pred[k]*y[k])+1e-5) / (torch.sum(pred[k])+1e-5) for k in range(pred.shape[0])], dtype=torch.float))
            hit1.append(hit.item())
            hit = torch.mean(torch.tensor([(torch.sum(pred[k]*y[k])+1e-5) / (torch.sum(y[k])+1e-5) for k in range(pred.shape[0])], dtype=torch.float))
            hit2.append(hit.item())
        acc1 = sum(acc1) / len(acc1)
        acc2 = sum(acc2) / len(acc2)
        acc3 = sum(acc3) / len(acc3)
        hit1 = sum(hit1) / len(hit1)
        hit2 = sum(hit2) / len(hit2)
        hit3 = 2 / (1 / hit1 + 1 / hit2)
        return loss_tot, acc1, acc2, acc3, hit1, hit2, hit3

    def model_valid(self, glove, path):
        data = list()
        for item in glove.data:
            s = item['content'].upper()
            temp, words = dict(), s.split()
            temp['content'] = item['content']
            temp['title'] = item['questionTitle']
            item = [glove.word_idx[word] for word in words]
            item = torch.LongTensor(item).unsqueeze(0)
            pred = torch.round(self.model(item))
            temp['class'] = pred.detach().cpu().numpy().tolist()[0]
            data.append(temp)
        with open(path, 'w') as json_file:
            json_file.write(json.dumps(data, indent=4))

