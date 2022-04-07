import torch
from torch import optim
import torch.nn as nn


class Trainer(object):

    def __init__(self, model, epoch_num, alpha, batch_size, **kwargs):
        self.model = model
        self.alpha = alpha
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        self.max_grad_norm = kwargs['max_grad_norm'] if 'max_grad_norm' in kwargs else -1
        self.optimizer = optim.Adam(model.parameters(), lr=alpha, weight_decay=self.weight_decay)
        self.loss_fun = nn.BCELoss(reduction='sum')
        self.n_question = model.n_question

    def model_save(self, epoch, path):
        state = {'kt_net': self.model.state_dict(),
                 'kt_opt': self.optimizer.state_dict(),
                 'lr_kt': self.alpha,
                 'epoch_num': epoch,
                 'bs_kt': self.batch_size,
                 'l2_kt': self.weight_decay,
                 'max_grad_norm': self.max_grad_norm,
                 }
        torch.save(state, path)

    def train_func(self, batch_data):
        self.model.train()
        loss_tra = torch.Tensor([0]).cuda()
        for batch in batch_data:
            q, qa = batch
            q, qa = q.cuda(), qa.cuda()
            y = qa[:, 1:, self.n_question:]
            pred = self.model(qa)[:, :-1, :]
            mask = torch.sum(q[:, 1:, :], dim=-1) != 0
            mask = mask.unsqueeze(-1)
            # print(y.shape, pred.shape, mask.shape)
            y, pred = y * mask, pred * mask
            self.optimizer.zero_grad()
            loss = self.loss_fun(pred.reshape((-1, )), y.reshape((-1, ))).cuda()
            loss_tra += loss
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
        return loss_tra

    def model_eval(self, batch_data):
        self.model.eval()
        loss_tot, acc1, acc2, acc3, acc4 = 0, [], [], [], []
        for batch in batch_data:
            q, qa = batch
            q, qa = q.cuda(), qa.cuda()
            y = qa[:, 1:, self.n_question:]
            pred = self.model(qa)[:, :-1, :]
            mask = torch.sum(q[:, 1:, :], dim=-1) != 0
            mask = mask.unsqueeze(-1)
            y, pred = y * mask, pred * mask
            loss = self.loss_fun(pred.reshape((-1, )), y.reshape((-1, )))
            loss_tot += loss.item()
            pred = torch.round(pred)
            acc = torch.mean(torch.tensor(pred == y, dtype=torch.float))
            acc1.append(acc.item())
            acc = torch.mean(torch.tensor([pred[k].equal(y[k])
                                           for k in range(y.shape[0])], dtype=torch.float))
            acc2.append(acc.item())
            pr, yr = pred, y
            acc = torch.mean(torch.tensor([(torch.sum(pr[k] * yr[k]) + 1e-5) / (torch.sum(yr[k]) + 1e-5)
                                           for k in range(yr.shape[0])], dtype=torch.float))
            acc3.append(acc.item())
            pw, yw = 1 - pred, 1 - y
            acc = torch.mean(torch.tensor([(torch.sum(pw[k] * yw[k]) + 1e-5) / (torch.sum(yw[k]) + 1e-5)
                                           for k in range(yw.shape[0])], dtype=torch.float))
            acc4.append(acc.item())
        acc1 = sum(acc1) / len(acc1)
        acc2 = sum(acc2) / len(acc2)
        acc3 = sum(acc3) / len(acc3)
        acc4 = sum(acc4) / len(acc4)
        return loss_tot, acc1, acc2, acc3, acc4

    def model_valid(self, valid_from, valid_to):
        self.model.eval()
        L, Q, P, A = 0, [], [], []
        acc1, acc2 = [], []
        f_data, t_data = open(valid_from, 'r'), open(valid_to, 'w+')
        for lineID, line in enumerate(f_data):
            if lineID % 4 == 0:
                L = eval(line)
            else:
                line = '[' + line.strip() + ']'
            if lineID % 4 == 1:
                P = eval(line)
            elif lineID % 4 == 2:
                Q = eval(line)
                while -1 in Q:
                    Q.remove(-1)
            elif lineID % 4 == 3:
                A = eval(line)
                a_seq = []
                for i in range(len(A)):
                    if len(Q[i]) > 0:
                        answer = [0] * (2 * self.n_question)
                        for q in Q[i]:
                            answer[int(q) + int(A[i]) * self.n_question] = int(A[i])
                        a_seq.append(answer)
                if len(a_seq) > 0:
                    a_seq = torch.FloatTensor(a_seq).unsqueeze(0).cuda()
                    y = a_seq[:, 1:, self.n_question:]
                    pred = self.model(a_seq)
                    pred = torch.round(pred[:, :-1, :])
                    print(L, file=t_data)
                    print(','.join([str(p) for p in P]), file=t_data)
                    print(','.join([str(q).replace(' ', '') for q in Q]), file=t_data)
                    print(','.join([str(a) for a in A]), file=t_data)
                    print(','.join([str(r) for r in y[0].tolist()]), file=t_data)
                    print(','.join([str(r).replace(' ', '') for r in pred[0].tolist()]), file=t_data)
                    pr, yr = pred[0], y[0]
                    acc = (torch.sum(pr * yr) + 1e-5) / (torch.sum(yr) + 1e-5)
                    acc1.append(acc.item())
                    pw, yw = 1 - pred, 1 - y
                    acc = (torch.sum(pw * yw) + 1e-5) / (torch.sum(yw) + 1e-5)
                    acc2.append(acc.item())
        acc1 = torch.mean(torch.tensor(acc1, dtype=torch.float))
        acc2 = torch.mean(torch.tensor(acc2, dtype=torch.float))
        print(acc1.item(), ', ', acc2.item(), file=t_data)
        f_data.close(), t_data.close()

