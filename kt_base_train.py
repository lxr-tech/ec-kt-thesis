import torch
from torch import optim
import torch.nn as nn


class Trainer(object):

    def __init__(self, model, train_data, epoch_num, alpha, batch_size, **kwargs):
        self.model = model
        self.alpha = alpha
        self.epoch_num = epoch_num
        self.train_data = train_data
        self.batch_size = batch_size
        self.weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0
        self.max_grad_norm = kwargs['max_grad_norm'] if 'max_grad_norm' in kwargs else -1
        self.optimizer = optim.Adam(model.parameters(), lr=alpha, weight_decay=self.weight_decay)
        self.loss_fun = nn.MSELoss(reduce=False)
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

    def train_func(self):
        self.model.train()
        loss_tra = torch.Tensor([0]).cuda()
        for i, batch in enumerate(self.train_data):
            q, qa, pid = batch
            q, qa, pid = q.cuda(), qa.cuda(), pid.cuda()
            y = 2 * torch.sign(torch.sum(qa[:, :, self.n_question:], dim=-1)) - 1
            pred = self.model(q, qa, pid).squeeze(-1)
            mask = torch.sum(q, dim=-1) != 0
            y, pred = y * mask, pred * mask
            self.optimizer.zero_grad()
            loss = torch.sum(self.loss_fun(pred, y))
            loss_tra += loss.item()
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.step()
        return loss_tra

    def model_eval(self, batch_data):
        self.model.eval()
        loss_tot, acc1, acc2, acc3, acc4 = 0, [], [], [], []
        for i, batch in enumerate(batch_data):
            q, qa, pid = batch
            q, qa, pid = q.cuda(), qa.cuda(), pid.cuda()
            y = 2 * torch.sign(torch.sum(qa[:, :, self.n_question:], dim=-1)) - 1
            pred = self.model(q, qa, pid).squeeze(-1)
            loss = torch.sum(self.loss_fun(pred, y))
            loss_tot += loss.item()
            pred = torch.round(pred)
            mask = torch.sum(qa, dim=-1) != 0
            pred, y = pred * mask, y * mask
            acc = torch.mean(torch.tensor(pred == y, dtype=torch.float))
            acc1.append(acc.item())
            acc = torch.mean(torch.tensor([(pred[k] * (y[k] != 0)).equal(y[k])
                                           for k in range(y.shape[0])], dtype=torch.float))
            acc2.append(acc.item())
            pr, yr = torch.clip(pred, min=0), torch.clip(y, min=0)
            acc = torch.mean(torch.tensor([(torch.sum(pr[k] * yr[k]) + 1e-5) / (torch.sum(yr[k]) + 1e-5)
                                           for k in range(yr.shape[0])], dtype=torch.float))
            acc3.append(acc.item())
            pw, yw = torch.clip(-pred, min=0), torch.clip(-y, min=0)
            acc = torch.mean(torch.tensor([(torch.sum(pw[k] * yw[k]) + 1e-5) / (torch.sum(yw[k]) + 1e-5)
                                           for k in range(yw.shape[0])], dtype=torch.float))
            acc4.append(acc.item())
        acc1 = torch.mean(torch.tensor(acc1, dtype=torch.float)).item()
        acc2 = torch.mean(torch.tensor(acc2, dtype=torch.float)).item()
        acc3 = torch.mean(torch.tensor(acc3, dtype=torch.float)).item()
        acc4 = torch.mean(torch.tensor(acc4, dtype=torch.float)).item()
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
                q_seq, p_seq, a_seq = [], [], []
                for i in range(len(A)):
                    if len(Q[i]) > 0:
                        question = [0] * self.n_question
                        answer = [0] * (2 * self.n_question)
                        for q in Q[i]:
                            question[int(q)] = 1
                            answer[int(q) + int(A[i]) * self.n_question] = int(A[i]) * 2 - 1
                        q_seq.append(question)
                        a_seq.append(answer)
                        p_seq.append(int(P[i]))
                if len(q_seq) > 0:
                    q_seq = torch.FloatTensor(q_seq).unsqueeze(0).cuda()
                    a_seq = torch.FloatTensor(a_seq).unsqueeze(0).cuda()
                    p_seq = torch.LongTensor(p_seq).unsqueeze(0).cuda()
                    y = 2 * torch.sign(torch.sum(a_seq[:, :, self.n_question:], dim=-1)) - 1
                    pred = self.model(q_seq, a_seq, p_seq).squeeze(-1)
                    pred = torch.round(pred)
                    print(L, file=t_data)
                    print(','.join([str(p) for p in P]), file=t_data)
                    print(','.join([str(q).replace(' ', '') for q in Q]), file=t_data)
                    print(','.join([str(a) for a in A]), file=t_data)
                    print(','.join([str(int(r)) for r in y[0].tolist()]), file=t_data)
                    print(','.join([str(int(r)) for r in pred[0].tolist()]), file=t_data)
                    pr, yr = torch.clip(pred, min=0), torch.clip(y, min=0)
                    acc = torch.mean(torch.tensor([(torch.sum(pr[k]*yr[k])+1e-5)/(torch.sum(yr[k])+1e-5)
                                                   for k in range(yr.shape[0])], dtype=torch.float))
                    acc1.append(acc.item())
                    pw, yw = torch.clip(-pred, min=0), torch.clip(-y, min=0)
                    acc = torch.mean(torch.tensor([(torch.sum(pw[k]*yw[k])+1e-5)/(torch.sum(yw[k])+1e-5)
                                                   for k in range(yw.shape[0])], dtype=torch.float))
                    acc2.append(acc.item())
        acc1 = torch.mean(torch.tensor(acc1, dtype=torch.float))
        acc2 = torch.mean(torch.tensor(acc2, dtype=torch.float))
        print(acc1.item(), ', ', acc2.item(), file=t_data)
        f_data.close(), t_data.close()

