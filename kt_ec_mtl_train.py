import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

idx = [0, 2, 3, 4]
name_list = ['array', 'hash-table', 'math', 'string', 'dynamic-programming', 'greedy', 'sorting', 'depth-first-search']


class Trainer(object):

    def __init__(self, ec_net, kt_net, epoch_num, lr_ec, lr_kt, bs_ec, bs_kt, l2_ec, l2_kt, max_grad_norm=-1, pre='kt'):
        self.ec_net = ec_net
        self.kt_net = kt_net
        self.epoch_num = epoch_num
        self.lr_ec, self.lr_kt = lr_ec, lr_kt
        self.bs_ec, self.bs_kt = bs_ec, bs_kt
        self.l2_ec, self.l2_kt = l2_ec, l2_kt
        self.max_grad_norm = max_grad_norm
        if self.ec_net is not None:
            self.opt_ec = optim.Adam(ec_net.parameters(), lr=lr_ec, weight_decay=self.l2_ec)
            self.loss_ec = F.cosine_similarity
        if self.kt_net is not None:
            self.opt_kt = optim.Adam(kt_net.parameters(), lr=lr_kt, weight_decay=self.l2_kt)
            self.loss_kt = nn.MSELoss(reduction='sum')
            self.n_question = self.kt_net.n_question
        pre = None if self.ec_net is None or self.kt_net is None else pre
        if pre == 'ec':
            self.kt_net.q_embed.weight = nn.Parameter(copy.copy(self.ec_net.k_net.weight.t())).cuda()
        elif pre == 'kt':
            self.ec_net.k_net.weight = nn.Parameter(copy.copy(self.kt_net.q_embed.weight.t())).cuda()

    def model_save(self, epoch, n_ec, n_kt, path):
        state = {'ec_net': self.ec_net.state_dict(),
                 'kt_net': self.kt_net.state_dict(),
                 'opt_ec': self.opt_ec.state_dict(),
                 'opt_kt': self.opt_kt.state_dict(),
                 'epoch_num': epoch,
                 'n_ec': n_ec, 'n_kt': n_kt,
                 'lr_ec': self.lr_ec, 'lr_kt': self.lr_kt,
                 'bs_ec': self.bs_ec, 'bs_kt': self.bs_kt,
                 'l2_ec': self.l2_ec, 'l2_kt': self.l2_kt,
                 'max_grad_norm': self.max_grad_norm,
                 }
        torch.save(state, path)

    def ec_func(self, batch_data):
        self.ec_net.train()
        loss_tra = torch.Tensor([0]).cuda()
        for i, batch in enumerate(batch_data):
            x, y = batch
            y = y.cuda()
            pred = self.ec_net(x).cuda()
            self.opt_ec.zero_grad()
            loss = 1 - self.loss_ec(pred, y).cuda()
            loss = torch.sum(loss, dim=0)
            loss_tra += loss.cpu().item()
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.ec_net.parameters(), max_norm=self.max_grad_norm)
            self.opt_ec.step()
        return loss_tra

    def kt_func(self, batch_data):
        self.kt_net.train()
        loss_tra = torch.Tensor([0]).cuda()
        for i, batch in enumerate(batch_data):
            q, qa, pid = batch
            q, qa, pid = q.cuda(), qa.cuda(), pid.cuda()
            y = 2 * torch.sign(torch.sum(qa[:, :, self.n_question:], dim=-1)) - 1
            pred = self.kt_net(q, qa, pid).squeeze(-1)
            self.opt_kt.zero_grad()
            loss = torch.sum(self.loss_kt(pred, y))
            loss_tra += loss.item()
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.kt_net.parameters(), max_norm=self.max_grad_norm)
            self.opt_kt.step()
        return loss_tra

    def ec_eval(self, batch_data):
        self.ec_net.eval()
        loss_tot, acc1, acc2, acc3 = 0, [], [], []
        hit1, hit2 = [], []
        for i, batch in enumerate(batch_data):
            x, y = batch
            y = y.cuda()
            pred = self.ec_net(x).cuda()
            loss = 1 - self.loss_ec(pred, y).cuda()
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

    def kt_eval(self, batch_data):
        self.kt_net.eval()
        loss_tot, acc1, acc2, acc3, acc4 = 0, [], [], [], []
        for i, batch in enumerate(batch_data):
            q, qa, pid = batch
            q, qa, pid = q.cuda(), qa.cuda(), pid.cuda()
            y = 2 * torch.sign(torch.sum(qa[:, :, self.n_question:], dim=-1)) - 1
            pred = self.kt_net(q, qa, pid).squeeze(-1)
            loss = self.loss_kt(pred, y).cuda()
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
        acc1 = sum(acc1) / len(acc1)
        acc2 = sum(acc2) / len(acc2)
        acc3 = sum(acc3) / len(acc3)
        acc4 = sum(acc4) / len(acc4)
        return loss_tot, acc1, acc2, acc3, acc4

    def ec_valid(self, glove, path):
        data = list()
        for item in glove.data:
            s = item['content'].upper()
            temp, words = dict(), s.split()
            temp['content'] = item['content']
            temp['title'] = item['questionTitle']
            item = [glove.word_idx[word] for word in words]
            item = torch.LongTensor(item).unsqueeze(0)
            pred = torch.round(self.ec_net(item))
            temp['class'] = pred.detach().cpu().numpy().tolist()[0]
            data.append(temp)
        with open(path, 'w') as json_file:
            json_file.write(json.dumps(data, indent=4))

    def kt_valid(self, valid_from, valid_to):
        self.kt_net.eval()
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
                    pred = self.kt_net(q_seq, a_seq, p_seq).squeeze(-1)
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
