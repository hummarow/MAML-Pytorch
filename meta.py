import torch
import numpy as np
import os
import configs
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import linalg as LA
from learner import Learner
from copy import deepcopy


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.reg = args.reg
        self.ord = args.ord
        self.aug = args.aug
        self.qry_aug = args.qry_aug

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        log_path = configs.get_path(args.reg, args.ord, args.log_dir, args.aug)
        self.writer = SummaryWriter(log_path)


    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry, t, spt_aug=None, qry_aug=None):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        assert (self.aug and torch.is_tensor(spt_aug) and torch.is_tensor(qry_aug)) or (not self.aug)
        x_qry_orig, y_qry_orig = x_qry, y_qry
        if self.qry_aug:
            x_qry = torch.cat([x_qry, qry_aug], dim=1)
            y_qry = torch.cat([y_qry, y_qry], dim=1)
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        phi = self.net.parameters()

        fast_weights = [phi] * task_num
#        fast_weights = torch.empty(len(self.net.parameters()), task_num)
        if self.aug:
            fast_weights_aug = [phi] * task_num
            losses_q_aug = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        for i in range(task_num):
            if self.aug:
                # Augmented data
                logits = self.net(spt_aug[i], vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights_aug[i] = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

                with torch.no_grad():
                    logits_q = self.net(qry_aug[i], self.net.parameters(), bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry_orig[i])
                    losses_q_aug[0] += loss_q

                # this is the loss and accuracy after the first update
#                with torch.no_grad():
                    # [setsz, nway]
                    logits_q = self.net(qry_aug[i], fast_weights_aug[i], bn_training=True)
                    loss_q = F.cross_entropy(logits_q, y_qry_orig[i])
                    losses_q_aug[1] += loss_q
                    # [setsz]

                for k in range(1, self.update_step):
                    # 1. run the i-th task and compute loss for k=1~K-1
                    logits = self.net(spt_aug[i], fast_weights_aug[i], bn_training=True)
                    loss = F.cross_entropy(logits, y_spt[i])
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(loss, fast_weights_aug[i], create_graph=True, retain_graph=True)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights_aug[i] = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights_aug[i])))

                    logits_q = self.net(qry_aug[i], fast_weights_aug[i], bn_training=True)
                    # loss_q will be overwritten and just keep the loss_q on last update step.
                    loss_q = F.cross_entropy(logits_q, y_qry_orig[i])
                    losses_q_aug[k + 1] += loss_q


            # 1. run the i-th task and compute loss for k=0
            # self.net = Learner()
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters(), retain_graph=True, create_graph=True)
            fast_weights[i] = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights[i], bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights[i], bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights[i], create_graph=True, retain_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights[i] = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights[i])))

                logits_q = self.net(x_qry[i], fast_weights[i], bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
############################################################################
# Reptile
##########################################################################
#            state_dict = self.net.state_dict()
#            key_list = list(state_dict.keys())
#            for key in state_dict.keys():
#                if 'bn' in key:
#                    key_list.remove(key)
#            for j, key in enumerate(key_list):
#                state_dict[key] = state_dict[key] - self.reg * (state_dict[key] - fast_weights[i][j])
#           
#            self.net.load_state_dict(state_dict)


        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

####################################################################################
# Weight Clustering
####################################################################################

        weight_flat = []
        for i in range(len(fast_weights)):
            w = None
            for fw in fast_weights[i]:
                if not torch.is_tensor(w):
                    w = torch.flatten(fw)
                else:
                    w = torch.cat([w, torch.flatten(fw)], dim=0)
            weight_flat.append(w)

        weight_flat = torch.stack(weight_flat, axis=1)
        if not self.aug:
            average_weight = torch.mean(weight_flat, dim=1, keepdim=True)

            # Reset origin
            weight_flat = weight_flat - average_weight
#            norm = torch.norm(weight_flat, p='fro')
#
            norm = LA.vector_norm(weight_flat, ord=self.ord)

        else:
            weight_flat_aug = []
            for i in range(len(fast_weights_aug)):
                w = None
                for fw in fast_weights_aug[i]:
                    if not torch.is_tensor(w):
                        w = torch.flatten(fw)
                    else:
                        w = torch.cat([w, torch.flatten(fw)], dim=0)
                weight_flat_aug.append(w)
            weight_flat_aug = torch.stack(weight_flat_aug, axis=1)
#            diff = weight_flat - weight_flat_aug
            st = torch.stack([weight_flat, weight_flat_aug])
            diff = torch.norm(st, p='fro', dim=0)
            diff = torch.norm(diff, p='fro', dim=0)

            norm = torch.mean(diff)

            # 다른 방식
#            average_weight = torch.mean(weight_flat, dim=1, keepdim=True)
#            average_weight_aug = torch.mean(weight_flat_aug, dim=1, keepdim=True)
#            diff = average_weight - average_weight_aug
#            norm = LA.vector_norm(diff, ord=self.ord)

        self.writer.add_scalar("Distance", norm, t)
        self.writer.add_scalar("loss", loss_q, t)

        loss_q += self.reg * norm

###############################################################################
# End Weight Clustering
####################################################################################

        self.writer.add_scalar("loss+Distance", loss_q, t)
# MAML
        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()

        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())

        self.meta_optim.step()

# Reptile
#        state_dict = self.net.state_dict()
#        key_list = list(state_dict.keys())
#        for key in state_dict.keys():
#            if 'bn' in key:
#                key_list.remove(key)
#        for fw in fast_weights:
#            for i, key in enumerate(key_list):
#                state_dict[key] = state_dict[key] - fw[i]
#        self.net.load_state_dict(state_dict)

        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs


def main():
    pass


if __name__ == '__main__':
    main()

