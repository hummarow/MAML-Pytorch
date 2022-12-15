import torch
import numpy as np
import os
import utils
import typing
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
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.reg = args.reg
        self.aug = args.aug
        self.qry_aug = args.qry_aug
        self.traditional_augmentation = args.traditional_augmentation
        self.need_aug = args.need_aug
        self.rm_augloss = args.rm_augloss
        self.prox_lam = args.prox_lam
        self.prox_task = args.prox_task
        self.chaser_lam = args.chaser_lam
        self.chaser_task = args.chaser_task
        self.chaser_lr = args.chaser_lr
        self.bmaml = args.bmaml
        if self.bmaml:
            self.chaser_lam = 1.0

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        log_path = utils.get_log_path(args)
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
        total_norm = total_norm ** (1.0 / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def weight_clustering(self, weight_1, weight_2):
        # Flatten weight_1
        weight_flat = []
        for i in range(len(weight_1)):
            w = None
            for fw in weight_1[i]:
                if not torch.is_tensor(w):
                    w = torch.flatten(fw)
                else:
                    w = torch.cat([w, torch.flatten(fw)], dim=0)
            weight_flat.append(w)
        weight_flat = torch.stack(weight_flat, axis=1)

        # Flatten weight_2
        weight_flat_2 = []
        for i in range(len(weight_2)):
            w = None
            for fw in weight_2[i]:
                if not torch.is_tensor(w):
                    w = torch.flatten(fw)
                else:
                    w = torch.cat([w, torch.flatten(fw)], dim=0)
            weight_flat_2.append(w)
        weight_flat_2 = torch.stack(weight_flat_2, axis=1)

        st = torch.stack([weight_flat, weight_flat_2])
        diff = torch.norm(st, p="fro", dim=0)
        diff = torch.norm(diff, p="fro", dim=0)

        norm = torch.mean(diff)
        return norm

    def proximal_reg(self, weight_1, weight_2, lam):
        # return self.weight_clustering(weight_1, weight_2)
        reg = 0
        for i in range(len(weight_1)):
            delta = weight_1[i].view(-1) - weight_2[i].view(-1)
            reg += 0.5 * lam * torch.sum(delta**2)
        return reg

    def chaser_loss(self, weight_1, weight_2, lam):
        return self.proximal_reg(weight_1, weight_2, lam)

    def forward(self, x_spt, y_spt, x_qry, y_qry, t, spt_aug=None, qry_aug=None) -> float:
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        """
        3 conditions of forward function
        1. aug flag is set, and augmented support set and query set are given.
           The learning procedure is done with original dataset,
           and the distances between parameters from original dataset and augmented dataset are used as regularizer.
        2. traditional_augmentation flag is set, and augmented support set and query set are given.
           The learning procedure is done with augmented dataset,
           and the distances are calculated.
        2. aug flag is not set.
           The learning procedure is done without augmentation.
        """
        assert (
            self.need_aug  # Needs augmentation.
            and torch.is_tensor(spt_aug)  # Augmented data are given.
            and torch.is_tensor(qry_aug)
        ) or (not self.aug)

        x_qry_orig, y_qry_orig = x_qry, y_qry
        if self.qry_aug:
            x_qry = torch.cat([x_qry, qry_aug], dim=1)
            y_qry = torch.cat([y_qry, y_qry], dim=1)

        task_num = x_spt.size()[0]
        querysz = x_qry.size(1)

        loss_q = 0
        loss_q_aug = 0
        num_corrects = 0

        phi = self.net.parameters()

        finetuned_parameter = [phi] * task_num
        finetuned_parameter_aug = [phi] * task_num

        for i in range(task_num):
            # model with original data
            w_0 = finetuned_parameter[i]
            for k in range(self.update_step):
                # Update parameter with support data
                logits = self.net(x_spt[i], finetuned_parameter[i], bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(
                    loss,
                    finetuned_parameter[i],
                    create_graph=True,
                    retain_graph=True,
                )
                finetuned_parameter[i] = list(
                    map(
                        lambda p: p[1] - self.update_lr * p[0],
                        zip(grad, finetuned_parameter[i]),
                    )
                )

            # Calculate loss with query data and updated parameter
            logits_q = self.net(x_qry[i], finetuned_parameter[i], bn_training=True)
            _task_loss = F.cross_entropy(logits_q, y_qry[i])
            if not self.bmaml:
                loss_q += _task_loss  # Sum of losses of all tasks

            # iMAML proximal regularizer
            if (self.prox_task == 0 or self.prox_task == 2) and self.prox_lam > 0:
                prox_reg = self.proximal_reg(w_0, finetuned_parameter[i], self.prox_lam)
                loss_q += prox_reg

            # bMAML chaser loss
            if (self.chaser_task == 0 or self.chaser_task == 2) and self.chaser_lam > 0:
                chaser = finetuned_parameter[i]
                grad = torch.autograd.grad(
                    _task_loss,
                    finetuned_parameter[i],
                    create_graph=True,
                    retain_graph=True,
                )
                leader = list(
                    map(
                        lambda p: p[1] - self.chaser_lr * p[0],
                        zip(grad, finetuned_parameter[i]),
                    )
                )
                chaser_reg = self.chaser_loss(chaser, leader, self.chaser_lam)
                loss_q += chaser_reg

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                num_corrects += torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy

            # model with augmented data
            if self.need_aug:
                w_0 = finetuned_parameter_aug[i]
                for k in range(self.update_step):
                    # Update parameter with support data
                    logits = self.net(
                        spt_aug[i],
                        finetuned_parameter_aug[i],
                        bn_training=True,
                    )
                    loss = F.cross_entropy(logits, y_spt[i])
                    grad = torch.autograd.grad(
                        loss,
                        finetuned_parameter_aug[i],
                        create_graph=True,
                        retain_graph=True,
                    )
                    finetuned_parameter_aug[i] = list(
                        map(
                            lambda p: p[1] - self.update_lr * p[0],
                            zip(grad, finetuned_parameter_aug[i]),
                        )
                    )
                # Calculate loss with query data
                # with no grad
                logits_q = self.net(qry_aug[i], finetuned_parameter_aug[i], bn_training=True)
                loss_q_aug += F.cross_entropy(logits_q, y_qry_orig[i])

                # iMAML proximal regularizer
                if (self.prox_task == 1 or self.prox_task == 2) and self.prox_lam > 0:
                    prox_reg = self.proximal_reg(w_0, finetuned_parameter_aug[i], self.prox_lam)
                    loss_q_aug += prox_reg

                # bMAML chaser loss
                if (self.chaser_task == 1 or self.chaser_task == 2) and self.chaser_lam > 0:
                    chaser = finetuned_parameter_aug[i]
                    grad = torch.autograd.grad(
                        _task_loss,
                        finetuned_parameter_aug[i],
                        create_graph=True,
                        retain_graph=True,
                    )
                    leader = list(
                        map(
                            lambda p: p[1] - self.chaser_lr * p[0],
                            zip(grad, finetuned_parameter_aug[i]),
                        )
                    )
                    chaser_reg = self.chaser_loss(chaser, leader, self.chaser_lam)
                    loss_q += chaser_reg

        loss_q = torch.div(loss_q, task_num)
        self.writer.add_scalar("loss/original", loss_q, t)
        loss_q_aug = torch.div(loss_q_aug, task_num)
        self.writer.add_scalar("loss/augmented", loss_q_aug, t)
        self.writer.add_scalar("loss_difference", abs(loss_q - loss_q_aug), t)
        # Total Loss = Loss + Loss(aug) + Regularizer
        if not self.bmaml and not self.rm_augloss:
            loss_q += loss_q_aug

        # Overwrite loss with the one calculated from augmented dataset if trad. augmentation is set.
        if self.traditional_augmentation:
            loss_q = loss_q_aug

        # Weight Clustering
        if self.need_aug:
            norm = self.weight_clustering(finetuned_parameter, finetuned_parameter_aug)
        else:
            norm = 0

        self.writer.add_scalar("Distance", norm, t)
        loss_q += self.reg * norm  # norm is calculated even when self.reg == 0, for the record
        self.writer.add_scalar("loss+Distance", loss_q, t)
        # optimize theta parameters
        self.meta_optim.zero_grad()
        if self.bmaml:
            loss_q.requires_grad = True
        loss_q.backward()

        self.meta_optim.step()
        acc = num_corrects / (querysz * task_num)

        return acc

    def finetunning(self, x_spt, y_spt, x_qry, y_qry) -> float:
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        # assert len(x_spt.shape) == 4
        if len(x_spt.shape) == 4:
            x_spt = x_spt.unsqueeze(0)
            y_spt = y_spt.unsqueeze(0)
            x_qry = x_qry.unsqueeze(0)
            y_qry = y_qry.unsqueeze(0)

        task_num = x_spt.size(0)
        querysz = x_qry.size(1)
        num_corrects = 0

        net = deepcopy(self.net)

        for i in range(task_num):
            # 1. run the i-th task and compute loss for k=0
            logits = net(x_spt[i])
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, net.parameters())
            finetuned_parameter = list(
                map(
                    lambda p: p[1] - self.update_lr * p[0],
                    zip(grad, net.parameters()),
                )
            )

            for k in range(self.update_step_test):
                logits = net(x_spt[i], finetuned_parameter, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                grad = torch.autograd.grad(loss, finetuned_parameter)
                finetuned_parameter = list(
                    map(
                        lambda p: p[1] - self.update_lr * p[0],
                        zip(grad, finetuned_parameter),
                    )
                )

                logits_q = net(x_qry[i], finetuned_parameter, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                num_corrects += correct
        del net

        acc = num_corrects / (querysz * task_num)
        return acc


class ResNet(Meta):
    def __init__(self, args, config):
        super(ResNet, self).__init__(args, config)
        self.net = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)


def main():
    pass


if __name__ == "__main__":
    main()
