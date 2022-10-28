from mimetypes import init
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

torch.autograd.set_detect_anomaly(True)

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
#        self.k_spt = args.k_spt
#        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.reg = args.reg
        self.aug = args.aug
        self.qry_aug = args.qry_aug
        self.original_augmentation = args.original_augmentation

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        log_path = configs.get_path(args.log_dir, args.aug, args.reg)
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

    def gradient_descent(param, grad):
        pass

    def forward(self, x_spt, y_spt, x_qry, y_qry, t, spt_aug=None, qry_aug=None):
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
           and the distances between original dataset and augmented dataset are used as regularizer.
        2. aug flag is not set.
           The learning procedure is done without augmentation.
        3. original_augmentation flag is set, and augmented support set and query set are NOT given.
           The learning procedure is done with augmented dataset.
        """
        assert (self.aug and torch.is_tensor(spt_aug) and torch.is_tensor(qry_aug)) or (not self.aug) or (self.original_augmentation and not (torch.is_tensor(spt_aug) or torch.is_tensor(qry_aug)))

        x_qry_orig, y_qry_orig = x_qry, y_qry
        if self.qry_aug:
            x_qry = torch.cat([x_qry, qry_aug], dim=1)
            y_qry = torch.cat([y_qry, y_qry], dim=1)

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)
        # TODO: Remove losses of all steps, and instead use final losses per tasks
        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        sum_loss = 0

        losses_per_task = [0] * task_num
        losses_per_task_aug = [0] * task_num

        phi = self.net.parameters()

        finetuned_param = [phi] * task_num
        finetuned_param_aug = [phi] * task_num

        sgd = optim.SGD(self.net.parameters(), self.update_lr) 

        if self.aug and not self.original_augmentation:
            losses_q_aug = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

        # Save initial parameters
        init_param = deepcopy(self.net.state_dict())

        for i in range(task_num):
            # # Load initial parameters
            # self.net.load_state_dict(init_param)
            # # model with augmented data
            # if self.aug and not self.original_augmentation:
            #     for k in range(self.update_step):
            #         self.net.train()    # Put train() inside of for-loop for safety
            #         sgd.zero_grad()
            #         # Augmented data
            #         # Update parameter with support data
            #         logits = self.net(spt_aug[i], vars=None, bn_training=True)
            #         loss = F.cross_entropy(logits, y_spt[i])
            #         loss.backward()
            #         sgd.step()
            #         finetuned_param_aug.append(self.net.parameters())
            #     # Calculate loss with query data
            #     # Only calculate on the final step
            #     self.net.eval()
            #     logits_q = self.net(qry_aug[i], self.net.parameters(), bn_training=True)
            #     # loss_q will be overwritten and just keep the loss_q on last update step.
            #     loss_q = F.cross_entropy(logits_q, y_qry_orig[i])
            #     # losses_q_aug[k + 1] += loss_q
            #     losses_per_task_aug[i] = loss_q

            # self.net.load_state_dict(init_param)
            # model with original data
            for k in range(self.update_step):
                self.net.train()
                # Augmented data
                # Update parameter with support data
                x = torch.tensor(x_spt[i], requires_grad=True)
                logits = self.net(x, vars=None, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # loss.backward(retain_graph=True)
                # grad = x.grad
                grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
                print(grad.sum())
                input()
                # loss.backward()
                finetuned_param.append(self.net.parameters())
            # Calculate loss with query data
            # Only calculate on the final step
            qry = torch.tensor(x_qry[i], requires_grad=True)
            logits_q = self.net(qry, self.net.parameters(), bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry_orig[i])
            sum_loss += loss_q
            losses_per_task[i] = loss_q
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry[i]).sum().item()
            corrects[k+1] = corrects[k+1] + correct

# Total Loss = Loss + Loss(aug) + Regularizer
        # avg_loss = torch.mean(losses_per_task)
        # avg_loss_aug = torch.mean(losses_per_task_aug)
        loss_q = torch.div(sum_loss, task_num)
        # loss_q = sum_loss   # TODO: Add loss_aug and regularizer
        # loss_q += loss_q_aug * 0.001

        self.writer.add_scalar("loss+Distance", loss_q, t)

        # self.net.load_state_dict(init_param)
        # optimize theta parameters
        # self.meta_optim.zero_grad()
        loss_q.backward()
        # self.meta_optim.step()
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

