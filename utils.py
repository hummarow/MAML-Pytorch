import typing
import sys
import os
import argparse
import numpy as np
import scipy.stats
import torch


class EarlyStopping:
    # https://quokkas.tistory.com/37
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt", monitor="loss"):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
            monitor (str): performance measure. "loss" or "acc"
                            Default: loss
        """
        # TODO: best_score and baseline_measure seems to have similar perposes
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.monitor = monitor
        if monitor == "loss":
            self.baseline_measure = np.Inf
        elif monitor == "acc":
            self.baseline_measure = -np.Inf
        else:
            sys.exit("Monitor of early stopping should be 'loss' or 'acc', but is {}.".format(mode))
        self.delta = delta
        self.path = path

    def __call__(self, measure, model):

        if self.monitor == "loss":
            score = -measure
        elif self.monitor == "acc":
            score = measure

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(measure, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(measure, model)
            self.counter = 0

    def save_checkpoint(self, measure, model):
        """validation loss가 감소하면 모델을 저장한다."""
        if self.verbose:
            print(
                f"Performance decreased ({self.baseline_measure:.6f} --> {measure:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.baseline_measure = measure


def get_model_dir(args):
    path = "logs"
    path = os.path.join(path, args.logdir, args.date)

    dir_name = ""

    if args.aug:
        dir_name += "aug_"
    if args.flip:
        dir_name += "flip_"
    if args.reg > 0:
        dir_name += str(args.reg)
        dir_name += "_"
    if args.prox_lam > 0:
        dir_name += "imaml_"
    if args.chaser_lam > 0:
        dir_name += "bmaml_"
    if dir_name == "":
        dir_name = "org"
    else:
        dir_name = dir_name[:-1]

    path = os.path.join(path, dir_name, args.TIME)

    return path


def get_log_path(args):
    model_path = get_model_dir(args)
    return os.path.join(model_path, "_logs")


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


def parse_argument(kwargs):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--epoch", type=int, help="epoch number", default=6)
    argparser.add_argument("--n_way", type=int, help="n way", default=5)
    argparser.add_argument("--k_spt", type=int, help="k shot for support set", default=1)
    argparser.add_argument("--k_qry", type=int, help="k shot for query set", default=15)
    argparser.add_argument("--imgsz", type=int, help="imgsz", default=84)
    argparser.add_argument("--imgc", type=int, help="imgc", default=3)
    argparser.add_argument(
        "--task_num",
        type=int,
        help="meta batch size, namely task num",
        default=4,
    )
    argparser.add_argument(
        "--meta_lr",
        type=float,
        help="meta-level outer learning rate",
        default=1e-3,
    )
    argparser.add_argument(
        "--update_lr",
        type=float,
        help="task-level inner update learning rate",
        default=0.01,
    )
    argparser.add_argument(
        "--update_step",
        type=int,
        help="task-level inner update steps",
        default=5,
    )
    argparser.add_argument(
        "--episode",
        type=int,
        help="Number of training episodes per epoch",
        default=10000,
    )
    argparser.add_argument(
        "--repeat",
        type=int,
        help="number of validations",
        default=5,
    )
    argparser.add_argument(
        "--update_step_test",
        type=int,
        help="update steps for finetunning",
        default=10,
    )
    argparser.add_argument("--logdir", type=str, help="log directory for tensorboard", default="")
    argparser.add_argument(
        "--seed",
        type=int,
        default=-1,
    )
    # Augmentation
    argparser.add_argument(
        "--traditional_augmentation",
        "--trad_aug",
        action="store_true",
        help="train with augment data in traditional way",
        default=False,
    )
    argparser.add_argument(
        "--aug",
        action="store_true",
        help="add augmentation and measure weight distance between original data and augmented data",
        default=False,
    )
    argparser.add_argument(
        "--qry_aug",
        action="store_true",
        help="use augmented query set when meta-updating parameters",
        default=False,
    )
    argparser.add_argument(
        "--flip",
        action="store_true",
        help="add random horizontal flip augmentation",
        default=False,
    )
    # Regularizer
    argparser.add_argument("--reg", type=float, help="coefficient for regularizer", default=0.01)
    argparser.add_argument(
        "--rm_augloss",
        action="store_true",
        default=False,
    )
    # proximal regularizer for imaml
    argparser.add_argument(
        "--prox_lam",
        type=float,
        help="Lambda for imaml proximal regularizer",
        default=0,
    )
    argparser.add_argument(
        "--prox_task",
        type=int,
        help="Apply proximal regularizer at task 0 (original), task 1 (augmented), or 2 (both)",
        default=-1,
    )
    # Chaser loss for bmaml
    argparser.add_argument(
        "--chaser_lam",
        type=float,
        help="Lambda for bmaml chaser loss",
        default=0,
    )
    argparser.add_argument(
        "--chaser_task",
        type=int,
        help="Apply proximal regularizer at task 0 (original), task 1 (augmented), or 2 (both)",
        default=-1,
    )
    argparser.add_argument(
        "--chaser_lr",
        type=float,
        default=1e-3,
    )
    argparser.add_argument(
        "--bmaml",
        action="store_true",
        help="Bmaml loss only",
        default=False,
    )

    args = argparse.Namespace()
    for i, k in kwargs.items():
        vars(args)[i] = k
    args = argparser.parse_args(namespace=args)
    return args


def print_args(args):
    msg = ""
    if args.traditional_augmentation:
        msg += "Traditional Augmentation\n"
    if args.aug:
        msg += "Augmentation\n"
        if args.reg > 0:
            msg += "reg: {}\n".format(args.reg)
    if args.flip:
        msg += "Flip\n"
    if args.rm_augloss:
        msg += "Original Loss only\n"
    if args.prox_lam > 0:
        msg += "iMAML {}\n".format(args.prox_lam)
    if args.prox_task != -1:
        msg += "Prox Reg applied to "
        if args.prox_task == 0:
            msg += "original dataset only\n"
        elif args.prox_task == 1:
            msg += "augmented dataset only\n"
        else:
            msg += "both of the datasets\n"
    if args.bmaml:
        msg += "bMAML\n"
    if args.chaser_lam > 0:
        msg += "Chaser {}\n".format(args.chaser_lam)
    if args.chaser_task != -1:
        msg += "Chaser Reg applied to "
        if args.chaser_task == 0:
            msg += "original dataset only\n"
        elif args.chaser_task == 1:
            msg += "augmented dataset only\n"
        else:
            msg += "both of the datasets\n"
    if msg == "":
        msg = "Original MAML\n"
    msg += "{} Way {} Shot".format(args.n_way, args.k_spt)
    print(msg)
    return msg


if __name__ == "__main__":
    import ast

    a = input()
    a = ast.literal_eval(a)
    print(mean_confidence_interval(a))
