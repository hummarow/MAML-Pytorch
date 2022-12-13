import typing
import os
import argparse
import numpy as np
import scipy.stats


def get_path(logdir: str, aug=False, reg=0.0):
    if logdir != "":
        logdir = "_" + logdir

    if aug:
        _aug_log_path_name = "Reg" + str(reg) + logdir
        log_path = os.path.join("logs", "aug", _aug_log_path_name)
    else:
        _log_path_name = "org" + logdir
        log_path = os.path.join("logs", _log_path_name)
    return log_path


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
        default=10,
    )
    argparser.add_argument(
        "--update_step_test",
        type=int,
        help="update steps for finetunning",
        default=10,
    )

    argparser.add_argument("--reg", type=float, help="coefficient for regularizer", default=1.0)
    argparser.add_argument("--logdir", type=str, help="log directory for tensorboard", default="")
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
        "--traditional_augmentation",
        "--trad_aug",
        action="store_true",
        help="...",
        default=False,
    )
    argparser.add_argument(
        "--flip",
        action="store_true",
        help="add random horizontal flip augmentation",
        default=False,
    )
    argparser.add_argument(
        "--rm_augloss",
        action="store_true",
        default=False,
    )
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
        "--seed",
        type=int,
        default=-1,
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
        msg += "iMAML\n"
    if args.prox_task != -1:
        msg += "Prox Reg applied to "
        if args.prox_task == 0:
            msg += "original dataset only\n"
        elif args.prox_task == 1:
            msg += "augmented dataset only\n"
        else:
            msg += "both of the datasets\n"
    if msg == "":
        print("Original MAML")
    else:
        print(msg)
    print("{} Way {} Shot".format(args.n_way, args.k_spt))


if __name__ == "__main__":
    import ast

    a = input()
    a = ast.literal_eval(a)
    print(mean_confidence_interval(a))
