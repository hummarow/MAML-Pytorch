import torch, os
import numpy as np
import scipy.stats
import random, sys, pickle
import argparse
import tensorboard
import matplotlib.pyplot as plt
import os
import configs
import typing
import time
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from MiniImagenet import MiniImagenet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from collections import defaultdict
from tqdm import tqdm
from meta import Meta
from meta import ResNet


TRAIN_EPISODES = 10000
VALIDATION_EPISODES = 100
TEST_EPISODES = 600

PBAR_UPDATE_CYCLE = 1000
VALIDATION_CYCLE = 500

VALIDATION_STARTING_POINT = 10000

now = datetime.now()

imagenet_path = "/home/bjk/Datasets/mini-imagenet/"
TIME = now.strftime("%y%m%d_%H%M%S")

# Tensorboard custom scalar layout
layout = {
    "Accuracy": {
        "Accuracy": [
            "Multiline",
            ["Accuracy/Train", "Accuracy/Val", "Accuracy/AugTest"],
        ],
    },
    "loss": {
        "loss": [
            "Multiline",
            ["loss/original", "loss/augmented"],
        ]
    },
}


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
        "--seed",
        type=int,
        default=-1,
    )

    args = argparse.Namespace()
    for i, k in kwargs.items():
        vars(args)[i] = k
    args = argparser.parse_args(namespace=args)
    return args


def main(**kwargs):
    """
    return list of validation loss
    """
    args = parse_argument(kwargs)
    TRAIN_EPISODES = args.episode
    validation_num = args.repeat
    args.need_aug = args.aug | args.traditional_augmentation

    MODEL_DIR = "./logs/models/" + (
        "trad_aug" if args.traditional_augmentation else "aug" if args.aug else "org"
    )
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    model_name = (
        TIME
        + "_"
        # + ("tradAug_" if args.traditional_augmentation else "")
        # + ("augmented_" if args.aug else "")
        + ((str(args.reg) + "_" if args.reg > 0 else "") if args.aug else "")
        + "model.pt"
    )
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)

    # seed = int(time.time())
    if args.seed == -1:
        seed = random.choice([1004, 8282, 486])
    else:
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(args)
    print("Seed: {}".format(seed))

    config = [
        ("conv2d", [32, 3, 3, 3, 1, 0]),  # [ch_out, ch_in, kernel, kernel, stride, pad]
        ("relu", [True]),  # [inplace]
        ("bn", [32]),  # [ch_out]
        ("max_pool2d", [2, 2, 0]),  # [kernel, stride, padding]
        ("conv2d", [32, 32, 3, 3, 1, 0]),
        ("relu", [True]),
        ("bn", [32]),
        ("max_pool2d", [2, 2, 0]),
        ("conv2d", [32, 32, 3, 3, 1, 0]),
        ("relu", [True]),
        ("bn", [32]),
        ("max_pool2d", [2, 2, 0]),
        ("conv2d", [32, 32, 3, 3, 1, 0]),
        ("relu", [True]),
        ("bn", [32]),
        ("max_pool2d", [2, 1, 0]),
        ("flatten", []),
        ("linear", [args.n_way, 32 * 5 * 5]),
    ]

    resnet_config = [
        ("conv2d", [32, 3, 3, 3, 1, 0]),
        ("max_pool2d", [3, 2, 0]),
        ("res", []),
        ("res", []),
        ("res", []),
        ("res", []),
        ("avg_pool2d",),
        ("flatten", []),
        ("linear", [args.n_way, 32 * 5 * 5]),
    ]

    device = torch.device("cuda")

    # num_episodes here means total episode number
    mini = MiniImagenet(imagenet_path, mode="train", num_episodes=TRAIN_EPISODES, args=args)
    mini_val = MiniImagenet(imagenet_path, mode="val", num_episodes=VALIDATION_EPISODES, args=args)
    mini_test = MiniImagenet(imagenet_path, mode="test", num_episodes=TEST_EPISODES, args=args)

    log_path = configs.get_path(args.logdir, args.aug, args.reg)

    writer = SummaryWriter(log_path)
    writer.add_custom_scalars(layout)
    best_val_acc = -float("inf")
    mean_val_accs = []
    val_accs = [0] * VALIDATION_EPISODES
    test_accs = [0] * TEST_EPISODES
    best_val_accs = [0] * validation_num
    best_test_accs = [0] * validation_num
    best_step = 0
    checkpoint = None
    ##########################
    t = 0
    maml = Meta(args, config).to(device)
    for epoch in tqdm(range(args.epoch)):
        # seed = int(time.time())
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # np.random.seed(seed)
        mini.create_batch(TRAIN_EPISODES)
        mini_val.create_batch(VALIDATION_EPISODES)
        mini_test.create_batch(TEST_EPISODES)
        # fetch meta_num_episodes num of episode each time
        db = DataLoader(
            mini,
            args.task_num,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
        for step, data in enumerate(db):
            t += args.task_num  # Timestep Update
            assert (len(data) == 4 and not args.need_aug) or (
                len(data) == 6 and args.need_aug
            )  # if augmentation is necessary, data contains x_spt_aug, x_qry_aug additionally.
            if not args.need_aug:
                (x_spt, y_spt, x_qry, y_qry) = data
                x_spt_aug = None
                x_qry_aug = None
            else:
                (
                    x_spt,
                    y_spt,
                    x_qry,
                    y_qry,
                    x_spt_aug,
                    x_qry_aug,
                ) = data  # Length asserted (safe)
                x_spt_aug = x_spt_aug.to(device)
                x_qry_aug = x_qry_aug.to(device)

            x_spt, y_spt, x_qry, y_qry = (
                x_spt.to(device),
                y_spt.to(device),
                x_qry.to(device),
                y_qry.to(device),
            )
            accs = maml(
                x_spt,
                y_spt,
                x_qry,
                y_qry,
                t,
                spt_aug=x_spt_aug,
                qry_aug=x_qry_aug,
            )

            if step % 30 == 0:
                writer.add_scalar("Accuracy/Train", accs, t)

            # Evaluation with validation data
            if step % VALIDATION_CYCLE == 0:
                db_val = DataLoader(
                    mini_val,
                    1,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True,
                )

                for i, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_val):
                    x_spt, y_spt, x_qry, y_qry = (
                        x_spt.squeeze(0).to(device),
                        y_spt.squeeze(0).to(device),
                        x_qry.squeeze(0).to(device),
                        y_qry.squeeze(0).to(device),
                    )

                    acc = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                    val_accs[i] = acc

                # [b, update_step+1]
                mean_acc = np.array(val_accs).mean(axis=0).astype(np.float16)
                writer.add_scalar("Accuracy", mean_acc, t)
                writer.add_scalar("Accuracy/Val", mean_acc, t)
                mean_val_accs.append(mean_acc)

                # Save the best model of given validation step
                if mean_val_accs[-1] > best_val_acc:
                    best_val_acc = mean_val_accs[-1]
                    if checkpoint:
                        del checkpoint
                    checkpoint = deepcopy(maml.state_dict())
                    torch.save(checkpoint, MODEL_PATH)
                    best_step = t
    # Choose the best model

    # maml = Meta(args, config).to(device)
    # Get the best model and test
    maml.load_state_dict(checkpoint)
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
    for i, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
        x_spt, y_spt, x_qry, y_qry = (
            x_spt.squeeze(0).to(device),
            y_spt.squeeze(0).to(device),
            x_qry.squeeze(0).to(device),
            y_qry.squeeze(0).to(device),
        )
        # Save all accs
        acc = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
        test_accs[i] = acc
    del maml
    # Get mean and std from all accs
    mean, ci = mean_confidence_interval(test_accs)
    # print("Test Accuracies: ")
    # print(test_accs)
    print(
        "Shot: {}\nAug: {}\nReg: {}\nFlip: {}\nTradAug: {}".format(
            str(args.k_spt),
            str(args.aug),
            str(args.reg),
            str(args.flip),
            str(args.traditional_augmentation),
        )
    )
    print("Step: {}".format(best_step))
    print("Test Accuracy: {:.2f}% +- {:.2f}%".format(mean * 100, ci * 100))

    return mean, ci


if __name__ == "__main__":
    main()
