import torch, os
from torch import nn
import numpy as np
import scipy.stats
import random, sys, pickle
import argparse
import tensorboard
import matplotlib.pyplot as plt
import os
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
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import utils
from meta import Meta

# Train Configuration
TRAIN_EPISODES = 10000
VALIDATION_EPISODES = 100
TEST_EPISODES = 600
VALIDATION_CYCLE = 500

device = torch.device("cuda")

now = datetime.now()
TIME = now.strftime("%y%m%d_%H%M%S")
seed_list = [1004, 8282, 486, 404, 9797, 1010235, 2848, 10288, 8255, 5825]

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

imagenet_path = "./datasets/mini-imagenet/"
# Ray Configuration
ray_dir = "./raydata/"
ray_config = {
    "prox_lam": tune.uniform(0.005, 1.0),
    "reg": tune.sample_from(lambda _: 0.0005 * np.random.randint(0, 20)),
    "update_step": tune.choice([5, 10, 15]),
}


def train(val_iter: int, args, config, ray_config, model_path):
    t = 0
    best_val_acc = -float("inf")
    mean_val_accs = []
    num_test = 5
    val_accs = [0] * VALIDATION_EPISODES
    test_accs = [0] * TEST_EPISODES * num_test
    best_val_acc = 0
    best_step = 0
    checkpoint = None
    log_path = utils.get_path(args.logdir, args.aug, args.reg)
    writer = SummaryWriter(log_path)
    writer.add_custom_scalars(layout)
    MODEL_PATH = model_path

    maml = Meta(args, config, ray_config["prox_lam"], ray_config["reg"], ray_config["update_step"])
    if torch.cuda.device_count() > 1:
        maml = torch.nn.DataParallel(maml)
    maml.to(device)
    print(val_iter)
    seed = seed_list[val_iter]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if val_iter == 0:
        # num_episodes here means total episode number
        mini = MiniImagenet(imagenet_path, mode="train", num_episodes=args.episode, args=args)
        mini_val = MiniImagenet(
            imagenet_path, mode="val", num_episodes=VALIDATION_EPISODES, args=args
        )
        mini_test = MiniImagenet(imagenet_path, mode="test", num_episodes=TEST_EPISODES, args=args)

    else:
        mini.create_batch(TRAIN_EPISODES)
        mini_val.create_batch(VALIDATION_EPISODES)
        mini_test.create_batch(TEST_EPISODES)

    for epoch in tqdm(range(args.epoch)):
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
    for i in range(num_test):
        maml.load_state_dict(checkpoint)
        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
        for j, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
            x_spt, y_spt, x_qry, y_qry = (
                x_spt.squeeze(0).to(device),
                y_spt.squeeze(0).to(device),
                x_qry.squeeze(0).to(device),
                y_qry.squeeze(0).to(device),
            )

            acc = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            test_accs[len(db_test) * i + j] = acc
    del maml

    mean_test_acc = np.array(test_accs).mean(axis=0).astype(np.float16)
    print("Step: {}".format(best_step))
    print("Test: {}%".format(mean_test_acc * 100))
    print(mean_test_acc)
    return mean_test_acc


def main(**kwargs):
    """
    return list of validation loss
    """
    args = utils.parse_argument(kwargs)
    TRAIN_EPISODES = args.episode
    validation_num = args.repeat
    best_val_accs = [0] * validation_num
    args.need_aug = args.aug | args.traditional_augmentation
    MODEL_DIR = "./logs/models/" + (
        "trad_aug" if args.traditional_augmentation else "aug" if args.aug else "org"
    )
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model_name = (
        TIME
        + "_"
        + ("tradAug_" if args.traditional_augmentation else "")
        + ("augmented_" if args.aug else "")
        + ((str(args.reg) + "_" if args.reg > 0 else "") if args.aug else "")
        + "model.pt"
    )
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)

    utils.print_args(args)

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

    best_test_accs = [0] * validation_num

    for val_iter in range(validation_num):
        mean_test_acc = train(val_iter, args, config, ray_config, MODEL_PATH)
        best_test_accs[val_iter] = mean_test_acc
    utils.print_args(args)
    mean, ci = utils.mean_confidence_interval(best_test_accs)
    print("Test Accuracies: ")
    print(best_test_accs)
    print("Test Accuracy: {:.2f}% +- {:.2f}%".format(mean * 100, ci * 100))

    return best_test_accs


if __name__ == "__main__":
    main()
