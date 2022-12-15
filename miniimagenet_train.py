import torch, os
from torch import nn
import numpy as np
import random, sys, pickle
import tensorboard
import matplotlib.pyplot as plt
import os
import typing
import pprint
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from MiniImagenet import MiniImagenet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from collections import defaultdict
from tqdm import tqdm

import utils
from meta import Meta

# Train Configuration
VALIDATION_EPISODES = 100
TEST_EPISODES = 600
VALIDATION_CYCLE = 500

device = torch.device("cuda")

now = datetime.now()
TIME = now.strftime("%y%m%d_%H%M%S")
date = TIME[:6]
TIME = TIME[7:]
seed_list = [1004, 8282, 486, 404, 9797, 1010235, 2848, 10288, 8255, 5825, 101101, 58486, 660660]


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


def train(val_iter, args, model_config, dataloaders, writer):
    t = 0
    best_val_acc = -float("inf")
    mean_val_accs = []
    num_test = 5
    val_accs = [0] * VALIDATION_EPISODES
    test_accs = [0] * TEST_EPISODES * num_test
    best_val_acc = 0
    best_step = 0
    checkpoint = None

    maml = Meta(args, model_config)
    if torch.cuda.device_count() > 1:
        maml = torch.nn.DataParallel(maml)
    maml.to(device)
    print(val_iter)

    seed = seed_list[val_iter % len(seed_list)]
    if val_iter >= len(seed_list):
        seed += 1

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    mini, mini_val, mini_test = dataloaders

    if val_iter > 0:
        mini.create_batch()
        mini_val.create_batch()

    # Start training
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
            t += 1  # Timestep Update
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
                torch.save(checkpoint, args.MODEL_PATH)
                best_step = t

    # maml = Meta(args, config).to(device)
    for i in range(num_test):
        maml.load_state_dict(checkpoint)
        mini_test.create_batch()
        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
        for j, (x_spt, y_spt, x_qry, y_qry) in enumerate(db_test):
            x_spt, y_spt, x_qry, y_qry = (
                x_spt.squeeze(0).to(device),
                y_spt.squeeze(0).to(device),
                x_qry.squeeze(0).to(device),
                y_qry.squeeze(0).to(device),
            )

            acc = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            test_accs[i * len(db_test) + j] = acc
    del maml

    # mean_test_acc = np.array(test_accs).mean(axis=0).astype(np.float16)
    mean_test_acc, ci = utils.mean_confidence_interval(test_accs)
    print("Step: {}".format(best_step))
    print("Test Accuracy: {:.2f}% +- {:.2f}%".format(mean_test_acc * 100, ci * 100))
    with open(args.INFO_PATH, "a") as f:
        f.write("Step: {}".format(best_step))
        f.write("Test Accuracy: {:.2f}% +- {:.2f}%".format(mean_test_acc * 100, ci * 100))
    # print("Test: {}%".format(mean_test_acc * 100))
    # print(mean_test_acc)
    return mean_test_acc


def main(**kwargs):
    """
    return list of validation loss
    """
    args = utils.parse_argument(kwargs)
    TRAIN_EPISODES = args.episode * args.task_num
    validation_num = args.repeat
    best_val_accs = [0] * validation_num
    args.need_aug = args.aug | args.traditional_augmentation
    args.TIME = TIME
    args.date = date

    MODEL_DIR = "./logs/models/" + (
        "trad_aug" if args.traditional_augmentation else "aug" if args.aug else "org"
    )
    MODEL_DIR = utils.get_model_dir(args)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model_name = (
        TIME
        # + "_"
        # + ("tradAug_" if args.traditional_augmentation else "")
        # + ("augmented_" if args.aug else "")
        # + ((str(args.reg) + "_" if args.reg > 0 else "") if args.aug else "")
        + ".pt"
    )
    MODEL_PATH = os.path.join(MODEL_DIR, model_name)
    info_name = TIME + ".info"
    INFO_PATH = os.path.join(MODEL_DIR, info_name)
    args.MODEL_PATH = MODEL_PATH
    args.INFO_PATH = INFO_PATH

    print(TIME)
    model_config = [
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

    log_path = utils.get_log_path(args)
    writer = SummaryWriter(log_path)
    writer.add_custom_scalars(layout)

    best_test_accs = [0] * validation_num

    mini = MiniImagenet(imagenet_path, mode="train", num_episodes=TRAIN_EPISODES, args=args)
    mini_val = MiniImagenet(imagenet_path, mode="val", num_episodes=VALIDATION_EPISODES, args=args)
    mini_test = MiniImagenet(imagenet_path, mode="test", num_episodes=TEST_EPISODES, args=args)

    with open(INFO_PATH, "w") as f:
        pprint.pprint(vars(args), f)
        f.write("\n------------------------------\n\n")
        f.write(utils.print_args(args))
        f.write("\n------------------------------\n\n")

    for val_iter in range(validation_num):
        mean_test_acc = train(val_iter, args, model_config, [mini, mini_val, mini_test], writer)
        best_test_accs[val_iter] = mean_test_acc
    utils.print_args(args)
    mean, ci = utils.mean_confidence_interval(best_test_accs)
    print("Test Accuracies: ")
    print(best_test_accs)
    print("Test Accuracy: {:.2f}% +- {:.2f}%".format(mean * 100, ci * 100))
    with open(INFO_PATH, "a") as f:
        f.write("Test Accuracies: {}".format(best_test_accs))
        f.write("Test Accuracy: {:.2f}% +- {:.2f}%".format(mean * 100, ci * 100))

    return mean, ci


if __name__ == "__main__":
    main()
