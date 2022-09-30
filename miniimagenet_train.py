import torch, os
import numpy as np
import scipy.stats
import random, sys, pickle
import argparse
import tensorboard
import matplotlib.pyplot as plt
import os
import configs

from MiniImagenet import MiniImagenet, MiniImagenet_aug
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from collections import defaultdict
from meta import Meta


imagenet_path = '/home/bjk/Datasets/mini-imagenet/'

# Tensorboard custom scalar layout
layout = {
    "Accuracy": {
        "Accuracy": ["Multiline", ["Accuracy/Train", "Accuracy/Test", "Accuracy/AugTest"]],
    }
}


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * 5 * 5])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    if args.aug:
        mini_aug = MiniImagenet_aug(imagenet_path, mode='train',
                                n_way=args.n_way, k_shot=args.k_spt,
                                k_query=args.k_qry,
                                batchsz=10000, resize=args.imgsz, get_original=True)
    else:
        # batchsz here means total episode number
        mini = MiniImagenet(imagenet_path, mode='train',
                            n_way=args.n_way, k_shot=args.k_spt,
                            k_query=args.k_qry,
                            batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet(imagenet_path, mode='test',
                             n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    
    mini_test_aug = MiniImagenet_aug(imagenet_path, mode='test',
                                 n_way=args.n_way, k_shot=args.k_spt,
                                 k_query=args.k_qry,
                                 batchsz=100, resize=args.imgsz, get_original=False)

    log_path = configs.get_path(args.reg, args.ord, args.log_dir, args.aug)

    writer = SummaryWriter(log_path)
    writer.add_custom_scalars(layout)

    for epoch in range(args.epoch//10000):
        if args.aug:
            # fetch meta_batchsz num of episode each time
            db = DataLoader(mini_aug, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
            for step, (spt_aug, y_spt, qry_aug, y_qry, x_spt, x_qry) in enumerate(db):
                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
                spt_aug, qry_aug = spt_aug.to(device), qry_aug.to(device)
                accs = maml(x_spt, y_spt, x_qry, y_qry, step+len(db)*epoch, spt_aug=spt_aug, qry_aug=qry_aug)
                if step % 30 == 0:
                    print('step:', step, '\ttraining acc:', accs)
                    writer.add_scalar('Accuracy/Train',
                                      accs[-1],
                                      step + epoch*len(db))

                if step % 500 == 0:  # evaluation
                    if args.test in ['original', 'both']:
                        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                        accs_all_test = []

                        for x_spt, y_spt, x_qry, y_qry in db_test:
                            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                            accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                            accs_all_test.append(accs)

                        # [b, update_step+1]
                        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                        print('Test acc:', accs)
                        writer.add_scalar('Accuracy',
                                          accs[-1],
                                          step + epoch*len(db))
                        writer.add_scalar('Accuracy/Test',
                                          accs[-1],
                                          step + epoch*len(db))
                    if args.test in ['aug', 'both']:
                        db_test = DataLoader(mini_test_aug, 1, shuffle=True, num_workers=1, pin_memory=True)
                        accs_all_test = []

                        for x_spt, y_spt, x_qry, y_qry in db_test:
                            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                            accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                            accs_all_test.append(accs)

                        # [b, update_step+1]
                        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                        print('Test acc:', accs)
                        writer.add_scalar('Accuracy',
                                          accs[-1],
                                          step + epoch*len(db))
                        writer.add_scalar('Accuracy/AugTest',
                                          accs[-1],
                                          step + epoch*len(db))

        else:
            db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
            for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

                accs = maml(x_spt, y_spt, x_qry, y_qry, step+len(db)*epoch)

                if step % 30 == 0:
                    print('step:', step, '\ttraining acc:', accs)
                    writer.add_scalar('Accuracy/Train',
                                      accs[-1],
                                      step + epoch*len(db))

                if step % 500 == 0:  # evaluation
                    if args.test in ['original', 'both']:
                        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                        accs_all_test = []

                        for x_spt, y_spt, x_qry, y_qry in db_test:
                            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                            accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                            accs_all_test.append(accs)

                        # [b, update_step+1]
                        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                        print('Test acc:', accs)
                        writer.add_scalar('Accuracy',
                                          accs[-1],
                                          step + epoch*len(db))
                        writer.add_scalar('Accuracy/Test',
                                          accs[-1],
                                          step + epoch*len(db))
                    if args.test in ['aug', 'both']:
                        db_test = DataLoader(mini_test_aug, 1, shuffle=True, num_workers=1, pin_memory=True)
                        accs_all_test = []

                        for x_spt, y_spt, x_qry, y_qry in db_test:
                            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                            accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                            accs_all_test.append(accs)

                        # [b, update_step+1]
                        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                        print('Test acc:', accs)
                        writer.add_scalar('Accuracy/AugTest',
                                          accs[-1],
                                          step + epoch*len(db))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--reg', type=float, help='coefficient for regularizer', default=1.0)
    argparser.add_argument('--log_dir', type=str, help='log directory for tensorboard', default='')
    argparser.add_argument('--ord', type=int, help='order of norms among fine-tuned weights', default=2)
    argparser.add_argument('--aug', action='store_true',
                           help='add augmentation and measure weight distance between original data and augmented data',
                           default=False)
    argparser.add_argument('--test', type=str, help='use original test set, or augmented set, or both', default='original')
    argparser.add_argument('--qry_aug', action='store_true', help='use augmented query set when meta-updating parameters', default=False)
    argparser.add_argument('--original_augmentation', action='store_true', help='...', default=False)

    args = argparser.parse_args()
    if args.log_dir != '':
        args.log_dir = '_' + args.log_dir

    main()
