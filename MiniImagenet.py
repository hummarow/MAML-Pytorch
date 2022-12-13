import os
import torch
import numpy as np
import collections
import csv
import random
import glob
import sys
import functools
import typing
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from copy import deepcopy
from imagenet_c import corrupt


class Rotate:
    def __init__(self, angles: list):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)


class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    #    def __init__(self, root, mode, num_episodes, n_way, k_shot, k_query, resize, startidx=0, aug=False):
    def __init__(self, root, mode, num_episodes, args, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param num_episodes: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of quy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.num_episodes = num_episodes  # batch of set, not batch of imgs
        self.n_way = args.n_way  # n-way
        self.k_shot = args.k_spt  # k-shot
        self.k_query = args.k_qry  # for evaluation
        self.mode = mode
        if self.mode == "val" or self.mode == "test":
            self.k_query = 1
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = args.imgsz  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        self.need_aug = args.need_aug  # Needs augmentation or not
        self.flip = args.flip
        print(
            "shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d"
            % (self.mode, self.num_episodes, self.n_way, self.k_shot, self.k_query, self.resize)
        )

        self.transform = self.get_transform(mode, aug=False, flip=self.flip)
        self.transform_aug = self.get_transform(mode, aug=True, flip=self.flip)

        self.path = os.path.join(root, "images")  # image path
        images = glob.glob(root + "/*/*/*.jpg")
        self.images = {}
        for img in images:
            _, name = os.path.split(img)
            self.images[name] = img
        csvdata = self.loadCSV(os.path.join(root, mode + ".csv"))  # csv path
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(csvdata.items()):
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]]
            self.img2label[k] = i + self.startidx  # {"img_name[:9]":label}
        self.cls_num = len(self.data)

        self.create_batch()

    def get_transform(self, mode, aug, flip=False):
        """
        Return the transform function with or without augmentation
        """
        if mode == "train":
            transform = transforms.Compose(
                [
                    lambda x: np.asarray(Image.open(x).convert("RGB")),
                    lambda x: Image.fromarray(x),
                    transforms.Resize((self.resize, self.resize)),
                    lambda x: self.augmentation(x, aug, flip),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    lambda x: Image.open(x).convert("RGB"),
                    transforms.Resize((self.resize, self.resize)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )

        return transform

    def augmentation(self, img, aug, flip=False):
        """
        Return a [flipped, rotated] image.
        if no augmentation is required,
        return input image without any transformation.
        """
        if not aug:
            return img
        else:
            #            transform = transforms.Compose([
            #                                            transforms.RandomHorizontalFlip(p=prob),
            ##                                            transforms.RandomRotation((0,70)),
            #                                            ])
            augmentation_list = [Rotate(angles=[-90, 0, 90, 180])]
            if flip:
                augmentation_list.append(transforms.RandomHorizontalFlip(p=0.5))
            transform = transforms.Compose(
                # random.choice(augmentation_list)
                augmentation_list
            )
            return transform(img)

    def loadCSV(self, csvf):
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {}
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            next(csvreader, None)  # skip (filename, label)
            for i, row in enumerate(csvreader):
                filename = row[0]
                label = row[1]
                # append filename to current label
                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]
        return dictLabels

    def create_batch(self):
        """
        create batch for meta-learning.
        episode here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in range(self.num_episodes):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(
                    len(self.data[cls]), self.k_shot + self.k_query, False
                )
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[: self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot :])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist()
                )  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)

            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets
        # print("batch length ", len(self.support_x_batch))

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= num_episodes-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int)

        flatten_support_x = [
            self.images[item] for sublist in self.support_x_batch[index] for item in sublist
        ]
        support_y = np.array(
            [
                self.img2label[
                    item[:9]
                ]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
                for sublist in self.support_x_batch[index]
                for item in sublist
            ]
        ).astype(np.int32)

        flatten_query_x = [
            self.images[item] for sublist in self.query_x_batch[index] for item in sublist
        ]
        query_y = np.array(
            [self.img2label[item[:9]] for sublist in self.query_x_batch[index] for item in sublist]
        ).astype(np.int32)

        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)

        if self.need_aug:
            support_x_aug = deepcopy(support_x)
            query_x_aug = deepcopy(query_x)
        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)
            if self.need_aug:
                support_x_aug[i] = self.transform_aug(path)

        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
            if self.need_aug:
                query_x_aug[i] = self.transform_aug(path)
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

        if self.need_aug and self.mode == "train":
            return (
                support_x,
                torch.LongTensor(support_y_relative),
                query_x,
                torch.LongTensor(query_y_relative),
                support_x_aug,
                query_x_aug,
            )
        return (
            support_x,
            torch.LongTensor(support_y_relative),
            query_x,
            torch.LongTensor(query_y_relative),
        )

    def __len__(self):
        # as we have built up to num_episodes of sets, you can sample some small batch size of sets.
        return self.num_episodes


if __name__ == "__main__":
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion()

    tb = SummaryWriter("runs", "mini-imagenet")
    mini = MiniImagenet(
        "../Datasets/mini-imagenet/",
        mode="train",
        n_way=5,
        k_shot=1,
        k_query=1,
        num_episodes=1000,
        resize=168,
        aug=True,
    )
    #    mini_aug = MiniImagenet('../Datasets/mini-imagenet/', mode='train', n_way=5, k_shot=1, k_query=1, num_episodes=1000, resize=168, aug=True)

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_

        support_x = make_grid(support_x, nrow=2)
        #        support_x_aug = make_grid(mini_aug[0][0], nrow=2)
        query_x = make_grid(query_x, nrow=2)

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).numpy())
        #        plt.savefig('asd.png')
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).numpy())
        plt.pause(0.5)
        plt.figure(3)
        #        plt.imshow(support_x_aug.transpose(2, 0).numpy())
        #        plt.savefig('asd_aug.png')
        #        plt.pause(0.5)

        tb.add_image("support_x", support_x)
        tb.add_image("query_x", query_x)

        time.sleep(5)

    tb.close()
