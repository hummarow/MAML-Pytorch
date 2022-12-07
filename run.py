import numpy as np
import pprint
import scipy.stats
import miniimagenet_train
from datetime import datetime

EPOCH = 10


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h


if __name__ == "__main__":
    pp = pprint.PrettyPrinter()
    settings = {
        0: {
            "epoch": EPOCH,
            "reg": 0.0,
            "qry_aug": False,
            "aug": True,
        },
        1: {
            "epoch": EPOCH,
            "reg": 0.01,
            "qry_aug": False,
            "aug": True,
        },
        2: {
            "epoch": EPOCH,
            "reg": 0.01,
            "qry_aug": False,
            "aug": True,
            "flip": True,
        },
        3: {
            "epoch": EPOCH,
            "reg": 0.0,
            "qry_aug": False,
            "aug": False,
            "traditional_augmentation": True,
        },
        4: {
            "epoch": EPOCH,
            "aug": False,
            "qry_aug": False,
            "traditional_augmentation": False,
            "reg": 0.0,
        },
        5: {
            "epoch": EPOCH,
            "aug": False,
            "qry_aug": False,
            "traditional_augmentation": False,
            "reg": 0.0,
            "prox_lam": 0.5,
            "prox_task": 2,
            "update_step": 10,
        },
    }

    while True:
        pp.pprint(settings)
        setting_number = input("Setting #")
        if not setting_number:
            setting_number = 0
            break
        else:
            try:
                setting_number = int(setting_number)
                break
            except ValueError:
                print("Input Number")

    setting = settings[setting_number]

    miniimagenet_train.main(**setting)
