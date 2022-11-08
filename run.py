import numpy as np
import pprint
import scipy.stats
import miniimagenet_train
from datetime import datetime

EPOCH = 6


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
            "reg": 0.05,
            "qry_aug": True,
            "aug": True,
        },
        2: {
            "epoch": EPOCH,
            "reg": 0.0,
            "qry_aug": False,
            "aug": False,
            "traditional_augmentation": True,
        },
        3: {
            "epoch": EPOCH,
            "aug": False,
            "qry_aug": False,
            "traditional_augmentation": False,
            "reg": 0.0,
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

    best_test_accs = miniimagenet_train.main(**setting)
    if len(best_test_accs) < 10:
        print("Test Accuracies: ", best_test_accs)
