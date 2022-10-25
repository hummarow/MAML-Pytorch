import numpy as np
import scipy.stats
import miniimagenet_train


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


if __name__ == '__main__':
    settings = []
    setting = {'epoch': 120000,
               'reg': 0.0,
               'qry_aug': False,
               'aug': True,
               }
    settings.append(setting)
    setting = {'epoch': 120000,
               'reg': 0.05,
               'qry_aug': True,
               'aug': True,
               }
    settings.append(setting)
    setting = {'epoch': 120000,
               'reg': 0.0,
               'qry_aug': False,
               'aug': False,
               'original_augmentation': True,
               'log_dir': 'baseline_aug',
               }
    settings.append(setting)
    a = int(input('setting'))
    setting = settings[a]
    best_accs = []
    for i in range(1):
        print(i)
        accs = miniimagenet_train.main(**setting)
        best_accs.append(max(accs))
    mean, ci = mean_confidence_interval(best_accs)
    print(setting)
    print("{}% +- {}%".format(mean*100, ci*100))
