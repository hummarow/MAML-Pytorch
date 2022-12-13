import pprint
import miniimagenet_train

EPOCH = 8


if __name__ == "__main__":
    pp = pprint.PrettyPrinter()
    settings = {
        0: {
            "epoch": EPOCH,
            "reg": 0.0,
            "aug": True,
        },
        1: {
            "epoch": EPOCH,
            "reg": 0.01,
            "aug": True,
        },
        2: {
            "epoch": EPOCH,
            "reg": 0.01,
            "aug": True,
            "flip": True,
        },
        3: {
            "epoch": EPOCH,
            "reg": 0.0,
            "aug": False,
            "traditional_augmentation": True,
        },
        4: {
            "epoch": 6,
            "aug": False,
            "traditional_augmentation": False,
            "reg": 0.0,
        },
        5: {
            "epoch": EPOCH,
            "aug": False,
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
