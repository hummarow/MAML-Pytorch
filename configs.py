import typing
import os


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
