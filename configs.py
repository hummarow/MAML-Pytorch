import typing
import os


def get_path(log_dir: str, aug=False, reg=0.0):
    if log_dir != '':
        log_dir = '_' + log_dir

    if aug:
        _aug_log_path_name = 'Reg' + str(reg) + log_dir
        log_path = os.path.join('logs', 'aug', _aug_log_path_name)
    else:
        _log_path_name = 'org' + log_dir
        log_path = os.path.join('logs', _log_path_name)
    return log_path
