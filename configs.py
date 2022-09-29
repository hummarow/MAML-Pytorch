import typing
import os


def get_path(reg: float, ord: int, log_dir: str, aug: bool):
    if aug:
        _aug_log_path_name = 'Reg' + str(reg) + log_dir
        log_path = os.path.join('logs', 'aug', _aug_log_path_name)
    else:
        _log_path_name = 'Reg' + str(reg) + log_dir
        log_path = os.path.join('logs', 'L'+str(ord), _log_path_name)
    return log_path
