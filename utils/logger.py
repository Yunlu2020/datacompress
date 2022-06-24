import logging
import os
import torch.distributed as dist
from collections import OrderedDict
import numpy as np

from src.utils.dist import get_rank, is_local_master

logger_initialized = dict()


def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = get_logger(name='PMR', log_file=log_file, log_level=log_level)
    return logger


def get_logger(name, log_file=None, log_level=logging.INFO):
    logger = logging.getLogger(name)
    logger.propagate = False
    
    if name in logger_initialized:
        return logger
    
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]
    
    rank = get_rank()
    
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    
    log_level = log_level if is_local_master() else logging.ERROR
    logger.setLevel(log_level)
    
    logger_initialized[name] = True
    
    return logger
