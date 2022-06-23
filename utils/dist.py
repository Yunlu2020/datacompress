import pickle
import torch
import torch.distributed as dist
from functools import lru_cache


def is_distributed():
    return get_world_size() > 1


@lru_cache()
def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


@lru_cache()
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def master_only(func):
    def warp(*args, **kwargs):
        # do somethings.
        if is_master():
            res = func(*args, **kwargs)  # 执行函数
        return res
    return warp


def local_master_only(func):
    def warp(*args, **kwargs):
        # do somethings.
        if is_local_master():
            res = func(*args, **kwargs)  # 执行函数
        return res
    return warp


@lru_cache()
def is_local_master():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return get_rank() % 8 == 0


@lru_cache()
def is_master():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return get_rank() == 0


def synchronize():
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    dist.barrier()
    

def all_gather(data):
    to_device = torch.device('cuda')
    
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(to_device)
    
    local_size = torch.LongTensor([tensor.numel()]).to(to_device)
    size_list = [torch.LongTensor([0]).to(to_device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to(to_device))
        
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size-local_size, )).to(to_device)
        tensor = torch.cat((tensor, padding), dim=0)
    
    dist.all_gather(tensor_list, tensor)
    
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
        
    return data_list


def broadcast(data, src=0, group=None):
    to_device = torch.device('cuda')
    
    world_size = get_world_size()
    if world_size == 1:
        return data

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(to_device)

    local_size = torch.LongTensor([tensor.numel()]).to(to_device)
    size_list = [torch.LongTensor([0]).to(to_device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(to_device)
        tensor = torch.cat((tensor, padding), dim=0)
        
    dist.broadcast(tensor, src=src, group=group, async_op=False)
    
    buffer = tensor.cpu().numpy().tobytes()
    return pickle.loads(buffer)