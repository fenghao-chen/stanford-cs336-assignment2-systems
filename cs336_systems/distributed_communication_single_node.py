import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

is_gpu = torch.cuda.is_available()
device = "cuda" if is_gpu else "cpu"

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if is_gpu:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, tensor_size):
    setup(rank, world_size)
    data = torch.randn(int(tensor_size)).to(device)
    dist.all_reduce(data, async_op=False)

if __name__ == "__main__":
    world_size_list = [2, 4, 6]
    mb_over_float32 = (2 ** 20) / 4
    tensor_size_list = [mb_over_float32, 10 * mb_over_float32, 100 * mb_over_float32, 1000 * mb_over_float32]

    # warm up
    for _ in range(5):
        mp.spawn(fn=distributed_demo, args=(6, 100), nprocs=6, join=True)

    for tensor_size in tensor_size_list:
        for world_size in world_size_list:
            start = time.time()
            mp.spawn(fn=distributed_demo, args=(world_size, tensor_size), nprocs=world_size, join=True)
            if is_gpu:
                torch.cuda.synchronize()
            end = time.time()
            print(f"{tensor_size/mb_over_float32} MB, {world_size=} {end - start:2f} seconds")