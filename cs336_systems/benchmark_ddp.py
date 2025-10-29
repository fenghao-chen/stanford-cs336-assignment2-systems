import time
from typing import Type

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from cs336_basics.model import BasicsTransformerLM

from cs336_systems.ddp_overlap_bucketed import DDPOverlapBucketed
from cs336_systems.ddp_overlap_individual_parameters import DDPOverlapIndividualParameters
from cs336_systems.naive_ddp import DDPIndividualParameters

from tests.common import (
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def get_backend() -> str:
    if torch.cuda.is_available():
        return "nccl"
    else:
        return "gloo"

world_size = 2
batch_size = 4
vocab_size = 10000
rope_theta = 10000
num_steps = 10
d_model = 1600
num_layers = 48
num_heads = 25
d_ff = 6400
context_length = 256

use_ddp_flatten = False
use_ddp_overlap = True
use_ddp_bucketed = True

bucket_size = 1

def _naive_DistributedDataParallelIndividualParameters(rank: int, world_size: int, base_model: torch.nn.Module, all_x: torch.Tensor, all_y: torch.Tensor):
    device_str = _setup_process_group(rank=rank, world_size=world_size, backend=get_backend())
    device = torch.device(device_str)
    # Execute barrier prior to running test to ensure that every process
    # has finished initialization and that the following test
    # immediately exiting due to a skip doesn't cause flakiness.
    if torch.cuda.is_available():
        # Ensure each rank only participates with its local device to avoid NCCL duplicate GPU errors.
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()

    # Seed to ensure that ranks are initialized with different initial models.
    torch.manual_seed(rank)

    # This is our non-parallel baseline.
    non_parallel_model = deepcopy(base_model).to(device)

    # Create a DDP model. Note that the weights of this model should
    # match the non-parallel baseline above.
    ddp_base = deepcopy(non_parallel_model)

    if use_ddp_bucketed:
        ddp_model = DDPOverlapBucketed(ddp_base, bucket_size)
    elif use_ddp_overlap:
        ddp_model = DDPOverlapIndividualParameters(ddp_base)
    else:
        ddp_model = DDPIndividualParameters(ddp_base)

    # Make sure all the ranks have the same model state
    validate_ddp_net_equivalence(ddp_model)

    # Each rank will see only 10 examples (out of the total dataset size of 20)
    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = nn.MSELoss()

    # Optimizer for the DDP model
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)

    total_training_time = 0.
    total_comm_time = 0.
    for i in range(50):
        training_start = time.time()
        ddp_optimizer.zero_grad()

        # While the non-parallel model does a forward pass on all the data,
        # each DDP rank only sees n = batch_size / world_size examples.
        # However, the end result should be the same as doing a forward pass on all examples.
        offset = rank * local_bs
        ddp_data = all_x[offset : offset + local_bs, :].to(device)
        ddp_labels = all_y[offset : offset + local_bs, :].to(device)
        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        print(f"rank: {rank}, loss: {ddp_loss}")
        ddp_loss.backward()

        # Run student-written code that needs to execute after the backward pass,
        # but before the optimizer step (e.g., to wait for all DDP ranks to sync gradients)
        comm_start = time.time()
        if use_ddp_flatten:
            ddp_model.finish_gradient_synchronization_flatten()
        else:
            ddp_model.finish_gradient_synchronization()
        comm_end = time.time()

        ddp_optimizer.step()

        # Shuffle the data so that during the next iteration, each DDP rank sees a different set of inputs.
        # We make sure to use the same seed when shuffling (else the per-rank examples might not be disjoint).
        torch.manual_seed(42 + i)
        shuffle_idxs = torch.randperm(all_x.size(0))
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]

        training_end = time.time()
        total_training_time += training_end - training_start
        total_comm_time += comm_end - comm_start

    if use_ddp_bucketed:
        print(f"{bucket_size=}")

    print(f"{total_training_time=}, {total_comm_time=}, percentage={total_comm_time/total_training_time}")
    _cleanup_process_group()

if __name__ == '__main__':
    torch.random.manual_seed(42)
    model_class = BasicsTransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta)
    all_x = torch.randint(0, vocab_size, (batch_size, context_length))
    all_y = torch.randn(batch_size, context_length, vocab_size)

    if use_ddp_bucketed:
        for size in [1, 10, 100, 1000]:
            bucket_size = size
            mp.spawn(
                _naive_DistributedDataParallelIndividualParameters,
                args=(world_size, model_class, all_x, all_y),
                nprocs=world_size,
                join=True,
            )
    else:
        mp.spawn(
            _naive_DistributedDataParallelIndividualParameters,
            args=(world_size, model_class, all_x, all_y),
            nprocs=world_size,
            join=True,
        )
