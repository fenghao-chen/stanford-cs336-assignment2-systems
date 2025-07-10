import math
import time
from typing import Callable

from cs336_basics.model import BasicsTransformerLM
import torch
import numpy as np
from torch import Tensor
import argparse

from torch.optim import AdamW


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")           # NVIDIA / AMD ROCm
    elif torch.backends.mps.is_available():     # Apple-silicon (M-series) Metal backend
        device =  torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    return device

def run_transformer(
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: int,
        input: Tensor, num_steps: int, forward_only: bool = False) -> Callable:

    # Define a model (with random weights)
    model = BasicsTransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta).to(get_device())

    optimizer = AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            # Forward
            y = model(input).mean()
            # Backward
            if not forward_only:
                y.backward()
                optimizer.step()
    return run

def benchmark(description: str, run: Callable, num_warmups: int = 5, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    # Time it for real now!
    times: list[float] = [] # @inspect times, @inspect description
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()
        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
        end_time = time.time()
        times.append((end_time - start_time) * 1000) # @inspect times
    mean_time = np.mean(times) # @inspect mean_time
    std_time = np.std(times)
    return mean_time, std_time

if __name__ == '__main__':
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--context_length", default=256, type=int)
    ap.add_argument("--d_model", default=512, type=int)
    ap.add_argument("--d_ff", default=1344, type=int)
    ap.add_argument("--num_layers", default=4, type=int)
    ap.add_argument("--num_heads", default=16, type=int)
    ap.add_argument('--forward_only', action='store_true', default=False)

    args = ap.parse_args()

    batch_size = 4
    vocab_size = 10000
    rope_theta = 10000
    num_steps = 10
    context_length = args.context_length
    d_model = args.d_model
    num_heads = args.num_heads
    d_ff = args.d_ff
    num_layers = args.num_layers
    forward_only = args.forward_only

    input = torch.randint(0, vocab_size, (batch_size, context_length))

    print(f"benchmarking using the following hyperparameters: context_length: {context_length}, d_model: {d_model}, d_ff: {d_ff}, num_heads: {num_heads}, num_layers: {num_layers}, forward_only: {forward_only}")
    mean, std = benchmark("run_transformer",
              run_transformer(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_heads=num_heads, num_layers=num_layers, rope_theta=rope_theta, d_ff=d_ff, input=input, num_steps=num_steps, forward_only=forward_only))
    print(f"result: average {mean} millisecond, standard deviation {std}")