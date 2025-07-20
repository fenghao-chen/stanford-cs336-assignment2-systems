import torch
import torch.distributed as dist

class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        for param in self.module.parameters():
            dist.broadcast(tensor=param.data, src=0, async_op=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def finish_gradient_synchronization(self) -> None:
        for param in self.module.parameters():
            if param.grad is not None:
                dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM, async_op=False)
                param.grad = param.grad / dist.get_world_size()  # dist.ReduceOp.AVG is not supported on Gloo