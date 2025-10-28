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

    def finish_gradient_synchronization_flatten(self) -> None:
        grad_list: list[torch.Tensor] = []
        for param in self.module.parameters():
            if param.grad is not None:
                grad_list.append(param.grad)

        flatten = torch._utils._flatten_dense_tensors(grad_list)
        dist.all_reduce(tensor=flatten, op=dist.ReduceOp.SUM, async_op=False)
        flatten = flatten / dist.get_world_size()
        un_flatten = torch._utils._unflatten_dense_tensors(flatten, tuple(grad_list))
        count = 0
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad = un_flatten[count]
                count += 1

