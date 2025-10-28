import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []

        def send_gradient(param: torch.Tensor) -> None:
            if param.grad is not None:
                handle = dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append(handle)

        for param in self.module.parameters():
            dist.broadcast(tensor=param.data, src=0, async_op=False)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(send_gradient)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def finish_gradient_synchronization(self) -> None:
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        for param in self.module.parameters():
            if param.grad is not None:
                param.grad = param.grad / dist.get_world_size()  # dist.ReduceOp.AVG is not supported on Gloo


