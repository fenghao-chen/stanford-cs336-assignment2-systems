import torch
import torch.distributed as dist

class DDPOverlapBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb * 1024 * 1024
        self.current_grads : list[torch.Tensor] = []
        self.handles = [] # (count, handle, flatten)

        def send_gradient(param: torch.Tensor) -> None:
            grad: torch.Tensor = param.grad
            if grad is not None:
                grad_mem_size = grad.element_size() * grad.nelement()
                current_mem_size = sum([t.element_size() * t.nelement() for t in self.current_grads])
                self.current_grads.append(grad)

                if grad_mem_size + current_mem_size >= self.bucket_size_mb:
                    flatten = torch._utils._flatten_dense_tensors(self.current_grads)
                    count = len(self.current_grads)
                    handle = dist.all_reduce(tensor=flatten, op=dist.ReduceOp.SUM, async_op=True)
                    self.handles.append((count, handle, flatten))
                    self.current_grads.clear()

        for param in reversed(list(self.module.parameters())):
            dist.broadcast(tensor=param.data, src=0, async_op=False)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(send_gradient)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def finish_gradient_synchronization(self) -> None:
        # Deal with the remaining gradients
        if len(self.current_grads) > 0:
            flatten = torch._utils._flatten_dense_tensors(self.current_grads)
            count = len(self.current_grads)
            handle = dist.all_reduce(tensor=flatten, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((count, handle, flatten))
            self.current_grads.clear()

        current_params: list[torch.Tensor] = []
        for param in reversed(list(self.module.parameters())):
            if len(self.handles) == 0:
                break

            if param.grad is not None:
                current_params.append(param)

            if len(current_params) == self.handles[0][0]:
                _, handle, flatten = self.handles.pop(0)
                handle.wait()
                un_flatten = torch._utils._unflatten_dense_tensors(flatten, [p.grad for p in current_params])
                for p, g in zip(current_params, un_flatten):
                    p.grad = g / dist.get_world_size()  # dist.ReduceOp.AVG is not supported on Gloo

                current_params.clear()

