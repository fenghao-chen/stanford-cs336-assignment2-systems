import torch
import torch.distributed as dist

class DDPOverlapBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        if bucket_size_mb is None:
            self.bucket_size_mb = float("inf")
        else:
            self.bucket_size_mb = bucket_size_mb * 1024 * 1024
        self.current_grads: list[torch.Tensor] = []
        self.current_params: list[torch.nn.Parameter] = []
        self.handles = []  # (handle, bucket_flatten)
        self.world_size = dist.get_world_size()

        def send_gradient(param: torch.Tensor) -> None:
            grad: torch.Tensor = param.grad
            if grad is not None:
                grad_mem_size = grad.element_size() * grad.nelement()
                current_mem_size = sum([t.element_size() * t.nelement() for t in self.current_grads])
                self.current_grads.append(grad)
                self.current_params.append(param)

                if grad_mem_size + current_mem_size >= self.bucket_size_mb:
                    self._flush_bucket()

        for param in reversed(list(self.module.parameters())):
            dist.broadcast(tensor=param.data, src=0, async_op=False)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(send_gradient)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def _flush_bucket(self) -> None:
        if not self.current_grads:
            return

        bucket_grads = list(self.current_grads)
        bucket_params = list(self.current_params)
        bucket_flatten = torch._utils._flatten_dense_tensors(bucket_grads)
        unflatten_views = torch._utils._unflatten_dense_tensors(bucket_flatten, bucket_grads)

        for param, view in zip(bucket_params, unflatten_views):
            param.grad = view

        handle = dist.all_reduce(tensor=bucket_flatten, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append((handle, bucket_flatten))
        self.current_grads.clear()
        self.current_params.clear()

    def finish_gradient_synchronization(self) -> None:
        self._flush_bucket()

        while self.handles:
            handle, bucket_flatten = self.handles.pop(0)
            handle.wait()
            bucket_flatten /= self.world_size  # dist.ReduceOp.AVG is not supported on Gloo
