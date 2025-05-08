import torch.distributed as dist

import torch

import os

from pathlib import Path

import copy
import datetime
import builtins
import time
import io


def get_wandb_id():
    obj = wandb.run.id if is_main_process() else None
    obj = broadcast_object(obj)
    return obj


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def is_rank_zero():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        setup_for_distributed(is_master=True)
        return False

    torch.cuda.set_device(local_rank)
    dist_backend = "nccl"
    dist_url = "env://"

    print(
        f"| distributed init (rank {rank}): {dist_url}, gpu {local_rank}",
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=dist_backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
    return True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == float("inf"):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
            or "diffloss" in name
        ):
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def broadcast_object(obj):
    """
    Broadcast a Python object from rank 0 to all processes more efficiently.
    Uses cuda tensors when available and reduces memory copies.
    """
    if dist.get_rank() == 0:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = buffer.getvalue()
        length_tensor = torch.LongTensor([len(data)]).cuda()
        data_tensor = torch.frombuffer(data, dtype=torch.uint8).cuda()
    else:
        length_tensor = torch.LongTensor([0]).cuda()

    dist.broadcast(length_tensor, src=0)
    print(f"Rank {dist.get_rank()} broadcasting length tensor {length_tensor}")

    if dist.get_rank() != 0:
        data_tensor = torch.empty(length_tensor.item(), dtype=torch.uint8).cuda()

    dist.broadcast(data_tensor, src=0)

    if dist.get_rank() != 0:
        buffer = io.BytesIO(data_tensor.cpu().numpy().tobytes())
        obj = torch.load(buffer)

    return obj
