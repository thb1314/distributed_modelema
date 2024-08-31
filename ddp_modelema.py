import math
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Tuple
import torch.distributed as dist
from collections import OrderedDict


def is_parallel(model):
    # return true if model is of type DP or DDP
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))

def de_parallel(model):
    # de-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


class ModelEMA:
    """
    Distributed Version Updated Exponential Moving Average (EMA)
    Keeps a moving average of everything in the model state_dict
    """

    def __init__(self, model:nn.Module, decay=0.9999, updates=0, parameters_as_bucket_view=True, group=None):
        self._ori_state_dict:Dict[str, nn.Parameter] = de_parallel(model).state_dict()
        # replace to original parameter
        ori_param_dict = {param.data_ptr():param for param in de_parallel(model).parameters()}
        ori_param_dict.update({buffer.data_ptr():buffer for buffer in de_parallel(model).buffers()})

        self._no_need_ema_dict = dict()
        for name, param in self._ori_state_dict.items():
            if param.data_ptr() in ori_param_dict and param.dtype.is_floating_point:
                self._ori_state_dict[name] = ori_param_dict[param.data_ptr()]
            else:
                self._no_need_ema_dict[name] = param

        for rm_name in self._no_need_ema_dict:
            self._ori_state_dict.pop(rm_name)

        self._partition_parameters_cache = []
        self.group = group if group is not None else dist.group.WORLD

        self.rank = 0
        self.world_size = 1
        if dist.is_initialized():
            self.world_size = dist.get_world_size(self.group)
            self.rank = dist.get_rank(self.group)
        # dispatch parameter to each rank
        self.partition_parameters()

        self._ori_cur_rank_param: Dict[str, nn.Parameter] = self._partition_parameters_cache[self.rank]
        self._ori_cur_rank_bucket: Optional[nn.Parameter] = None
        self._bucket_data_info_dict: Dict[str, Dict] = dict()

        item_size_dict = {
            torch.float16:2,
            torch.float32:4,
            torch.float64:8
        }
        if parameters_as_bucket_view and self._ori_cur_rank_param:
            device = next(iter(self._ori_cur_rank_param.values())).device
            dtype = next(iter(self._ori_cur_rank_param.values())).dtype
            buffer_size = 0
            # 8 bytes aligned
            grid_size = 8 // item_size_dict[dtype]

            for key, param in self._ori_cur_rank_param.items():
                offset_start = buffer_size
                buffer_size += (param.numel() + grid_size - 1) // grid_size * grid_size
                self._bucket_data_info_dict[key] = {
                    "offset_start": offset_start,
                    "offset_end": buffer_size,
                    "real_size": param.numel()
                }

            bucket = nn.Parameter(torch.empty(buffer_size, dtype=dtype, device=device), requires_grad=False)
            self._ori_cur_rank_bucket = bucket

            for key, param in self._ori_cur_rank_param.items():
                data_info_dict = self._bucket_data_info_dict[key]
                offset = data_info_dict['offset_start']
                offset_next = offset + data_info_dict['real_size']
                bucket[offset:offset_next].copy_(param.data.flatten(), non_blocking=False)
                param.data = bucket[offset:offset_next].view_as(param.data)
            
        self._cur_rank_param:Dict[str, nn.Parameter] = dict()
        self._cur_rank_bucket:Optional[nn.Parameter] = None
        if self._ori_cur_rank_bucket is not None:
            self._cur_rank_bucket = self._ori_cur_rank_bucket.detach().clone()
        
        for name, param in self._ori_cur_rank_param.items():
            param = param.detach().clone()
            self._cur_rank_param[name] = param
            param.requires_grad_(False)
            if self._cur_rank_bucket is not None:
                data_info_dict = self._bucket_data_info_dict[name]
                offset = data_info_dict["offset_start"]
                offset_next = offset + data_info_dict["real_size"]
                param.data = self._cur_rank_bucket[offset:offset_next].view_as(param.data)
        
        self._other_rank_param:Dict[str, nn.Parameter] = {k:v for k, v in self._ori_state_dict.items() if k not in self._ori_cur_rank_param}
        self._other_param2rank:Dict[nn.Parameter, int] = dict()
        for rank, state_dict in enumerate(self._partition_parameters_cache):
            if rank == self.rank:
                continue
            for name, param in state_dict.items():
                self._other_param2rank[param] = rank
        
        # number of EMA updates
        self.updates = updates 
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))

    @torch.no_grad()
    def update(self):
        # update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        if self._cur_rank_bucket is not None and self._ori_cur_rank_bucket is not None:
            self._cur_rank_bucket *= d
            self._cur_rank_bucket += (1 - d) * self._ori_cur_rank_bucket.detach()
        else:
            for ema_k, ema_v in self._cur_rank_param.items():
                model_v = self._ori_cur_rank_param[ema_k]
                ema_v *= d
                ema_v += (1 - d) * model_v.detach()
                self._cur_rank_param[ema_k] = ema_v

    
    def partition_parameters(self) -> List[Dict[str, nn.Parameter]]:
        r"""
        Partitions parameters across distributed data parallel ranks.

        Returns:
            a list of ``param_groups`` (which is a list of dict) where each
            element of the list contains the param_groups for a rank. Element 0
            corresponds to rank 0, etc. We need all the ranks for the broadcast
            inside ``get_model_state_dict()``.
        """
        if len(self._partition_parameters_cache) == 0:
            self._partition_parameters_cache = [dict() for _ in range(self.world_size)]
            sizes = [0] * self.world_size

            param_lists: List[List[Tuple[str, nn.Parameter]]] = [list() for _ in range(self.world_size)]
            for name, param in self._ori_state_dict.items():
                # add this param to rank with smallest size
                rank = sizes.index(min(sizes))
                param_lists[rank].append((name, param))
                sizes[rank] += param.numel()
            
            for rank, param_tuple_list in enumerate(param_lists):
                for name, param in param_tuple_list:
                    self._partition_parameters_cache[rank][name] = param
        
        return self._partition_parameters_cache

    def state_dict(self):
        state_dict, _ = self.get_model_state_dict(strict=False)
        return {
            "param": state_dict,
            "updates": self.updates
        }
    
    def load_state_dict(self, state_dict, strict=True):
        self.updates = state_dict["updates"]
        all_keys = set(self._cur_rank_param.keys())
        load_keys = set()
        for key, param in state_dict["param"].items():
            if key in self._cur_rank_param:
                self._cur_rank_param[key].copy_(param)
                load_keys.add(key)
        if strict:
            assert all_keys == load_keys
        return all_keys == load_keys
    
    def get_model_state_dict(self, strict=True):
        ema_state_dict = OrderedDict()
        ori_state_dict = OrderedDict()
        handles = []

        for key in self._ori_state_dict:
            if key in self._no_need_ema_dict:
                if not strict:
                    continue
                # adopt its original reference
                ema_state_dict[key] = self._no_need_ema_dict[key]
                ori_state_dict[key] = self._no_need_ema_dict[key]
            elif key in self._ori_state_dict:
                # send parameters
                if key in self._cur_rank_param:
                    param_value = self._cur_rank_param[key]
                    ema_state_dict[key] = param_value
                    ori_state_dict[key] = self._ori_cur_rank_param[key].detach().clone()
                    if self.world_size > 1:
                        handles.append(dist.broadcast(tensor=param_value.data, src=self.rank, group=self.group, async_op=True))
                elif key in self._other_rank_param:
                    param_value = self._other_rank_param[key]
                    src_rank = self._other_param2rank[param_value]
                    ori_state_dict[key] = param_value.detach().clone()
                    param_value = param_value.detach().clone()
                    ema_state_dict[key] = param_value
                    if self.world_size > 1:
                        handles.append(dist.broadcast(tensor=param_value.data, src=src_rank, group=self.group, async_op=True))
                else:
                    raise RuntimeError(f"{key} not in parameter list")
            else:
                raise RuntimeError(f"{key} not in parameter list")
        
        _ = list(map(lambda x: x.wait(), handles))
        return ema_state_dict, ori_state_dict
    

if __name__ == "__main__":
    """
    8 gpus
    torchrun --standlone --nproc_per_node=8 ddp_modelema.py

    1 gpu
    python ddp_modelema.py
    """
    import os
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    rank = int(os.getenv('RANK', -1))
    world_size = int(os.getenv('WORLD_SIZE', 1))

    is_ddp = local_rank != -1
    if is_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(f"cuda:0")
    
    is_master = rank in (-1, 0)
    if is_master and is_ddp:
        print('dist backend', dist.get_backend())
    
    import torchvision.transforms as transforms
    import torchvision
    import time
    import torch.optim as optim

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    if is_master:
        trainset = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transform_train)
        if is_ddp:
            dist.barrier()
    else:
        if is_ddp:
            dist.barrier()
        trainset = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transform_train)

    import torch.utils.data as torch_data
    if is_ddp:
        import torch.utils.data.distributed as data_dist
        
        train_sampler = data_dist.DistributedSampler(trainset, shuffle=True)
        trainloader = torch_data.DataLoader(trainset, batch_size=128, sampler=train_sampler, num_workers=8)
    else:
        trainloader = torch_data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./cifar', train=False, download=True, transform=transform_test)
    testloader = torch_data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    from torchvision.models import resnet50
    if is_master:
        model = resnet50(pretrained=True)
        if is_ddp:
            dist.barrier()
    else:
        if is_ddp:
            dist.barrier()
        model = resnet50(pretrained=True)
    
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    train_model = model
    if is_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        train_model = ddp_model
    train_model.train()

    ema = ModelEMA(train_model, parameters_as_bucket_view=True)

    train_loss = 0
    epoch = 100

    optimizer = optim.SGD(train_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    
    import torch.nn.functional as F
    def eval_model(test_model):
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = test_model(inputs)
                loss = F.cross_entropy(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        if is_master:
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} {:.0f}%'.format(
                test_loss, correct, len(testloader.dataset), 100. * correct / total
            ))

    
    for e in range(epoch):
        if is_ddp:
            train_sampler.set_epoch(e)
        
        train_model.train()
        ema_cost = 0.0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = train_model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            start = time.perf_counter()
            ema.update()
            ema_cost += time.perf_counter() - start
        
        if is_master:
            print(f"\nepoch:{e:03d}, loss:{loss.item():.04f} ema_cost:{ema_cost:.04f}")
        
        train_model.eval()
        ema_state_dict, ori_state_dict = ema.get_model_state_dict()
        de_parallel(train_model).load_state_dict(ema_state_dict, strict=True)
        # EMA evaluation
        if is_master:
            print('ema model')
        eval_model(train_model)
        # restore original weights
        de_parallel(train_model).load_state_dict(ori_state_dict, strict=True)

        del ori_state_dict
        del ema_state_dict

        if is_master:
            print('ori model')
        eval_model(train_model)

        scheduler.step()


