# Distributed ModelEMA

启发于ZeroOptimizer2的原理，本项目给出一种分布式ModelEMA的实现，相比于原有的ModelEMA实现，其可以将待计算的参数均匀分配到每个计算卡上，再进行计算，极大节省了计算量。由于EMA更新过程不需要进行模型评估，所以仅需在进行模型评估之前同步所有节点的`EMA Model`参数。毫不夸张地讲，本项目提出的分布式ModelEMA可以无缝替换其之前的版本，极大地促进了训练步骤中ModelEMA步骤的训练效率。

Inspired by ZeroOptimizer2, this project presents a distributed implementation of ModelEMA. 
Compared to the its original implementation, it evenly distributes the parameters to be calculated across each computing card for processing, which greatly reduces computational cost. Since the EMA update process does not require model evaluation, we only need to synchronize the EMA Model parameters across all nodes before model evaluation step. Without bells and whistles, the distributed ModelEMA proposed in this project can seamlessly replace its original version, greatly boosting the efficiency of ModelEMA step in training pipeline.

## 接口设计(Interface)

- ModelEMA
  - update：替代原来的update(replace original update method)
  - state_dict：用于ModelEMA的checkpoint生成(used for checkpoint generation)
  - load_state_dict：用于加载ModelEMA的checkpoint(used for checkpoint loading)
  - get_model_state_dict：获取`ema model`的`state_dict`(get ema model statedict)

## 案例代码（Example）


```python

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

```
