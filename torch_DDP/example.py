"""
example.py

Simple example demonstrating DistributedDataParallel training using DDP with
PyTorch.
"""
import argparse
import os
import sys
import tempfile
from urllib.parse import urlparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic(local_world_size, local_rank):
    """Basic demo of DDP."""
    # setup devices for this process
    # For local_world_size = 2, num_gpus = 8
    # rank1 uses GPUs [0, 1, 2, 3] and
    # rank2 uses GPUs [4, 5, 6, 7]
    n = torch.cuda.device_count() // local_world_size
    device_ids = list(range(local_rank * n, (local_rank + 1) * n))

    print(
        f'[{os.getpid()}] rank = {dist.get_rank()}, '
        f'world_size: {dist.get_world_size()}, '
        f'n = {n}, device_ids = {device_ids} \n'
    )

    model = ToyModel().cuda(device_ids[0])
    ddp_model = DDP(model, device_ids)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device_ids[0])
    loss_fn(outputs, labels).backward()
    optimizer.step()


def spmd_main(local_world_size, local_rank):
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE')
    }
    print(f'[{os.getpid()}] Initializing process group with: {env_dict}')
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend=backend)
    print(f'[{os.getpid()}] world_size = {dist.get_world_size()}, '
          f'rank = {dist.get_rank()}, backend = {dist.get_backend()}')

    demo_basic(local_world_size, local_rank)

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # This is passed in via launch.py
    parser.add_argument('--local_rank', type=int, default=0)
    # This needs to be explicitly passed in
    parser.add_argument('--local_world_size', type=int, default=1)
    args = parser.parse_args()
    # The main entry point is called directly without using subprocess
    spmd_main(args.local_world_size, args.local_rank)

