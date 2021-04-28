"""
torch_ddp.py

Modified from original: https://leimao.github.io/blog/PyTorch-Distributed-Training/
"""
import sys
import torch
import time
import json
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.distributed as dist
from typing import Callable
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import random
import numpy as np

here = os.path.abspath(os.path.dirname(__file__))
modulepath = os.path.dirname(here)
if modulepath not in sys.path:
    sys.path.append(modulepath)

from utils.io import Logger, DistributedDataObject, prepare_datasets
from utils.parse_args import parse_args_ddp

logger = Logger()

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def metric_average(x: torch.tensor, with_ddp: bool) -> (torch.tensor):
    """Compute global averages across all workers if using DDP. """
    x = torch.tensor(x)
    if with_ddp:
        # Sum everything and divide by total size
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= SIZE
    else:
        pass

    return x.item()


def train(
        epoch: int,
        data: DistributedDataObject,
        device: torch.device,
        rank: int,
        model: nn.Module,
        loss_fn: Callable[[torch.tensor], torch.tensor],
        optimizer: optim.Optimizer,
        args: dict,
        scaler: GradScaler=None,
):
    model.train()
    # Horovod: set epoch to sampler for shuffling
    data.sampler.set_epoch(epoch)
    running_loss = 0.0
    training_acc = 0.0
    #  running_loss = torch.tensor(0.0)
    #  training_acc = torch.tensor(0.0)

    for batch_idx, (batch, target) in enumerate(data.loader):
        if args.cuda:
            batch, target = batch.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        acc = pred.eq(target.data.view_as(pred)).cpu().float().sum()

        training_acc += acc
        running_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            metrics_ = {
                'epoch': epoch,
                'batch_loss': loss.item() / args.batch_size,
                'running_loss': running_loss / len(data.sampler),
                'batch_acc': acc.item() / args.batch_size,
                'training_acc': training_acc / len(data.sampler),
            }

            jdx = batch_idx * len(batch)
            frac = 100. * batch_idx / len(data.loader)
            pre = [f'[{rank}]',
                   f'[{jdx}/{len(data.sampler)} ({frac}%)]']
                   #  f'[{jdx:05}/{len(data.sampler):05} ({frac:>4.3g}%)]']
            mstr = ' '.join([
                #  f'{str(k):>5}: {v:<7.4g}' for k, v in metrics_.items()
                f'{k}: {v}' for k, v in metrics_.items()
            ])
            logger.log(' '.join([*pre, mstr]))

            #  str0 = f'[{jdx:5<}/{len(data.sampler):5<} ({frac:>3.3g}%)]'
            #  str1 = ' '.join([
            #      f'{str(k):>5}: {v:<7.4g}' for k, v in metrics_.items()
            #  ])
            #  logger.log(' '.join([str0, str1]))

    running_loss = running_loss / len(data.sampler)
    training_acc = training_acc / len(data.sampler)
    #  loss_avg = metric_average(running_loss, args.cuda)
    #  training_acc = metric_average(training_acc, args.cuda)
    if rank == 0:
        logger.log(f'training set; avg loss: {running_loss:.4g}, '
                   f'accuracy: {training_acc * 100:.2f}%')


def main(*args):
    args = parse_args_ddp()
    with_cuda = torch.cuda.is_available()

    args.cuda = with_cuda

    local_rank = args.local_rank
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    random_seed = args.random_seed
    model_dir = args.model_dir
    model_filename = args.model_filename
    resume = args.resume

    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''

    model_filepath = os.path.join(model_dir, model_filename)

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    backend = 'nccl' if with_cuda else 'gloo'
    dist.init_process_group(backend=backend)
    #  torch.distributed.init_process_group(backend=backend)
    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    model = torchvision.models.resnet18(pretrained=False)

    if args.cuda:
        device = torch.device("cuda:{}".format(local_rank))
        num_workers = torch.cuda.device_count()
    else:
        device = torch.device(int(local_rank))

    model = model.to(device)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume == True:
        if with_cuda:
            map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        else:
            map_location = {'0': f'{local_rank}'}

        state_dict = torch.load(model_filepath, map_location=map_location)
        ddp_model.load_state_dict(state_dict)

    # Prepare dataset and dataloader
    data = prepare_datasets(args, rank=local_rank,
                            num_workers=num_workers,
                            data='cifar10')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=lr, weight_decay=1e-5)
    #  optimizer = optim.SGD(ddp_model.parameters(), lr=lr, momentum=0.9,
    #                        weight_decay=1e-5)

    # Loop over the dataset multiple times
    epoch_times = []
    for epoch in range(epochs):
        t0 = time.time()
        train(epoch, data['training'], device=device, rank=local_rank,
              model=ddp_model, loss_fn=criterion,
              optimizer=optimizer, args=args, scaler=None)

        if epoch > 2:
            epoch_times.append(time.time() - t0)

        if epoch % 10 == 0:
            if local_rank == 0:
                accuracy = evaluate(model=ddp_model, device=device,
                                    test_loader=data['testing'].loader)
                torch.save(ddp_model.state_dict(), model_filepath)
                logger.log('-' * 75)
                logger.log(f'Epoch: {epoch}, Accuracy: {accuracy}')
                logger.log('-' * 75)


    if local_rank == 0:
        epoch_times_str = ', '.join(str(x) for x in epoch_times)
        logger.log('Epoch times:')
        logger.log(epoch_times_str)

        args_file = os.path.join(os.getcwd(), f'args_size{num_workers}.json')
        logger.log(f'Saving args to: {args_file}.')

        with open(args_file, 'at') as f:
            json.dump(args.__dict__, f, indent=4)

        times_file = os.path.join(os.getcwd(),
                                  f'epoch_times_size{num_workers}.csv')
        logger.log(f'Saving epoch times to: {times_file}')
        with open(times_file, 'a') as f:
            f.write(epoch_times_str + '\n')

        #  with open('./args.json', 'wt') as f:
        #      json.dump(args.__dict__, f, indent=4)
        #
        #  with open('./epoch_times.csv', 'w') as f:
        #      f.write(', '.join(str(x) for x in epoch_times) + '\n')




if __name__ == "__main__":
    main()
