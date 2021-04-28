import argparse
import torch


def parse_args_ddp(*args):
    num_epochs_default = 10000
    batch_size_default = 256 # 1024
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "saved_models"
    model_filename_default = "model_distributed.pth"

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--local_rank", type=int, default=0,
        help="Local rank. Necessary for using the torch.distributed.launch utility."
    )
    parser.add_argument(
        '--num_threads', type=int, default=0, required=False,
        help='set number of threads per worker'
    )
    parser.add_argument(
        '--backend', type=str, default='gloo',
        help='Backend to use'
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs."
    )
    parser.add_argument(
        '--log_interval', type=int, default=10, required=False,
        help='how many batches to wait before logging training status',
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Training batch size for one process."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=256,
        help="Test batch size for one process."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--random_seed", type=int, default=0,
        help="Random seed."
    )
    parser.add_argument(
        "--model_dir", type=str, default='saved_models',
        help="Directory for saving models."
    )
    parser.add_argument(
        "--model_filename", type=str, default='model_distributed.pth',
        help="Model filename."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from saved checkpoint."
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Whether or not training is to be done on GPU'
    )
    args = parser.parse_args()

    return args


def parse_args_torch(*args):
    """Parse command line arguments containing settings for training."""
    description = 'PyTorch CIFAR10 Example using DDP'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '--batch_size', type=int, default=64, required=False,
        help='input `batch_size` for training (default: 64)',
    )
    parser.add_argument(
        '--dataset', type=str, default='cifar10', required=False,
        choices=['cifar10', 'mnist'],
        help='whether this is running on gpu or cpu',
    )
    parser.add_argument(
        '--test_batch_size', type=int, default=64, required=False,
        help='input `batch_size` for testing (default: 64)',
    )
    parser.add_argument(
        '--epochs', type=int, default=10, required=False,
        help='training epochs (default: 10)',
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, required=False,
        help='learning rate (default: 0.001)',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.5, required=False,
        help='SGD momentum (default: 0.5)',
    )
    parser.add_argument(
        '--seed', type=int, default=42, required=False,
        help='random seed (default: 42)',
    )
    parser.add_argument(
        '--log_interval', type=int, default=10, required=False,
        help='how many batches to wait before logging training status',
    )
    parser.add_argument(
        '--fp16_allreduce', action='store_true', default=False, required=False,
        help='use fp16 compression during allreduce',
    )
    parser.add_argument(
        '--device', default='cpu', choices=['cpu', 'gpu'], required=False,
        help='whether this is running on gpu or cpu'
    )
    parser.add_argument(
        '--num_threads', type=int, default=0, required=False,
        help='set number of threads per worker'
    )
    args = parser.parse_args()
    args.__dict__['cuda'] = torch.cuda.is_available()

    return args
