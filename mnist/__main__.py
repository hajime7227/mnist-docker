# https://github.com/pytorch/examples/blob/main/mnist/main.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from mnist.configs.train_settings import get_args
from mnist.models.cnn import Net
from mnist.training.test import test
from mnist.training.train import train
from mnist.utils.data_loadoer import get_data_loaders


def main():
    # Training settings
    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(device)

    train_loader, test_loader = get_data_loaders(
        args.batch_size, args.test_batch_size, use_cuda
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
