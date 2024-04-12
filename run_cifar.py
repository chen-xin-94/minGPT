import numpy as np
import argparse
import torchvision
import torch
import logging
from mingpt.utils import set_seed
from mingpt.utils_image import ImageDataset, TrainerConfig, kmeans
from mingpt.model import GPT
from mingpt.trainer import Trainer

def add_argument():
    parser = argparse.ArgumentParser(description='CIFAR10')
    # model config
    parser.add_argument('--model_type', type=str, default=None, help='model type')
    parser.add_argument('--n_layer', type=int, default=12, help='number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of heads')
    parser.add_argument('--n_embd', type=int, default=256, help='embedding dimension')
    # trainer config
    parser.add_argument('--max_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    args = parser.parse_args()
    # either model_type or (n_layer, n_head, n_embd) must be given in the config
    if args.model_type == '':
        if args.n_layer is None or args.n_head is None or args.n_embd is None:
            parser.error("If model_type is not given, n_layer, n_head, n_embd must be given.")
    return args

def main():
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )
    set_seed(42)


    # get the data CIFAR10
    root = './'
    train_data = torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=True)
    test_data  = torchvision.datasets.CIFAR10(root, train=False, transform=None, target_transform=None, download=True)
    print(len(train_data), len(test_data))

    # +
    # get random 3 pixels per image and stack them all up as rgb values to get half a million random pixels
    pluck_rgb = lambda x: torch.from_numpy(np.array(x)).view(32*32, 3)[torch.randperm(32*32)[:3], :]
    px = torch.cat([pluck_rgb(x) for x, y in train_data], dim=0).float()

    # run kmeans to get our codebook
    ncluster = 512
    with torch.no_grad():
        clusters = kmeans(px, ncluster, niter=5)

    train_dataset = ImageDataset(train_data, clusters)
    test_dataset = ImageDataset(test_data, clusters)

    args = add_argument()
    # config model
    m_config = GPT.get_default_config()
    m_config.model_type = args.model_type
    m_config.n_layer = args.n_layer
    m_config.n_head = args.n_head
    m_config.n_embd = args.n_embd
    m_config.vocab_size = train_dataset.vocab_size
    m_config.block_size = train_dataset.block_size
    model = GPT(m_config)

    # config trainer
    t_config = Trainer.get_default_config()
    t_config.max_iters = args.max_epochs
    t_config.batch_size = args.batch_size
    trainer = Trainer(t_config, model, train_dataset)
    trainer.run()
    print('done training')

if __name__ == '__main__':
    main()