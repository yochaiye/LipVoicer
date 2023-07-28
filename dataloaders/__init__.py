# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE


import torch
from torch.utils.data.distributed import DistributedSampler
from .dataset_lipvoicer import LipVoicerDataset

def dataloader(dataset_cfg, batch_size, num_gpus):

    dataset_name = dataset_cfg.pop("_name_")
    dataset = LipVoicerDataset(split='train', **dataset_cfg)
    dataset_cfg["_name_"] = dataset_name # Restore

    # distributed sampler
    train_sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=False,
        drop_last=True,
    )
    return trainloader
