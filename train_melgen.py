# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE

import os
import time
# import warnings
# warnings.filterwarnings("ignore")
from functools import partial
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import hydra
# import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from dataloaders import dataloader
from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory

from distributed_util import init_distributed, apply_gradient_allreduce, reduce_tensor
from generate_melgen import generate

from models.model_builder import ModelBuilder
from models.audioVisual_model import AudioVisualModel

def distributed_train(rank, num_gpus, group_name, cfg):
    # Initialize logger
    if rank == 0 and cfg.wandb is not None:
        # wandb_cfg = cfg.pop("wandb")
        # wandb.init(
        #     **wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True)
        # )
        pass

    # Distributed running initialization
    dist_cfg = cfg.pop("distributed")
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_cfg)

    train(
        rank=rank, num_gpus=num_gpus,
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        generate_cfg=cfg.generate,
        **cfg.train,
    )

def train(
    rank, num_gpus, save_dir,
    diffusion_cfg, model_cfg, dataset_cfg, generate_cfg, # dist_cfg, wandb_cfg, # train_cfg,
    ckpt_iter, n_iters, iters_per_ckpt, iters_per_logging,
    learning_rate, batch_size_per_gpu,
    name=None,
):
    
    """
    Parameters:
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automitically selects the maximum iteration if 'max' is selected
    n_iters (int):                  number of iterations to train, default is 1M
    iters_per_ckpt (int):           number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate
    batch_size_per_gpu (int):       batchsize per gpu, default is 2 so total batchsize is 16 with 8 gpus
    n_samples (int):                audio samples to generate and log per checkpoint
    name (str):                     prefix in front of experiment name
    mel_path (str):                 for vocoding, path to mel spectrograms (TODO generate these on the fly)
    """

    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, save_dir, 'checkpoint')

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_cfg, fast=False)  # dictionary of all diffusion hyperparameters

    # load training data
    trainloader = dataloader(dataset_cfg, batch_size=batch_size_per_gpu, num_gpus=num_gpus)
    print('Data loaded')

    # predefine model
    builder = ModelBuilder()
    net_lipreading = builder.build_lipreadingnet()
    net_facial = builder.build_facial(fc_out=128, with_fc=True)
    net_diffwave = builder.build_diffwave_model(model_cfg)
    net = AudioVisualModel((net_lipreading, net_facial, net_diffwave)).cuda()
    print_size(net, verbose=False)

    criterion = nn.L1Loss()

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(checkpoint_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(checkpoint_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # HACK to reset learning rate
                optimizer.param_groups[0]['lr'] = learning_rate

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            print(f"Model checkpoint found at iteration {ckpt_iter}, but was not successfully loaded - training from scratch.")
            ckpt_iter = -1
    else:
        print('No valid checkpoint model found - training from scratch.')
        ckpt_iter = -1

    # training
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        epoch_loss = 0.
        for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}') if rank==0 else trainloader:
        # for data in tqdm(trainloader, desc=f'Epoch {n_iter // len(trainloader)}'):
            melspec, mouthroi, face_image = data
            melspec, mouthroi, face_image = melspec.cuda(), mouthroi.cuda(), face_image.cuda()

            # back-propagation
            optimizer.zero_grad()
            loss = training_loss(net, criterion, melspec, mouthroi, face_image, diffusion_hyperparams)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            epoch_loss += reduced_loss

            # output to log
            # if n_iter % iters_per_logging == 0 and rank == 0:
                # save training loss to tensorboard
                # print("iteration: {} \treduced loss: {} \tloss: {}".format(n_iter, reduced_loss, loss.item()))
                # wandb.log({
                #     'train/loss': reduced_loss,
                #     'train/log_loss': np.log(reduced_loss),
                # }, step=n_iter)

            # save checkpoint
            if n_iter % iters_per_ckpt == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(checkpoint_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

                # Generate samples
                generate_cfg["ckpt_iter"] = n_iter
                samples = generate(
                    rank, # n_iter,
                    diffusion_cfg, model_cfg, dataset_cfg,
                    name=name,
                    save_dir=generate_cfg.save_dir,
                    ckpt_iter="max",
                    n_samples=generate_cfg.n_samples,
                    guidance_factor=generate_cfg.guidance_factor,
                )
                # samples = [wandb.Audio(sample.squeeze().cpu(), sample_rate=dataset_cfg['sampling_rate']) for sample in samples]
                # wandb.log(
                #     {'inference/audio': samples},
                #     step=n_iter,
                #     # commit=False,
                # )

            n_iter += 1
        if rank == 0:
            epoch_loss /= len(trainloader)
            # wandb.log({'train/loss_epoch': epoch_loss, 'train/log_loss_epoch': np.log(epoch_loss)}, step=n_iter)

    # Close logger
    if rank == 0:
        # tb.close()
        # wandb.finish()
        pass

def training_loss(net, loss_fn, melspec, mouthroi, face_image, diffusion_hyperparams):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    training loss
    """
    # Predict melspectrogram from visual features using diffusion model
    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    B, C, L = melspec.shape  # B is batchsize, C=80, L is number of melspec frames
    diffusion_steps = torch.randint(T, size=(B,1,1)).cuda()  # randomly sample diffusion steps from 1~T
    z = torch.normal(0, 1, size=melspec.shape).cuda()
    transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    cond_drop_prob = 0.2
    epsilon_theta = net(transformed_X, mouthroi, face_image, diffusion_steps.view(B,1), cond_drop_prob)
    return loss_fn(epsilon_theta, z)



@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    if not os.path.isdir("exp/"):
        os.makedirs("exp/")
        os.chmod("exp/", 0o775)

    num_gpus = torch.cuda.device_count()
    print(f'there are {num_gpus} gpus')
    train_fn = partial(
        distributed_train,
        num_gpus=num_gpus,
        group_name=time.strftime("%Y%m%d-%H%M%S"),
        cfg=cfg,
    )

    if num_gpus <= 1:
        train_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=train_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
