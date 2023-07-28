# this file is an adapated version https://github.com/albertfgu/diffwave-sashimi, licensed
# under https://github.com/albertfgu/diffwave-sashimi/blob/master/LICENSE


import os
import time
# import warnings
# warnings.filterwarnings("ignore")

from functools import partial
import multiprocessing as mp

import matplotlib.image
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from models.model_builder import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from dataloaders.dataset_lipvoicer import LipVoicerDataset
from dataloaders.stft import denormalise_mel

from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory

def sampling(net, diffusion_hyperparams, guidance_factor, condition=None):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the model
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated melspec(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T

    print('begin sampling, total number of reverse steps = %s' % T)

    mouthroi, face_image = condition
    x = torch.normal(0, 1, size=(mouthroi.shape[0], 80, mouthroi.shape[2]*4)).cuda()
    with torch.no_grad():
        for t in tqdm(range(T-1, -1, -1)):
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net(x, mouthroi, face_image, diffusion_steps, cond_drop_prob=0)   # predict \epsilon according to \epsilon_\theta
            epsilon_theta_uncond = net(x, mouthroi, face_image, diffusion_steps, cond_drop_prob=1)
            epsilon_theta = (1+guidance_factor) * epsilon_theta - guidance_factor * epsilon_theta_uncond

            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()  # add the variance term to x_{t-1}
    return x


@torch.no_grad()
def generate(
        rank,
        diffusion_cfg,
        model_cfg,
        dataset_cfg,
        save_dir,
        ckpt_iter="max",
        name=None,
        n_samples=None,
        guidance_factor=0,
    ):
    """
    Generate melspectrograms based on lips movement

    Parameters:
    output_directory (str):         checkpoint path
    n_samples (int):                number of samples to generate, default is 4
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded;
                                    automatically selects the maximum iteration if 'max' is selected
    mel_path, mel_name (str):       condition on spectrogram "{mel_path}/{mel_name}.wav.pt"
    # dataloader:                     condition on spectrograms provided by dataloader
    """

    if rank is not None:
        print(f"rank {rank} {torch.cuda.device_count()} GPUs")
        torch.cuda.set_device(rank % torch.cuda.device_count())

    local_path, checkpoint_directory = local_directory(name, model_cfg, diffusion_cfg, save_dir, 'checkpoint')

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams  = calc_diffusion_hyperparams(**diffusion_cfg, fast=True)  # dictionary of all diffusion hyperparameters

    # predefine model
    builder = ModelBuilder()
    net_lipreading = builder.build_lipreadingnet()
    net_facial = builder.build_facial(fc_out=128, with_fc=True)
    net_diffwave = builder.build_diffwave_model(model_cfg)
    net = AudioVisualModel((net_lipreading, net_facial, net_diffwave)).cuda()
    print_size(net)
    net.eval()

    # load checkpoint
    print('ckpt_iter', ckpt_iter)
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(checkpoint_directory)
    ckpt_iter = int(ckpt_iter)

    try:
        model_path = os.path.join(checkpoint_directory, '{}.pkl'.format(ckpt_iter))
        checkpoint = torch.load(model_path, map_location='cpu')
        net.load_state_dict(checkpoint['model_state_dict'])
        print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    except:
        raise Exception('No valid model found')

    # Add checkpoint number to output directory
    # output_directory = os.path.join(output_directory, str(ckpt_iter))
    output_directory = checkpoint_directory.replace('checkpoint', 'generated_mels')
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("saving to output directory", output_directory)

    ds_name = dataset_cfg.pop('_name_')
    dataset = LipVoicerDataset('test', **dataset_cfg)
    dataset_indices = torch.arange(n_samples)
    groundtruth_melspec, mouthroi, face_image = [], [], []
    for i in dataset_indices:
        _gt_melspec, _, _mouthroi, _face_image, _, _ = dataset[i]
        _gt_melspec = denormalise_mel(_gt_melspec)
        groundtruth_melspec.append(_gt_melspec.unsqueeze(0))
        mouthroi.append(_mouthroi.unsqueeze(0))
        face_image.append(_face_image.unsqueeze(0))
    dataset_cfg['_name_'] = ds_name

    print(f'begin generating melspectrograms | {n_samples} samples')

    # inference
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    generated_melspec = []

    for i in range(n_samples):
        _melspec = sampling(
            net,
            diffusion_hyperparams,
            guidance_factor,
            condition=(mouthroi[i].cuda(), face_image[i].cuda()),
        )
        generated_melspec.append(denormalise_mel(_melspec))

    end.record()
    torch.cuda.synchronize()
    print('generated {} samples at iteration {} in {} seconds'.format(n_samples,
        ckpt_iter,
        int(start.elapsed_time(end)/1000)))

    # save audio to .wav
    for i in range(n_samples):
        # save as image
        outfile = f'{ckpt_iter // 1000}k_{i}_w={guidance_factor}.png'
        matplotlib.image.imsave(os.path.join(output_directory, outfile),
                                generated_melspec[i].squeeze(0).cpu().numpy()[::-1])     # squeeze is for getting rid of the batch dim
        matplotlib.image.imsave(os.path.join(output_directory, f'gt{i}.png'),
                                groundtruth_melspec[i].squeeze(0).numpy()[::-1])

        # save as file
        outfile = outfile.replace(".wav", ".wav.spec")
        torch.save(generated_melspec[i].squeeze(0).cpu(),                   # squeeze is for getting rid of the batch dim
                                os.path.join(output_directory, outfile).replace(".png", ".wav.spec"))
        torch.save(groundtruth_melspec[i].squeeze(0),
                                os.path.join(output_directory, f'gt{i}.wav.spec'))
        
    print('saved generated samples at iteration %s' % ckpt_iter)
    return generated_melspec


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    num_gpus = torch.cuda.device_count()
    generate_fn = partial(
        generate,
        name=cfg.train['name'],
        diffusion_cfg=cfg.diffusion,
        model_cfg=cfg.model,
        dataset_cfg=cfg.dataset,
        **cfg.generate,
    )

    if num_gpus <= 1:
        generate_fn(0)
    else:
        mp.set_start_method("spawn")
        processes = []
        for i in range(num_gpus):
            p = mp.Process(target=generate_fn, args=(i,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


if __name__ == "__main__":
    main()
