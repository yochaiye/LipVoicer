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
from asr_models import asr_guidance_net, tokenizer, decoder
from dataloaders.dataset_lipvoicer import LipVoicerDataset
from dataloaders.stft import denormalise_mel

from utils import find_max_epoch, print_size, calc_diffusion_hyperparams, local_directory


def sampling(net, diffusion_hyperparams,
            guidance_factor, condition=None, 
            asr_guidance_net=None,
            asr_scale=None,
            asr_start=None,
            guidance_text=None,
            tokenizer=None,
            decoder=None
            ):
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

    # tokenize text
    if asr_guidance_net is not None:
        text_tokens = torch.LongTensor(tokenizer.encode(guidance_text))
        text_tokens = text_tokens.unsqueeze(0).cuda()
        # asr_guid_start = 300 # 120 # 150

    print('begin sampling, total number of reverse steps = %s' % T)

    mouthroi, face_image = condition
    x = torch.normal(0, 1, size=(mouthroi.shape[0], 80, mouthroi.shape[2]*4)).cuda()
    with torch.no_grad():
        for t in range(T-1, -1, -1):
            diffusion_steps = (t * torch.ones((x.shape[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = net(x, mouthroi, face_image, diffusion_steps, cond_drop_prob=0)   # predict \epsilon according to \epsilon_\theta
            epsilon_theta_uncond = net(x, mouthroi, face_image, diffusion_steps, cond_drop_prob=1)
            epsilon_theta = (1+guidance_factor) * epsilon_theta - guidance_factor * epsilon_theta_uncond
            
            if asr_guidance_net is not None and t <= asr_start:
                with torch.enable_grad():
                    length_input = torch.tensor([x.shape[2]]).cuda()
                    inputs = x.detach().requires_grad_(True), length_input
                    targets = text_tokens, torch.tensor([text_tokens.shape[1]]).cuda()
                    asr_guidance_net.device = torch.device("cuda")
                    batch_losses = asr_guidance_net.forward_model(inputs, diffusion_steps, targets, compute_metrics=True, verbose=0)[0]
                    asr_grad = torch.autograd.grad(batch_losses["loss"], inputs[0])[0]
                    asr_guidance_net.device = torch.device("cpu")
                grad_normaliser = torch.norm(epsilon_theta / torch.sqrt(1 - Alpha_bar[t])) / torch.norm(asr_grad)
                epsilon_theta = epsilon_theta + torch.sqrt(1 - Alpha_bar[t]) * asr_scale * grad_normaliser * asr_grad

            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])  # update x_{t-1} to \mu_\theta(x_t)
            if t > 0:
                x = x + Sigma[t] * torch.normal(0, 1, size=x.shape).cuda()  # add the variance term to x_{t-1}
            # x = x.clip(-1, 1.5)
            
            # if t % 10 == 0:
            #     if asr_guidance_net is not None and t <= asr_start:
            #         inputs = x, length_input
            #         outputs_ao = asr_guidance_net(inputs, diffusion_steps)["outputs"]
            #         preds_ao = decoder(outputs_ao)[0]
            #         print(preds_ao)

    return x


@torch.no_grad()
def generate(
        rank,
        diffusion_cfg,
        model_cfg,
        dataset_cfg,
        dataset_div,
        ckpt_iter="max",
        name=None,
        guidance_factor=0,
        asr_scale=1.1,
        asr_start=250,
        save_dir=None,
        lipread_text_dir=None,
        **kwargs
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
    output_directory = checkpoint_directory.replace('checkpoint', 'generated_mels')
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("saving to output directory", output_directory)

    # Add checkpoint number to output directory
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)

    ds_name = dataset_cfg.pop('_name_')
    dataset = LipVoicerDataset('test', **dataset_cfg)
    

    guidance_dir_name = f'w={guidance_factor}'
    guidance_dir_name += f'_asr_scale={asr_scale}_asr_start={asr_start}'
    _output_directory = os.path.join(output_directory, guidance_dir_name)
    os.makedirs(_output_directory, exist_ok=True)
    print("saving to output directory", _output_directory)

    # Generate samples on several gpus in parallel
    if dataset_div is not None:
        dataiter = list(range(dataset_div, len(dataset), 4))
    else:   # use a single gpu
        dataiter = list(range(len(dataset)))
    print(f"Generating from sample {dataiter[0]} to {dataiter[-1]}")
    for i in tqdm(dataiter):
        gt_melspec, _, mouthroi, face_image, gt_text, video_id = dataset[i]
        with open(os.path.join(lipread_text_dir, video_id+".txt"), 'r') as f:
            text = f.readline()
        gt_melspec = denormalise_mel(gt_melspec)
        gt_melspec = gt_melspec.unsqueeze(0)
        mouthroi = mouthroi.unsqueeze(0)        # add batch dimension
        face_image = face_image.unsqueeze(0)

        # inference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        melspec = sampling(net, 
                        diffusion_hyperparams,
                        guidance_factor,
                        condition=(mouthroi.cuda(), face_image.cuda()),
                        asr_guidance_net=asr_guidance_net,
                        asr_scale=asr_scale,
                        asr_start=asr_start,
                        guidance_text=text,
                        tokenizer=tokenizer,
                        decoder=decoder
                        )
        melspec = denormalise_mel(melspec)
        end.record()
        torch.cuda.synchronize()
        print('generated {} in {} seconds'.format(video_id, int(start.elapsed_time(end)/1000)))
        
        # save as image
        video_dir = video_id.split('/')[0]
        os.makedirs(os.path.join(_output_directory, video_dir), exist_ok=True)
        matplotlib.image.imsave(os.path.join(_output_directory, video_id+'.png'),
                                melspec.squeeze(0).cpu().numpy()[::-1])     # squeeze is for getting rid of the batch dim
        matplotlib.image.imsave(os.path.join(_output_directory, video_id+'_gt.png'),
                                gt_melspec.squeeze(0).numpy()[::-1])

        # save as file
        torch.save(melspec.squeeze(0).cpu(),                   # squeeze is for getting rid of the batch dim
                                os.path.join(_output_directory, video_id + '.wav.spec'))
        torch.save(gt_melspec.squeeze(0),
                                os.path.join(_output_directory, video_id + '_gt.wav.spec'))
        
        # save text
        text_filename = os.path.join(_output_directory, video_id+'.txt')
        with open(text_filename, 'w') as f:
            f.write("gt       :  " + gt_text+"\n")
            f.write("lipreader:  " + text)
    dataset_cfg['_name_'] = ds_name

        
    return


@hydra.main(version_base=None, config_path="configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    num_gpus = torch.cuda.device_count()
    generate_fn = partial(
        generate,
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
