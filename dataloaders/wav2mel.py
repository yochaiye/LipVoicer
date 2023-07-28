# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************\
import os
import torch
import torch.utils.data
from scipy.io.wavfile import read
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from glob import glob
from pathlib import Path

# We're using the audio processing from TacoTron2 to make sure it matches
from stft import TacotronSTFT, normalise_mel


def get_all_filenames(data_path):
    """
    Load all .wav files in data_path
    """
    files = glob(os.path.join(data_path, '**/*.wav'), recursive=True)
    return files

def load_wav_to_torch(full_path):
    """
    Loads wavdata into torch array
    """
    sampling_rate, data = read(full_path)
    return torch.from_numpy(data).float(), sampling_rate


class STFT():
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, filter_length, hop_length, win_length, sampling_rate, mel_fmin, mel_fmax):
        
        self.stft = TacotronSTFT(filter_length=filter_length,
                                 hop_length=hop_length,
                                 win_length=win_length,
                                 sampling_rate=sampling_rate,
                                 mel_fmin=mel_fmin, mel_fmax=mel_fmax)

    def get_mel(self, audio):
        audio = audio.unsqueeze(0)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)
        return melspec

# ===================================================================
# Takes directory of clean audio and makes directory of spectrograms

@hydra.main(version_base=None, config_path="../configs/", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg.dataset))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    stft = STFT(**cfg.spec)

    filepaths = get_all_filenames(cfg.dataset["audio_dir"])
    filepaths = sorted(filepaths)
    
    max_val = 0
    min_val = 0
    for filepath in tqdm(filepaths):
        audio, sr = load_wav_to_torch(filepath)
        audio = audio / 1.1 / audio.abs().max()     # normalise max amplitude to be ~0.9
        melspectrogram = stft.get_mel(audio)
        if melspectrogram.max() > max_val:
            max_val = melspectrogram.max()
        if melspectrogram.min() < min_val:
            min_val = melspectrogram.min()

        new_filepath = filepath + '.spec'
        os.makedirs(Path(new_filepath).parent, exist_ok=True)
        # print(new_filepath)
        torch.save(melspectrogram, new_filepath)
    print(f"max_val={max_val},\n min_val={min_val}")


if __name__ == "__main__":
    main()
