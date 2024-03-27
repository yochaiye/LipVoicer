
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
def main(cfg):

    stft = STFT(**cfg.audio)

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
        torch.save(melspectrogram, new_filepath)
    print(f"max_val={max_val},\n min_val={min_val}")


if __name__ == "__main__":
    main()
