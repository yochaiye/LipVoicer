# this file is an adapated version of https://github.com/burchim/AVEC/tree/master, licensed
# under https://github.com/burchim/AVEC/blob/master/LICENSE


import gdown
import os
from zipfile import ZipFile

# Gdrive Ids
repo_files = {
    "MelGen LRS2 checkpoint": {
        "gdrive_id": "1gTIpxaMx31ZUhPd2jW8Bt8u4QknieO31",
        "dir_path": "exp/LRS2/wnet_h512_d12_T400_betaT0.02/checkpoint",
        "filename": "1000000.pkl"
    },
    "MelGen LRS3 checkpoint": {
        "gdrive_id": "1MmmPGsrRyVH9hyuYc2HC625G6mFgQM_G",
        "dir_path": "exp/LRS3/wnet_h512_d12_T400_betaT0.02/checkpoint",
        "filename": "1000000.pkl"
    },
    "ASR LRS2 checkpoint": {
        "gdrive_id": "1adeCf4NzhshJVU-JndlKpC34rRwJOQ2B",
        "dir_path": "ASR/callbacks/LRS23/AO/EffConfCTC",
        "filename": "checkpoints_ft_lrs2.ckpt"
    },
    "ASR LRS3 checkpoint": {
        "gdrive_id": "18F1xsk0A4cqoxlOE58LB5Rl6pBoCftCL",
        "dir_path": "ASR/callbacks/LRS23/AO/EffConfCTC",
        "filename": "checkpoints_ft_lrs3.ckpt"
    },
    "TransformerLM checkpoint": {
        "gdrive_id": "1PSo4ZQIZPWEI_S5LHkJBo0gYhQpWzRnh",
        "dir_path": "ASR/callbacks/LRS23/LM/GPT-Small",
        "filename": "checkpoints_epoch_10_step_2860.ckpt"
    },
    "Tokenizer model": {
        "gdrive_id": "1u3U3aHaTWvR_NTftkUGv1JXkxpX1pkOL",
        "dir_path": "ASR/media",
        "filename": "tokenizerbpe256.model"
    },
    "TokenizerLM model": {
        "gdrive_id": "1zKp376kItVhceTFSi2_-EMG3oeYbSC0U",
        "dir_path": "ASR/media",
        "filename": "tokenizerbpe1024.model"
    },
    "6gramLM": {
        "gdrive_id": "1l71jUmRdQMFO2AVezxweENpZgdvL7TyD",
        "dir_path": "ASR/media",
        "filename": "6gram_lrs23.arpa"
    },
    "HiFi-GAN checkpoint": {
        "gdrive_id": "1h0gcgifwe5HVM76rlREHj1daBNItWh7e",
        "dir_path": "hifi_gan",
        "filename": "g_02400000"
    },
    "Lip reading checkpoint": {
        "gdrive_id": "1t8RHhzDTTvOQkLQhmK1LZGnXRRXOXGi6",
        "dir_path": "mouthroi_processing/benchmarks/LRS3/models",
        "filename": "LRS3_V_WER19.1.zip"
    },
    "Lip reading LM": {
        "gdrive_id": "1g31HGxJnnOwYl17b70ObFQZ1TSnPvRQv",
        "dir_path": "mouthroi_processing/benchmarks/LRS3/language_models",
        "filename": "lm_en_subword.zip"
    },
}

# Download pretrained models checkpoints
for key, value in repo_files.items():
    
    # Print
    print("Download {}".format(key))
    
    # Create model callback directory
    if not os.path.exists(value["dir_path"]):
      os.makedirs(value["dir_path"], exist_ok=True)
    
    # Download
    gdown.download("https://drive.google.com/uc?id=" + value["gdrive_id"], os.path.join(value["dir_path"], value["filename"]), quiet=False)
    print()
    if ".zip" in value["filename"]:
       print('Unzipping model')
       with ZipFile(os.path.join(value["dir_path"], value["filename"])) as z:
          z.extractall(path=os.path.join(value["dir_path"]))