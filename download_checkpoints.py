# this file is an adapated version of https://github.com/burchim/AVEC/tree/master, licensed
# under https://github.com/burchim/AVEC/blob/master/LICENSE


import gdown
import os

# Gdrive Ids
repo_files = {
    "Audio-Only Efficient Conformer CTC checkpoint": {
        "gdrive_id": "1TPlqoSythY90xJrZRquJmMUwK4sVeAHc",
        "dir_path": "ASR/callbacks/LRS23/AO/EffConfCTC",
        "filename": "checkpoints_swa-equal-200-210.ckpt"
    },
    "TransformerLM checkpoint": {
        "gdrive_id": "1PSo4ZQIZPWEI_S5LHkJBo0gYhQpWzRnh",
        "dir_path": "ASR/callbacks/LRS23/LM/GPT-Small",
        "filename": "checkpoints_epoch_10_step_2860.ckpt"
    },
    "TransformerLM pre-trained checkpoint": {
        "gdrive_id": "1V4-GMlh8dh0LXYniZb72pYqgBkVdpND6",
        "dir_path": "ASR/callbacks/LibriSpeechCorpus/GPT-Small",
        "filename": "checkpoints_epoch_13_step_512924.ckpt"
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
}

# Download pretrained models checkpoints
for key, value in repo_files.items():
    
    # Print
    print("Download {}".format(key))
    
    # Create model callback directory
    if not os.path.exists(value["dir_path"]):
      os.makedirs(value["dir_path"])
    
    # Download
    gdown.download("https://drive.google.com/uc?id=" + value["gdrive_id"], os.path.join(value["dir_path"], value["filename"]), quiet=False)
    print()