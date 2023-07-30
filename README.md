# LipVoicer: Generating Speech from Silent Videos Guided by Lip Reading

<div align="center">

[Paper](https://arxiv.org/pdf/2306.03258) |
[Introduction](#Introduction) |
[Preparation](#Preparation) |
[Benchmark](#Benchmark-evaluation) |
[Inference](#Speech-prediction) |
[Model zoo](#Model-Zoo) |
</div>

## Authors
Yochai Yemini, Aviv Shamsian, Lior Bracha, Sharon Gannot and Ethan Fetaya

## Introduction
Official implementation of LipVoicer, a lip-to-speech method. Given a silent video, we first predict the spoken text using a pre-trained lip-reading network. We then condition a diffusion model on the video and use the extracted text through a classifier-guidance mechanism where a pre-trained ASR serves as the classifier. LipVoicer outperforms multiple lip-to-speech baselines on LRS2 and LRS3, which are in-the-wild datasets with hundreds of unique speakers in their test set and an unrestricted vocabulary.

The lip reading network used in LipVoicer is taken from the [Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) repository.
The ASR system is adapted from [Audio-Visual Efficient Conformer for Robust Speech Recognition](https://github.com/burchim/AVEC/tree/master).

## Installation
1. Clone the repository:
```
git clone https://github.com/yochaiye/LipVoicer.git
cd LipVoicer
```
2. Install the required packages and ffmpeg
```
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
cd ..
```
3. Install `ibug.face_detection`
```
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..
```
4. Install `ibug.face_alignment`
```
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e .
cd ..
```
5. Install [RetinaFace](https://pypi.org/project/retina-face/) or [MediaPipe](https://pypi.org/project/mediapipe/) face tracker
6. Install ctcdecode for the ASR beam search
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .
cd ..
```

## Training
For training LipVoicer on the benchmark datasets, please download [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) or [LRS3](https://mmai.io/datasets/lip_reading/). 
**In all next steps, make sure to adhere to the dataset's structure.**

### Data Preparation
Perform the following steps inside the LipVoicer directory:
1. Extract the audio files from the videos (audio files will be saved in a WAV format)
```
python ...
```
2. Compute the log mel-spectrograms and save them
   ```
   cd dataloaders
   python wav2mel.py dataset.audio_dir=<audio_dir>
   cd ..
   ```
<!---
3. Detect mouth regions in the videos, convert to greyscale and save to `<mouthrois_dir>`
Download option?

### Train
4. Train MelGen
```
CUDA_VISIBLE_DEVICES=<...> python train_melgen.py train.save_dir=<save_dir> \
                                                  dataset.dataset_path=<dataset_path> \
                                                  dataset.audio_dir=<audio_dir> \
                                                  dataset.mouthrois_dir=<mouthrois_dir>
```
5. Finetune the modified ASR, which now includes the diffusion time-step embedding
```
```
--->

## Inference
### Random (In-the-Wild) Video
### For Benchmark Datasets
If you wish to generate audio for all of the test videos of LRS2/LRS3, use the following
```
python generate_full_test_split.py generate.save_dir=<save_dir> \
                                   generate.lipread_text_dir=<lipread_text_dir> \
                                   dataset.dataset_path=<dataset_path> \
                                   dataset.audio_dir=<audio_dir> \
                                   dataset.mouthrois_dir=<mouthrois_dir
```
