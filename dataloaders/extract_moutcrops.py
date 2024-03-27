#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from argparse import ArgumentParser
import os
import glob
import pickle
import numpy as np
# from pipelines.detectors.mediapipe.detector import LandmarksDetector
from pipelines.data.data_module import AVSRDataLoader
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial


def get_mouth_crop(videos_dir, landmarks_filename, out_dir, dataloader):
    with open(landmarks_filename, "rb") as f:
        landmarks = pickle.load(f)
    video_id = landmarks_filename.split("/")[-2:]
    video_id = ("/").join(video_id).replace(".pkl", "")
    video_filename = os.path.join(videos_dir, video_id+".mp4")
    try:
        data = dataloader.load_data(video_filename, landmarks)
    except:
        return
    outdir = os.path.join(out_dir, video_id.split("/")[-2])
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, video_id.split("/")[-1]+".npz")
    np.savez(outfile, data=data)
    print(f"The mouth images have been cropped and saved to {outfile}")
    return
    

def main(args):
    dataloader = AVSRDataLoader(modality="video", speed_rate=1, transform=False, detector='retinaface', convert_gray=True)

    landmarks_filenames = glob.glob(os.path.join(args.landmarks_dir, "**/*.pkl"), recursive=True)

    _get_mouth_crop = partial(get_mouth_crop, 
                              videos_dir=args.videos_dir, 
                              out_dir=args.out_dir,
                              dataloader=dataloader)
    with Pool(processes=20) as pool:
        r = list(tqdm(pool.imap(_get_mouth_crop, landmarks_filenames), total=len(landmarks_filenames)))


if __name__ == "__main__":
    parser = ArgumentParser('')
    parser.add_argument('--landmarks_dir', type=str, help='directory with landmarks')
    parser.add_argument('--videos_dir', type=str, help='directory with videos in .mp4 format')
    parser.add_argument('--out_dir', type=str, help='directory where the output mouth crops will be saved')
    main()
