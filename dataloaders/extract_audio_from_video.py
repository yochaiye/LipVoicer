from multiprocessing import Pool
import os
import subprocess
import glob
from tqdm import tqdm
from argparse import ArgumentParser
from functools import partial

def extract_audio(v, outdir, ds_dir):
    outfile = v.replace(ds_dir, outdir).replace(".mp4", ".wav")
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    cmd = f"ffmpeg -loglevel error -y -i {v} -map 0:a {outfile}"
    os.system(cmd)


def main(args):
    video_filenames = glob.glob(os.path.join(args.ds_dir, args.split, "**/*.mp4"), recursive=True)
    _extract_audio = partial(extract_audio, outdir=args.outdir)
    with Pool(4) as p:
        r = list(tqdm(p.imap(_extract_audio, video_filenames), total=len(video_filenames)))

if __name__ == "__main__":
    parser = ArgumentParser('')
    parser.add_argument('--ds_dir', help='path to root directory of the dataset')
    parser.add_argument('--split', help='which split (e.g. pretrain, train, test etc)')
    parser.add_argument('--out_dir', help='path to output directory where the audio files will be saved')
    args = parser.parse_args()
    main(args)