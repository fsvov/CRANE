"""Extract 16kHz mono WAV audio from video files.

Usage:
    python extract_audio.py --dataset mosi \\
        --input data/MOSI/Raw \\
        --output data/MOSI/wav
"""

import os
import argparse
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip


def extract(dataset, input_path, output_path):
    dataset = dataset.upper()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    os.makedirs(output_path, exist_ok=True)

    for folder in tqdm(os.listdir(input_path)):
        in_subdir = os.path.join(input_path, folder)
        out_subdir = os.path.join(output_path, folder)
        if not os.path.isdir(in_subdir):
            continue
        os.makedirs(out_subdir, exist_ok=True)

        for file in os.listdir(in_subdir):
            if not file.endswith(".mp4"):
                continue
            if "-edited" in file:
                continue

            input_file = os.path.join(in_subdir, file)
            output_file = os.path.join(out_subdir, os.path.splitext(file)[0] + ".wav")

            if os.path.exists(output_file):
                continue

            try:
                video = VideoFileClip(input_file)
                audio = video.audio
                resampled = audio.set_fps(16000)
                resampled.write_audiofile(
                    output_file, codec='pcm_s16le', verbose=False, logger=None
                )
                video.close()
            except Exception as exc:
                print(f"  Failed: {input_file} ({exc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio from videos")
    parser.add_argument('--dataset', type=str, default='mosi', help='dataset name')
    parser.add_argument('--input', type=str, required=True,
                        help='path to raw video directory')
    parser.add_argument('--output', type=str, required=True,
                        help='path to output WAV directory')
    args = parser.parse_args()

    extract(args.dataset, args.input, args.output)
