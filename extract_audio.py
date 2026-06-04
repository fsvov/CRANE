from moviepy.editor import *
import os
import argparse
import cv2
import time
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
import subprocess

def extract(dataset):
    # Set input and output path
    dataset = dataset.upper()
    input_path = f'your input path' #e.g. '...data/{dataset}/Raw'
    output_path = (f'your output path') #'...data/{dataset}/wav'
    if os.path.exists(input_path):
        print("Path exists!")
    else:
        print("Path does NOT exist!")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for folder in tqdm(os.listdir(input_path)):
        
        input_subdirectory_path = os.path.join(input_path, folder)
        output_subdirectory_path = os.path.join(output_path, folder)
        if not os.path.exists(output_subdirectory_path):
            os.makedirs(output_subdirectory_path)
        
        for file in os.listdir(input_subdirectory_path):
            if file.split(".")[-1] != "mp4" or file.split(".")[1] != "mp4":
                continue
            input_file_path = os.path.join(input_subdirectory_path, file)
            output_file_path = os.path.join(output_subdirectory_path, file)
            if os.path.exists(input_file_path.replace(".mp4", "-edited.mp4")):
                continue
            # Load the video file
            video = VideoFileClip(input_file_path)
            # Extract the audio from the video
            audio = video.audio
            # Set the desired sampling rate
            desired_sampling_rate = 16000  # Replace this value with your desired sampling rate
            # Resample the audio to the desired sampling rate
            resampled_audio = audio.set_fps(desired_sampling_rate)
            if "-edited.mp4" in output_file_path:
                output_file_path = output_file_path.replace("-edited.mp4", ".mp4")
            output_file_path = output_file_path.split(".")[0] + '.wav'
            try:
                # Save the extracted and resampled audio to a WAV file
                resampled_audio.write_audiofile(output_file_path, codec='pcm_s16le', verbose=False, logger=None)
            except:
                print(input_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sims', help='dataset name')
    args = parser.parse_args()

    extract(args.dataset)