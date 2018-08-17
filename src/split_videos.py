"""

This script splits the downloaded video into frames.

Author: Kung-hsiang, Huang, 2018
"""
import glob
import subprocess
from joblib import Parallel, delayed
import argparse
import os
import shutil
import multiprocessing
from util import *

def video2frames(video_id, video_path, processed_videos_dir, frame_path, sample_rate):
    """
    Execute shell command which calls ffmpeg to split video into frames.

    Parameters
    ----------
    video_id : str
        The video id of a video
    video_path: str
        The directory storing the videos
    processed_video_path: str
        The directory storing videos that have been split.
    sample_rate: int
        The sample rate for splitting the video.
        
    """
    
    #concat path and video id
    path_video = video_path + video_id + '.mp4'
    video_duration = get_duration(path_video)
    
    # split only the main part of the video
    starting_time = 0.2 * video_duration
    split_duration = 0.6 * video_duration
    
    try:
        #-loglevel panic: slience
        #-hwaccel vdpau: hardware acceleration with GPU
        # -ss starting time
        # -t duration
        cmd = f'ffmpeg -ss {starting_time} -t {split_duration} -i {path_video} -r {sample_rate} {frame_path}/{video_id}-%07d-{sample_rate}.png'.split(" ")
        subprocess.run(cmd)
        shutil.move(path_video, f"{processed_videos_dir}/{video_id}.mp4")
    except Exception as e:
        print(f'Failed to cut videos {video_id}: {e}')

    
def get_duration(path_video):
    
    """
    Get the duration of a video

    Parameters
    ----------
    path_video : str
        The path of the video (path+file name)
    
    Returns
    --------
    the duration of path_video in second
    """
    
    #execute the command
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path_video}'
    return float(subprocess.Popen(cmd.split(), stdout=subprocess.PIPE).communicate()[0].strip())

def path2id(path):
    """
    Extract video id from path 
    e.g. ADFBSDFG.mp4 --> ADFBSDFG

    Parameters
    ----------
    path : str
        The path of the video (only file name)
    
    Returns
    --------
    str
        The extracted video id
    """
    return path.split('.')[0]



def main():
    parser = argparse.ArgumentParser("Script for splitting youtube videos")
    parser.add_argument("--thread_count", type=int)
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--processed_videos_dir", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--video_id_file", type=str, required=True)
    
    args = parser.parse_args()
    
    # get all video_ids
    video_ids = get_video_id_from_file(args.video_id_file)
    
    # *********WE HAVE TO SET (# OF THREADS) = (LENGTH OF THE VIDEOS) SO THAT THE PROGRAM PROCESS ALL THE FILES******
    if args.thread_count == None:
        thread_count = len(video_ids)
    else:
        thread_count = args.thread_count
    
    os.makedirs(args.frames_dir, exist_ok=True)
    
    # parallel processing to increase speed
    parallel = Parallel(thread_count, backend="threading", verbose=0)

    try: 
        #split video into frames
        parallel(delayed(video2frames)(video_id, video_path=args.videos_dir, processed_videos_dir=args.processed_videos_dir, frame_path=args.frames_dir, sample_rate=args.sample_rate) for video_id in video_ids)
    
    except Exception as e:
        print('Failed to split videos: {}'.format(e))
        
if __name__ == "__main__":
    main()    
