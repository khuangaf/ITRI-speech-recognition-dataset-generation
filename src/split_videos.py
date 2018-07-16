import glob
import subprocess
from joblib import Parallel, delayed
import argparse
import os
import shutil
import multiprocessing

def video2frames(video_id, video_path, processed_videos_dir, frame_path, sample_rate):
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
        cmd = f'ffmpeg -ss {starting_time} -t {split_duration} -i {path_video} -r {sample_rate} {frame_path}/{video_id}-%07d-{sample_rate}.png'
        subprocess.call(cmd, shell=True)
    
    except Exception as e:
        print(f'Failed to cut videos {video_id}: {e}')

    shutil.move(path_video, processed_videos_dir)
def get_duration(path_video):
    # return the duration of path_video in second
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path_video}'
    return float(subprocess.Popen(cmd.split(), stdout=subprocess.PIPE).communicate()[0].strip())
def path2id(path):
    return path.split('.')[0]

def gather_video_ids(video_path):
    video_ids = [ path2id(f) for f in os.listdir(f'{video_path}/') if '.mp4' in f]
    return video_ids

def main():
    parser = argparse.ArgumentParser("Script for splitting youtube videos")
    parser.add_argument("--thread_count", type=int)
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--processed_videos_dir", type=str, required=True)
    parser.add_argument("--frames_dir", type=str, required=True)
    parser.add_argument("--sample_rate", type=float, required=True)
    
    args = parser.parse_args()
    
    if args.thread_count == None:
        thread_count = multiprocessing.cpu_count() // 4
    else:
        thread_count = args.thread_count
    os.makedirs(args.frames_dir, exist_ok=True)
    # parallel processing to increase speed
    parallel = Parallel(thread_count, backend="threading", verbose=0)
    
    video_ids = gather_video_ids(args.videos_dir)
    
    try: 
        #split video into frames
        parallel(delayed(video2frames)(video_id, video_path=args.videos_dir, processed_videos_dir=args.processed_videos_dir, frame_path=args.frames_dir, sample_rate=args.sample_rate) for video_id in video_ids)
    
    except Exception as e:
        print('Failed to split videos: {}'.format(e))
        
if __name__ == "__main__":
    main()    
