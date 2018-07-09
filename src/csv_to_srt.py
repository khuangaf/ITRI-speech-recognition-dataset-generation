import numpy as np
import pandas as pd
import glob
import subprocess
import os
from tqdm import tqdm
import argparse

def get_duration(path_video):
    # return the duration of path_video in second
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path_video}'
    return float(subprocess.Popen(cmd.split(), stdout=subprocess.PIPE).communicate()[0].strip())

def get_sample_rate(frame_id):
    return float(frame_id.split('-')[-1])
    
def get_all_inputs(dir):
    return [fn for fn in os.listdir(dir) if 'csv' in fn]
    
def get_time(id, sample_rate, start_time):
    sample_index = int(id.split('-')[-2])
#     print(sample_rate)
    return sample_index / sample_rate + start_time


def second2timecode(time):
    hours = time // 3600
    time -= 3600 * hours
    minutes = time // 60
    time -= 60 * minutes
    secs = np.floor(time % 60)
    msecs = (time % 60 - secs) * 1000
    timecode = "%02d:%02d:%02d,%03d" % (hours, minutes, secs, msecs)
    return timecode

def main():
    parser = argparse.ArgumentParser("Script for processing csv to srt")
    parser.add_argument("--thread-count", type=int, default=3)
    parser.add_argument("--srts-dir", type=str, required=True)
    parser.add_argument("--csvs-dir", type=str, required=True)
    parser.add_argument("--videos-dir", type=str, required=True)
    args = parser.parse_args()
    
    srts_dir = args.srts_dir
    csvs_dir = args.csvs_dir
    videos_dir = args.videos_dir
    os.makedirs(args.srts_dir, exist_ok=True)
    video_csvs = get_all_inputs(csvs_dir)
    for video_csv in tqdm(video_csvs):
        video_id = video_csv.split('.csv')[0]
#         print(f'{csvs_dir}/{video_csv}')
        df = pd.read_csv(f'{csvs_dir}/{video_csv}')
        first_sample = df.id.values[0]
        sample_rate = get_sample_rate(first_sample)
        path_video = f'{videos_dir}/{video_id}.mp4'
        duration = get_duration(path_video)
        
        current_subtitle= None
        # seconds per frame
        
        spf = 1.0 / sample_rate
        
        # records the cut-off of the video
        video_start_time = 0.2 * duration  - spf*2
        subtitle_start_time = video_start_time
        # get only subtitles with confidence > 0.95
        df = df.loc[df.confidence >= 0.95, :].set_index('id')
        
        # a string that gathers subtitles for srt output
        subtitles = ''
        subtitle_count = 1
        for index, row in df.iterrows():
            if current_subtitle == None:
                
                current_subtitle = row.prediction
#                 continue
            elif current_subtitle == row.prediction:
                subtitle_end_time = get_time(index, sample_rate, video_start_time) 
            elif current_subtitle != row.prediction:
                
                subtitles += f'{subtitle_count}\n'
                subtitles += second2timecode(subtitle_start_time) + ' --> ' + second2timecode(subtitle_end_time) + '\n'
                subtitles += current_subtitle + '\n\n'
                current_subtitle = row.prediction
                subtitle_start_time = get_time(index, sample_rate, video_start_time) - spf
                subtitle_end_time = get_time(index, sample_rate, video_start_time) 
                subtitle_count +=1

        with open(f'{srts_dir}/{video_id}.srt', 'wb') as f:
            f.write(subtitles.encode('utf-8'))
    
    
    
if __name__ == "__main__":
    main()    