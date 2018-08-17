"""
This script aggregates the results from Google OCR API, and output SRT.

Author: Kung-hsiang, Huang, 2018
"""
import numpy as np
import pandas as pd
import glob
import subprocess
import os
from tqdm import tqdm
import argparse
import operator
import editdistance
from util import *

def get_duration(path_video):
    '''
    Return the duration of path_video in seconds.
    
    Parameters
    ----------
    path_video : str 
        The path of the video 
        e.g.
            path/to/video/abc.mp4
    Returns
    --------
    float
        The length of the video in seconds.
    
    '''
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {path_video}'
    return float(subprocess.Popen(cmd.split(), stdout=subprocess.PIPE).communicate()[0].strip())

def get_sample_rate(frame_id):
    '''
    Return the sample rate from id
    e.g.
        qebmksd-000356-2.0 --> 2.0
    
    '''
    return float(frame_id.split('-')[-1])
    
def get_time(id, sample_rate, start_time):
    '''
    Return the time in the video with regard to the start time in seconds

    Parameters
    ----------
    id: str
        e.g.
            qebmksd-000356-2.0
    sample_rate: float
        Sample rate of the current video
    
    start_time: int
        Start time of the current video
        
    Returns
    --------
    float
        The converted time in seconds.
    
    '''
    # e.g. qebmksd-000356-2.0 --> 356
    sample_index = int(id.split('-')[-2])

    return sample_index / sample_rate + start_time

def get_max_value_from_dict(d):
    '''
    Return the key in a dictionary with the maximum value
    
    '''
    return max(d.items(), key=operator.itemgetter(1))[0]

def get_max_value_from_df(pred_conf_df):
    '''
    Return the key in a dataframe with the maximum confidence value
    
    '''
    
    max_confidence = pred_conf_df.confidence.max()
    
    max_df = pred_conf_df.loc[pred_conf_df.confidence == max_confidence]
    
     
    # highest voting prediction
    result = max_df.prediction.value_counts(ascending=False).index[0]
    return result
    
    

def get_end_bestprediction(i, all_predictions, all_confidences):
    
    '''
    This function starts searching from index i in all predictions. 
    Loop over the predictions until it reachs a prediction that is not the the same subtitle.
    Return the ending index so that the current subtitle corresponds to the time range (start_i, end_i).
    Parameters
    ----------
    i: int
        The starting index for a given subtitle.
        
    all_predictions: list(str)
        A list of text containing all predicted subtitles. (The 'prediction' column of the csv)
    
    all_confidences: list(float)
        A list of float containing all predicted confidences. (The 'confidence' column of the csv)
        
    Returns
    --------
    next_i: int
        The ending index for a given subtitle.
    
    best_prediction: str
        Since the subtitles cover with this starting and ending index can be different a little, we decide which one to represents
    '''
    
    
    current_subtitle = all_predictions[i]
    current_confidence = all_confidences[i]
    

    pred_conf_df = pd.DataFrame(columns=['prediction', 'confidence'])
    
    pred_conf_df.loc[0] = [current_subtitle, current_confidence]
    if i+1 >= len(all_predictions):
        next_i = i
        best_prediction = current_subtitle
        return next_i,best_prediction
    
    next_i = i+1
    next_subtitle = all_predictions[next_i]
    next_confidence = all_confidences[next_i]

    while same_subtitle(current_subtitle, next_subtitle):
            
        pred_conf_df.loc[next_i] = [current_subtitle, current_confidence]
        
        next_i += 1
        if next_i >= len(all_predictions):
            break
        next_subtitle = all_predictions[next_i]
        next_confidence = all_confidences[next_i]
    next_i -= 1

    best_prediction = get_max_value_from_df(pred_conf_df)
    return next_i, best_prediction


# based on heuristic
def same_subtitle(current_subtitle, next_subtitle):
    '''Return true if the two given subtitles are the same (but can tolerate a bit difference)'''
    # convert the two subtitle into set e.g. '我很乖' -> {'我','很','乖'}
    current_set = set(current_subtitle)
    next_set = set(next_subtitle)
    current_set_len = len(current_set)
    next_set_len = len(next_set)
    intersect_set = current_set & next_set
    intersect_set_len = len(intersect_set)
    
    # if any of the two subtitle are of 70% the same with the intersected set return True
    if intersect_set_len >= 0.7 * current_set_len or intersect_set_len >= 0.7 * next_set_len:
        return True
    else:
        return False

def second2timecode(time):
    '''Convert second into time code for srt.'''
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
    parser.add_argument("--srts_dir", type=str, required=True)
    parser.add_argument("--csvs_dir", type=str, required=True)
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=False)
    parser.add_argument("--video_id_file", type=str, required=True)
    args = parser.parse_args()
    
    srts_dir = args.srts_dir
    csvs_dir = args.csvs_dir
    videos_dir = args.videos_dir
    input_csv = args.input_csv
    os.makedirs(args.srts_dir, exist_ok=True)
    
    if input_csv != None:
        video_csvs = [input_csv]
    else:
        video_ids = get_video_id_from_file(args.video_id_file)
        video_csvs = [vid+".csv" for vid in video_ids]
    
    for video_csv in tqdm(video_csvs):
        
        video_id = video_csv.split('.csv')[0]
        print(f'{videos_dir}/{video_id}.mp4')
        if not os.path.isfile(f'{videos_dir}/{video_id}.mp4'):
            continue

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
        
        # get only subtitles with confidence > 0.98
        df = df.loc[df.confidence >= 0.98, :]
        
        # a string that gathers subtitles for srt output
        subtitles = ''
        subtitle_count = 1
        
        # get the confidences and predictions from dataframe
        all_predictions = df.prediction.values
        all_confidences = df.confidence.values
        all_frame_indexes = df.id.values

        start_i = 0
        while start_i < len((all_predictions)):

            # get the end_index given a subtitle of the start_index
            end_i, current_subtitle = get_end_bestprediction(start_i, all_predictions, all_confidences)
        
            end_frame_index = all_frame_indexes[end_i]
            start_frame_index = all_frame_indexes[start_i]
            
            subtitle_end_time = get_time(end_frame_index, sample_rate, video_start_time) 
            subtitle_start_time = get_time(start_frame_index, sample_rate, video_start_time)  - spf 
                
            subtitles += f'{subtitle_count}\n'
            subtitles += second2timecode(subtitle_start_time) + ' --> ' + second2timecode(subtitle_end_time) + '\n'
            subtitles += current_subtitle + '\n\n'
            
            # let start_index = previous end_index + 1
            start_i = end_i +1
            subtitle_count += 1

        with open(f'{srts_dir}/{video_id}.srt', 'wb') as f:
            f.write(subtitles.encode('utf-8'))
    
    
    
if __name__ == "__main__":
    main()    