import numpy as np
import pandas as pd
import glob
import subprocess
import os
from tqdm import tqdm
import argparse
import operator
import editdistance

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

    return sample_index / sample_rate + start_time

def get_max_value_from_dict(d):
    return max(d.items(), key=operator.itemgetter(1))[0]

def get_end_bestprediction(i, all_predictions, all_confidences):
    
    '''
    input:
        i: starting index

    output: 
        next_i,
        ending index and best prediction with highest confidence
    '''
    # map prediction to highest confidence
    prediction_confidence_dict = {}
    
    current_subtitle = all_predictions[i]
    current_confidence = all_confidences[i]
    
    prediction_confidence_dict[current_subtitle] = current_confidence
    
    if i+1 >= len(all_predictions):
        next_i = i
        best_prediction = current_subtitle
        return next_i,best_prediction
    
    next_i = i+1
    next_subtitle = all_predictions[next_i]
    next_confidence = all_confidences[next_i]
#     print(f"current {current_subtitle.encode('utf-8')}")
#     print([p.encode('utf-8')  for p in all_predictions[30:]])
    while same_subtitle(current_subtitle, next_subtitle):

        # update best confidence if it is higher than the original one in the dictionary
        if next_subtitle in prediction_confidence_dict and next_confidence > prediction_confidence_dict[next_subtitle]:
            prediction_confidence_dict[next_subtitle] = next_confidence
        elif next_subtitle not in prediction_confidence_dict:
            prediction_confidence_dict[next_subtitle] = next_confidence
            
        next_i += 1
        if next_i >= len(all_predictions):
            break
        next_subtitle = all_predictions[next_i]
        next_confidence = all_confidences[next_i]
    next_i -= 1
    best_prediction = get_max_value_from_dict(prediction_confidence_dict)
    return next_i, best_prediction


# based on heuristic
def same_subtitle(current_subtitle, next_subtitle):
    current_set = set(current_subtitle)
    next_set = set(next_subtitle)
    current_set_len = len(current_set)
    next_set_len = len(next_set)
    intersect_set = current_set & next_set
    intersect_set_len = len(intersect_set)
    

    if intersect_set_len >= 0.7 * current_set_len or intersect_set_len >= 0.7 * next_set_len:
        return True
    else:
        return False

# edit distance based
# def same_subtitle(current_subtitle, next_subtitle):
#     score = editdistance.eval(current_subtitle,next_subtitle)
#     if score < max(len(current_subtitle), len(next_subtitle)) * 0.5:
#         return True
#     else:
#         return False
      
# jaccard distance based    
# def same_subtitle(current_subtitle, next_subtitle):
#     current_set = set(current_subtitle)
#     next_set = set(next_subtitle)
#     current_set_len = len(current_set)
#     next_set_len = len(next_set)
#     intersect_set = current_set & next_set
#     union_set =  current_set | next_set
#     union_set_len = len(union_set)
#     intersect_set_len = len(intersect_set)
#     jaccard_distance = intersect_set_len/ union_set_len
    
#     if jaccard_distance > 0.3:
#         return True
#     else:
#         return False

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
    parser.add_argument("--thread_count", type=int, default=3)
    parser.add_argument("--srts_dir", type=str, required=True)
    parser.add_argument("--csvs_dir", type=str, required=True)
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=False)
    
    args = parser.parse_args()
    
    srts_dir = args.srts_dir
    csvs_dir = args.csvs_dir
    videos_dir = args.videos_dir
    input_csv = args.input_csv
    os.makedirs(args.srts_dir, exist_ok=True)
    if input_csv != None:
        video_csvs = [input_csv]
    else:
        video_csvs = get_all_inputs(csvs_dir)
    
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
        
        # get only subtitles with confidence > 0.7
        df = df.loc[df.confidence >= 0.98, :]
        
        # a string that gathers subtitles for srt output
        subtitles = ''
        subtitle_count = 1
        
        all_predictions = df.prediction.values
        all_confidences = df.confidence.values
        all_frame_indexes = df.id.values

        start_i = 0
        while start_i < len((all_predictions)):

            end_i, current_subtitle = get_end_bestprediction(start_i, all_predictions, all_confidences)
        
            end_frame_index = all_frame_indexes[end_i]
            start_frame_index = all_frame_indexes[start_i]
            
            subtitle_end_time = get_time(end_frame_index, sample_rate, video_start_time) 
            subtitle_start_time = get_time(start_frame_index, sample_rate, video_start_time)  - spf 
                
            subtitles += f'{subtitle_count}\n'
            subtitles += second2timecode(subtitle_start_time) + ' --> ' + second2timecode(subtitle_end_time) + '\n'
            subtitles += current_subtitle + '\n\n'
            start_i = end_i +1
            subtitle_count += 1

        with open(f'{srts_dir}/{video_id}.srt', 'wb') as f:
            f.write(subtitles.encode('utf-8'))
    
    
    
if __name__ == "__main__":
    main()    