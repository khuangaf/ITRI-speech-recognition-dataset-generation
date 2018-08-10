import cv2
import numpy as np
import pandas as pd
import os
import skimage
from skimage.filters import try_all_threshold, threshold_minimum
from glob import glob 
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import re
import argparse
from joblib import Parallel, delayed
from util import *

def run_bgs(all_frame_names, batch_id, batch_size, video_id):
    
    
    current_frames=[]
    current_frame_names = all_frame_names[batch_id * batch_size: (batch_id+1) * batch_size]
    for cfn in current_frame_names:
        image_fn = f'{cfn}/image.png'
        img= cv2.imread(image_fn).astype(np.float32)
        current_frames.append(img)
    current_frames = np.array(current_frames)
    next_frames = current_frames[1:]
    current_frames = current_frames[:-1]
    current_diff = np.mean(np.mean(np.square(current_frames - next_frames), axis=0), axis=-1)

    mm = MinMaxScaler(feature_range=(0, 255))
    scale_diff = mm.fit_transform(current_diff).astype(np.uint8)



    diff_filter = scale_diff > 5
    zeros = np.zeros(img.shape[:2])
    for cfn in current_frame_names:

        tta_fn = f'{cfn}/tta.png'
        image_id = cfn.split('/')[-1]
        os.makedirs(f'{results_dir}/{video_id}/{image_id}', exist_ok=True)
        image_path = f'{results_dir}/{video_id}/{image_id}/tta.png'
        filter_path = f'{results_dir}/{video_id}/{image_id}/dif.png'


        img= cv2.imread(tta_fn)
        cv2.imwrite(image_path, img)

        for c in range(3):
            img[:,:,c] = np.where(diff_filter, img[:,:,c], zeros)

        cv2.imwrite(filter_path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script for performing ocr on frames")
    parser.add_argument("--inputs_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--video_id_file", type=str, required=True)
    args = parser.parse_args()
    
    

    inputs_dir = args.inputs_dir
    results_dir = args.results_dir
    os.makedirs(args.results_dir, exist_ok=True)
    
    video_ids = get_video_id_from_file(args.video_id_file)

    
    for video_id in tqdm(video_ids):
        
        batch_size = 20
        
        all_frame_names = glob(f"{inputs_dir}/{video_id}/*")
        
        total_batches = len(all_frame_names) // batch_size +1
        parallel = Parallel(40, backend="threading", verbose=0)
        parallel(delayed(run_bgs)(all_frame_names, batch_id, batch_size, video_id) for batch_id in range(total_batches))

    


    