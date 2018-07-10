import cv2
import os
import subprocess 
from joblib import Parallel, delayed
import argparse
import glob

INPUT_DIR = 'frames/'
RESULTS_DIR = 'crop_frames/'
os.makedirs(RESULTS_DIR, exist_ok=True)

def crop(img):
    img_height, img_width, _ = img.shape
    crop_height = int(img_height * 0.2)
    crop_img = img[img_height-crop_height:,:,:]
    return crop_img
    
def hist_equal(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def crop_store(f):
    #change_dir
    crop_file_name = RESULTS_DIR + '/' + f.split('/')[1]
    #change name
    crop_file_name = '_crop.'.join(crop_file_name.split('.'))
    
    #check if file exist in the result dir
    if os.path.isfile(crop_file_name):
        return
    
    
    
    img = cv2.imread(f)
    
    crop_img = crop(img)
    
    
    #write to disk
    cv2.imwrite(crop_file_name,crop_img)

def get_input_names():
    '''
    return a list of frame names
    '''
    return glob.glob(INPUT_DIR + '*')

def main():
    image_names = get_input_names()
    # use multi-thread to speed up cropping
    parallel = Parallel(6, backend="threading", verbose=100)

    #download video
    parallel(delayed(crop_store)(image_name) for image_name in image_names)

if __name__ == "__main__":
    main()    
    
    