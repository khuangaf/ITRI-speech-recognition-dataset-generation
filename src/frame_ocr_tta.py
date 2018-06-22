from base64 import b64encode
from os import makedirs
from os.path import join, basename
from sys import argv
import json
import requests
import re
import cv2
import numpy as np
import io
import pandas as pd

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
import os
import operator
import glob
from tqdm import tqdm
from time import gmtime
import argparse

ENDPOINT_URL = 'https://vision.googleapis.com/v1p1beta1/images:annotate'


os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/home/ashutosh/ITRI-Youtube-c9cd1c1d883d.json'
client = vision.ImageAnnotatorClient()


def get_max_value_from_dict(d):
    return max(d.items(), key=operator.itemgetter(1))[0]

def get_bounding_box_ratio(bounding_box):
    
    '''
    return ratio width / height of the bounding box
    '''
    top_left = bounding_box.vertices[0]
    bottom_right = bounding_box.vertices[2]
    x1 = top_left.x
    y1 = top_left.y

    x2 = bottom_right.x
    y2 = bottom_right.y

    return float(x2 - x1) / float(y2-y1)

def bounding_box_is_centered(bounding_box, fn):
    '''
    return False if the bounding box is not centered
    '''
    bottom_right = bounding_box.vertices[2]
    top_left = bounding_box.vertices[0]
    bottom_right = bounding_box.vertices[2]
    
    # get the center.x of the detected bounding box
    x1 = top_left.x
    x2 = bottom_right.x
    
    middle_bb = (x1 + x2) /2.0
    img = cv2.imread(fn)
    middle_img = img.shape[1] / 2.0
    img_width = float(img.shape[1])
#     print((middle_bb - middle_img) / img_width)
    
    # if the center of the bounding box is shifted
    if abs(middle_bb - middle_img) / img_width > 0.02:
        return False
    else:
        return True
def get_best_prediction(fn):
    
    with io.open(fn, 'rb') as f:
        content = f.read()
    image = vision.types.Image(content=content)    
    image_context = vision.types.ImageContext(language_hints=['zh-TW'])

    response = client.document_text_detection(image=image,image_context=image_context)

    word_texts = {}
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            block_box_ratio =get_bounding_box_ratio(block.bounding_box)            

            # get rid of the weird text block
            if block_box_ratio < 3 or block_box_ratio > 18: 
                continue
            if not bounding_box_is_centered(block.bounding_box, fn):
                    continue 

            block_words = []
            for paragraph in block.paragraphs:
                block_words.extend(paragraph.words)

            
            block_symbols = []
            for word in block_words:
                word_box_ratio = get_bounding_box_ratio(word.bounding_box)    
                if word_box_ratio < 3 or word_box_ratio > 18: 
                    continue
                if not bounding_box_is_centered(block.bounding_box, fn):
                    continue    
                block_symbols.extend(word.symbols)
                word_text = ''
                
                
                for symbol in word.symbols:
                    word_text = word_text + symbol.text
                    

                word_texts[word_text] = word.confidence
    if (len(word_texts)) == 0:
        return None, 0
    best_prediction = get_max_value_from_dict(word_texts)            
    return  best_prediction, word_texts[best_prediction]

def get_input_names(dir_):
    '''
    return a list of frame names
    '''
    return glob.glob(dir_ + '/*')

def filename_to_id(filename):
    filename = filename.split('_crop.')[0]
    return filename.split('crop_frames/')[1]

def hist_equal(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def rgb_shift(img):
    return  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
# equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def remove_intermediate_files(dir_):
    file_list = glob.glob(f'{dir_}/*TEMP*')
    [os.remove(f) for f in file_list]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script for ocr on frames and TTA")
    parser.add_argument("--inputs-dir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--results-file", type=str, required=True)
    args = parser.parse_args()
    
    
    # remove temp files created by tta
    remove_intermediate_files(args.inputs_dir)
    
    makedirs(args.inputs_dir, exist_ok=True)
    makedirs(args.results_dir, exist_ok=True)
    file_names = get_input_names(args.inputs_dir)
    image_ids = []
    predictions = []
    confidences = []
    for i, fn in enumerate(tqdm(file_names)):
        if i >= 50:break
        
        
        
        image_id = filename_to_id(fn)
        
        # test time augmentatin 
        temp_fn = args.inputs_dir + image_id + '-TEMP.png'
        img = cv2.imread(fn)
#         temp_img = hist_equal(img)
        temp_img = rgb_shift(img)
        cv2.imwrite(temp_fn, temp_img)
        
        prediction, confidence = get_best_prediction(temp_fn)
#         except:
#             print(f'Cannot predict {fn}')
#             continue
        image_ids.append(image_id)
        predictions.append(prediction)
        confidences.append(confidence)
        
        
        # remove temp file
        os.remove(temp_fn)
    df = pd.DataFrame(columns=['id', 'prediction', 'confidence'])
    df['id'] = image_ids
    df['prediction'] = predictions
    df['confidence'] = confidences
    df.to_csv(args.results_dir + args.results_file, index=None)
    