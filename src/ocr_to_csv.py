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

ID = 'l8xMMvrgW2I'
ENDPOINT_URL = 'https://vision.googleapis.com/v1p1beta1/images:annotate'


os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/home/A60174/ITRI-Youtube-c9cd1c1d883d.json'
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

    return float(x2 - x1) / float(y2-y1 + 0.000000001) 

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
    img_width = float(img.shape[1] ) + 0.000000001
#     print((middle_bb - middle_img) / img_width)
    
    # if the center of the bounding box is shifted
    if abs(middle_bb - middle_img) / img_width > 0.02:
        return False
    else:
        return True
def remove_intermediate_files(dir_):
    file_list = glob.glob(f'{dir_}/*TEMP*')
    [os.remove(f) for f in file_list]
    
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
                if not bounding_box_is_centered(word.bounding_box, fn):
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

def get_input_dirs(dir_):
    '''
    return a list of frame names
    '''
    return glob.glob(dir_ + '/*')

def dir_to_id(dir_):
    return dir_.split('/')[-1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script for performing ocr on frames")
    parser.add_argument("--inputs_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    args = parser.parse_args()
    
    

    makedirs(args.inputs_dir, exist_ok=True)
    makedirs(args.results_dir, exist_ok=True)
    dirs = get_input_dirs(args.inputs_dir)
    image_ids = []
    predictions = []
    confidences = []

    filtered_predictions = []
    filtered_confidences = []
    # remove temp files created by tta
    remove_intermediate_files(args.inputs_dir)
    for i, dir_ in enumerate(tqdm(dirs)):
#         if i >= 10:break
        
        image_id = dir_to_id(dir_)
        image_fn = f'{dir_}/image.png'
        filtered_fn = f'{dir_}/tta.png'        
#         try:
#         prediction, confidence = get_best_prediction(image_fn)
        filtered_prediction, filtered_confidence = get_best_prediction(filtered_fn)
#         except:
#             print(f'Cannot predict {fn}')
#             continue
        image_ids.append(image_id)
#         predictions.append(prediction)
#         confidences.append(confidence)
        filtered_predictions.append(filtered_prediction)
        filtered_confidences.append(filtered_confidence)

    df = pd.DataFrame(columns=['id', 'prediction', 'confidence'])
    df['id'] = image_ids
#     df['prediction'] = predictions
#     df['confidence'] = confidences
    df['prediction'] = filtered_predictions
    df['confidence'] = filtered_confidences
    df.to_csv(args.results_dir + args.results_file, index=None)