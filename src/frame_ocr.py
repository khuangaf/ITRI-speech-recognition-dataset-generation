from base64 import b64encode
from os import makedirs
from os.path import join, basename
from sys import argv
import json
import requests
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
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

ENDPOINT_URL = 'https://vision.googleapis.com/v1p1beta1/images:annotate'


os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/home/ashutosh/ITRI-Youtube-c9cd1c1d883d.json'
client = vision.ImageAnnotatorClient()

INPUT_DIR = 'crop_frames/'
RESULTS_DIR = 'results/'
# get current_time
# y, m, d, h, m, s, _,_,_ = gmtime()
RESULT_NAME = f'result.csv'

makedirs(INPUT_DIR, exist_ok=True)
makedirs(RESULTS_DIR, exist_ok=True)

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

def get_best_prediction(fn):
    
    with io.open(fn, 'rb') as f:
        content = f.read()
    image = vision.types.Image(content=content)    
    image_context = vision.types.ImageContext(language_hints=['zh-TW'])

    response = client.document_text_detection(image=image,image_context=image_context)
#     print(response)
    word_texts = {}
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            block_box_ratio =get_bounding_box_ratio(block.bounding_box)            
            # get rid of the weird text block
            if block_box_ratio < 3 or block_box_ratio > 18: 
                continue
                
#             print(block_ratio)
            block_words = []
            for paragraph in block.paragraphs:
                block_words.extend(paragraph.words)
#                 print(u'Paragraph Confidence: {}\n'.format(
#                     paragraph.confidence))
            
            block_symbols = []
            for word in block_words:
                word_box_ratio = get_bounding_box_ratio(word.bounding_box)    
                if word_box_ratio < 3 or word_box_ratio > 18: 
                    continue
                block_symbols.extend(word.symbols)
                word_text = ''
                
                
                for symbol in word.symbols:
                    word_text = word_text + symbol.text
                    
#                 print(u'Word text: {} (confidence: {})\n'.format(
#                     word_text, word.confidence))

                word_texts[word_text] = word.confidence
    if (len(word_texts)) == 0:
        return None, 0
    best_prediction = get_max_value_from_dict(word_texts)            
    return  best_prediction, word_texts[best_prediction]

def get_input_names():
    '''
    return a list of frame names
    '''
    return glob.glob(INPUT_DIR + '/*')

def filename_to_id(filename):
    filename = filename.split('_crop.')[0]
    return filename.split('crop_frames/')[1]

if __name__ == '__main__':
    
    file_names = get_input_names()
    image_ids = []
    predictions = []
    confidences = []
    for i, fn in enumerate(tqdm(file_names)):
        if i >= 50:break

        image_id = filename_to_id(fn)
        prediction, confidence = get_best_prediction(fn)
        image_ids.append(image_id)
        predictions.append(prediction)
        confidences.append(confidence)

    df = pd.DataFrame(columns=['id', 'prediction', 'confidence'])
    df['id'] = image_ids
    df['prediction'] = predictions
    df['confidence'] = confidences
    df.to_csv(RESULTS_DIR + RESULT_NAME, index=None)
    