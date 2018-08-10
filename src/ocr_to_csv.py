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
from util import *
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
from joblib import Parallel, delayed

ENDPOINT_URL = 'https://vision.googleapis.com/v1p1beta1/images:annotate'

os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/data/ITRI-create-speech-recognition-dataset/itriaccount.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS']='/home/A60174/ITRI-Youtube-c9cd1c1d883d.json'
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

def bounding_box_is_bottom(bounding_box, img):
    '''
    return False if the bounding box is not at the bottom
    '''
    bottom_right = bounding_box.vertices[2]
    top_left = bounding_box.vertices[0]
    bottom_right = bounding_box.vertices[2]
    
    # get the center.x of the detected bounding box
    y1 = top_left.y
    

    image_y = img.shape[0]
    
    if y1 < image_y * 0.6:
        return False
    else:
        return True

    
def bounding_box_is_peripheral(bounding_box, img):
    '''
    return True if the bounding box is at the peripheral
    '''
    bottom_right = bounding_box.vertices[2]
    top_left = bounding_box.vertices[0]
    bottom_right = bounding_box.vertices[2]
    
    # get the center.x of the detected bounding box
    x1 = top_left.x
    x2 = bottom_right.x
    
    middle_bb = (x1 + x2) /2.0

    # prevent devision by zero
    img_width = float(img.shape[1] ) + 0.000000001
    
    if middle_bb < img_width * 0.16 or middle_bb > img_width * (1 - 0.16):
        return True
    else:
        return False
    
def bounding_box_is_centered(bounding_box, img):
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

    middle_img = img.shape[1] / 2.0
    img_width = float(img.shape[1] ) + 0.000000001
    
    # if the center of the bounding box is shifted
    if abs(middle_bb - middle_img) / img_width > 0.02:
        return False
    else:
        return True

def boundingbox_neighbor(bounding_box_1, bounding_box_2):
    bottom_right_1 = bounding_box_1.vertices[2]
    bottom_right_2 = bounding_box_2.vertices[2]    
    
    top_left_1 = bounding_box_1.vertices[0]
    top_left_2 = bounding_box_2.vertices[0]
    
    bb_height =  bottom_right_1.y - top_left_1.y
    

    # if the second bbox is at the next line
    if top_left_2.y - bottom_right_1.y > 0 and top_left_2.y - bottom_right_1.y  < 0.5 * bb_height:
        return True
    # if the second bbox is next to the first horizontally
    elif abs(bottom_right_1.y - bottom_right_2.y) < 0.2 *bb_height :
        return True
    else:
        return False
        
def get_merge_list(prediction_boundingbox_dict, prediction_confidence_dict):
    '''
    Return the list of text that should be merged.
    If there is only one box that should be returned, output a list with length 1.
    '''
    predictions = list(prediction_boundingbox_dict.keys())
    if len(predictions) == 1:
        prediction = predictions[0]
        return [prediction]
    
    merge_list = []
    for i in range(len(predictions) -1):
        # if both predictions are neighbor and they are of high confidence
        if boundingbox_neighbor(prediction_boundingbox_dict[predictions[i]], prediction_boundingbox_dict[predictions[i+1]]):
            if len(merge_list) == 0:
                merge_list.append(predictions[i])
                merge_list.append(predictions[i+1])
            elif merge_list[-1] == predictions[i]:
                merge_list.append(predictions[i+1])
            elif merge_list[-1] == predictions[i]:
                merge_list.append(predictions[i])
                merge_list.append(predictions[i+1])
    # if there is no neighbor, get prediction with highest confidence
    if len(merge_list) == 0:
        merge_list = get_max_value_from_dict(prediction_confidence_dict)
        merge_list = [merge_list] 
    return merge_list
    
def merge_get_confidence(merge_list, prediction_confidence_dict, prediction_boundingbox_dict):
    '''
    return the merged string and the average confidence
    '''
    average_confidence = np.mean([prediction_confidence_dict[word] for word in merge_list])
    
    
    # set y value to min if they are > min_y + 0.5* font_height or
    # set y value to max if they are < max_y - 0.5* font_height
    
    font_height = None
    df = pd.DataFrame(columns=['topleft_x', 'topleft_y'])
    for word in merge_list:
        topleft = prediction_boundingbox_dict[word].vertices[0]
        topleft_x = topleft.x
        topleft_y = topleft.y
        
        if font_height == None:
            bottomright = prediction_boundingbox_dict[word].vertices[2]
            bottomright_y = bottomright.y
            font_height = bottomright_y - topleft_y

        df.loc[word] = [topleft_x, topleft_y]

    min_y = df.topleft_y.min()
    max_y  = df.topleft_y.max()
    
    df.loc[df.topleft_y < min_y + 0.5* font_height, "topleft_y"] = min_y
    df.loc[df.topleft_y > max_y - 0.5* font_height, "topleft_y"] = max_y
    
    df.sort_values(["topleft_y", "topleft_x"], inplace=True)
    
    merge_list = df.index.tolist()
    
    merged_string = ' '.join(merge_list)
    return merged_string, average_confidence
    
    
def get_best_prediction(fn):
    
#     print(fn)
    '''
    Send OCR request with Google Vision Client
    Input: File name
    Output: (best prediction, confidence) 
    '''
    #check if image has text
    
    original_image = cv2.imread(fn)
    try:
        height = original_image.shape[0]
    except:
        print(fn)
    crop_height = int(0.7*height)
    PIXEL_THRESHOLD = 30000
    
    # if the bottom part of the image has sum of rgb value less than thresohld, return None, -1
    if original_image[crop_height:,:,:].sum() < PIXEL_THRESHOLD:
#         print("None")
        return None, -1
    
    with io.open(fn, 'rb') as f:
        content = f.read()
    image = vision.types.Image(content=content)    
    image_context = vision.types.ImageContext(language_hints=['zh-TW'])

    response = client.document_text_detection(image=image,image_context=image_context)

    prediction_confidence_dict = {}
    prediction_boundingbox_dict = {}
    
    
    
    
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            block_box_ratio =get_bounding_box_ratio(block.bounding_box)            
            
            # get rid of the weird text block
            if block_box_ratio < 0.7 or block_box_ratio > 18: 
                continue
            if bounding_box_is_peripheral(block.bounding_box, original_image):
                continue                
            if not bounding_box_is_bottom(block.bounding_box, original_image):
                continue 
            

            block_words = []
            for paragraph in block.paragraphs:
                block_words.extend(paragraph.words)

            

            for word in block_words:
                word_box_ratio = get_bounding_box_ratio(word.bounding_box)    
                if word_box_ratio > 18: 
                    continue
                if bounding_box_is_peripheral(block.bounding_box, original_image):
                    continue 
                if not bounding_box_is_bottom(word.bounding_box, original_image):
                    continue    

                word_text = ''
                
                
                for symbol in word.symbols:
#                     clean_text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", symbol.text)
                    
                    if symbol.confidence < 0.98:
                        return None, -3
                    word_text += symbol.text
                    
                    

                prediction_confidence_dict[word_text] = word.confidence
                prediction_boundingbox_dict[word_text] = word.bounding_box
    
    if (len(prediction_confidence_dict)) == 0:
        return None, -2
    
    merge_list = get_merge_list(prediction_boundingbox_dict, prediction_confidence_dict)
    if len(merge_list) == 1:
        best_prediction = merge_list[0]
        confidence = prediction_confidence_dict[best_prediction]
    else:
        best_prediction, confidence = merge_get_confidence(merge_list, prediction_confidence_dict, prediction_boundingbox_dict)
          
    return  best_prediction, confidence

def get_input_dirs(dir_):
    '''
    return a list of frame names
    '''
    return glob.glob(dir_ + '/*')



def dir_to_id(dir_):
    return dir_.split('/')[-1]

def get_result(dir_):
    image_id = dir_to_id(dir_)
    image_fn = f'{dir_}/image.png'
    filtered_fn = f'{dir_}/dif.png'
    prediction, confidence = get_best_prediction(filtered_fn)
    
    return image_id, prediction, confidence

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script for performing ocr on frames")
    parser.add_argument("--inputs_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--video_id_file", type=str, required=True)
    args = parser.parse_args()
    
    


    makedirs(args.results_dir, exist_ok=True)
    dirs = get_input_dirs(args.inputs_dir)
    image_ids = []
    predictions = []
    confidences = []

#     filtered_predictions = []
#     filtered_confidences = []
    
    # remove temp files created by tta
#     remove_intermediate_files(args.inputs_dir)
    print(args.inputs_dir)
    parallel = Parallel(40, backend="threading", verbose=10)

#     for i, dir_ in enumerate(tqdm(dirs)):
        
#         if i not in range(760,770): continue
#         image_id = dir_to_id(dir_)

#         image_fn = f'{dir_}/image.png'
#         filtered_fn = f'{dir_}/tta.png'
        
        
#         prediction, confidence = get_best_prediction(filtered_fn)


#         image_ids.append(image_id)

#         predictions.append(prediction)
#         confidences.append(confidence)
        
    results = parallel(delayed(get_result)(dir_) for dir_ in dirs)
    for r in results:
        image_ids.append(r[0])
        predictions.append(r[1])
        confidences.append(r[2])
        
    image_ids, predictions, confidences
    df = pd.DataFrame(columns=['id', 'prediction', 'confidence'])
    df['id'] = image_ids
    df['prediction'] = predictions
    df['confidence'] = confidences
    df['frame_id'] = df.id.apply(lambda x: int(x.split('-')[-2]))
    df.sort_values('frame_id', inplace=True)
    df = df.loc[:,['id', 'prediction', 'confidence']]
    df.to_csv(args.results_dir + args.results_file, index=None)