"""

This script runs Mask-RCNN models stored in the Mask_RCNN directory.

Author: Kung-hsiang, Huang, 2018
"""

#restrict gpu
import shutil
import os
import tensorflow as tf
from tqdm import tqdm
from util import *
from keras.backend.tensorflow_backend import set_session
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import skimage

# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
import cv2
from glob import glob
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


def get_all_id(dir_):
    files = os.listdir(dir_)
    ids = set([ '-'.join(f.split('-')[:-2]) for f in files])
    return list(ids)


############################################################
#  Configurations
############################################################

class SubtitleConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "subtitle"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + subtitle

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 30

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 50


class SubtitleInferenceConfig(SubtitleConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9


############################################################
#  Dataset
############################################################

class SubtitleDataset(utils.Dataset):

    def load_subtitle(self, dataset_dir, subset, image_ids=None):
        """Load a subset of the nuclei dataset.
        
        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * test: Run prediction on the frames associated with the input (video) id 
                * test_manual: Run prediction on the manually labelled data
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("subtitle", 1, "subtitle")
        self.dataset_dir = dataset_dir
        
        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test", "test_manual"]


        if subset in ["test", "test_manual"]:
            image_ids = image_ids
        else:
            raise ValueError('Subset must be test or test_manual')

        # Add images
        for image_id in image_ids:
            self.add_image(
                "subtitle",
                image_id=image_id,
                path="{}".format(image_id))
            

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        id_ = info['id']
        # Get mask directory from image path
        mask_path = os.path.join( f"{self.dataset_dir}/multi_masks/{id_}.npy")


        # Read mask files from .png image
        mask = []

        multi_masks = np.load(mask_path)
        num_masks = multi_masks.max()

        for i in range(num_masks):
            mask.append(multi_masks == i+1)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "subtitle":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)






############################################################
#  Detection
############################################################

def detect(models, dataset_dir, subset, processed_frames_dir, input_id, results_dir):
    """Run detection on images in the given directory.
    
    Parameters
    ----------
    models: list(Models)
        A list of Mask-RCNN models for running prediction
    dataset_dir: str
        The directory storing the frames
    subset: str
        Subset to load. One of:
            * test: Run prediction on the frames associated with the input (video) id 
            * test_manual: Run prediction on the manually labelled data
    processed_frames_dir: str
        Directory storing processed frames
    
    input_id
    
    
    """
    print("Running on {}".format(dataset_dir))

    
    ########################### Helper functions for Test Time Augmentation #########################
    def crop_bottom(image, crop_ratio=0.7):
        if len(image.shape)==3:
            height, width, _ = image.shape
        else:
            height, width = image.shape
        return image[int(height*crop_ratio):,:]
    
    def uncrop_bottom(current_mask, original_mask):
        padded_mask = np.zeros(original_mask.shape)
        current_mask_height, _ = current_mask.shape
        padded_mask[-current_mask_height:,:] = current_mask
        return padded_mask
    
    def scale(image):
        return cv2.resize(image,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_CUBIC)
    
    def unscale(current_mask, original_mask):
        return skimage.transform.resize(current_mask, original_mask.shape)
    
    def rotate90(image):
        return skimage.transform.rotate(image, 90)
    def unrotate90(image):
        return skimage.transform.rotate(image, -90)
    # Create directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    
    
    ############################# test mode: for real collecting data ###################################
    if subset=='test':
        # if you have specify which video_id to run mask-rcnn
        if input_id != None:
            ids = [input_id]
        else:
            ids = get_all_id(dataset_dir)

        for video_id in ids:
            image_ids = glob(dataset_dir + f'/{video_id}*')
            # Read dataset
            dataset = SubtitleDataset()
            dataset.load_subtitle(dataset_dir, subset, image_ids)

            print(f'{video_id}')
            dataset.prepare()



            submit_dir = video_id
            submit_dir = os.path.join(results_dir, submit_dir)
            os.makedirs(submit_dir, exist_ok=True)

            # Load over images
            for image_id in tqdm(dataset.image_ids):
                # Load image and run detection
                image = dataset.load_image(image_id)
            
                height, width, _ = image.shape

                #resize for faster process and higher accuracy
                if width >= 1000 or height >= 700:
                    image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
                elif width >= 1500 or height >= 1000:
                    image = cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)

                
                # in the latest version for test, we replace Test-time augmentation with ensemble of two models
                for model_i, model in enumerate(models):
                    
                    # Detect objects
                    r = model.detect([image], verbose=0)[0]

                    source_id = dataset.image_info[image_id]["id"].split('/')[-1].split('.png')[0]
                    source_path = dataset.image_info[image_id]['id']

                    masks = r['masks']
                    num_mask = masks.shape[2]
                    if model_i == 0:
                        multi_masks = np.zeros(masks.shape[:2])
                        masked_image = np.zeros(image.shape)
                    
                    #gather all mask together
                    for i in range(num_mask):
                        mask = masks[:, :, i]
                        multi_masks[np.where(mask==1)] = 1
                    
                    # keep a copy of image
                    original_image = image.copy()
                    for c in range(3):
                        masked_image[:,:,c] = np.where(multi_masks==1,original_image[:,:,c], multi_masks)
                    
                
                    if model_i == 0:
                        os.makedirs(f'{submit_dir}/{source_id}', exist_ok=True)
                        result_file_name = f'{submit_dir}/{source_id}/filtered.png'
                        original_file_name = f'{submit_dir}/{source_id}/image.png'
                        cv2.imwrite(original_file_name, original_image.astype(np.uint8))
                        cv2.imwrite(result_file_name, masked_image)
                    else:
                        
                        ensemble_file_name = f'{submit_dir}/{source_id}/ensemble.png'
                        
                        cv2.imwrite(ensemble_file_name, masked_image)  
                
                
                # move the processed frame to processed frames directory
                shutil.move(source_path,f"{processed_frames_dir}/{source_id}.png")
                
    
    ############################## test_manual mode: run on manully labelled data ###################################
    elif subset=='test_manual':
        
        
        image_ids = glob(f'{dataset_dir}/*')
        
        # Read dataset
        dataset = SubtitleDataset()
        dataset.load_subtitle(dataset_dir, subset, image_ids)
        dataset.prepare()

        submit_dir = results_dir

        # Load over images
        for image_id in tqdm(dataset.image_ids):
            # Load image and run detection
            image = dataset.load_image(image_id)
            


            height, width, _ = image.shape
            crop_height = int(0.6*height)
            image = image[crop_height:,:]
            height, width, _ = image.shape

            #resize for faster process and higher accuracy
            if width >= 1000 or height >= 700:
                image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
            elif width >= 1500 or height >= 1000:
                image = cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)


            # Detect objects
            r = model.detect([image], verbose=0)[0]

            source_id = dataset.image_info[image_id]["id"].split('/')[-1].split('.png')[0]
            source_path = dataset.image_info[image_id]['id']
            masks = r['masks']
            num_mask = masks.shape[2]

            multi_masks = np.zeros(masks.shape[:-1])

            #gather all mask together
            for i in range(num_mask):
                mask = masks[:, :, i]
                multi_masks[np.where(mask==1)] = 1

            multi_masks_tta = multi_masks
            
            original_image = image.copy()
            
            # test time augmentation
            tta_transforms = ['flipud', 'fliplr']
            for tta in tta_transforms:
                if tta == 'flipud':
                    transform = np.flipud
                    un_transform = np.flipud
                elif tta == 'fliplr':
                    transform = np.fliplr
                    un_transform = np.fliplr
                elif tta == 'scale':
                    transform = scale
                    un_transform = np.array
                elif tta == 'rotate90':
                    transform = rotate90
                    un_transform = unrotate90
                else:
                    continue

                image = original_image.copy()
                image = transform(image)
                r = model.detect([image], verbose=0)[0]


                masks = r['masks']
                num_mask = masks.shape[2]

                multi_masks_1 = np.zeros(masks.shape[:-1])

                #gather all mask together
                for i in range(num_mask):
                    mask = masks[:, :, i]
                    multi_masks_1[np.where(mask==1)] = 1

                
                multi_masks_1 = un_transform(multi_masks_1)
                
                if tta == 'scale':
                    multi_masks_1 = unscale(multi_masks_1, multi_masks_tta)
                # get union of the masks
                multi_masks_tta = np.logical_or(multi_masks_1, multi_masks_tta)

            masked_image = original_image.astype(np.uint8).copy()    

            for c in range(3):
                masked_image[:,:,c] = np.where(multi_masks==1,masked_image[:,:,c], multi_masks)

            os.makedirs(f'{submit_dir}/{source_id}', exist_ok=True)
            result_file_name = f'{submit_dir}/{source_id}/filtered.png'
            original_file_name = f'{submit_dir}/{source_id}/image.png'
            cv2.imwrite(original_file_name, original_image.astype(np.uint8))
            cv2.imwrite(result_file_name, masked_image)


            # store tta result
            masked_image = original_image.astype(np.uint8).copy()
            for c in range(3):
                masked_image[:,:,c] = np.where(multi_masks_tta==1,masked_image[:,:,c], multi_masks_tta)
            tta_file_name = f'{submit_dir}/{source_id}/tta.png'
            cv2.imwrite(tta_file_name, masked_image)    

            
    print("Saved to ", submit_dir)

def str2bool(v):
    '''Convert boolean string to python bool variable'''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for subtitle counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--input_id', required=False,
                        metavar="video id",
                        help="Specify video id")
    parser.add_argument('--processed_frames_dir', required=True,
                        metavar="/path/to/processed_frames_dir/",
                        help="move frames to processed frames")
    parser.add_argument('--results_dir', required=True,
                        metavar="/path/to/results_dir/",
                        help="move frames to processed frames")
    parser.add_argument('--disable_gpu', type=str2bool, required=False, 
                        default=False,
                        metavar="True",
                        help="whether to use cpu only")
    parser.add_argument("--video_id_file", type=str, required=True)
    
    args = parser.parse_args()
    
    
    # make gpu invisible if disable gpu
    if args.disable_gpu == True:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
    # make sure the processed frames directory exists
    processed_frames_dir = args.processed_frames_dir
    os.makedirs(processed_frames_dir, exist_ok=True)
    
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)
    

    
    
    # restrict GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess) 
    
    # Configurations
    config = SubtitleInferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif args.command=='detect':
        # ensemble of two models
        models = []
        model.load_weights('Mask_RCNN/logs/subtitle20180731T0943/mask_rcnn_subtitle_0091.h5', by_name=True)
        models.append(model)
        model1 = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
        model1.load_weights('Mask_RCNN/logs/subtitle20180806T1527/mask_rcnn_subtitle_0089.h5', by_name=True)
        models.append(model1)
    else:
        model.load_weights(weights_path, by_name=True)
    
    
    if args.subset=='test_manual':
        now = datetime.datetime.now()
        results_dir = args.results_dir + f'/{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}'
    else:
        results_dir = args.results_dir
        
    # inferencing
    # if specify video id, process that one only (this is for debugging)
    if args.input_id:
        detect(models, args.dataset, args.subset, processed_frames_dir, args.input_id, results_dir)
    else:
        input_ids = get_video_id_from_file(args.video_id_file)

        for input_id in input_ids:
            detect(models, args.dataset, args.subset, processed_frames_dir, input_id, results_dir)
