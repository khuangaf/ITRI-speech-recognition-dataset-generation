"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python subtitle.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python subtitle.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    

    

#restrict gpu
import shutil
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
from tqdm import tqdm

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

from keras.backend.tensorflow_backend import set_session
set_session(sess) 

from augment import *
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

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
# Results directory
# Save submission files here
RESULTS_DIR = '/data/ITRI-create-speech-recognition-dataset/mandarin/maskrcnn_results/'

#Be sure to run the Dataset Generation Mask-RCNN.ipynb file to generate these two
with open('/data/steeve/mask_rcnn/data/split/valid_ids', 'r') as f:
    VAL_IMAGE_IDS = [id.strip() for id in f.readlines()]

with open('/data/steeve/mask_rcnn/data/split/train_ids', 'r') as f:
    TRAIN_IMAGE_IDS = [id.strip() for id in f.readlines()]   
    
# with open('/data/steeve/mask_rcnn//split/test_ids', 'r') as f:
#     TEST_IMAGE_IDS = [id.strip() for id in f.readlines()]

# TEST_IMAGE_IDS = get_all_id('mandarin/frames/') 

############################################################
#  Configurations
############################################################

class SubtitleConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "subtitle"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + subtitle

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (len(TRAIN_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
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
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class SubtitleDataset(utils.Dataset):

    def load_subtitle(self, dataset_dir, subset, image_ids=None):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("subtitle", 1, "subtitle")
        self.dataset_dir = dataset_dir
        
        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "test"]


        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        elif subset == "test":
            image_ids = image_ids
        else:
            # Get image ids from directory names
            image_ids = [ fn.split('.png')[0] for fn in os.listdir(dataset_dir+'/images/')]

            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in image_ids:
            # get only the file names
            if image_id == '1EM1aaJfPck-0000474-0.25':
                continue
            if subset=='test':
                self.add_image(
                    "subtitle",
                    image_id=image_id,
                    path="{}".format(image_id))
            else:
                self.add_image(
                    "subtitle",
                    image_id=image_id,
                    path=os.path.join(dataset_dir, "images/{}.png".format(image_id)))

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
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = SubtitleDataset()
    dataset_train.load_subtitle(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SubtitleDataset()
    dataset_val.load_subtitle(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
#     model.train(dataset_train, dataset_val,
#                 learning_rate=config.LEARNING_RATE,
#                 epochs=20,
#                 augmentation=augmentation,
#                 layers='heads')

#     print("Train all layers")
#     model.train(dataset_train, dataset_val,
#                 learning_rate=config.LEARNING_RATE,
#                 epochs=60,
#                 augmentation=augmentation,
#                 layers='all')
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/5,
                epochs=100,
                augmentation=augmentation,
                layers='all')




############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset, processed_frames_dir, input_id):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    
    
    # if testing, 
    if subset=='test':
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



            submit_dir = f"{video_id}"
            submit_dir = os.path.join(RESULTS_DIR, submit_dir)
            os.makedirs(submit_dir, exist_ok=True)

            # Load over images
            for image_id in tqdm(dataset.image_ids):
                # Load image and run detection
                image = dataset.load_image(image_id)
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

                # TTA: shift color
                tta_transforms = ['flipud', 'fliplr']
                for tta in tta_transforms:
                    if tta == 'flipud':
                        transform = np.flipud
                        un_transform = np.flipud
                    elif tta == 'fliplr':
                        transform = np.fliplr
                        un_transform = np.fliplr
                    else:
                        continue

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
                    image = un_transform(image)

                    # get union of the masks
                    multi_masks_tta = np.logical_or(multi_masks_1, multi_masks_tta)

                    masked_image = image.astype(np.uint8).copy()    

                for c in range(3):
                    masked_image[:,:,c] = np.where(multi_masks==1,masked_image[:,:,c], multi_masks)

                os.makedirs(f'{submit_dir}/{source_id}', exist_ok=True)
                result_file_name = f'{submit_dir}/{source_id}/filtered.png'
                original_file_name = f'{submit_dir}/{source_id}/image.png'
                cv2.imwrite(original_file_name, image.astype(np.uint8))
                cv2.imwrite(result_file_name, masked_image)


                # store tta result
                masked_image = image.astype(np.uint8).copy()
                for c in range(3):
                    masked_image[:,:,c] = np.where(multi_masks_tta==1,masked_image[:,:,c], multi_masks_tta)
                tta_file_name = f'{submit_dir}/{source_id}/tta.png'
                cv2.imwrite(tta_file_name, masked_image)    


                # move the processed frame to processed_frames_dir
                shutil.move(source_path,processed_frames_dir)
    print("Saved to ", submit_dir)


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
    parser.add_argument('--dataset', required=False,
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
    parser.add_argument('--processed_frames_dir', required=False,
                        metavar="/path/to/processed_frames_dir/",
                        help="move frames to processed frames")
    args = parser.parse_args()
    
    processed_frames_dir = args.processed_frames_dir
    
    if processed_frames_dir != None:
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
    

    # Configurations
    if args.command == "train":
        config = SubtitleConfig()
    else:
        config = SubtitleInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
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
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset, processed_frames_dir, args.input_id)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
