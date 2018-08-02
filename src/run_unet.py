from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from PIL import Image
import glob
import datetime
from torch.utils.data import DataLoader
from time import gmtime
import os
import skimage
from imgaug import augmenters as iaa
import imgaug
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
from configuration import *
Configuration()
from UNet import *
import torch.backends.cudnn as cudnn
from tqdm import tqdm
cudnn.enabled = False
cfg = Configuration()


class SubtitleDataset(Dataset):
    """Cat Dog dataset."""

    def __init__(self, dataset_dir, ids, mode='train', transform=None, image_ids=None):
        """
        Args:

            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        """

        self.dataset_dir = dataset_dir
        self.ids = ids
        self.mode = mode
        self.transform = transform
        self.augmentation = iaa.SomeOf((0, 2), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                       iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)]),
            iaa.Multiply((0.8, 1.5)),
#             iaa.GaussianBlur(sigma=(0.0, 5.0))
        ])
        self.image_ids = image_ids
#         self.labels = labels
#         self.load_memory = load_memory
        

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        
        #
        min_dim = cfg.min_dim
        max_dim = cfg.max_dim
        min_scale = cfg.min_scale
        
        if self.mode in ['train','eval']:
            image_id = self.ids[idx]
            img_name = os.path.join(self.dataset_dir, "images/{}.png".format(image_id))
            image = cv2.imread(img_name)
            mask_path = os.path.join( f"{self.dataset_dir}/multi_masks/{image_id}.npy")
            label = np.load(mask_path)
            label = np.expand_dims(label, 2)
            label = np.where(label >= 1, np.ones(label.shape), np.zeros(label.shape))
            det = self.augmentation.to_deterministic()


            MASK_AUGMENTERS = ["Sequential", #"SomeOf", "OneOf", "Sometimes",
                               "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return (augmenter.__class__.__name__ in MASK_AUGMENTERS)
        elif self.mode in ['test', 'test_manual']:
            image_path = self.image_ids[idx]
            image = cv2.imread(image_path)
            image = resize_image(image, min_dim, max_dim, min_scale, mode='pad64')
#             print(image.shape)
            image = np.transpose(image, (2, 0, 1))
            image_id =  image_path.split('.png')[0].split('/')[-1]
            return image, image_id
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        if self.mode == 'train':
            image = det.augment_image(image.astype(np.uint8))
            label = det.augment_image(label.astype(np.uint8),
                                     hooks=imgaug.HooksImages(activator=hook))
        
        
        
        if self.mode == 'train':
            resize_mode= 'crop'
        else:
            resize_mode='pad64'
        image = resize_image(image, min_dim, max_dim, min_scale, mode=resize_mode)
        label = resize_image(label, min_dim, max_dim, min_scale, mode=resize_mode)    
            

#         cv2.imwrite(f'temp/{image_id}.png', image)
        image = np.transpose(image, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        if not train:
            return self.transform(image)
        
        
        image = image.astype(np.float32)
        image = image / 255.0
        label = label.astype(np.float32)
        return image, label

    
def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype)

with open('/data/steeve/mask_rcnn/data/split/valid_ids', 'r') as f:
    valid_ids = [id.strip() for id in f.readlines()]

with open('/data/steeve/mask_rcnn/data/split/train_ids', 'r') as f:
    train_ids = [id.strip() for id in f.readlines()]   

    
dataset_dir='/data/steeve/mask_rcnn/data/'
# train_images = [cv2.imread(f'{dataset_dir}/images/{id}.png') for id in train_ids]
# valid_images = [cv2.imread(f'{dataset_dir}/images/{id}.png') for id in valid_ids]


# train_labels = [np.load(f"{dataset_dir}/multi_masks/{id}.npy") for id in train_ids]
# valid_labels = [np.load(f"{dataset_dir}/multi_masks/{id}.npy") for id in valid_ids]
    
train_dataset = SubtitleDataset(dataset_dir=dataset_dir, ids= train_ids, mode='train', transform=None)
valid_dataset = SubtitleDataset(dataset_dir=dataset_dir,ids= valid_ids, mode='eval', transform=None)




train_loader = DataLoader(
                        train_dataset,
                        shuffle=True,
                        batch_size  = cfg.batch_size,
                        drop_last   = True,
                        num_workers = 0,
                        pin_memory  = True,)

valid_loader = DataLoader(
                        valid_dataset,
                        shuffle=True,
                        batch_size  = 1,
                        drop_last   = True,
                        num_workers = 0,
                        pin_memory  = True,)





def inference(model, device, submit_dir, test_loader):
    model.set_mode('test')
    outputs = []

    for batch_idx, (original_image, source_id) in enumerate(test_loader):  
        original_image = original_image.type(torch.FloatTensor).to(device)
    
        source_id = source_id[0]
        os.makedirs(f'{submit_dir}/{source_id}', exist_ok=True)
        result_file_name = f'{submit_dir}/{source_id}/filtered.png'
        original_file_name = f'{submit_dir}/{source_id}/image.png'
                
        with torch.no_grad(): 
            model.forward(original_image)
            multi_masks = F.sigmoid(model.logits).cpu().numpy()[0]
            multi_masks = np.where(multi_masks >= 0.5, np.ones(multi_masks.shape), np.zeros(multi_masks.shape))[0]
            original_image = original_image.cpu().numpy()[0]
            
            masked_image = original_image.astype(np.uint8).copy()
            for c in range(3):

                masked_image[c,:,:] = np.where(multi_masks>0.5, original_image[c,:,:], multi_masks)
                
                
            original_image = np.transpose(original_image, (1, 2, 0))
            masked_image = np.transpose(masked_image, (1, 2, 0))

            cv2.imwrite(original_file_name, original_image.astype(np.uint8))
            cv2.imwrite(result_file_name, masked_image)
            
        

def evaluate( model, device, val_loader):
    model.set_mode('eval')
    losses = []

    for batch_idx, (data, target) in enumerate(val_loader):  
        data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            model.forward(data)
        model.criterion(target)
        
        losses.append(model.loss)
    return np.mean(losses)

def train(model, device, train_loader, optimizer, epoch):

    model.set_mode('train')
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):

        data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)

        
        losses = []
        model.forward(data)
        model.criterion(target)
        
        losses.append(model.loss.detach().cpu().numpy())
        
        model.loss.backward()
        if batch_idx % iter_accum == 0:
            
            optimizer.step()
            optimizer.zero_grad()
        

    return model, np.mean(losses)

iter_accum = cfg.iter_accum
best_valid_loss = np.inf
patience = 2
not_improve_count = 0
epochs = 100
model_path= 'unet_weights/'

def run_train():
    model = UNet(cfg).to('cuda')
    for epoch in range(1, epochs):
        print(f'Epoch {epoch}')
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay = cfg.weight_decay)
        model, train_loss = train(model, 'cuda', train_loader, optimizer, epoch)
        valid_loss = evaluate(model, 'cuda', valid_loader, optimizer)
        print(f'Training loss :{train_loss}; validation loss: {valid_loss}')
#         if best_valid_loss == None or valid_loss < best_valid_loss:
#             torch.save(model.state_dict(),model_path +'/%03d_model.pth'%(epoch))
#             torch.save({
#                 'optimizer': optimizer.state_dict(),
#                 'epoch'    : epoch,
#             }, model_path +'/%03d_optimizer.pth'%(epoch))
#             valid_loss = best_valid_loss
def run_test():
    model = UNet(cfg).to('cuda')
    model.load_state_dict(torch.load('unet_weights/098_model.pth', map_location=lambda storage, loc: storage))
    
    dataset_dir = '/data/ITRI-create-speech-recognition-dataset/manual_label_data/'
    image_ids = glob.glob(f'{dataset_dir}/*')
    test_dataset = SubtitleDataset(dataset_dir=dataset_dir,ids= valid_ids, mode='test', transform=None, image_ids=image_ids)
    test_loader = DataLoader(
                        test_dataset,
                        shuffle=False,
                        batch_size  = 1,
                        drop_last   = False,
                        num_workers = 0,
                        pin_memory  = True,)
    now = datetime.now()
    result_dir = 'manual_label_result/'
    result_dir += f'/{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}'
    os.makedirs(f'{result_dir}', exist_ok=True)
    inference(model, 'cuda', result_dir, test_loader)
    
if __name__ == '__main__':
    run_train()