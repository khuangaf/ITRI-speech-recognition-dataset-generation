# Automatic Speech Recognition Dataset Generation

__Author: Kung-hsiang, Huang (Steeve), 2018__

Although there exists an abundance of English speech recognition datasets publicly available, the opposite is true for the Mandarin ones, espically for Mandarin datasets that contain a little Taiwanese or English speaking. We want to leverage the copiousness of the Taiwanese dramas uploaded to Youtube to collect Speech Recognition dataset. The pipeline is shown as the following figure:

![Pipeline](/docs/DataCollectionPipeline.png)

## Requirements

First, [install FFMPEG from its website](https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg)

* Python==3.6
* joblib==0.12.0
* numpy==1.13.3
* pandas==0.23.3
* tensorflow-gpu==1.4.0
* keras==2.1.3
* google-cloud-vision==0.32.0
* pafy==0.5.4
* youtube-dl==2017.12.2
* tqdm==4.23.4
* editdistance==0.4

To set up all the requirements, prepare a python 3.6 environment with conda and install packages with pip.

```
conda create -n py36 python=3.6 anaconda
source activate py36
pip install -r requirements.txt
```

## Directories

```
|-- src
|-- mandarin
|   |-- audios
|   |-- bgs_results
|   |-- frames
|   |-- maskrcnn_results
|   |-- ocr_results
|   |-- processed_frames
|   |-- processed_videos
|   |-- split_audios
|   |-- srts
|   `-- videos
|-- Mask_RCNN
|   |-- assets
|   |-- images
|   |-- logs
|   |-- mrcnn
|   `-- samples
|-- docs

```

**src/**: Directory that stores all the code.

**mandarin/**: Directory that stores all the intermediate and final results, including the sub-directories as the follows:

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; **videos/**: Directory that stores the downloaded videos.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; **audios/**: Directory that stores the extracted audio from the videos.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; **frames/**: Directory that stores the split frames from the videos.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; **maskrcnn_result/**: Directory that stores the resulted frames processed by Mask-RCNN.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; **ocr_results/**: Directory that stores the OCR results in CSV files for each video.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; **srts/**: Directory that stores the SRT files for each video.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; **processed_videos/**: Directory that stores the videos that have been split into frames.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; **processed_frames/**: Directory that stores the frames that have been processed by Mask-RCNN.


**/Mask-RCNN**: Directory of Mask-RCNN.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; **logs/**: Directory that stores the training logs (in tensorboard format) and Mask-RCNN's weights.

&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp; **samples/subtitle/**: Directory that stores the training logs (in tensorboard format) and Mask-RCNN's weights.

**docs/**: Presentation to the General Director.

## Files 

**mandarin_drama.txt**: Input file for __download_video.py__. Each row contains a drama (playlist) name.

__download_videos.py__ : Download videos with Youtube API.

__split_videos.py__ : Split videos with  FFMPEG.

__run_mask_rcnn.py__ : Run Mask-RCNN to remove everything in the images except the subtitles.

__ocr_to_csv.py__ : Detect text in frames with Google OCR API.

__csv_to_srt.py__ : Aggregate OCR results to SRT files.

__automatic_script.sh__ : Run the script to run through the whole pipeline.

__Dataset Generation Mask-RCNN .ipynb__ : Jupyter notebook for generating Mask-RCNN training dataset.


## Usage 

```
bash automatic_script.sh

```