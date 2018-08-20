ROOT_DIR=/data/ITRI-create-speech-recognition-dataset/
MANDARIN_DIR=$ROOT_DIR/mandarin/
FRAMES_DIR=$MANDARIN_DIR/frames/
VIDEOS_DIR=$MANDARIN_DIR/videos/
AUDIOS_DIR=$MANDARIN_DIR/audios/
SRTS_DIR=$MANDARIN_DIR/srts
WEIGHTS_DIR=$ROOT_DIR/Mask_RCNN/logs/subtitle20180806T1527/mask_rcnn_subtitle_0089.h5
# WEIGHTS_DIR=Mask_RCNN/logs/subtitle20180731T0943/mask_rcnn_subtitle_0091.h5
# WEIGHTS_DIR=Mask_RCNN/logs/subtitle20180802T1555/mask_rcnn_subtitle_0082.h5 
DRAMA_FILE=mandarin_drama_list.txt
VIDEO_ID_FILE=video_ids.txt
SAMPLE_RATE=2
PROCESSED_FRAMES_DIR=$MANDARIN_DIR/processed_frames/
PROCESSED_VIDEOS_DIR=$MANDARIN_DIR/processed_videos/
MASKRCNN_RCNN_RESULTS_DIR=$MANDARIN_DIR/maskrcnn_results/
BGS_RESULTS_DIR=$MANDARIN_DIR/bgs_results/
OCR_RESULTS_DIR=$MANDARIN_DIR/ocr_results/


python src/download_videos.py --drama_file=$DRAMA_FILE  --audios_dir=$AUDIOS_DIR --videos_dir=$VIDEOS_DIR --video_id_file=$VIDEO_ID_FILE

python src/split_videos.py --videos_dir=$VIDEOS_DIR --sample_rate=$SAMPLE_RATE --frames_dir=$FRAMES_DIR --processed_videos_dir=$PROCESSED_VIDEOS_DIR --video_id_file=$VIDEO_ID_FILE


python src/run_mask_rcnn.py detect --dataset=$FRAMES_DIR --weights=$WEIGHTS_DIR --processed_frames_dir=$PROCESSED_FRAMES_DIR --subset=test --results_dir=$MASKRCNN_RCNN_RESULTS_DIR --video_id_file=$VIDEO_ID_FILE 


# don't use BGS
# python src/generate_BGS_filters.py --inputs_dir=$MASKRCNN_RCNN_RESULTS_DIR --results_dir=$BGS_RESULTS_DIR --video_id_file=$VIDEO_ID_FILE

python src/ocr_to_csv.py --inputs_dir=$MASKRCNN_RCNN_RESULTS_DIR --results_dir=$OCR_RESULTS_DIR --video_id_file=$VIDEO_ID_FILE

python src/csv_to_srt.py --srts_dir=$SRTS_DIR --csvs_dir=$OCR_RESULTS_DIR --video_id_file=$VIDEO_ID_FILE --videos_dir=$PROCESSED_VIDEOS_DIR