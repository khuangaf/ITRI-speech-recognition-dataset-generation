# Automatic Speech Recognition Dataset Creation

## Files & Directories

**src/**: Directory that stores all the code.

**mandarin_drama.txt**: File for Mandarin video inputs.

**taiwanese_drama.txt**: File for Taiwanese video inputs.

__crop_frames.py__ : Crop frames with openCV.

__download_videos.py__ : Download videos with Youtube API.

__frame_ocr.py__ : Detect text in frames with OCR.

__frame_ocr_tta.py__ : Detect text in TTAed framed with OCR.

__split_videos.py__ : Split videos with  FFMPEG.
