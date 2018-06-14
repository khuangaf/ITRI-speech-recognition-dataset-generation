import glob
import subprocess
from joblib import Parallel, delayed
import argparse

FRAMES_DIR = 'frames'

os.makedirs(FRAMES_DIR, exist_ok=True)

def video2frames(video_id, video_path = 'videos/', frame_path='frames/'):
    path_video = video_path + video_id + '.mp4'

    #sample_rate = 1 1/s
    try:
        cmd = f'ffmpeg -i {path_video} -r 1 {frame_path}/{video_id}-%07d.png'
        subprocess.call(cmd, shell=True)
    
    except Exception as e:
        print(f'Failed to cut videos {video_id}: {e}')
        


def path2id(path):
    return path.split('.')[0].split('/')[1]

def gather_video_ids(video_path = 'videos/'):
    video_ids = [ path2id(p) for p in glob.glob(f'{video_path}/*')]
    return video_ids

def main():
    parser = argparse.ArgumentParser("Script for downloading youtube video")
    parser.add_argument("--thread-count", type=int, default=3)
    args = parser.parse_args()
    
    # parallel processing to increase speed
    parallel = Parallel(args.thread_count, backend="threading", verbose=10)
    
    #get video_ids from videos/
    video_ids = gather_video_ids()
    try: 
        #split video into frames
        parallel(delayed(video2frames)(video_id) for video_id in video_ids)
    
    except Exception as e:
        print('Failed to download videos: {}'.format(e))
        
if __name__ == "__main__":
    main()    
