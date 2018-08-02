
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import os
import errno
import json
import pafy
from pprint import pprint
from pathlib import Path
import getpass
import datetime
from googleapiclient.discovery import build
import subprocess 
from joblib import Parallel, delayed
import argparse
import glob
# arguments to be passed to build function

DEVELOPER_KEY = 'AIzaSyCxMsbfq_AXhzIf40L5MvaZr2KEVV0lc6s'
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"



# In[56]:


# creating youtube resource object for interacting with api
youtube = build(YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION,
                developerKey=DEVELOPER_KEY)


def get_youtube_video_url(video_id):
    return "https://www.youtube.com/watch?v=" + video_id


playlist_url = 'https://www.youtube.com/watch?v=5rf8Z-b22Ac&list=PLtbJJuU1O8j_uECxOs1v2RIUFfLGEadHB'

def get_playlist_id(name):
    
    #search for the first playlist result given a drama name
    search_response = youtube.search().list(q=name,type="playlist",part="id",maxResults=1).execute()
    result = search_response.get("items", [])
    playlist_id = result[0]['id']['playlistId']
    return playlist_id


def get_video_ids(playlist_id):
    
    #search for all the videos given a playlist id
    search_response = youtube.playlistItems().list(part='contentDetails',maxResults=1,playlistId=playlist_id).execute()
    all_videos = search_response['items']
    video_ids = []
    for vid in all_videos:
        video_id = vid['contentDetails']['videoId']
        video_ids.append(video_id)
    return video_ids



def get_youtube_video_url(video_id):
    return "https://www.youtube.com/watch?v=" + video_id


def download_video(video_id, path="videos", verbose=True):
    try:
        # get video url
        video_url = get_youtube_video_url(video_id)

        try:
            video = pafy.new(video_url)
            # get best video format
            best = video.getbest(preftype="mp4")
            # download video
            best.download(filepath=path + "/" + video_id + "." + best.extension,
                          quiet=False)
            # log
            if verbose == True:
                print("- {id} video downloaded.".format(id=video_id))

            return True
        except Exception as e:
            print("- {exception}".format(exception=e))
            print("- {id} video cannot be downloaded.".format(id=video_id))
            return False
    except Exception as e:
        print('Failed to download video: {}'.format(e))
        return False



def download_audio(video_id, path="audios", verbose=True):
    try:
        # get video url
        video_url = get_youtube_video_url(video_id)

        try:
            video = pafy.new(video_url)
            # get best audio format
            bestaudio = None
            bestaudio = video.getbestaudio(preftype="ogg")
            if bestaudio == None:
                bestaudio = video.getbestaudio(preftype="m4a")
            # download audio
            bestaudio.download(filepath=path + "/" + video_id + "." + bestaudio.extension,
                               quiet=False)
            # log
            if verbose:
                print("- {id} audio downloaded.".format(id=video_id))

            return True
        except Exception as e:
            print("- {exception}".format(exception=e))
            print("- {id} audio cannot be downloaded.".format(id=video_id))
            return False
    except Exception as e:
        print('Failed to download audio: {}'.format(e))
        return False

def extract_audio(video_id,videos_dir, audios_dir):
    video_path = f'{videos_dir}/{video_id}.mp4'
    audio_path = f'{audios_dir}/{video_id}.mp3'
    
    #-i: it is the path to the input file. The second option -f mp3 tells ffmpeg that the ouput is in mp3 format. 
    #-ab 192000: we want the output to be encoded at 192Kbps 
    #-vn :we dont want video. 

    cmd = f'ffmpeg -i {video_path} -f mp3 -ab 192000 -vn -y {audio_path}'
    subprocess.call(cmd, shell=True)
    

def read_drama_names(drama_file):
    with open(drama_file, 'rb') as f:
        drama_list = [d.decode('utf-8').strip() for d in f.readlines()]
    return drama_list

def remove_intermediate_files(dir_):
    file_list = glob.glob(f'{dir_}/*temp*')
    [os.remove(f) for f in file_list]
    
def main():
    parser = argparse.ArgumentParser("Script for downloading youtube video")
    parser.add_argument("--thread_count", type=int, default=50)
    parser.add_argument("--drama_file", type=str, required=True)
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--audios_dir", type=str, required=True)
    args = parser.parse_args()
    
    
    os.makedirs(args.videos_dir, exist_ok=True)
    os.makedirs(args.audios_dir, exist_ok=True)
    
    remove_intermediate_files(args.videos_dir)
    remove_intermediate_files(args.audios_dir)
    dramas = read_drama_names(args.drama_file)
    try: 
        for drama in dramas:
            playlist_id = get_playlist_id(drama)
            video_ids = get_video_ids(playlist_id)
            video_ids = ['Ymnux9cBUVg']
            video_threads = min(args.thread_count//2, len(video_ids))
            audio_threads = min(args.thread_count, len(video_ids))
            # use multi-thread because downloading audios sometimes can be slow
            parallel = Parallel(video_threads, backend="threading", verbose=10)
            
            #download video
            parallel(delayed(download_video)(video_id, path=args.videos_dir) for video_id in video_ids)
            
            # use multi-thread because downloading audios sometimes can be slow
            parallel = Parallel(audio_threads, backend="threading", verbose=10)
            
            #download audio
#             parallel(delayed(download_audio)(video_id, path=args.audios_dir) for video_id in video_ids)

            # extract audio
            parallel(delayed(extract_audio)(video_id, args.videos_dir, args.audios_dir) for video_id in video_ids)
        
            
    except Exception as e:
        print('Failed to download videos: {}'.format(e))
        
          

if __name__ == "__main__":
    main()