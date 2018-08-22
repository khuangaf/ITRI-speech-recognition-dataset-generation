"""

This script allows the user to download videos from Youtube given a list of drama names

Author: Kung-hsiang, Huang, 2018
"""
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


# developer keys for Youtube V3 API
DEVELOPER_KEY = 'YOUR_API_KEY'
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"



# creating youtube resource object for interacting with api
youtube = build(YOUTUBE_API_SERVICE_NAME,
                YOUTUBE_API_VERSION,
                developerKey=DEVELOPER_KEY)



def get_youtube_video_url(video_id):
    
    """
    Append the video with the base Youtube URL.

    Parameters
    ----------
    video_id : str
        The video id of a video

    Returns
    -------
    url: str
        Concatenated video URL. 
    """
    url = "https://www.youtube.com/watch?v=" + video_id
    return url



def get_playlist_id(name):
    """
    Search for the playlist id given a drama name.

    Parameters
    ----------
    name : str
        The name of the drama

    Returns
    -------
    playlist_id: str
        Resulted playlist id returned by Youtube API
    """
    
    #search for the first playlist result given a drama name
    search_response = youtube.search().list(q=name,type="playlist",part="id",maxResults=1).execute()
    result = search_response.get("items", [])
    playlist_id = result[0]['id']['playlistId']
    return playlist_id


def get_video_ids(playlist_id):
    """
    Get the video ids given a playlist id.

    Parameters
    ----------
    playlist_id : str
        A Youtube playlist id. (up to 50 results)

    Returns
    -------
    video_ids: list(str)
        The video ids associated with the input playlist
    """
    
    #search for all the videos given a playlist id
    search_response = youtube.playlistItems().list(part='contentDetails',maxResults=50,playlistId=playlist_id).execute()
    all_videos = search_response['items']
    video_ids = []
    for vid in all_videos:
        video_id = vid['contentDetails']['videoId']
        video_ids.append(video_id)

    return video_ids


def download_video(video_id, path="videos", verbose=True):
    
    """
    Download the videos

    Parameters
    ----------
    video_id : str
        A Youtube video id. 
    path: str
        The directory which stores videos.
    verbose: bool
        Whether to log the intermediate results.
        
    Returns
    -------
    True/False
        Successfully downloaded the video or not.
    """
    
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





def extract_audio(video_id,videos_dir, audios_dir):
    """
    Download the videos

    Parameters
    ----------
    video_id : str
        A Youtube video id. 
    videos_dir: str
        The directory which stores videos.
    audios_dir: str
        The directory which stores audios.
        
    """
    
    video_path = f'{videos_dir}/{video_id}.mp4'
    audio_path = f'{audios_dir}/{video_id}.mp3'
    
    #-i: it is the path to the input file. The second option -f mp3 tells ffmpeg that the ouput is in mp3 format. 
    #-ab 192000: we want the output to be encoded at 192Kbps 
    #-vn :we dont want video. 

    # execute the command
    cmd = f'ffmpeg -i {video_path} -f mp3 -ab 192000 -vn -y {audio_path}'.split(" ")
    subprocess.call(cmd, shell=False)
    

def read_drama_names(drama_file):
    """
    Get drama names in the input file

    Parameters
    ----------
    drama_file : str
        The file which you read the drama names.
    
    Returns
    -------
    drama_list: list(str)
        The read drama list.
        
    """
    with open(drama_file, 'rb') as f:
        drama_list = [d.decode('utf-8').strip() for d in f.readlines()]
    return drama_list

def remove_intermediate_files(dir_):
    """
    Clean up all the intermediate files which contain "temp" in their file names.

    Parameters
    ----------
    dir_ : str
        The directory you are cleaning.
        
    """
    file_list = glob.glob(f'{dir_}/*temp*')
    [os.remove(f) for f in file_list]
    
def main():
    parser = argparse.ArgumentParser("Script for downloading youtube video")
    parser.add_argument("--thread_count", type=int, default=50)
    parser.add_argument("--drama_file", type=str, required=True)
    parser.add_argument("--video_id_file", type=str, required=True)
    parser.add_argument("--videos_dir", type=str, required=True)
    parser.add_argument("--audios_dir", type=str, required=True)
    args = parser.parse_args()
    
    
    os.makedirs(args.videos_dir, exist_ok=True)
    os.makedirs(args.audios_dir, exist_ok=True)
    
    try:
        os.remove(args.video_id_file)
    except:
        print("previous file does not exist")
        
    remove_intermediate_files(args.videos_dir)
    dramas = read_drama_names(args.drama_file)
    
    try: 
        for drama in dramas:
            
            playlist_id = get_playlist_id(drama)
            
            video_ids = get_video_ids(playlist_id)
            
            # write all the video id into a file
            with open(args.video_id_file, 'a') as f:
                for vid in video_ids:
                    f.write(f'{vid}\n')
                    
            video_threads = min(args.thread_count, len(video_ids))
            
            audio_threads = min(args.thread_count, len(video_ids))
            
            
            # multi-processing to speed up
            parallel = Parallel(video_threads, backend="threading", verbose=10)
            
            #download video
            parallel(delayed(download_video)(video_id, path=args.videos_dir) for video_id in video_ids)
            
            # multi-processing to speed up
            parallel = Parallel(audio_threads, backend="threading", verbose=10)

            # extract audio
            parallel(delayed(extract_audio)(video_id, args.videos_dir, args.audios_dir) for video_id in video_ids)
        
            
    except Exception as e:
        print('Failed to download videos: {}'.format(e))
        
          

if __name__ == "__main__":
    main()
