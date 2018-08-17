def get_video_id_from_file(video_id_file):
    """
    Extract video id from path 
    e.g. ADFBSDFG.mp4 --> ADFBSDFG

    Parameters
    ----------
    video_id_file : str
        The name of the file which stores all the video ids to be processed
    
    Returns
    --------
    list(dir)
        A list of video ids to be processed
    """
    
    with open(video_id_file, 'rb') as f: 
        return [vid.decode('utf-8').strip() for vid in f.readlines()]