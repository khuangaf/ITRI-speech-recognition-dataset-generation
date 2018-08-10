def get_video_id_from_file(video_id_file):
    with open(video_id_file, 'rb') as f: 
        return [vid.decode('utf-8').strip() for vid in f.readlines()]