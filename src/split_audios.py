
# coding: utf-8

# In[74]:


import os
import numpy
import subprocess
import re

# In[18]:


import json


# In[24]:


with open("srtAndFrames", 'rb') as f:
    lines = f.readlines()
lines


# In[17]:


json_lines = [json.loads(re.sub("'", '"', "{"+l.decode('utf-8').split("{")[-1].strip())) for l in lines]


# In[66]:

result_dir = "mandarin/split_audios/"
video_id = 'Awb_koyyc7o'
audio_dir = "mandarin/audios/"
os.makedirs(result_dir, exist_ok=True)
os.makedirs(result_dir +"/"+video_id, exist_ok=True)

for i, jl in enumerate(json_lines):
    start = int(jl["s"])
    end = int(jl["e"])
    
    mil_sec_start = start % 100
    start = start// 100
    sec_start = start % 60
    start = start //60
    min_start = start % 60
    start = start // 60
    hour_start = start% 60
    
    start_time = "%02d:%02d:%02d.%02d0" %(hour_start, min_start, sec_start, mil_sec_start)
    
    mil_sec_end = end % 100
    end = end// 100
    sec_end = end % 60
    end = end //60
    min_end = end % 60
    end = end // 60
    hour_end = end% 60
    
    
    end_time = "%02d:%02d:%02d.%02d0" %(hour_end, min_end, sec_end, mil_sec_end)
    cmd = f"ffmpeg -y -i {audio_dir}/{video_id}.mp3 -ss {start_time} -to {end_time} -acodec copy {result_dir}/{video_id}/{video_id}-{i:05d}.mp3".split(" ")

#     subprocess.run(cmd)
#     print(start_time +" ~ "+ end_time)
#     if i ==3:
#         exit()

#     print(jl["srt"])


# In[56]:


with open('mandarin/srts/Awb_koyyc7o.srt', 'rb') as f:
    times = []
    for index, l in enumerate(f.readlines()):
        if index % 4  == 1:
            times.append(l.decode('utf-8').strip())
    


# In[75]:


import os


# In[88]:



for i in range(len(times)):
    t = times[i]
    t = re.sub(",", ".", t)
#     print(t)
    
    start, end = t.split(" --> ")
    
#     print("before")
#     print(start, end)
    
    ms = int(start.split(".")[-1])
    s = int(start.split(".")[0].split(":")[-1])
    
    ms -= 200
    if ms < 0:
        ms +=1000
        s -= 1
    start = ":".join(start.split(":")[:2]) + ":"+ str(s) +"."+ str(ms)
    
    ms = int(end.split(".")[-1])
    s = int(end.split(".")[0].split(":")[-1])
    
    ms += 200
    if ms > 1000:
        ms -=1000
        s += 1
    end = ":".join(end.split(":")[:2]) + ":"+ str(s) +"."+ str(ms)
    
#     print("after")
#     print(start, end)
    cmd = f"ffmpeg -y -i {audio_dir}/{video_id}.mp3 -ss {start} -to {end} -acodec copy {result_dir}/{video_id}/{video_id}-{i:05d}.mp3".split(" ")

    subprocess.run(cmd)

