import numpy as np
import pandas as pd
import os
import cv2


output_file = 'labelled_subtitle.csv'
input_path = 'manual_label_data/'


all_image_ids = os.listdir(input_path)

if not os.path.exists(output_file):
	with open(output_file, 'w') as f:
		f.write('id,subtitle\n')
else:
	annotated_image_ids = [f+'.png' for f in pd.read_csv(output_file).id.values]
	all_image_ids = list(set(all_image_ids) - set(annotated_image_ids))

print(len(all_image_ids))

for image_id in all_image_ids:
	image_path = input_path +  image_id
	img = cv2.imread(image_path)
	
	image_id = image_id.split('.png')[0]
	print(image_id)
	cv2.namedWindow(image_id)
	cv2.moveWindow(image_id, 400,5)
	cv2.imshow(f'{image_id}',img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	subtitle = input('')
	with open(output_file, 'a') as f:
		f.write(f'{image_id},{subtitle}\n')