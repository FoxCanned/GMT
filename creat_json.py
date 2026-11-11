import os
import numpy as np
import json
import cv2

DATA_PATH = ''
OUT_PATH = DATA_PATH + 'annotations/'

SPLITS = ['train', 'train_stage1', 'test']

SAVE_JSON = True

if __name__ == '__main__':

  if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH, exist_ok=True)

  for split in SPLITS:


    if split in ['train', 'train_stage1']:
      data_path = DATA_PATH + 'train'
      IMAGE_DIR = DATA_PATH + 'train/'
    else:
      data_path = DATA_PATH + 'test'
      IMAGE_DIR = DATA_PATH + 'test/'

    print('data_path', data_path)
    out_path = OUT_PATH + '{}.json'.format(split)

    out = {
      'images': [],
      'annotations': [],
      'categories': [{'id': 1, 'name': 'person'}],
      'videos': []
    }

    seqs = sorted(os.listdir(data_path))

    image_cnt = 0
    ann_cnt = 0
    video_cnt = 1


    total_id = 0
    max_id = 0

    for idx, seq in enumerate(seqs):

      seq_path = f'{data_path}/{seq}/'
      img_path = seq_path + 'img1/'
      ann_path = seq_path + 'gt/gt.txt'

      view = seq.split('_')[-1]
      view_num = int(view[-1])
      view_name = seq.split('_')[0]

      if idx == len(seqs) - 1 or view_name != seqs[idx+1].split('_')[0]:
        out['videos'].append({
          'id': video_cnt,
          'file_name': view_name,
          'view_num': view_num
        })

      images = sorted(os.listdir(img_path))
      num_images = len(images)

      # ---------- images ----------
      for i in range(num_images):
        image_info = {
          'file_name': f'{seq}/img1/{images[i]}',
          'id': image_cnt + i + 1,
          'frame_id': i + 1,
          'prev_image_id': image_cnt + i if i > 0 else -1,
          'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
          'video_id': video_cnt,
          'view_id': view_num
        }

        if SAVE_JSON:
          img = cv2.imread(IMAGE_DIR + image_info['file_name'])
          h, w = img.shape[:2]
          image_info['height'] = h
          image_info['width'] = w

        out['images'].append(image_info)

      print(f'{seq}: {num_images} images')

      # ---------- annotations ----------
      if os.path.exists(ann_path):
        anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')

        for i in range(anns.shape[0]):
          frame_id = int(anns[i][0])
          track_id = int(anns[i][1])

          ann_cnt += 1


          if split == 'train_stage1':
            instance_id = track_id + total_id
            max_id = max(max_id, track_id)
          else:
            instance_id = track_id

          ann = {
            'id': ann_cnt,
            'category_id': 1,
            'image_id': image_cnt + frame_id,
            'instance_id': instance_id,
            'bbox': anns[i][2:6].tolist(),
            'conf': float(anns[i][6]),
            'iscrowd': 0,
            'view_id': view_num
          }

          ann['area'] = ann['bbox'][2] * ann['bbox'][3]
          out['annotations'].append(ann)

      image_cnt += num_images


      if split == 'train_stage1':
        if idx == len(seqs) - 1 or view_name != seqs[idx+1].split('_')[0]:
          total_id += max_id
          max_id = 0

      if idx == len(seqs) - 1 or view_name != seqs[idx+1].split('_')[0]:
        video_cnt += 1

    print(f'loaded {split}: {len(out["images"])} images, {len(out["annotations"])} anns')

    if SAVE_JSON:
      json.dump(out, open(out_path, 'w'))