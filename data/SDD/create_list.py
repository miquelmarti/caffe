import argparse
import os
from random import shuffle
import shutil
import subprocess
import sys
import itertools

import numpy as np

import cv2

HOMEDIR = os.path.expanduser("~")
DATASETS = os.environ['DATASETS']

# If true, re-create all list files.
redo = False
# The root directory which holds all information of the dataset.
data_dir = os.path.join(DATASETS, 'SDD')
os.chdir(data_dir)
# The directory name which holds the video sets.
vidset_dir = "videos"
# The direcotry which contains the images.
vid_name = "video"
vid_ext = ".mov"
# The directory name which holds the video sets.
anno_dir = "annotations"
# The directory which contains the annotations.
anno_name = "annotations"
anno_ext = ".txt"

# Dataset splits relative sizes
train_ratio = 0.6
val_ratio = 0.2

# Files where to save lists
train_list_file = 'train_small.txt'
val_list_file = 'val_small.txt'
test_list_file = 'test_small.txt'

# Extract frames from videos
video_gen = ((dirpath, filenames[0]) for dirpath,
 dirnames, filenames in os.walk(vidset_dir) if vid_name + vid_ext in filenames)
print "Extracting frames from videos"
for dirpath, filename in video_gen:
    frames_dir = os.path.join(dirpath, 'frames')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        print "Created frame dir: ", dirpath
    else:
        print "Skipping: frames folder for ", dirpath, " already exists!"
        continue
    vidcap = cv2.VideoCapture(os.path.join(dirpath, filename))
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
      success, image = vidcap.read()
      print 'Read frame: #', str(count), success
      if success:
          cv2.imwrite(os.path.join(frames_dir, str(count)+'.jpg'), image)
      count += 1

# Get annotation for each frame
anno_gen = ((dirpath, filenames[0]) for dirpath,
 dirnames, filenames in os.walk(anno_dir) if anno_name + anno_ext in filenames)
print "Extracting annotation for each frame"
for dirpath, filename in anno_gen:
    frames_dir = os.path.join(dirpath, 'frames')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    elif not redo:
        print "Skipping: frames folder for ", dirpath, " already exists!"
        continue
    else:
        print "Redoing annotations for ", dirpath
        pass

    bboxes = np.genfromtxt(os.path.join(dirpath, anno_name + anno_ext), dtype=[
    ('TrackID', int), ('xmin', int), ('ymin', int), ('xmax', int),
    ('ymax', int), ('frame', int), ('lost', int), ('occluded', int),
    ('generated', int), ('label', 'S15')])

    bboxes_sorted = np.sort(bboxes, order='frame')
    skip_frames = 100 # Use only each 10th frame
    bboxes_sorted = bboxes_sorted[bboxes_sorted['frame'] % skip_frames == 0] # Do not use the interpolated annotations
    bboxes_visible = bboxes_sorted[bboxes_sorted['lost']==0] # Do not use the tracks that are not visible
    bboxes_visible = bboxes_visible[bboxes_visible['occluded']==0] # Or occluded

    unique_frames = set(bboxes_visible['frame'])

    # Get image size (assume each video gives same sized frames)
    img = cv2.imread(os.path.join(dirpath.replace("annotations","videos"), 'frames', '0.jpg'))
    height, width, depth = img.shape

    for frame in sorted(unique_frames):
        # Create the annotation file for each frame in bboxes_visible
        with open(os.path.join(dirpath, 'frames', str(frame) + '.xml'), 'w') as f:
            print >>f, "<annotation>"
            print >>f, "<folder>", dirpath.replace("annotations","videos")+"/frames", "</folder>"
            print >>f, "<filename>", str(frame)+".jpg", "</filename>"
            print >>f, "<size>"
            print >>f, "<width>", str(width), "</width>"
            print >>f, "<height>", str(height), "</height>"
            print >>f, "<depth>", str(depth), "</depth>"
            print >>f, "</size>"
            for bbox in bboxes_visible[bboxes_visible['frame']==frame]:
                print >>f, "<object>"
                print >>f, "<name>"+bbox["label"].replace('''"''',"")+"</name>"
                print >>f, "<pose>Unspecified</pose>"
                print >>f, "<truncated>0</truncated>"
                print >>f, "<difficult>0</difficult>"
                print >>f, "<bndbox>"
                print >>f, "<xmin>", str(bbox["xmin"]), "</xmin>"
                print >>f, "<xmax>", str(bbox["xmax"]), "</xmax>"
                print >>f, "<ymin>", str(bbox["ymin"]), "</ymin>"
                print >>f, "<ymax>", str(bbox["ymax"]), "</ymax>"
                print >>f, "</bndbox>"
                print >>f, "</object>"
            print >>f, "</annotation>"

# Obtain list of frames
frames_list_vid = [[os.path.join(dirpath, f) for f in filenames] for dirpath,
 dirnames, filenames in os.walk(anno_dir) if 'frames' in dirpath]
frames_list = list(itertools.chain.from_iterable(frames_list_vid[::10]))
print "Total frames extracted: ", str(len(frames_list))

# Shuffle list and divide in splits
shuffle(frames_list)
train_split_size = int(round(train_ratio*len(frames_list)))
val_split_size = int(round(val_ratio*len(frames_list)))
train_split = frames_list[:train_split_size]
val_split = frames_list[train_split_size : train_split_size + val_split_size]
test_split = frames_list[train_split_size + val_split_size :]

print "Created 3 splits"
print "Train size: ", str(train_split_size)
print "Validation size: ", str(val_split_size)
print "Test size: ", str(len(frames_list) - train_split_size - val_split_size)

os.chdir(os.path.join(os.environ['CAFFE_ROOT'], 'data', 'SDD'))
# Save lists
with open(train_list_file, 'w') as f:
    f.writelines(map(lambda x: x.replace("annotations","videos").replace("xml","jpg")
    +' '+x+'\n', train_split))
with open(val_list_file, 'w') as f:
    f.writelines(map(lambda x:x.replace("annotations","videos").replace("xml","jpg")
    +' '+x+'\n', val_split))
with open(test_list_file, 'w') as f:
    f.writelines(map(lambda x: x.replace("annotations","videos").replace("xml","jpg")
    +' '+x+'\n', test_split))
