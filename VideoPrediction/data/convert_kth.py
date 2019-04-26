import os
import argparse
import glob
import subprocess

classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
frame_rate = 25

parser = argparse.ArgumentParser()
parser.add_argument('image_size', nargs='?', type=int, default=64, help='size of image')
parser.add_argument('data_root', nargs='?', type=str, default='./data/kth/', help='data root directory')

args = parser.parse_args()

if not os.path.exists(args.data_root):
    print('Error with data directory: %s' % args.data_root)

for single_class in classes:
    print(' ---- ')
    print(single_class)
    path = os.path.join(args.data_root, 'raw', single_class, '*.avi')
    print(path)
    for vid in glob.iglob(path):
        print(vid)
        fname = os.path.basename(vid[:-11])
        print(fname)
        dirpath = os.path.join(args.data_root, 'processed', single_class, fname)
        print(dirpath)
        subprocess.call('mkdir -p %s' % dirpath, shell=True)
        subprocess.call('ffmpeg -i %s -r %d -f image2 -s %dx%d  %s/image-%%03d_%dx%d.png' % \
                        (vid, frame_rate, args.image_size, args.image_size, dirpath, args.image_size, args.image_size), shell=True)

